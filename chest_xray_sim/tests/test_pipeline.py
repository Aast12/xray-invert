"""Essential tests for the pipeline module using synthetic data."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from chest_xray_sim.pipeline.config import PRESETS, PipelineConfig
from chest_xray_sim.pipeline.forward import forward_batch, forward_single
from chest_xray_sim.pipeline.loss import compute_loss
from chest_xray_sim.pipeline.projections import (
    build_projection_fn,
    default_weights,
)
from chest_xray_sim.pipeline.inputs import full_pytree, prepare_inputs
from chest_xray_sim.pipeline.run import PipelineResult


@pytest.fixture
def config():
    return PipelineConfig(n_steps=5, lr=0.01, windowing_type="sigmoid")


@pytest.fixture
def linear_config():
    return PipelineConfig(n_steps=5, lr=0.01, windowing_type="linear")


@pytest.fixture
def synthetic_batch():
    """Small synthetic batch for testing."""
    return jnp.ones((2, 32, 32)) * 0.5


@pytest.fixture
def synthetic_weights(config):
    w0 = default_weights(config)
    return full_pytree(w0, 2)


@pytest.fixture
def synthetic_segmentation():
    """Synthetic segmentation masks (2 images, 3 regions, 32x32)."""
    masks = jnp.zeros((2, 3, 32, 32), dtype=jnp.int8)
    masks = masks.at[:, 0, :16, :].set(1)  # lung region
    masks = masks.at[:, 1, 16:, :16].set(1)  # soft tissue region
    masks = masks.at[:, 2, 16:, 16:].set(1)  # bone region
    return masks


@pytest.fixture
def synthetic_priors():
    return jnp.array([[0.1, 0.6], [0.2, 0.7], [0.3, 0.8]])


def test_forward_single_output_range(config, synthetic_weights):
    txm = jnp.ones((32, 32)) * 0.5
    # Extract single-image weights
    w = {k: v[0] for k, v in synthetic_weights.items()}
    result = forward_single(txm, w, config)

    assert result.shape == (32, 32)
    assert jnp.all(result >= 0.0)
    assert jnp.all(result <= 1.0)


def test_forward_batch_matches_single(config, synthetic_batch, synthetic_weights):
    batch_result = forward_batch(synthetic_batch, synthetic_weights, config)

    assert batch_result.shape == (2, 32, 32)

    # Each image should match forward_single
    for i in range(2):
        w_i = {k: v[i] for k, v in synthetic_weights.items()}
        single_result = forward_single(synthetic_batch[i], w_i, config)
        assert jnp.allclose(batch_result[i], single_result, atol=1e-5)


def test_compute_loss_finite(config, synthetic_weights, synthetic_batch, synthetic_segmentation, synthetic_priors):
    w = {k: v[0] for k, v in synthetic_weights.items()}
    loss = compute_loss(
        w,
        synthetic_batch[0],
        synthetic_batch[0],
        config,
        synthetic_priors,
        synthetic_segmentation[0],
    )
    assert jnp.isfinite(loss)


def test_projection_enforces_bounds(config):
    proj_fn = build_projection_fn(config)
    # Create out-of-bounds values
    txm = jnp.full((2, 32, 32), 1.5)
    weights = {
        "enhance_factor": jnp.array([2.0, 2.0]),
        "high_enhance_factor": jnp.array([2.0, 2.0]),
        "window_center": jnp.array([0.9, 0.9]),
        "window_width": jnp.array([2.0, 2.0]),
        "gamma": jnp.array([25.0, 25.0]),
    }

    new_txm, new_weights = proj_fn(txm, weights)

    # TXM should be clipped to [0, 1]
    assert jnp.all(new_txm <= 1.0)
    assert jnp.all(new_txm >= 0.0)

    # Weights should be within spec bounds
    assert jnp.all(new_weights["enhance_factor"] <= config.max_enhancement)
    assert jnp.all(new_weights["window_center"] <= 0.5)
    assert jnp.all(new_weights["window_width"] <= 1.0)
    assert jnp.all(new_weights["gamma"] <= 20.0)


def test_default_weights_keys(config, linear_config):
    sigmoid_w = default_weights(config)
    assert "enhance_factor" in sigmoid_w
    assert "high_enhance_factor" in sigmoid_w
    assert "window_center" in sigmoid_w
    assert "window_width" in sigmoid_w
    assert "gamma" in sigmoid_w

    linear_w = default_weights(linear_config)
    assert "gamma" not in linear_w
    assert "enhance_factor" in linear_w


def test_pipeline_config_presets_exist():
    assert "dual_band" in PRESETS
    assert "dual_band_constant" in PRESETS
    assert isinstance(PRESETS["dual_band"], PipelineConfig)
    assert isinstance(PRESETS["dual_band_constant"], PipelineConfig)


def test_pipeline_result_save_load_roundtrip():
    result = PipelineResult(
        txm=jnp.ones((2, 32, 32)),
        weights={"enhance_factor": jnp.array([0.5, 0.5])},
        predictions=jnp.ones((2, 32, 32)) * 0.8,
        losses=[1.0, 0.5, 0.1],
        eval_metrics=None,
        config=PipelineConfig(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_result.joblib"
        result.save(path)
        loaded = PipelineResult.load(path)

        assert jnp.allclose(loaded.txm, result.txm)
        assert jnp.allclose(
            loaded.weights["enhance_factor"],
            result.weights["enhance_factor"],
        )
        assert jnp.allclose(loaded.predictions, result.predictions)
        assert loaded.losses == result.losses
        assert loaded.config.n_steps == result.config.n_steps


def test_prepare_inputs_shapes(config, synthetic_batch, synthetic_segmentation, synthetic_priors):
    prior_labels = ["lung", "soft", "bone"]
    txm0, w0 = prepare_inputs(
        synthetic_batch,
        synthetic_segmentation,
        synthetic_priors,
        prior_labels,  # type: ignore
        config,
    )

    assert txm0.shape == synthetic_batch.shape
    # Weights should be batched to match batch size
    for v in w0.values():
        assert v.shape[0] == synthetic_batch.shape[0]
