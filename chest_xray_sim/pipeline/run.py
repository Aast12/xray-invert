"""High-level pipeline runner and result container."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import joblib
from jaxtyping import Array, Float

from chest_xray_sim.inverse.eval import EvalMetrics, batch_evaluation
from chest_xray_sim.segmentation.model import MaskGroupsT
from chest_xray_sim.types import ValueRangeT
from .config import PRESETS, PipelineConfig
from .experiment import InversePipelineExperiment
from .forward import forward_batch
from .inputs import prepare_inputs


@dataclass
class PipelineResult:
    """Container for pipeline outputs with save/load support."""

    txm: Float[Array, "batch height width"]
    weights: dict
    predictions: Float[Array, "batch height width"]
    losses: list[float]
    eval_metrics: EvalMetrics | None
    config: PipelineConfig

    def save(self, path: str | Path) -> None:
        """Serialize result to disk."""
        joblib.dump(
            {
                "txm": self.txm,
                "weights": self.weights,
                "predictions": self.predictions,
                "losses": self.losses,
                "eval_metrics": self.eval_metrics,
                "config": self.config,
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> "PipelineResult":
        """Deserialize result from disk."""
        data = joblib.load(path)
        return PipelineResult(**data)


def run_pipeline(
    images: Float[Array, "batch height width"],
    segmentations: Float[Array, "batch reduced_labels height width"],
    priors: ValueRangeT,
    prior_labels: list[MaskGroupsT],
    config: PipelineConfig | None = None,
    preset: str = "dual_band",
) -> PipelineResult:
    """Run the full inverse pipeline.

    Args:
        images: Target processed images (B, H, W).
        segmentations: Exclusive segmentation masks (B, reduced_labels, H, W).
        priors: Value ranges per segmentation region.
        prior_labels: Labels for segmentation regions.
        config: Explicit pipeline config. If None, uses preset.
        preset: Name of preset config to use (ignored if config is provided).

    Returns:
        PipelineResult with recovered transmission maps, weights, predictions,
        losses, and evaluation metrics.
    """
    if config is None:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}"
            )
        config = PRESETS[preset]

    # Prepare inputs
    txm0, w0 = prepare_inputs(
        images, segmentations, priors, prior_labels, config
    )

    # Create and run experiment
    experiment = InversePipelineExperiment(config=config, priors=priors)
    result, losses = experiment.optimize(
        target=images,
        txm0=txm0,
        w0=w0,
        segmentation=segmentations,
    )

    if result is None:
        raise RuntimeError("Optimization failed to converge")

    txm, weights = result

    # Compute predictions
    predictions = forward_batch(txm, weights, config)

    # Evaluate
    eval_metrics = batch_evaluation(
        images, txm, predictions, segmentations, priors
    )

    return PipelineResult(
        txm=txm,
        weights=weights,
        predictions=predictions,
        losses=losses,
        eval_metrics=eval_metrics,
        config=config,
    )
