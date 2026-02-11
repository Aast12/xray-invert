"""Loss computation for the inverse pipeline."""

import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from chest_xray_sim.inverse import metrics
from chest_xray_sim.types import ValueRangeT
from .config import PipelineConfig
from .forward import forward_single


def compute_loss(
    weights: PyTree,
    txm: Float[Array, "rows cols"],
    target: Float[Array, "rows cols"],
    config: PipelineConfig,
    priors: ValueRangeT,
    segmentation: Float[Array, "labels rows cols"] | None = None,
) -> Float[Array, ""]:
    """Compute loss for a single image including forward pass.

    Combines data fidelity, smoothness regularization, and segmentation penalty.
    """
    pred = forward_single(txm, weights, config)

    # Data fidelity
    data_loss = jnp.float32(0.0)
    if config.data_fidelity == "l2":
        data_loss = 0.5 * jnp.mean((pred - target) ** 2)
    elif config.data_fidelity == "ssim":
        data_loss = metrics.ms_ssim_l1(pred, target)

    # Smoothness regularization
    tv = jnp.float32(0.0)
    if config.smooth_metric == "tikonov":
        tv = metrics.tikhonov(txm, reduction="mean")
    elif config.smooth_metric == "tv2":
        tv = metrics.total_generalized_variation(txm, reduction="mean")
    elif config.smooth_metric == "tv":
        tv = metrics.total_variation(txm[jnp.newaxis, ...], reduction="mean")

    # Segmentation penalty
    seg_penalty = jnp.float32(0.0)
    if segmentation is not None:
        seg_penalty = metrics.batch_segmentation_sq_penalty(
            txm[jnp.newaxis, ...], segmentation[jnp.newaxis, ...], priors
        ).sum()

    return data_loss + config.total_variation * tv + config.prior_weight * seg_penalty
