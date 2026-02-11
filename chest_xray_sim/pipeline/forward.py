"""Forward model for the dual-band pipeline."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import chest_xray_sim.inverse.operators as ops
from .config import PipelineConfig


def forward_single(
    txm: Float[Array, "rows cols"],
    weights: PyTree,
    config: PipelineConfig,
) -> Float[Array, "rows cols"]:
    """Single-image dual-band forward pipeline.

    Pipeline: window -> negative_log -> unsharp(high) -> clip -> unsharp(low) -> clip
    """
    x = ops.window(
        txm,
        weights["window_center"],
        weights["window_width"],
        weights.get("gamma", 0.0),
        config.windowing_type,
    )
    x = ops.negative_log(x)
    x = ops.unsharp_masking(x, config.high_sigma, weights["high_enhance_factor"])
    x = ops.clipping(x)
    x = ops.unsharp_masking(x, config.sigma, weights["enhance_factor"])
    x = ops.clipping(x)
    return x


def forward_batch(
    txm: Float[Array, "batch rows cols"],
    weights: PyTree,
    config: PipelineConfig,
) -> Float[Array, "batch rows cols"]:
    """Batch forward with per-image weights via vmap."""
    return jax.vmap(forward_single, in_axes=(0, 0, None))(txm, weights, config)
