"""Input preparation utilities for the pipeline."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import chest_xray_sim.utils.initialization as init
from chest_xray_sim.segmentation.model import MaskGroupsT
from chest_xray_sim.segmentation.utils import batch_get_exclusive_masks
from chest_xray_sim.types import ValueRangeT
from .config import PipelineConfig
from .projections import default_weights


def full_pytree(w0: dict, batch_size: int | tuple, dtype=jnp.float32) -> dict:
    """Broadcast scalar weights to per-image shape."""
    return jax.tree.map(lambda v: jnp.full(batch_size, v, dtype=dtype), w0)


def prepare_inputs(
    images: Float[Array, "batch height width"],
    masks: Float[Array, "batch labels height width"],
    priors: ValueRangeT,
    prior_labels: list[MaskGroupsT],
    config: PipelineConfig,
    mask_threshold: float = 0.6,
) -> tuple[Float[Array, "batch height width"], dict]:
    """Prepare optimization inputs from images and masks.

    Converts raw masks to exclusive segmentation masks, initializes
    transmission maps and broadcasts weights to per-image shape.

    Args:
        images: Batch of images (B, H, W).
        masks: Raw segmentation masks (B, labels, H, W).
        priors: Value ranges for segmentation regions.
        prior_labels: Labels for the segmentation regions.
        config: Pipeline configuration.
        mask_threshold: Threshold for binarizing masks.

    Returns:
        Tuple of (initial_txm, batched_weights).
    """
    # Initialize transmission maps
    init_mode, init_params = init.parse_init_params(config, images)
    txm0 = init.initialize(config.PRNGKey, images.shape, **init_params)

    # Broadcast weights to per-image shape
    w0 = default_weights(config)
    w0_batched = full_pytree(w0, images.shape[0])

    return txm0, w0_batched


def prepare_segmentation(
    masks: Float[Array, "batch labels height width"],
    threshold: float = 0.6,
) -> tuple[list[MaskGroupsT], Float[Array, "batch reduced_labels height width"]]:
    """Convert raw segmentation masks to exclusive masks.

    Args:
        masks: Raw segmentation output (B, num_classes, H, W).
        threshold: Threshold for binarizing masks.

    Returns:
        Tuple of (mask group labels, exclusive masks).
    """
    return batch_get_exclusive_masks(masks, threshold)
