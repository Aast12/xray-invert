"""Input preparation utilities for the pipeline."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from torch import Tensor

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
    w0_override: dict | None = None,
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
        w0_override: Optional weight dict to use instead of default_weights(config).

    Returns:
        Tuple of (initial_txm, batched_weights).
    """
    # Initialize transmission maps
    init_mode, init_params = init.parse_init_params(config, images)
    txm0 = init.initialize(config.PRNGKey, images.shape, **init_params)

    # Broadcast weights to per-image shape
    w0 = w0_override if w0_override is not None else default_weights(config)
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


def prepare_torch_inputs(
    images_batch: Tensor,
    masks_batch: Tensor,
    priors: ValueRangeT,
    hyperparams,
    w0: dict,
    common_weights: bool = False,
    segmentation_th: float = 0.6,
) -> tuple[
    Float[Array, "batch height width"],
    Float[Array, "batch labels height width"],
    list[MaskGroupsT],
    Float[Array, "batch height width"],
    dict,
]:
    """Convert torch batch to JAX arrays and prepare optimization inputs.

    Handles the boilerplate duplicated across experiment runners:
    torch->numpy->JAX conversion, channel squeezing, mask processing,
    TXM initialization, and weight broadcasting.

    Args:
        images_batch: (B, C, H, W) torch tensor from dataloader.
        masks_batch: (B, labels, H, W) torch tensor of segmentation masks.
        priors: Value ranges for segmentation regions.
        hyperparams: Experiment hyperparameters (ExperimentArgs or similar).
        w0: Initial weight dict (scalar values).
        common_weights: If True, weights are shared (scalar); if False, per-image.
        segmentation_th: Threshold for binarizing masks.

    Returns:
        (images, segmentations, seg_labels, txm0, w0_batched)
    """
    # Convert masks to JAX and compute exclusive segmentation masks
    segmentations = jnp.array(masks_batch.cpu().numpy())
    seg_labels, segmentations = batch_get_exclusive_masks(segmentations, segmentation_th)

    # Convert images to JAX and squeeze channel dim if needed
    images = jnp.array(images_batch.cpu().numpy())
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0, :, :]
    elif images.ndim != 3:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    # Initialize transmission maps
    init_mode, init_params = init.parse_init_params(hyperparams, images)
    txm0 = init.initialize(hyperparams.PRNGKey, images.shape, **init_params)

    # Broadcast weights
    batch_shape = images.shape[0] if not common_weights else ()
    w0_batched = full_pytree(w0, batch_shape, jnp.float32)

    return images, segmentations, seg_labels, txm0, w0_batched
