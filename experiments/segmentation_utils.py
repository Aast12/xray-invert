import numpy as np
import jax
import jax.numpy as jnp
import torch
from torch import Tensor
from jaxtyping import Array, Float, PyTree
from typing import TypedDict, Callable, Any

from chest_xray_sim.data.segmentation import (
    MASK_GROUPS,
    batch_get_exclusive_masks,
    MaskGroupsT,
)
from utils import (
    map_range,
)
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.data.segmentation import ChestSegmentation

DEBUG = True
DTYPE = jnp.float32

# TODO: factor out fixed paths
# Paths for real transmission maps and their corresponding processed images paths (used to feed segmentation model)
REAL_TM_PATHS = [
"/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
    # "/Volumes/T7/projs/thesis/data/conventional_transmissionmap.tif",
]
FWD_REAL_TM_PATHS = [
    "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
    # "/Volumes/T7/projs/thesis/data/conventional_transmissionmap.tif",
]


# TODO: duplicate definition segmentation.py
# TODO: some usages might expect different dimension sizes for mask_groups,
# some parts use the entire segmentation labels, others a reduced amount
TransmissionMapT = Float[Array, "batch height width"]
ValueRangeT = Float[Array, "mask_groups 2"]
SegmentationT = Float[Array, "batch mask_groups height width"]


def extract_region_value_ranges(
    images: Array,
    segmentation: Array,
    regions: list[MaskGroupsT] | MaskGroupsT,
    threshold: float = 0.6,
    collimated_area_bound: float = 0.8,
) -> tuple[list[MaskGroupsT], ValueRangeT]:
    """
    Extracts the value ranges for each region in the segmentation.

    Different transmission maps (maybe due to dose or different BMIs) will have different ranges
    in the collimated area (largest consecutive value area in the histogram). A way to normalize it is
    to take the range from 0 to the max value found in the lungs, that way we can stretch the transmission
    range to a common value, say 60% transmission.

    Args:
        images: The input images.
        segmentation: The segmentation mask.
        regions: The regions to extract the value ranges for.
        threshold: The threshold for the segmentation mask.
        collimated_area_bound: The bound for the collimated area.

    Returns:
        The region labels and their value ranges.
    """
    if isinstance(regions, str):
        regions = [regions]

    region_ranges = {}

    exclusive_mask_labels, exclusive_masks = batch_get_exclusive_masks(
        segmentation, threshold
    )
    merged_masks = exclusive_masks.sum(axis=1)

    # TODO: assumes hard masks (not-None threshold)
    masked_values = images * merged_masks
    min_masked_values = masked_values.min(axis=(1, 2))
    max_masked_values = masked_values.max(axis=(1, 2))

    for region_id in regions:
        mask = exclusive_masks[:, exclusive_mask_labels.index(region_id)]

        min_values = images.min(axis=(1, 2), where=mask > 0, initial=jnp.inf)
        max_values = images.max(axis=(1, 2), where=mask > 0, initial=0.0)

        # map the values to the range of the merged masks
        # TODO: double check this
        min_values = map_range(
            min_values, min_masked_values, max_masked_values, 0.0, collimated_area_bound
        )
        max_values = map_range(
            max_values, min_masked_values, max_masked_values, 0.0, collimated_area_bound
        )

        jax.debug.print(
            "region_id: {region_id} \n min: {min_values} \n max: {max_values} \n\n",
            region_id=region_id,
            min_values=min_values,
            max_values=max_values,
        )

        region_ranges[region_id] = jnp.array([min_values.mean(), max_values.mean()])

    jax.debug.print("region_ranges: {region_ranges}", region_ranges=region_ranges)

    return (regions, jnp.stack(list(region_ranges.values())))



def get_priors(
    segmentation_cache_dir: str,
    as_numpy: bool = False,
    collimated_region_bound: float = 0.4,
    real_tm_paths=REAL_TM_PATHS,
    fwd_img_paths=FWD_REAL_TM_PATHS,
) -> tuple[list[MaskGroupsT], ValueRangeT]:
    """Get priors for segmentation model.

    Args:
        segmentation_cache_dir: Directory to cache segmentation model.
        as_numpy: Whether to return numpy arrays.
        collimated_region_bound: Bound for collimated region. This determines the maximum intensity value for the entire collimated region.

    Returns:
        Tuple of list of mask groups and value range.
    """
    segmentation_model = ChestSegmentation(cache_dir=segmentation_cache_dir)

    real_tm_data = torch.stack([read_image(path) for path in real_tm_paths])
    processed_data = torch.stack([read_image(path) for path in fwd_img_paths])

    # segmentation model was trained on processed images, segmentations have to be
    # pulled from applying a forward image processing to the real transmission maps
    real_tm_segmentations = segmentation_model(processed_data)
    real_tm_images = jnp.array(real_tm_data.cpu().numpy()).squeeze(1)
    real_tm_segmentations = jnp.array(real_tm_segmentations.cpu().numpy())

    # TODO: complete label forwarding
    seg_labels, value_ranges = extract_region_value_ranges(
        real_tm_images,
        real_tm_segmentations,
        list(MASK_GROUPS),
        collimated_area_bound=collimated_region_bound,
    )

    del segmentation_model

    if as_numpy:
        return seg_labels, np.array(value_ranges)

    return seg_labels, value_ranges

@jax.jit
def compute_single_mask_penalty(
    mask_id: int,
    mask: TransmissionMapT,
    value_range: Float[Array, " 2"],
    txm: TransmissionMapT,
    penalties: Float[Array, " reduced_labels"],
) -> Float[Array, " reduced_labels"]:
    min_val, max_val = value_range

    # region_values = txm[mask.astype(bool)]
    region_values = txm * mask
    region_size = jnp.sum(mask, axis=(-2, -1))

    below_min = jnp.where(mask > 0, jnp.maximum(0.0, min_val - region_values), 0.0) ** 2
    above_max = jnp.where(mask > 0, jnp.maximum(0.0, region_values - max_val), 0.0) ** 2

    # below_min = jnp.maximum(0.0, min_val - region_values) ** 2
    # above_max = jnp.maximum(0.0, region_values - max_val) ** 2

    region_penalty = jnp.mean(
        (below_min + above_max) / (jnp.expand_dims(region_size, (1, 2)) + 1e6)
    )
    return penalties.at[mask_id].set(region_penalty)


@jax.jit
def segmentation_sq_penalty(
    txm: Float[Array, "batch height width"],
    segmentation: Float[Array, " batch reduced_labels height width"],
    value_ranges: Float[Array, " reduced_labels 2"],
):
    """
    Assumes batch arrays
    """
    penalties = jnp.ones(value_ranges.shape[0])

    # TODO: possibly improve by making broadcast operations
    for mask_id, val_range in enumerate(value_ranges):
        penalties = compute_single_mask_penalty(
            mask_id, segmentation[:, mask_id], val_range, txm, penalties
        )

    return jnp.sum(penalties)

def detached_segmentation_sq_penalty(
    txm: TransmissionMapT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
):
    penalties = jnp.ones((value_ranges.shape[0], txm.shape[0]))

    jax.debug.print("penalties shape: {x}", x=penalties.shape)
    # TODO: possibly improve by making broadcast operations
    for mask_id, val_range in enumerate(value_ranges):
        mask = segmentation[:, mask_id]
        min_val, max_val = val_range

        region_values = txm * mask
        region_size = jnp.sum(mask, axis=(-2, -1))

        jax.debug.print("region size: {x}", x=region_size.shape)

        below_min = jnp.maximum(0.0, min_val - region_values) ** 2
        above_max = jnp.maximum(0.0, region_values - max_val) ** 2

        region_penalty = (below_min + above_max) / (
            jnp.expand_dims(region_size, (1, 2)) + 1e6
        )
        jax.debug.print("region penalty: {x}", x=region_penalty.shape)
        penalties = penalties.at[mask_id].set(jnp.sum(region_penalty, axis=(-2, -1)))

    jax.debug.print("penalties shape: {x}", x=penalties.shape)

    return penalties
