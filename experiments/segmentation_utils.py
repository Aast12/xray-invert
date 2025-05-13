import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from utils import (
    map_range,
)

from chest_xray_sim.data.segmentation import (
    MASK_GROUPS,
    ChestSegmentation,
    MaskGroupsT,
    batch_get_exclusive_masks,
)
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.types import ValueRangeT

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
            min_values,
            min_masked_values,
            max_masked_values,
            0.0,
            collimated_area_bound,
        )
        max_values = map_range(
            max_values,
            min_masked_values,
            max_masked_values,
            0.0,
            collimated_area_bound,
        )

        jax.debug.print(
            "region_id: {region_id} \n min: {min_values} \n max: {max_values} \n\n",
            region_id=region_id,
            min_values=min_values,
            max_values=max_values,
        )

        region_ranges[region_id] = jnp.array(
            [min_values.mean(), max_values.mean()]
        )

    jax.debug.print(
        "region_ranges: {region_ranges}", region_ranges=region_ranges
    )

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
