import jax.numpy as jnp
from jaxtyping import Array
from chest_xray_sim.data.segmentation import (
    MASK_GROUPS,
    MaskGroupsT,
)
import matplotlib.pyplot as plt

from utils import map_range


def visualize_segmentation(
    images: Array,
    merged_masks: Array,
    exclusive_masks: Array,
    regions: list[MaskGroupsT] = list(MASK_GROUPS),
):
    _, ax = plt.subplots(images.shape[0], len(regions) + 2, figsize=(12, 6))
    last = len(regions) + 1

    # get collimated area by getting value range in segmented areas
    max_masked_values = jnp.max(images * merged_masks, axis=(1, 2))
    min_masked_values = jnp.min(images * merged_masks, axis=(1, 2))

    for j, region_id in enumerate(regions):
        mask = exclusive_masks[:, j]  # TODO: assumes full list MASK_GROUPS

        region_values = images * mask
        for i, mv in enumerate(region_values):
            if j == 0:
                ax[i, 0].imshow(images[i], cmap="gray")

            ax[i, j + 1].imshow(mv, cmap="jet", vmin=0, vmax=1)
            ax[i, 0].axis("off")
            ax[i, j + 1].axis("off")
            mm = mv[mv > 0]
            ax[i, last].hist(mm.flatten(), bins=50, alpha=0.3, label=region_id)
            if j == 0:
                ax[i, last].hist(images[i].flatten(), bins=50, alpha=0.3, label="total")

            ax[i, last].axvline(
                min_masked_values[i].item(), color="red", linestyle="--"
            )
            ax[i, last].axvline(
                max_masked_values[i].item(), color="red", linestyle="--"
            )

            ax[i, last].legend()

    plt.legend()
    plt.tight_layout()

    plt.show()


def visualize_segementation_ranges(
    images: Array,
    merged_masks: Array,
    exclusive_masks: dict[str, Array],
    region_ranges: dict[MaskGroupsT, tuple[float, float]],
    regions: list[MaskGroupsT] = list(MASK_GROUPS),
    collimated_area_bound=0.8,
):
    max_masked_values = jnp.max(images * merged_masks, axis=(1, 2))
    min_masked_values = jnp.min(images * merged_masks, axis=(1, 2))

    _, ax = plt.subplots(len(images), 2, figsize=(12, 6))

    colors: dict[MaskGroupsT, str] = {
        "bone": "blue",
        "lung": "green",
        "soft": "red",
    }

    for i, image in enumerate(images):
        ax[i, 0].imshow(image, cmap="gray")

        target_values = image * merged_masks[i]
        target_values = map_range(
            target_values,
            min_masked_values[i],
            max_masked_values[i],
            0.0,
            collimated_area_bound,
        )

        ax[i, 1].hist(
            target_values.flatten(),
            bins=50,
            alpha=0.3,
            label="total",
            color="gray",
        )

        for region_id in regions:
            min_val, max_val = region_ranges[region_id]
            print("region:", region_id, min_val, max_val)
            region_values = target_values * exclusive_masks[region_id]
            ax[i, 1].hist(
                region_values[region_values > 0].flatten(),
                bins=50,
                alpha=0.3,
                label=region_id,
                color=colors.get(region_id, "gray"),
            )
            ax[i, 1].axvline(
                min_val, color=colors.get(region_id, "yellow"), linestyle="--"
            )
            ax[i, 1].axvline(
                max_val, color=colors.get(region_id, "yellow"), linestyle="--"
            )
            ax[i, 1].legend()

    plt.legend()
    plt.tight_layout()
    plt.show()
