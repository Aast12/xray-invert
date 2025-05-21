import os

import jax.numpy as jnp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Array, Float
from matplotlib.colors import ListedColormap

from chest_xray_sim.data.segmentation import (
    ChestSegmentation,
    MaskGroupsT,
    batch_get_exclusive_masks,
)
from chest_xray_sim.data.segmentation_dataset import (
    get_segmentation_dataset,
    iter_as_numpy,
)
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.inverse import operators as ops

# data loading vars
model_path = "/Volumes/T7/datasets/torchxrayvision"
masks_path = "/Volumes/T7/datasets/chexpert_plus/masks/"
data_dir = "/Volumes/T7/datasets/chexpert_plus/PNG/PNG/"
meta_dir = "/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv"

mask_threshold = 0.5
curr_file_dir = os.path.dirname(os.path.abspath(__file__))


transmission_map_paths = [
    (
        "/Volumes/T7/projs/thesis/data/conventional_transmissionmap_8bit.tif",
        None,
    ),
    (
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
    ),
]


def f(filename):
    path = os.path.join(curr_file_dir, "build", filename)
    print(f"saving {filename} at {path}")
    return path


def copy_builds(src_dir, dst_dir):
    build_dir = os.path.join(src_dir, "build")
    os.system(f"cp {build_dir}/* ~/projs/master-thesis/figures/")
    os.system(f"cp -r {build_dir}/* {dst_dir}")


example_weights = {
    "window_center": 0.2,
    "window_width": 0.5,
    "gamma": 5.0,
    "low_sigma": 3.5,
    "low_enhance_factor": 0.1,
}

# Alternative windowing parameters
linear_windowing_params = {
    "mild_linear": {
        "window_center": 0.5,
        "window_width": 0.7,
        "gamma": 0.5,
    },  # Ends around 0.8
    "very_linear": {
        "window_center": 0.5,
        "window_width": 50.0,
        "gamma": 0.5,
    },  # Nearly linear, ends around 0.95
    "compressed": {
        "window_center": 0.3,
        "window_width": 4.0,
        "gamma": 0.4,
    },  # Compressed linear, ends around 0.7
    "shadow_enhance": {
        "window_center": 0.0,
        "window_width": 0.8,
        "gamma": 0.2,
    },  # Boosts shadows significantly
}


def forward(image, weights):
    """Forward processing function that converts transmission maps to processed X-rays"""
    x = ops.negative_log(image)
    x = ops.window(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)
    x = ops.unsharp_masking(
        x, weights["low_sigma"], weights["low_enhance_factor"]
    )
    x = ops.clipping(x)

    return x


memory = joblib.Memory(os.path.join(curr_file_dir, "cache"))


@memory.cache
def get_tm_masks(
    real_tm_paths, fwd_img_paths, threshold=0.5
) -> tuple[list[MaskGroupsT], Float[Array, "batch rows cols"]]:
    segmentation_model = ChestSegmentation(cache_dir=model_path)

    real_tm_data = torch.stack([read_image(path) for path in real_tm_paths])
    processed_data = torch.stack([read_image(path) for path in fwd_img_paths])

    real_tm_segmentations = segmentation_model(processed_data)
    real_tm_images = jnp.array(real_tm_data.cpu().numpy()).squeeze(1)
    real_tm_segmentations = jnp.array(real_tm_segmentations.cpu().numpy())

    exclusive_mask_labels, exclusive_masks = batch_get_exclusive_masks(
        real_tm_segmentations, threshold
    )

    del segmentation_model

    fig, ax = plt.subplots(len(exclusive_mask_labels) + 1)
    ax[0].imshow(processed_data[0, 0], cmap="gray")
    for i in range(len(exclusive_mask_labels)):
        a = ax[i + 1].imshow(exclusive_masks[0, i], cmap="jet")
        fig.colorbar(a)

    return exclusive_mask_labels, exclusive_masks


@memory.cache
def load_transmission_maps(paths):
    maps = []
    for path, proc_path in paths:
        raw_tm = read_image(path).squeeze(0)
        maps.append(
            (
                raw_tm,
                read_image(proc_path).squeeze(0) if proc_path else None,
            )
        )
    return [
        (
            jnp.array(map0.cpu().numpy()),
            jnp.array(map1.cpu().numpy())
            if map1 is not None
            else forward(jnp.array(map0.cpu().numpy()), example_weights),
        )
        for map0, map1 in maps
    ]


@memory.cache
def load_example_images(size=4):
    ds = get_segmentation_dataset(
        data_dir=data_dir,
        meta_dir=meta_dir,
        mask_dir=masks_path,
        cache_dir=model_path,
        split="train",
        frontal_lateral="Frontal",
        batch_size=size,
    )

    ds = iter_as_numpy(ds)
    images, masks, meta = next(iter(ds))

    labels, exclusive_masks = batch_get_exclusive_masks(
        jnp.array(masks), mask_threshold
    )
    exclusive_masks = exclusive_masks

    return (
        jnp.array(images.squeeze(1)),
        (masks, labels, exclusive_masks),
        meta,
    )


def vis_overlay_mask_example(labels, img, mask, path="segmentation_masks.png"):
    # assign matplotlib colors to labels
    color_choices = [
        "red",
        "green",
        "blue",
    ]

    label_colors = [(label, color_choices[i]) for i, label in enumerate(labels)]

    fig, ax = plt.subplots(figsize=(8, 8))
    # remove ticks
    ax.imshow(img, interpolation="none", cmap="gray")
    for i, (label, color) in enumerate(label_colors):
        ax.imshow(
            np.ma.masked_where(mask[i] == 0, np.ones_like(mask[i])),
            interpolation="none",
            alpha=0.5 * (mask[i] > 0),
            cmap=ListedColormap([color]),
            vmin=0,
            vmax=1,
        )
        ax.plot([], color=color, label=label, linewidth=2)

    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    fig.tight_layout()

    fig.savefig(f("segmentation_masks.png"))


def plot_real_transmission_maps(maps):
    fig, ax = plt.subplots(len(maps), 2, figsize=(16, 8 * len(maps)))

    for i, (map, proc) in enumerate(maps):
        ax[i, 0].imshow(map, cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(proc, cmap="gray")
        ax[i, 1].axis("off")

    fig.tight_layout()
    fig.savefig(f("transmission_maps.png"))


def plot_negative_log(image):
    values = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].set_title("Negative Log LUT")
    ax[0].plot(values, ops.negative_log(values))
    ax[0].set_xlabel("Pixel value")
    ax[0].set_ylabel("Normalized Negative log")

    ax[1].imshow(ops.negative_log(image), cmap="gray")
    ax[1].set_title("Negative Log Image")
    ax[1].axis("off")

    fig.tight_layout()
    fig.savefig(f("negative_log_lut.png"))


def plot_txm_histograms(maps):
    n = len(maps)
    fig, ax = plt.subplots(n, 2, figsize=(16, 8 * n))
    for i, (map_, proc_) in enumerate(maps):
        ax[i, 0].imshow(map_, cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1].hist(map_.flatten(), bins=70)
        ax[i, 1].set_xlim(0, 1)
        ax[i, 1].set_xlabel("Pixel value")
        ax[i, 1].set_ylabel("Frequency")

    fig.savefig(f("transmission_maps_histograms.png"))


def plot_window_function():
    values = np.linspace(0, 1, 500)
    window_centers = [0.2, 0.5, 0.7]
    window_widths = [0.2, 0.3, 0.5]
    gammas = [1, 4, 10]
    colors = ["red", "green", "blue"]

    fig, ax = plt.subplots(
        len(window_widths),
        len(window_centers),
        figsize=(16, 16),
        sharex="col",
        sharey="row",
        gridspec_kw={
            "left": 0.17
        },  # Add moderate space on the left for row titles
    )

    # Set column titles first
    for j, center in enumerate(window_centers):
        ax[0, j].set_title(f"Window Center: {center}")

    # Set row titles as separate text elements
    for i, width in enumerate(window_widths):
        # Calculate vertical position for row title (centered with the row)
        row_pos = ax[i, 0].get_position().y0 + (
            ax[i, 0].get_position().height / 2
        )
        fig.text(
            0.15,
            row_pos,
            f"Window Width: {width}",
            va="center",
            ha="right",
            fontsize=12,
            rotation="horizontal",
        )

        # Add vertical and horizontal lines to each subplot
        for j, center in enumerate(window_centers):
            ax[i, j].axvline(x=center - width / 2, color="gray", linestyle="--")
            ax[i, j].axvline(x=center + width / 2, color="gray", linestyle="--")
            ax[i, j].axvline(x=center, color="gray", linestyle="--")

            for color, gamma in zip(colors, gammas):
                ax[i, j].plot(
                    values,
                    ops.window(values, center, width, gamma),
                    label=f" Gamma: {gamma}",
                    color=color,
                )
                ax[i, j].set_ylim(0, 1)

            ax[i, j].legend()

    fig.savefig(f("window_params.png"))


def plot_windowing_fwd_image(map, img_window_params):
    fig, ax = plt.subplots(2, 3, figsize=(16, 16))

    showim = ops.negative_log(map)
    proc_image = ops.window(showim, **img_window_params)
    values = jnp.linspace(showim.min(), showim.max(), 300)

    ax[0, 0].imshow(showim, cmap="gray")
    ax[0, 1].hist(showim.flatten(), bins=70)
    ax[0, 1].axvline(
        x=img_window_params["window_center"], color="gray", linestyle="--"
    )
    ax[0, 1].axvline(
        x=img_window_params["window_center"]
        + img_window_params["window_width"] / 2,
        color="gray",
        linestyle="--",
    )
    ax[0, 1].axvline(
        x=img_window_params["window_center"]
        - img_window_params["window_width"] / 2,
        color="gray",
        linestyle="--",
    )

    ax[0, 2].plot(
        values,
        ops.window(values, **img_window_params),
    )
    ax[0, 2].axvline(
        x=img_window_params["window_center"]
        + img_window_params["window_width"] / 2,
        color="gray",
        linestyle="--",
    )
    ax[0, 2].axvline(
        x=img_window_params["window_center"]
        - img_window_params["window_width"] / 2,
        color="gray",
        linestyle="--",
    )

    ax[1, 0].imshow(proc_image, cmap="gray")
    ax[1, 1].hist(proc_image.flatten(), bins=70)

    fig.savefig(f("processed_windowing_fwd_image.png"))


def plot_processed_tm_histogram(map0, img_window_params):
    proc_image = ops.window(map0, **img_window_params)

    fig, ax = plt.subplots(2, 2, figsize=(16, 16), sharey="col")
    ax[0, 0].set_title("Transmission Map")
    ax[0, 0].imshow(map1, cmap="gray", vmax=1, vmin=0)
    ax[0, 1].hist(map1.flatten(), bins=70)
    ax[0, 1].set_xlabel("Transmission")
    ax[0, 1].set_ylabel("Frequency")
    ax[0, 0].axis("off")

    ax[1, 0].set_title("Processed X-ray Image")
    ax[1, 0].imshow(proc1, cmap="gray", vmin=0, vmax=1)
    ax[1, 1].hist(proc1.flatten(), bins=70)
    ax[1, 1].set_xlabel("Pixel value")
    ax[1, 1].set_ylabel("Frequency")
    ax[1, 0].axis("off")

    fig.savefig(f("processed_hist_cmp.png"))


if __name__ == "__main__":
    images, (masks, labels, exclusive_masks), meta = load_example_images()

    img, mask = images[0], exclusive_masks[0]
    maps = load_transmission_maps(transmission_map_paths)
    real_tm_path, real_fwd_path = transmission_map_paths[1]
    tm_mask_labels, tm_masks = get_tm_masks([real_tm_path], [real_fwd_path])
    matching_tm, matching_mask = maps[1][0], tm_masks[0]

    map0, proc0 = maps[0]
    map1, proc1 = maps[1]

    plot_real_transmission_maps(maps)
    plot_txm_histograms(maps)

    values = np.linspace(0, 1, 500)

    proc0 = forward(jnp.array(map0), example_weights)

    plot_window_function()

    img_window_params = {
        "window_center": 0.2,
        "window_width": 0.5,
        "gamma": 5,
    }

    proc_image = ops.window(map0, **img_window_params)

    plot_negative_log(map0)
    plot_processed_tm_histogram(map0, img_window_params)
    plot_windowing_fwd_image(map0, img_window_params)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    ax[0].imshow(matching_tm, cmap="gray", vmax=1, vmin=0)
    ax[1].hist(matching_tm.flatten(), bins=70, alpha=0.3, label="full")
    colors = ["red", "green", "blue"]

    for i, label in enumerate(tm_mask_labels):
        color = colors[i]
        mask_values = matching_tm[matching_mask[i] > 0]
        ax[0].imshow(
            np.ma.masked_where(
                matching_mask[i] == 0, np.ones_like(matching_mask[i])
            ),
            interpolation="none",
            alpha=0.5 * (matching_mask[i] > 0),
            cmap=ListedColormap([color]),
            vmin=0,
            vmax=1,
        )
        ax[0].plot([], color=color, label=label, linewidth=2)

        ax[1].hist(mask_values, bins=50, label=label, alpha=0.4, color=color)

    ax[0].axis("off")
    ax[1].set_xlabel("Transmission")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    fig.savefig(f("segmentation_txm_histograms.png"))

    plt.show()

    vis_overlay_mask_example(labels, img, mask)

    copy_builds(curr_file_dir, "~/projs/master-thesis/figures/")
