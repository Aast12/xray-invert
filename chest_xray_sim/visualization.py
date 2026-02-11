"""Reusable plotting functions for X-ray transmission map analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from jaxtyping import Array, Float

from chest_xray_sim.inverse import operators as ops


def plot_segmentation_overlay(
    image: Float[Array, "height width"],
    masks: Float[Array, "labels height width"],
    labels: list[str],
    colors: list[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Overlay segmentation masks on an image.

    Args:
        image: Grayscale image (H, W).
        masks: Binary masks per region (labels, H, W).
        labels: Label names for each mask.
        colors: Colors for each mask. Defaults to red/green/blue.
        ax: Optional axes to draw on.

    Returns:
        (fig, ax) tuple.
    """
    if colors is None:
        colors = ["red", "green", "blue"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig_ = ax.get_figure()
        assert isinstance(fig_, Figure)
        fig = fig_

    ax.imshow(image, interpolation="none", cmap="gray")
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.imshow(
            np.ma.masked_where(masks[i] == 0, np.ones_like(masks[i])),
            interpolation="none",
            alpha=0.5 * (masks[i] > 0),
            cmap=ListedColormap([color]),
            vmin=0,
            vmax=1,
        )
        ax.plot([], color=color, label=label, linewidth=2)

    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    return fig, ax


def plot_transmission_maps(
    maps: list[tuple],
) -> tuple[Figure, np.ndarray]:
    """Plot transmission maps alongside their processed versions.

    Args:
        maps: List of (transmission_map, processed_image) tuples.

    Returns:
        (fig, axes) tuple.
    """
    fig, ax = plt.subplots(len(maps), 2, figsize=(16, 8 * len(maps)))
    if len(maps) == 1:
        ax = ax[np.newaxis, :]

    for i, (tm, proc) in enumerate(maps):
        ax[i, 0].imshow(tm, cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(proc, cmap="gray")
        ax[i, 1].axis("off")

    fig.tight_layout()
    return fig, ax


def plot_histograms(
    maps: list[tuple],
) -> tuple[Figure, np.ndarray]:
    """Plot transmission map histograms.

    Args:
        maps: List of (transmission_map, processed_image) tuples.

    Returns:
        (fig, axes) tuple.
    """
    n = len(maps)
    fig, ax = plt.subplots(n, 2, figsize=(16, 8 * n))
    if n == 1:
        ax = ax[np.newaxis, :]

    for i, (map_, _) in enumerate(maps):
        ax[i, 0].imshow(map_, cmap="gray")
        ax[i, 0].axis("off")
        ax[i, 1].hist(np.asarray(map_).flatten(), bins=70)
        ax[i, 1].set_xlim(0, 1)
        ax[i, 1].set_xlabel("Pixel value")
        ax[i, 1].set_ylabel("Frequency")

    return fig, ax


def plot_windowing_lut(
    centers: list[float] | None = None,
    widths: list[float] | None = None,
    gammas: list[float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot windowing function look-up tables for different parameters.

    Args:
        centers: Window center values. Defaults to [0.2, 0.5, 0.7].
        widths: Window width values. Defaults to [0.2, 0.3, 0.5].
        gammas: Gamma values. Defaults to [1, 4, 10].

    Returns:
        (fig, axes) tuple.
    """
    if centers is None:
        centers = [0.2, 0.5, 0.7]
    if widths is None:
        widths = [0.2, 0.3, 0.5]
    if gammas is None:
        gammas = [1, 4, 10]

    values = np.linspace(0, 1, 500)
    colors = ["red", "green", "blue"]

    fig, ax = plt.subplots(
        len(widths),
        len(centers),
        figsize=(16, 16),
        sharex="col",
        sharey="row",
        gridspec_kw={"left": 0.17},
    )

    for j, center in enumerate(centers):
        ax[0, j].set_title(f"Window Center: {center}")

    for i, width in enumerate(widths):
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

        for j, center in enumerate(centers):
            ax[i, j].axvline(x=center - width / 2, color="gray", linestyle="--")
            ax[i, j].axvline(x=center + width / 2, color="gray", linestyle="--")
            ax[i, j].axvline(x=center, color="gray", linestyle="--")

            for color, gamma in zip(colors, gammas):
                ax[i, j].plot(
                    values,
                    ops.window(values, center, width, gamma),
                    label=f"Gamma: {gamma}",
                    color=color,
                )
                ax[i, j].set_ylim(0, 1)

            ax[i, j].legend()

    return fig, ax


def plot_results(
    images: Float[Array, "batch height width"],
    txm: Float[Array, "batch height width"],
    predictions: Float[Array, "batch height width"],
    segmentations: Float[Array, "batch labels height width"] | None = None,
    max_images: int = 4,
) -> tuple[Figure, np.ndarray]:
    """Show pipeline results side-by-side.

    Columns: Target image | Recovered TXM | Re-processed prediction
    Optionally adds a segmentation overlay column.

    Args:
        images: Original target images.
        txm: Recovered transmission maps.
        predictions: Forward-processed predictions.
        segmentations: Optional segmentation masks.
        max_images: Maximum number of images to display.

    Returns:
        (fig, axes) tuple.
    """
    n = min(len(images), max_images)
    ncols = 4 if segmentations is not None else 3
    fig, ax = plt.subplots(n, ncols, figsize=(5 * ncols, 5 * n))

    if n == 1:
        ax = ax[np.newaxis, :]

    col_titles = ["Target", "Recovered TXM", "Prediction"]
    if segmentations is not None:
        col_titles.append("Segmentation")

    for j, title in enumerate(col_titles):
        ax[0, j].set_title(title)

    for i in range(n):
        ax[i, 0].imshow(np.asarray(images[i]), cmap="gray")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(np.asarray(txm[i]), cmap="gray")
        ax[i, 1].axis("off")

        ax[i, 2].imshow(np.asarray(predictions[i]), cmap="gray")
        ax[i, 2].axis("off")

        if segmentations is not None:
            seg_overlay = np.asarray(segmentations[i]).sum(axis=0)
            ax[i, 3].imshow(np.asarray(images[i]), cmap="gray")
            ax[i, 3].imshow(seg_overlay, cmap="jet", alpha=0.3)
            ax[i, 3].axis("off")

    fig.tight_layout()
    return fig, ax
