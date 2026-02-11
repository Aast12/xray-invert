"""TXM histograms, segmentation histograms, and heatmaps."""

import math

import matplotlib.pyplot as plt
import numpy as np
from chest_xray_sim.segmentation.model import MASK_GROUPS


def segmentation_histogram(ax, data, mask_groups, priors, labels=MASK_GROUPS, bins=50, small_bins=30):
    """Plot a histogram of data with segmentation overlays.

    Args:
        ax: Matplotlib axes to plot on.
        data: 2D array of pixel values.
        mask_groups: List of segmentation masks (one per group).
        priors: Array of shape (n_groups, 2) with [min, max] prior bounds.
        labels: Group labels for the legend.
        bins: Number of bins for the total histogram.
        small_bins: Number of bins for per-group histograms.

    Returns:
        The axes object.
    """
    colors = ["red", "blue", "green"]
    ax.hist(data.ravel(), bins=bins, alpha=0.5, label="total")

    for i, m in enumerate(mask_groups):
        label = labels[i]
        ax.axvline(priors[i, 0], color=colors[i], linestyle="--")
        ax.axvline(priors[i, 1], color=colors[i], linestyle="--")
        ax.hist(
            data[m.squeeze() > 0].ravel(),
            bins=small_bins,
            alpha=0.3,
            color=colors[i],
            label=f"{label}",
        )
    ax.legend()
    return ax


def plot_histogram_heatmap(txm, save_fn=None):
    """Plot a heatmap of per-sample TXM histograms.

    Args:
        txm: Array of shape (num_samples, H, W).
        save_fn: Optional callable that maps filename -> path. If provided, saves figure.

    Returns:
        (fig, ax) tuple.
    """
    num_samples = txm.shape[0]
    num_bins = 50
    hist_matrix = []
    for i in range(num_samples):
        hist, bin_edges = np.histogram(txm[i].ravel(), bins=num_bins, range=(0, 1), density=True)
        hist_matrix.append(hist)

    hist_matrix = np.array(hist_matrix).T
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        hist_matrix, aspect="auto", origin="lower", cmap="viridis",
        extent=(0, num_samples, float(bin_edges[0]), float(bin_edges[-1])),
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Transmission Map Value")
    fig.colorbar(im, ax=ax, label="Density")
    plt.tight_layout()

    if save_fn is not None:
        fig.savefig(save_fn("histogram_heatmap.png"))
        plt.close(fig)

    return fig, ax


def plot_extreme_histograms(metric, txm, masks, priors, save_fn=None):
    """Plot histograms for samples with extreme metric values.

    Args:
        metric: EvalMetrics object with penalties and ms_ssim arrays.
        txm: Array of shape (num_samples, H, W).
        masks: Segmentation masks array.
        priors: Prior value ranges.
        save_fn: Optional callable for saving.

    Returns:
        (fig, ax) tuple.
    """
    import jax.numpy as jnp

    indices = {
        "penalty": (jnp.argmax(metric.penalties), jnp.argmin(metric.penalties)),
        "$\\mathcal{L}^{\\text{Mix}}$": (jnp.argmax(metric.ms_ssim), jnp.argmin(metric.ms_ssim)),
    }

    fig, ax = plt.subplots(2, len(indices), figsize=(10 * len(indices), 10))
    for i, (key, (max_idx, min_idx)) in enumerate(indices.items()):
        segmentation_histogram(ax[0, i], txm[max_idx].squeeze(), masks[max_idx], priors)
        ax[0, i].set_title(f"Max {key} Transmission Map Histogram")
        segmentation_histogram(ax[1, i], txm[min_idx].squeeze(), masks[min_idx], priors)
        ax[1, i].set_title(f"Min {key} Transmission Map Histogram")

    if save_fn is not None:
        fig.savefig(save_fn("extreme_histograms.png"))
        plt.close(fig)

    return fig, ax
