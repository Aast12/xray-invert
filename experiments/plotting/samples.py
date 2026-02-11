"""Image comparison and percentile sample plots."""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

from .histograms import segmentation_histogram


def plot_percentile_samples(pred, txm, ms_ssim_errors, save_fn=None, num_samples=5):
    """Plot predicted/recovered images at evenly-spaced error percentiles.

    Args:
        pred: Predictions array of shape (N, H, W).
        txm: Transmission maps array of shape (N, H, W).
        ms_ssim_errors: 1D array of per-image MS-SSIM values.
        save_fn: Optional callable for saving.
        num_samples: Number of percentile samples to show.

    Returns:
        (fig, ax) tuple.
    """
    indices_perc = np.argsort(ms_ssim_errors)
    step = math.ceil(ms_ssim_errors.shape[0] / num_samples)
    percentile_labels = np.linspace(0, 100, num_samples, dtype=int)
    indices_percentiles = indices_perc[::step]

    fig, ax = plt.subplots(2, num_samples, figsize=(3 * num_samples, 8))
    for i, idx in enumerate(indices_percentiles):
        ax[0, i].imshow(pred[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(
            "$\\mathcal{L}^{\\text{Mix}}$:"
            + f" {ms_ssim_errors[idx]:.3f}\nP{percentile_labels[i]}"
        )
        ax[0, i].axis("off")
        ax[1, i].imshow(txm[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[1, i].axis("off")

    if save_fn is not None:
        fig.savefig(save_fn("pred_ms_ssim_percentiles.png"))
        plt.close(fig)

    return fig, ax


def plot_sample_comparison(imgs, txm, pred, masks, priors, save_fn=None, sample_id=None):
    """Plot a single sample: original vs estimated, and recovered vs naive histograms.

    Args:
        imgs: Images array.
        txm: Transmission maps array.
        pred: Predictions array.
        masks: Segmentation masks array.
        priors: Prior value ranges.
        save_fn: Optional callable for saving.
        sample_id: Index of sample to plot. Random if None.

    Returns:
        (fig1, fig2) tuple — the comparison figure and histogram figure.
    """
    if sample_id is None:
        sample_id = random.choice(range(len(pred)))

    img_sample = imgs[sample_id].squeeze()
    txm_sample = txm[sample_id].squeeze()
    pred_sample = pred[sample_id].squeeze()
    mask = masks[sample_id].squeeze()

    fig1, ax1 = plt.subplots(1, 2, figsize=(15, 7))
    ax1[0].imshow(img_sample, cmap="gray", vmin=0, vmax=1)
    ax1[0].axis("off")
    ax1[0].set_title("Original Image")
    ax1[1].imshow(pred_sample, cmap="gray", vmin=0, vmax=1)
    ax1[1].axis("off")
    ax1[1].set_title("Estimated Processed Image")
    plt.tight_layout()

    if save_fn is not None:
        fig1.savefig(save_fn("sample_comparison.png"))
        plt.close(fig1)

    naive = (1 - img_sample) / 2.0

    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 6))
    ax2[0, 0].set_title("Recovered Transmission Map")
    ax2[1, 0].set_title("Trivial Proc. Transmission Map")
    segmentation_histogram(ax2[0, 0], txm_sample, mask, priors)
    segmentation_histogram(ax2[1, 0], naive, mask, priors)
    ax2[1, 1].imshow(naive.squeeze(), cmap="gray", vmin=0, vmax=1)
    ax2[0, 1].imshow(txm_sample.squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()

    if save_fn is not None:
        fig2.savefig(save_fn("trivial_hist_comparison.png"))
        plt.close(fig2)

    return fig1, fig2
