"""Ground truth vs recovered comparison plots."""

import matplotlib.pyplot as plt
import numpy as np

from .histograms import segmentation_histogram


def plot_gt_comparison(recovered_txm, gt_txm, masks, priors, save_fn=None):
    """Side-by-side recovered vs GT images with segmentation histograms.

    Args:
        recovered_txm: Recovered transmission map (2D array).
        gt_txm: Ground truth transmission map (2D array).
        masks: Segmentation masks for this sample.
        priors: Prior value ranges.
        save_fn: Optional callable for saving.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex="col", sharey="col")

    im = ax[0, 0].imshow(recovered_txm, cmap="gray", vmin=0, vmax=0.6)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("Recovered Transmission Map")
    ax[0, 0].axis("off")
    segmentation_histogram(ax[0, 1], np.array(recovered_txm), masks, priors)
    ax[0, 1].set_xlabel("Pixel Intensity")

    im = ax[1, 0].imshow(gt_txm, cmap="gray", vmin=0, vmax=0.6)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title("Original Transmission Map")
    ax[1, 0].axis("off")
    segmentation_histogram(ax[1, 1], np.array(gt_txm), masks, priors)
    ax[1, 1].set_xlabel("Pixel Intensity")

    plt.tight_layout()

    if save_fn is not None:
        fig.savefig(save_fn("gt_comparison.pdf"))
        plt.close(fig)

    return fig, ax


def plot_gt_vs_naive(gt_txm, processed_img, masks, priors, save_fn=None):
    """GT vs trivial (1-proc)/2 with histograms.

    Args:
        gt_txm: Ground truth transmission map (2D array).
        processed_img: Processed image (2D array).
        masks: Segmentation masks.
        priors: Prior value ranges.
        save_fn: Optional callable for saving.

    Returns:
        (fig, ax) tuple.
    """
    naive = (1 - np.array(processed_img)) / 2.0

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex="col", sharey="col")

    ax[0, 0].imshow(gt_txm, cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("Real Transmission Map")
    ax[0, 0].axis("off")
    segmentation_histogram(ax[0, 1], np.array(gt_txm), masks, priors)
    ax[0, 1].set_ylabel("Frequency")

    ax[1, 0].imshow(naive, cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title("Trivial Transmission Map")
    ax[1, 0].axis("off")
    segmentation_histogram(ax[1, 1], naive, masks, priors)
    ax[1, 1].set_ylabel("Frequency")

    plt.tight_layout()

    if save_fn is not None:
        fig.savefig(save_fn("gt_vs_naive.png"))
        plt.close(fig)

    return fig, ax


def plot_gt_metrics(recovered_txm, gt_txm, naive_txm, save_fn=None):
    """MSE/SSIM/MS-SSIM comparison bar chart for recovered vs naive.

    Args:
        recovered_txm: Recovered transmission map (2D array).
        gt_txm: Ground truth transmission map (2D array).
        naive_txm: Naive/trivial transmission map (2D array).
        save_fn: Optional callable for saving.

    Returns:
        (fig, ax) tuple.
    """
    from chest_xray_sim.inverse.metrics import mse, ssim, ms_ssim_l1

    recovered_mse = float(mse(recovered_txm, gt_txm))
    recovered_ssim = float(ssim(recovered_txm, gt_txm))
    recovered_ms_ssim = float(ms_ssim_l1(recovered_txm, gt_txm))

    naive_mse = float(mse(naive_txm, gt_txm))
    naive_ssim = float(ssim(naive_txm, gt_txm))
    naive_ms_ssim = float(ms_ssim_l1(naive_txm, gt_txm))

    metrics = ['MSE', 'SSIM', 'MS-SSIM+L1']
    recovered_vals = [recovered_mse, recovered_ssim, recovered_ms_ssim]
    naive_vals = [naive_mse, naive_ssim, naive_ms_ssim]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, recovered_vals, width, label='Recovered', alpha=0.8)
    ax.bar(x + width/2, naive_vals, width, label='Naive', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Ground Truth Comparison: Recovered vs Naive')
    ax.legend()
    plt.tight_layout()

    print(f"Recovered: MSE={recovered_mse:.6f}, SSIM={recovered_ssim:.4f}, MS-SSIM+L1={recovered_ms_ssim:.4f}")
    print(f"Naive:     MSE={naive_mse:.6f}, SSIM={naive_ssim:.4f}, MS-SSIM+L1={naive_ms_ssim:.4f}")

    if save_fn is not None:
        fig.savefig(save_fn("gt_metrics.png"))
        plt.close(fig)

    return fig, ax
