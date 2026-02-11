"""EvalMetrics-based visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_distributions(metric, save_fn=None):
    """Plot distributions of data fidelity and anatomical compliance.

    Args:
        metric: EvalMetrics object.
        save_fn: Optional callable that maps filename -> path.

    Returns:
        (fig, ax) tuple.
    """
    sns.set_theme()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].set_title("Data Fidelity")
    ax[0].set_xlabel("$\\mathcal{L}^{\\mathrm{Mix}}$")
    sns.histplot(np.array(metric.ms_ssim), kde=True, bins=25, ax=ax[0])

    ax[1].set_title("Anatomical Compliance")
    ax[1].set_xlabel("Penalty")
    sns.histplot(np.array(metric.penalties), kde=True, bins=25, ax=ax[1])

    if save_fn is not None:
        fig.savefig(save_fn("metric_distributions.pdf"))
        plt.close(fig)

    return fig, ax


def plot_metrics_boxplot(metric, save_fn=None):
    """Plot boxplots of penalties and MS-SSIM.

    Args:
        metric: EvalMetrics object.
        save_fn: Optional callable for saving.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].boxplot(metric.penalties.ravel(), showfliers=False)
    ax[0].set_title("Penalties")
    ax[1].boxplot(metric.ms_ssim.ravel(), showfliers=False)
    ax[1].set_title("$\\mathcal{L}^{\\text{Mix}}$")

    if save_fn is not None:
        fig.savefig(save_fn("metrics.png"))
        plt.close(fig)

    return fig, ax


def plot_batch_metrics(metrics_list, save_fn=None):
    """Plot metrics across batches (MSE, SSIM, PSNR distributions and consistency).

    Consolidation of plot_metrics_from_replicate from plot_replicate_results.py.

    Args:
        metrics_list: List of EvalMetrics objects (one per batch).
        save_fn: Optional callable for saving.

    Returns:
        (fig, axes) tuple.
    """
    sns.set_theme()

    all_mse = []
    all_psnr = []
    all_ssim = []
    all_tv = []
    batch_labels = []

    for batch_idx, metrics in enumerate(metrics_list):
        n_images = len(metrics.mse)
        all_mse.extend(metrics.mse.tolist())
        all_psnr.extend(metrics.psnr.tolist())
        all_ssim.extend(metrics.ssim.tolist())
        all_tv.extend(metrics.total_variation.tolist())
        batch_labels.extend([batch_idx] * n_images)

    all_mse = np.array(all_mse)
    all_psnr = np.array(all_psnr)
    all_ssim = np.array(all_ssim)
    batch_labels = np.array(batch_labels)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # MSE by batch
    ax1 = axes[0, 0]
    for batch_idx in range(len(metrics_list)):
        batch_mask = batch_labels == batch_idx
        batch_mse = all_mse[batch_mask]
        ax1.scatter([batch_idx] * len(batch_mse), batch_mse, alpha=0.6, s=50)
        ax1.bar(batch_idx, batch_mse.mean(), alpha=0.3, width=0.6)
        ax1.errorbar(batch_idx, batch_mse.mean(), yerr=batch_mse.std(),
                     fmt='none', color='black', capsize=5)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE Distribution Across Batches')
    ax1.set_xticks(range(len(metrics_list)))
    ax1.set_xticklabels([f'Batch {i+1}' for i in range(len(metrics_list))])

    # SSIM by batch
    ax2 = axes[0, 1]
    for batch_idx in range(len(metrics_list)):
        batch_mask = batch_labels == batch_idx
        batch_ssim = all_ssim[batch_mask]
        ax2.scatter([batch_idx] * len(batch_ssim), batch_ssim, alpha=0.6, s=50)
        ax2.bar(batch_idx, batch_ssim.mean(), alpha=0.3, width=0.6)
        ax2.errorbar(batch_idx, batch_ssim.mean(), yerr=batch_ssim.std(),
                     fmt='none', color='black', capsize=5)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Distribution Across Batches')
    ax2.set_xticks(range(len(metrics_list)))
    ax2.set_xticklabels([f'Batch {i+1}' for i in range(len(metrics_list))])

    # Overall distributions (normalized violin)
    ax3 = axes[1, 0]
    data_to_plot = [all_mse, all_psnr, all_ssim]
    labels = ['MSE', 'PSNR', 'SSIM']
    normalized_data = [(d - d.mean()) / d.std() for d in data_to_plot]
    parts = ax3.violinplot(normalized_data, positions=range(3), showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Overall Metric Distributions (Normalized)')

    # Consistency analysis (CV)
    ax4 = axes[1, 1]
    batch_means_mse = [all_mse[batch_labels == i].mean() for i in range(len(metrics_list))]
    batch_means_ssim = [all_ssim[batch_labels == i].mean() for i in range(len(metrics_list))]
    batch_means_psnr = [all_psnr[batch_labels == i].mean() for i in range(len(metrics_list))]

    cv_mse = np.std(batch_means_mse) / np.mean(batch_means_mse)
    cv_ssim = np.std(batch_means_ssim) / np.mean(batch_means_ssim)
    cv_psnr = np.std(batch_means_psnr) / np.mean(batch_means_psnr)

    cvs = [cv_mse, cv_psnr, cv_ssim]
    metric_names = ['MSE', 'PSNR', 'SSIM']
    colors = ['red', 'green', 'blue']
    bars = ax4.bar(metric_names, cvs, color=colors, alpha=0.7)
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Metric Stability Across Batches\n(Lower = More Consistent)')
    ax4.set_ylim(0, float(np.max(cvs)) * 1.2)

    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.4f}', ha='center', va='bottom')

    plt.suptitle('Hyperparameter Generalization Analysis', fontsize=16)
    plt.tight_layout()

    # Print summary
    print("\n=== HYPERPARAMETER GENERALIZATION SUMMARY ===")
    print(f"\nNumber of batches analyzed: {len(metrics_list)}")
    print(f"Total images processed: {len(all_mse)}")
    print(f"\nOverall Performance (mean +/- std):")
    print(f"MSE:  {all_mse.mean():.6f} +/- {all_mse.std():.6f}")
    print(f"PSNR: {all_psnr.mean():.4f} +/- {all_psnr.std():.4f}")
    print(f"SSIM: {all_ssim.mean():.4f} +/- {all_ssim.std():.4f}")

    if save_fn is not None:
        fig.savefig(save_fn("batch_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, axes


def analyze_batch_results(metrics, txm, weights, pred, save_fn=None):
    """Comprehensive batch analysis with violin plots and scatter.

    Consolidation of analyze_batch_results from analyze_replicate_results.py.

    Args:
        metrics: List of EvalMetrics objects.
        txm: List of transmission map arrays.
        weights: List of weight dicts.
        pred: List of prediction arrays.
        save_fn: Optional callable for saving.

    Returns:
        (fig, df) tuple where df is a pandas DataFrame of all metrics.
    """
    import pandas as pd
    sns.set_theme()

    all_metrics = []
    for batch_idx, batch_metrics in enumerate(metrics):
        batch_size = len(batch_metrics.mse)
        for img_idx in range(batch_size):
            all_metrics.append({
                'batch': batch_idx,
                'image': img_idx,
                'mse': float(batch_metrics.mse[img_idx]),
                'psnr': float(batch_metrics.psnr[img_idx]),
                'ssim': float(batch_metrics.ssim[img_idx]),
                'ms_ssim': float(batch_metrics.ms_ssim[img_idx]),
                'total_variation': float(batch_metrics.total_variation[img_idx]),
                'tikonov': float(batch_metrics.tikonov[img_idx]),
                'band_similarity': float(batch_metrics.band_similarity[img_idx]),
            })

    df = pd.DataFrame(all_metrics)

    fig = plt.figure(figsize=(20, 12))

    # Performance consistency across batches
    ax1 = plt.subplot(2, 3, 1)
    batch_means = df.groupby('batch')[['mse', 'psnr', 'ssim']].mean()
    batch_stds = df.groupby('batch')[['mse', 'psnr', 'ssim']].std()
    x = np.arange(len(batch_means))
    width = 0.25
    for i, metric in enumerate(['mse', 'psnr', 'ssim']):
        ax1.bar(x + i*width, batch_means[metric], width,
                yerr=batch_stds[metric], label=metric.upper(), alpha=0.8)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Average Performance Across Batches')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'Batch {i+1}' for i in range(len(batch_means))])
    ax1.legend()

    # SSIM violin
    ax2 = plt.subplot(2, 3, 2)
    sns.violinplot(data=df, x='batch', y='ssim', ax=ax2)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Distribution by Batch')

    # MSE boxplot
    ax3 = plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='batch', y='mse', ax=ax3)
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE Distribution by Batch')

    # TV vs Band Similarity scatter
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(df['total_variation'], df['band_similarity'],
                         c=df['batch'], cmap='viridis', alpha=0.6)
    ax4.set_xlabel('Total Variation')
    ax4.set_ylabel('Band Similarity')
    ax4.set_title('Regularization Metrics Relationship')
    plt.colorbar(scatter, ax=ax4, label='Batch')

    # CV bars
    ax6 = plt.subplot(2, 3, 6)
    cv_metrics = {}
    for metric in ['mse', 'psnr', 'ssim']:
        bm = df.groupby('batch')[metric].mean()
        cv_metrics[metric] = bm.std() / bm.mean() if bm.mean() != 0 else 0
    bars = ax6.bar(list(cv_metrics.keys()), list(cv_metrics.values()),
                    color=['red', 'green', 'blue'], alpha=0.7)
    ax6.set_ylabel('Coefficient of Variation')
    ax6.set_title('Metric Stability Across Batches\n(Lower is Better)')
    for bar, (_, cv) in zip(bars, cv_metrics.items()):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{cv:.3f}', ha='center', va='bottom')

    plt.suptitle('Hyperparameter Generalization Analysis', fontsize=16)
    plt.tight_layout()

    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print("\nOverall Performance (mean +/- std):")
    for metric in ['mse', 'psnr', 'ssim', 'ms_ssim']:
        print(f"{metric.upper()}: {df[metric].mean():.4f} +/- {df[metric].std():.4f}")

    if save_fn is not None:
        fig.savefig(save_fn("batch_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, df


def create_simple_summary_plot(metrics_list, save_fn=None):
    """Simple summary histograms of MSE, SSIM, and PSNR.

    Args:
        metrics_list: List of EvalMetrics objects.
        save_fn: Optional callable for saving.

    Returns:
        (fig, axes) tuple.
    """
    mse_values = []
    ssim_values = []
    psnr_values = []

    for batch_metrics in metrics_list:
        mse_values.extend(batch_metrics.mse.tolist())
        ssim_values.extend(batch_metrics.ssim.tolist())
        psnr_values.extend(batch_metrics.psnr.tolist())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, values, color, dark, name in [
        (axes[0], mse_values, 'red', 'darkred', 'MSE'),
        (axes[1], ssim_values, 'blue', 'darkblue', 'SSIM'),
        (axes[2], psnr_values, 'green', 'darkgreen', 'PSNR (dB)'),
    ]:
        ax.hist(values, bins=20, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(np.mean(values), color=dark, linestyle='--',
                    label=f'Mean: {np.mean(values):.4f}')
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Distribution Across All Images')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Reconstruction Quality Metrics', fontsize=14)
    plt.tight_layout()

    if save_fn is not None:
        fig.savefig(save_fn("summary_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, axes
