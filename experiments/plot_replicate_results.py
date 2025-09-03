import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_metrics_from_replicate(metrics_list, save_path=None):
    """
    Plot metrics from individual_replicate.py execution.
    
    Args:
        metrics_list: List of EvalMetrics objects from execute() function
        save_path: Optional path to save the figure
    """
    
    # Extract data from each batch
    all_mse = []
    all_psnr = []
    all_ssim = []
    all_tv = []
    batch_labels = []
    
    for batch_idx, metrics in enumerate(metrics_list):
        # Each metrics object contains arrays of values
        n_images = len(metrics.mse)
        
        all_mse.extend(metrics.mse.tolist())
        all_psnr.extend(metrics.psnr.tolist())
        all_ssim.extend(metrics.ssim.tolist())
        all_tv.extend(metrics.total_variation.tolist())
        batch_labels.extend([batch_idx] * n_images)
    
    # Convert to numpy arrays
    all_mse = np.array(all_mse)
    all_psnr = np.array(all_psnr)
    all_ssim = np.array(all_ssim)
    all_tv = np.array(all_tv)
    batch_labels = np.array(batch_labels)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MSE by batch
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
    ax1.grid(True, alpha=0.3)
    
    # 2. SSIM by batch
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
    ax2.grid(True, alpha=0.3)
    
    # 3. Overall distributions
    ax3 = axes[1, 0]
    data_to_plot = [all_mse, all_psnr, all_ssim]
    labels = ['MSE', 'PSNR', 'SSIM']
    
    # Normalize data for comparison
    normalized_data = []
    for data in data_to_plot:
        normalized = (data - data.mean()) / data.std()
        normalized_data.append(normalized)
    
    parts = ax3.violinplot(normalized_data, positions=range(3), showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Overall Metric Distributions (Normalized)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Consistency analysis
    ax4 = axes[1, 1]
    
    # Calculate coefficient of variation for each metric across batches
    batch_means_mse = []
    batch_means_ssim = []
    batch_means_psnr = []
    
    for batch_idx in range(len(metrics_list)):
        batch_mask = batch_labels == batch_idx
        batch_means_mse.append(all_mse[batch_mask].mean())
        batch_means_ssim.append(all_ssim[batch_mask].mean())
        batch_means_psnr.append(all_psnr[batch_mask].mean())
    
    # Calculate CV
    cv_mse = np.std(batch_means_mse) / np.mean(batch_means_mse)
    cv_ssim = np.std(batch_means_ssim) / np.mean(batch_means_ssim)
    cv_psnr = np.std(batch_means_psnr) / np.mean(batch_means_psnr)
    
    cvs = [cv_mse, cv_psnr, cv_ssim]
    metric_names = ['MSE', 'PSNR', 'SSIM']
    colors = ['red', 'green', 'blue']
    
    bars = ax4.bar(metric_names, cvs, color=colors, alpha=0.7)
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Metric Stability Across Batches\n(Lower = More Consistent)')
    ax4.set_ylim(0, max(cvs) * 1.2)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.4f}', ha='center', va='bottom')
    
    plt.suptitle('Hyperparameter Generalization Analysis\nSame Hyperparameters on Different Data Batches', 
                 fontsize=16)
    plt.tight_layout()
    
    # Print summary
    print("\n=== HYPERPARAMETER GENERALIZATION SUMMARY ===")
    print(f"\nNumber of batches analyzed: {len(metrics_list)}")
    print(f"Total images processed: {len(all_mse)}")
    
    print("\nOverall Performance (mean ± std):")
    print(f"MSE:  {all_mse.mean():.6f} ± {all_mse.std():.6f}")
    print(f"PSNR: {all_psnr.mean():.4f} ± {all_psnr.std():.4f}")
    print(f"SSIM: {all_ssim.mean():.4f} ± {all_ssim.std():.4f}")
    
    print("\nBatch-wise Consistency (Coefficient of Variation):")
    print(f"MSE:  {cv_mse:.4f}")
    print(f"PSNR: {cv_psnr:.4f}")
    print(f"SSIM: {cv_ssim:.4f}")
    
    print("\nBatch Means:")
    for i, (mse, psnr, ssim) in enumerate(zip(batch_means_mse, batch_means_psnr, batch_means_ssim)):
        print(f"Batch {i+1}: MSE={mse:.6f}, PSNR={psnr:.4f}, SSIM={ssim:.4f}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Usage in individual_replicate.py:
# from plot_replicate_results import plot_metrics_from_replicate
# metrics, txm, weights, pred = execute(dataset, config, path)
# fig = plot_metrics_from_replicate(metrics, "hyperparameter_generalization.png")
# plt.show()