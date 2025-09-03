import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_batch_results(metrics: List[Any], txm: List, weights: List, pred: List):
    """
    Analyze results from multiple batches to show hyperparameter generalization.
    
    Args:
        metrics: List of EvalMetrics objects for each batch
        txm: List of transmission maps for each batch
        weights: List of optimized weights for each batch
        pred: List of predictions for each batch
    """
    
    # Extract metrics into a structured format
    all_metrics = []
    for batch_idx, batch_metrics in enumerate(metrics):
        # batch_metrics is an EvalMetrics object with arrays for each metric
        # Each array has shape (batch_size,) for per-image metrics
        batch_size = len(batch_metrics.mse)
        
        for img_idx in range(batch_size):
            metric_dict = {
                'batch': batch_idx,
                'image': img_idx,
                'mse': float(batch_metrics.mse[img_idx]),
                'psnr': float(batch_metrics.psnr[img_idx]),
                'ssim': float(batch_metrics.ssim[img_idx]),
                'ms_ssim': float(batch_metrics.ms_ssim[img_idx]),
                'total_variation': float(batch_metrics.total_variation[img_idx]),
                'tikonov': float(batch_metrics.tikonov[img_idx]),
                'band_similarity': float(batch_metrics.band_similarity[img_idx])
            }
            
            all_metrics.append(metric_dict)
    
    df = pd.DataFrame(all_metrics)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Performance consistency across batches
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
    ax1.grid(True, alpha=0.3)
    
    # 2. SSIM distribution across all images
    ax2 = plt.subplot(2, 3, 2)
    sns.violinplot(data=df, x='batch', y='ssim', ax=ax2)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Distribution by Batch')
    ax2.set_xticklabels([f'B{i+1}' for i in range(len(metrics))])
    
    # 3. MSE distribution
    ax3 = plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='batch', y='mse', ax=ax3)
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE Distribution by Batch')
    ax3.set_xticklabels([f'B{i+1}' for i in range(len(metrics))])
    
    # 4. Total Variation vs Band Similarity
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(df['total_variation'], df['band_similarity'], 
                         c=df['batch'], cmap='viridis', alpha=0.6)
    ax4.set_xlabel('Total Variation')
    ax4.set_ylabel('Band Similarity')
    ax4.set_title('Regularization Metrics Relationship')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Batch')
    
    # 5. Weight parameter consistency
    ax5 = plt.subplot(2, 3, 5)
    
    # Analyze weight parameters across batches
    weight_params = []
    for batch_idx, batch_weights in enumerate(weights):
        # Assuming weights are [window_center, window_width, gamma, low_sigma, ...]
        for img_weights in batch_weights:
            weight_params.append({
                'batch': batch_idx,
                'window_center': img_weights[0],
                'window_width': img_weights[1],
                'gamma': img_weights[2],
                'low_sigma': img_weights[3] if len(img_weights) > 3 else None
            })
    
    weight_df = pd.DataFrame(weight_params)
    
    # Plot window center distribution
    sns.violinplot(data=weight_df, x='batch', y='window_center', ax=ax5)
    ax5.set_xlabel('Batch')
    ax5.set_ylabel('Window Center')
    ax5.set_title('Window Center Distribution by Batch')
    ax5.set_xticklabels([f'B{i+1}' for i in range(len(weights))])
    
    # 6. Performance stability metrics
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate coefficient of variation for each metric
    cv_metrics = {}
    for metric in ['mse', 'psnr', 'ssim']:
        batch_means = df.groupby('batch')[metric].mean()
        cv = batch_means.std() / batch_means.mean() if batch_means.mean() != 0 else 0
        cv_metrics[metric] = cv
    
    bars = ax6.bar(cv_metrics.keys(), cv_metrics.values(), 
                    color=['red', 'green', 'blue'], alpha=0.7)
    ax6.set_ylabel('Coefficient of Variation')
    ax6.set_title('Metric Stability Across Batches\n(Lower is Better)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, (metric, cv) in zip(bars, cv_metrics.items()):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Hyperparameter Generalization Analysis\nSame Hyperparameters Applied to Different Data Batches', 
                 fontsize=16)
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nOverall Performance (mean ± std):")
    for metric in ['mse', 'psnr', 'ssim', 'ms_ssim']:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nPer-Batch Performance:")
    print(batch_means)
    
    print("\nCoefficient of Variation (stability measure):")
    for metric, cv in cv_metrics.items():
        print(f"{metric.upper()}: {cv:.4f}")
    
    return fig, df

def create_simple_summary_plot(metrics: List[Any]):
    """Create a simple summary plot focusing on key metrics."""
    
    # Extract key metrics
    mse_values = []
    ssim_values = []
    psnr_values = []
    
    for batch_metrics in metrics:
        # batch_metrics is an EvalMetrics object
        mse_values.extend(batch_metrics.mse.tolist())
        ssim_values.extend(batch_metrics.ssim.tolist())
        psnr_values.extend(batch_metrics.psnr.tolist())
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE histogram
    axes[0].hist(mse_values, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0].axvline(np.mean(mse_values), color='darkred', linestyle='--', 
                    label=f'Mean: {np.mean(mse_values):.4f}')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('MSE Distribution Across All Images')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[1].hist(ssim_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(np.mean(ssim_values), color='darkblue', linestyle='--',
                    label=f'Mean: {np.mean(ssim_values):.4f}')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SSIM Distribution Across All Images')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # PSNR histogram
    axes[2].hist(psnr_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[2].axvline(np.mean(psnr_values), color='darkgreen', linestyle='--',
                    label=f'Mean: {np.mean(psnr_values):.4f}')
    axes[2].set_xlabel('PSNR (dB)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('PSNR Distribution Across All Images')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Reconstruction Quality Metrics - Same Hyperparameters on Multiple Batches', 
                 fontsize=14)
    plt.tight_layout()
    
    return fig

# Example usage
if __name__ == "__main__":
    # This would be called after running execute() in individual_replicate.py
    # metrics, txm, weights, pred = execute(dataset, config, path)
    
    # For demonstration, show how to use the functions
    print("Usage:")
    print("1. Run individual_replicate.py to get metrics, txm, weights, pred")
    print("2. Call analyze_batch_results(metrics, txm, weights, pred)")
    print("3. Or call create_simple_summary_plot(metrics) for a simpler view")