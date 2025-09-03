import os
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import joblib
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from individual_replicate import execute, ExperimentArgs
from utils import experiment_args
import pandas as pd
from chest_xray_sim.inverse.eval import batch_evaluation

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Environment setup
args_spec = experiment_args(
    batch_size=32,
    frontal_lateral="Frontal",
    split="train",
    data_dir=os.environ.get("IMAGE_PATH"),
    meta_dir=os.environ.get("META_PATH"),
    mask_dir=os.environ.get("MASK_DIR"),
    cache_dir=os.environ.get("CACHE_DIR"),
    save_dir=os.environ.get("OUTPUT_DIR"),
)

def extract_metrics_data(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract metrics from batch results into a DataFrame."""
    data = []
    
    for batch_idx, batch_metrics in enumerate(metrics_list):
        for image_idx, image_metrics in enumerate(batch_metrics):
            row = {
                'batch': batch_idx,
                'image': image_idx,
                'mse': image_metrics['mse'],
                'psnr': image_metrics['psnr'],
                'ssim': image_metrics['ssim'],
                'mae': image_metrics['mae'],
            }
            
            # Add segmentation-specific metrics
            for seg_type in ['bone', 'lung', 'soft_tissue']:
                if f'{seg_type}_mse' in image_metrics:
                    row[f'{seg_type}_mse'] = image_metrics[f'{seg_type}_mse']
                    row[f'{seg_type}_psnr'] = image_metrics[f'{seg_type}_psnr']
                    row[f'{seg_type}_ssim'] = image_metrics[f'{seg_type}_ssim']
            
            data.append(row)
    
    return pd.DataFrame(data)

def run_multiple_datasets(hyperparams: ExperimentArgs, n_datasets: int = 5, batches_per_dataset: int = 3):
    """Run the same hyperparameters on multiple different dataset samples."""
    args = args_spec()
    
    all_results = []
    
    for dataset_idx in range(n_datasets):
        print(f"Running on dataset sample {dataset_idx + 1}/{n_datasets}")
        
        # Create a new dataset instance (will get different random samples)
        dataset = get_segmentation_dataset(
            data_dir=args.data_dir,
            meta_dir=args.meta_dir,
            mask_dir=args.mask_dir,
            cache_dir=args.cache_dir,
            split=args.split,
            frontal_lateral=args.frontal_lateral,
            batch_size=32,
        )
        
        # Run optimization
        metrics, txm, weights, pred = execute(
            dataset, 
            hyperparams, 
            save_dir=None,  # Don't save intermediate results
            batches=batches_per_dataset
        )
        
        all_results.append({
            'dataset_idx': dataset_idx,
            'metrics': metrics,
            'txm': txm,
            'weights': weights,
            'pred': pred
        })
    
    return all_results

def plot_cross_dataset_performance(all_results: List[Dict], save_path: str = None):
    """Create comprehensive visualization of performance across different datasets."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Collect all metrics data
    all_metrics_data = []
    for result in all_results:
        df = extract_metrics_data(result['metrics'])
        df['dataset'] = result['dataset_idx']
        all_metrics_data.append(df)
    
    full_df = pd.concat(all_metrics_data, ignore_index=True)
    
    # 1. Overall performance distribution across datasets
    ax1 = fig.add_subplot(gs[0, :])
    metrics_to_plot = ['mse', 'psnr', 'ssim']
    positions = np.arange(len(all_results))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        means = full_df.groupby('dataset')[metric].mean()
        stds = full_df.groupby('dataset')[metric].std()
        ax1.bar(positions + i*width, means, width, yerr=stds, 
                label=metric.upper(), alpha=0.8)
    
    ax1.set_xlabel('Dataset Sample')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Performance Metrics Across Different Dataset Samples')
    ax1.set_xticks(positions + width)
    ax1.set_xticklabels([f'Dataset {i+1}' for i in range(len(all_results))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Segmentation-specific performance
    seg_types = ['bone', 'lung', 'soft_tissue']
    colors = ['coral', 'skyblue', 'lightgreen']
    
    for idx, seg_type in enumerate(seg_types):
        ax = fig.add_subplot(gs[1, idx])
        
        # Box plot for MSE across datasets
        mse_data = []
        for dataset_idx in range(len(all_results)):
            dataset_data = full_df[full_df['dataset'] == dataset_idx][f'{seg_type}_mse'].dropna()
            if len(dataset_data) > 0:
                mse_data.append(dataset_data.values)
        
        if mse_data:
            bp = ax.boxplot(mse_data, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[idx])
                patch.set_alpha(0.6)
        
        ax.set_xlabel('Dataset Sample')
        ax.set_ylabel('MSE')
        ax.set_title(f'{seg_type.replace("_", " ").title()} MSE Distribution')
        ax.set_xticklabels([f'D{i+1}' for i in range(len(all_results))])
        ax.grid(True, alpha=0.3)
    
    # 3. SSIM distribution violin plot
    ax3 = fig.add_subplot(gs[2, 0])
    sns.violinplot(data=full_df, x='dataset', y='ssim', ax=ax3)
    ax3.set_xlabel('Dataset Sample')
    ax3.set_ylabel('SSIM')
    ax3.set_title('SSIM Distribution Across Datasets')
    ax3.set_xticklabels([f'D{i+1}' for i in range(len(all_results))])
    
    # 4. Performance consistency heatmap
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Calculate correlation matrix of metrics across datasets
    metric_means = full_df.groupby(['dataset'])[['mse', 'psnr', 'ssim']].mean()
    correlation_matrix = metric_means.T.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                ax=ax4, vmin=-1, vmax=1, square=True)
    ax4.set_title('Dataset Performance Correlation')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Dataset')
    
    # 5. Convergence statistics
    ax5 = fig.add_subplot(gs[2, 2])
    
    # Calculate coefficient of variation for each metric
    cv_data = []
    for metric in ['mse', 'psnr', 'ssim']:
        means = full_df.groupby('dataset')[metric].mean()
        cv = means.std() / means.mean() if means.mean() != 0 else 0
        cv_data.append(cv)
    
    bars = ax5.bar(['MSE', 'PSNR', 'SSIM'], cv_data, color=['red', 'green', 'blue'], alpha=0.7)
    ax5.set_ylabel('Coefficient of Variation')
    ax5.set_title('Metric Stability Across Datasets')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cv in zip(bars, cv_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Hyperparameter Generalization Analysis\nPerformance Across Different Dataset Samples', 
                 fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return full_df

def plot_weight_consistency(all_results: List[Dict], save_path: str = None):
    """Analyze the consistency of learned weights across datasets."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Extract weight parameters to analyze
    weight_params = ['window_center', 'window_width', 'gamma', 'low_sigma']
    
    for idx, param in enumerate(weight_params):
        ax = axes[idx]
        
        # Collect weights for this parameter across all datasets and images
        param_values = []
        dataset_labels = []
        
        for result in all_results:
            for batch_weights in result['weights']:
                # batch_weights should have shape (batch_size, n_params)
                if param == 'window_center':
                    values = batch_weights[:, 0]
                elif param == 'window_width':
                    values = batch_weights[:, 1]
                elif param == 'gamma':
                    values = batch_weights[:, 2]
                elif param == 'low_sigma':
                    values = batch_weights[:, 3]
                
                param_values.extend(values)
                dataset_labels.extend([result['dataset_idx']] * len(values))
        
        # Create violin plot
        df = pd.DataFrame({
            'value': param_values,
            'dataset': dataset_labels
        })
        
        sns.violinplot(data=df, x='dataset', y='value', ax=ax)
        ax.set_title(f'{param.replace("_", " ").title()} Distribution')
        ax.set_xlabel('Dataset Sample')
        ax.set_ylabel('Parameter Value')
        ax.set_xticklabels([f'D{i+1}' for i in range(len(all_results))])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Learned Weight Parameters Across Different Datasets', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(all_results: List[Dict], save_path: str = None):
    """Generate summary statistics table."""
    
    # Collect all metrics
    all_metrics_data = []
    for result in all_results:
        df = extract_metrics_data(result['metrics'])
        df['dataset'] = result['dataset_idx']
        all_metrics_data.append(df)
    
    full_df = pd.concat(all_metrics_data, ignore_index=True)
    
    # Calculate summary statistics
    summary = full_df.groupby('dataset')[['mse', 'psnr', 'ssim', 'mae']].agg(['mean', 'std'])
    
    # Calculate overall statistics
    overall_mean = full_df[['mse', 'psnr', 'ssim', 'mae']].mean()
    overall_std = full_df[['mse', 'psnr', 'ssim', 'mae']].std()
    
    # Calculate cross-dataset consistency (standard deviation of means)
    dataset_means = full_df.groupby('dataset')[['mse', 'psnr', 'ssim', 'mae']].mean()
    consistency = dataset_means.std()
    
    print("\n=== HYPERPARAMETER GENERALIZATION SUMMARY ===\n")
    print("Per-Dataset Statistics:")
    print(summary)
    print("\nOverall Statistics (across all datasets):")
    print(f"Mean: {overall_mean}")
    print(f"Std:  {overall_std}")
    print("\nCross-Dataset Consistency (std of dataset means):")
    print(consistency)
    print("\nRelative Consistency (CV = std/mean):")
    print(consistency / overall_mean)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("=== HYPERPARAMETER GENERALIZATION SUMMARY ===\n\n")
            f.write("Per-Dataset Statistics:\n")
            f.write(summary.to_string())
            f.write("\n\nOverall Statistics (across all datasets):\n")
            f.write(f"Mean: {overall_mean}\n")
            f.write(f"Std:  {overall_std}\n")
            f.write("\nCross-Dataset Consistency (std of dataset means):\n")
            f.write(consistency.to_string())
            f.write("\n\nRelative Consistency (CV = std/mean):\n")
            f.write((consistency / overall_mean).to_string())
    
    return summary, overall_mean, overall_std, consistency

if __name__ == "__main__":
    # Use the same hyperparameters from individual_replicate.py
    conf = {
        "lr": 0.0067026164705251525,
        "eps": 1e-06,
        "sigma": 1.8581783667465153,
        "PRNGKey": 0,
        "n_steps": 500,
        "variant": "double-band",
        "momentum": 0.9,
        "max_sigma": 4,
        "optimizer": "amsgrad",
        "high_sigma": 5.73970154279467,
        "prior_weight": 0.8319818443566406,
        "weight_decay": 0.0001,
        "data_fidelity": "l2",
        "smooth_metric": "none",
        "tm_init_params": ["naive", None],
        "windowing_type": "sigmoid",
        "max_enhancement": 0.8,
        "segmentation_th": 0.6,
        "total_variation": 0,
        "constant_weights": False,
        "use_band_similarity": False,
        "log_samples": 10,
        "gmse_weight": 0.0
    }
    
    config = ExperimentArgs(**conf)
    
    # Run on multiple dataset samples
    print("Running hyperparameter generalization analysis...")
    all_results = run_multiple_datasets(config, n_datasets=5, batches_per_dataset=3)
    
    # Generate visualizations
    output_dir = "/Volumes/T7/projs/thesis/figures/hyperparameter_generalization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all visualizations
    plot_cross_dataset_performance(all_results, 
                                 save_path=os.path.join(output_dir, "cross_dataset_performance.png"))
    
    plot_weight_consistency(all_results,
                          save_path=os.path.join(output_dir, "weight_consistency.png"))
    
    # Generate summary statistics
    create_summary_statistics(all_results,
                            save_path=os.path.join(output_dir, "summary_statistics.txt"))
    
    print(f"\nAll figures saved to: {output_dir}")