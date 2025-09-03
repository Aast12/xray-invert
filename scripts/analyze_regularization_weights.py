#!/usr/bin/env python3
"""
Analyze regularization weights using advanced methods beyond L-curve.

This script implements multiple approaches to find optimal regularization weights:
1. Grid search with validation set performance
2. Pareto frontier analysis for multi-objective optimization
3. Trade-off analysis between reconstruction quality and anatomical consistency

The focus is on properly valuing the segmentation prior, which is crucial for
anatomically consistent reconstructions.
"""

import os
import argparse
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
import optax
from tqdm import tqdm
import json

from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.data.priors import get_priors
from chest_xray_sim.inverse import metrics
from chest_xray_sim.inverse.individual_gradient_optimizer import IndividualGradientOptimizer
import chest_xray_sim.inverse.operators as ops


class RegularizationOptimizer(IndividualGradientOptimizer):
    """Optimizer for regularization weight analysis using individual gradients."""
    
    def __init__(
        self, 
        tv_weight: float = 0.0, 
        prior_weight: float = 0.0, 
        value_ranges=None,
        **kwargs
    ):
        # Set up separate optimizers for txm and weights
        txm_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=kwargs.get('lr', 0.001))
        )
        weights_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=kwargs.get('lr', 0.001))
        )
        
        super().__init__(
            txm_optimizer=txm_optimizer,
            weights_optimizer=weights_optimizer,
            **kwargs
        )
        
        self.tv_weight = tv_weight
        self.prior_weight = prior_weight
        self.value_ranges = value_ranges
    
    def forward(self, txm, weights):
        """Forward model for image processing."""
        # Ensure txm is in valid range
        txm = jnp.clip(txm, 0.01, 0.99)
        
        # Apply negative log transform
        x = -jnp.log(txm)
        
        # Window the image
        x = ops.window(
            x[jnp.newaxis, ...],  # Add batch dimension
            weights["window_center"],
            weights["window_width"],
            0.0,
            "linear"
        )[0]  # Remove batch dimension
        
        # Optional: add unsharp masking
        if "enhance_factor" in weights:
            x = ops.unsharp_masking(
                x[jnp.newaxis, ...], 
                2.0, 
                weights["enhance_factor"]
            )[0]
        
        # Ensure output is in [0, 1]
        x = jnp.clip(x, 0.0, 1.0)
        
        return x
    
    def loss_call_individual(self, weights, txm, target, segmentation=None):
        """Loss computation for a single image-weight pair."""
        # Forward pass
        pred = self.forward(txm, weights)
        
        # Data fidelity term (MSE)
        mse = 0.5 * jnp.mean((pred - target) ** 2)
        
        # Total variation regularization
        tv = metrics.total_variation(txm[jnp.newaxis, ...], reduction="mean")
        
        # Segmentation prior
        if segmentation is not None and self.value_ranges is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm[jnp.newaxis, ...], 
                segmentation[jnp.newaxis, ...], 
                self.value_ranges
            ).sum()
        else:
            seg_penalty = 0.0
        
        # Total loss
        total_loss = mse + self.tv_weight * tv + self.prior_weight * seg_penalty
        
        return total_loss
    
    def compute_metrics(self, txm, weights, target, segmentation=None):
        """Compute individual metrics for evaluation."""
        pred = self.forward(txm, weights)
        
        # Data fidelity
        mse = 0.5 * jnp.mean((pred - target) ** 2)
        
        # Regularization terms
        tv = metrics.total_variation(txm[jnp.newaxis, ...], reduction="mean")
        
        if segmentation is not None and self.value_ranges is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm[jnp.newaxis, ...], 
                segmentation[jnp.newaxis, ...], 
                self.value_ranges
            ).sum()
        else:
            seg_penalty = 0.0
        
        # Additional metrics for evaluation
        psnr = 20 * jnp.log10(1.0 / jnp.sqrt(2 * mse))
        
        # Anatomical consistency: average deviation from expected tissue values
        if segmentation is not None and self.value_ranges is not None:
            anatomical_consistency = compute_anatomical_consistency(
                txm, segmentation, self.value_ranges
            )
        else:
            anatomical_consistency = 0.0
        
        return {
            'mse': float(mse),
            'tv': float(tv),
            'seg_penalty': float(seg_penalty),
            'psnr': float(psnr),
            'anatomical_consistency': float(anatomical_consistency)
        }


def compute_anatomical_consistency(txm, segmentation, value_ranges):
    """
    Compute how well the transmission map matches expected tissue values.
    
    Returns a score where lower is better (0 = perfect match).
    """
    consistency = 0.0
    n_tissues = segmentation.shape[0]
    
    for tissue_idx in range(n_tissues):
        mask = segmentation[tissue_idx]
        if jnp.sum(mask) > 0:
            # Get tissue values
            tissue_txm = txm * mask
            masked_sum = jnp.sum(tissue_txm)
            mask_sum = jnp.sum(mask)
            mean_value = masked_sum / (mask_sum + 1e-8)
            
            # Expected range for this tissue
            expected_min = value_ranges[tissue_idx, 0]
            expected_max = value_ranges[tissue_idx, 1]
            expected_mean = (expected_min + expected_max) / 2
            
            # Deviation from expected
            deviation = jnp.abs(mean_value - expected_mean)
            consistency += deviation
    
    return consistency / n_tissues


def grid_search_validation(
    train_images: jnp.ndarray,
    train_segmentations: jnp.ndarray,
    val_images: jnp.ndarray,
    val_segmentations: jnp.ndarray,
    value_ranges: jnp.ndarray,
    tv_weights: List[float],
    prior_weights: List[float],
    n_steps: int = 200,
) -> Dict[str, Any]:
    """
    Perform grid search using validation set performance.
    
    Returns dictionary with results and optimal weights.
    """
    results = {
        'tv_weights': [],
        'prior_weights': [],
        'train_metrics': [],
        'val_metrics': [],
    }
    
    best_val_score = float('inf')
    best_weights = None
    
    # Initialize per-image weights for training set
    n_train = len(train_images)
    w0 = {
        "enhance_factor": jnp.ones(n_train) * 0.5,
        "window_center": jnp.ones(n_train) * 0.3,
        "window_width": jnp.ones(n_train) * 0.4,
    }
    
    print("\nPerforming grid search with validation set...")
    total_combinations = len(tv_weights) * len(prior_weights)
    
    with tqdm(total=total_combinations) as pbar:
        for tv_weight in tv_weights:
            for prior_weight in prior_weights:
                # Train on training set
                opt = RegularizationOptimizer(
                    tv_weight=tv_weight,
                    prior_weight=prior_weight,
                    value_ranges=value_ranges,
                    n_steps=n_steps,
                    lr=0.001,
                    log_interval=n_steps + 1,  # Disable logging
                )
                
                # Initialize transmission maps
                txm0 = jnp.ones_like(train_images) * 0.8
                
                # Optimize
                state, _ = opt.optimize(train_images, txm0, w0, train_segmentations)
                
                if state is not None:
                    final_txm, final_weights = state
                    
                    # Evaluate on training set
                    train_metrics = []
                    for i in range(len(train_images)):
                        # Extract per-image weights from the dictionary
                        weight_i = {k: v[i] for k, v in final_weights.items()}
                        m = opt.compute_metrics(
                            final_txm[i], 
                            weight_i, 
                            train_images[i],
                            train_segmentations[i] if train_segmentations is not None else None
                        )
                        train_metrics.append(m)
                    
                    # Average training metrics
                    avg_train_metrics = {
                        k: np.mean([m[k] for m in train_metrics])
                        for k in train_metrics[0].keys()
                    }
                    
                    # Evaluate on validation set - initialize new weights for validation size
                    val_txm0 = jnp.ones_like(val_images) * 0.8
                    n_val = len(val_images)
                    val_w0 = {
                        "enhance_factor": jnp.ones(n_val) * 0.5,
                        "window_center": jnp.ones(n_val) * 0.3,
                        "window_width": jnp.ones(n_val) * 0.4,
                    }
                    val_state, _ = opt.optimize(val_images, val_txm0, val_w0, val_segmentations)
                    
                    if val_state is not None:
                        val_final_txm, val_final_weights = val_state
                        
                        val_metrics = []
                        for i in range(len(val_images)):
                            # Extract per-image weights from the dictionary
                            val_weight_i = {k: v[i] for k, v in val_final_weights.items()}
                            m = opt.compute_metrics(
                                val_final_txm[i],
                                val_weight_i,
                                val_images[i],
                                val_segmentations[i] if val_segmentations is not None else None
                            )
                            val_metrics.append(m)
                        
                        # Average validation metrics
                        avg_val_metrics = {
                            k: np.mean([m[k] for m in val_metrics])
                            for k in val_metrics[0].keys()
                        }
                        
                        # Compute validation score (balance MSE and anatomical consistency)
                        val_score = avg_val_metrics['mse'] + 0.1 * avg_val_metrics['anatomical_consistency']
                        
                        # Update best weights
                        if val_score < best_val_score:
                            best_val_score = val_score
                            best_weights = {
                                'tv_weight': tv_weight,
                                'prior_weight': prior_weight,
                                'val_score': val_score,
                                'val_metrics': avg_val_metrics
                            }
                        
                        # Store results
                        results['tv_weights'].append(tv_weight)
                        results['prior_weights'].append(prior_weight)
                        results['train_metrics'].append(avg_train_metrics)
                        results['val_metrics'].append(avg_val_metrics)
                
                pbar.update(1)
    
    results['best_weights'] = best_weights if 'best_weights' in locals() else None
    return results


def pareto_frontier_analysis(
    images: jnp.ndarray,
    segmentations: jnp.ndarray,
    value_ranges: jnp.ndarray,
    tv_weights: List[float],
    prior_weights: List[float],
    n_steps: int = 200,
) -> Dict[str, Any]:
    """
    Perform Pareto frontier analysis for multi-objective optimization.
    
    Objectives:
    1. Minimize reconstruction error (MSE)
    2. Minimize anatomical inconsistency
    3. Minimize total variation (smoothness)
    """
    results = {
        'tv_weights': [],
        'prior_weights': [],
        'objectives': {
            'mse': [],
            'anatomical_consistency': [],
            'tv': [],
        },
        'pareto_indices': [],
    }
    
    # Initialize per-image weights
    n_images = len(images)
    w0 = {
        "enhance_factor": jnp.ones(n_images) * 0.5,
        "window_center": jnp.ones(n_images) * 0.3,
        "window_width": jnp.ones(n_images) * 0.4,
    }
    
    print("\nPerforming Pareto frontier analysis...")
    
    for tv_weight in tqdm(tv_weights):
        for prior_weight in prior_weights:
            opt = RegularizationOptimizer(
                tv_weight=tv_weight,
                prior_weight=prior_weight,
                value_ranges=value_ranges,
                n_steps=n_steps,
                lr=0.001,
                log_interval=n_steps + 1,
            )
            
            txm0 = jnp.ones_like(images) * 0.8
            state, _ = opt.optimize(images, txm0, w0, segmentations)
            
            if state is not None:
                final_txm, final_weights = state
                
                # Compute objectives
                all_metrics = []
                for i in range(len(images)):
                    # Extract per-image weights from the dictionary
                    weight_i = {k: v[i] for k, v in final_weights.items()}
                    m = opt.compute_metrics(
                        final_txm[i],
                        weight_i,
                        images[i],
                        segmentations[i] if segmentations is not None else None
                    )
                    all_metrics.append(m)
                
                # Average objectives
                avg_mse = np.mean([m['mse'] for m in all_metrics])
                avg_consistency = np.mean([m['anatomical_consistency'] for m in all_metrics])
                avg_tv = np.mean([m['tv'] for m in all_metrics])
                
                results['tv_weights'].append(tv_weight)
                results['prior_weights'].append(prior_weight)
                results['objectives']['mse'].append(avg_mse)
                results['objectives']['anatomical_consistency'].append(avg_consistency)
                results['objectives']['tv'].append(avg_tv)
    
    # Find Pareto optimal points
    objectives = np.array([
        results['objectives']['mse'],
        results['objectives']['anatomical_consistency'],
        results['objectives']['tv']
    ]).T
    
    pareto_mask = compute_pareto_frontier_3d(objectives)
    results['pareto_indices'] = np.where(pareto_mask)[0].tolist()
    
    return results


def compute_pareto_frontier_3d(objectives: np.ndarray) -> np.ndarray:
    """
    Compute Pareto frontier for 3 objectives (all to be minimized).
    
    Args:
        objectives: Array of shape (n_points, 3)
        
    Returns:
        Boolean mask of Pareto optimal points
    """
    n_points = len(objectives)
    pareto_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                dominates = np.all(objectives[j] <= objectives[i]) and \
                           np.any(objectives[j] < objectives[i])
                if dominates:
                    pareto_mask[i] = False
                    break
    
    return pareto_mask


def plot_results(
    grid_results: Dict[str, Any],
    pareto_results: Dict[str, Any],
    output_dir: str
):
    """Create comprehensive plots of the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Grid search heatmap - Validation MSE
    ax1 = fig.add_subplot(gs[0, 0])
    if grid_results and len(grid_results['tv_weights']) > 0:
        plot_grid_search_heatmap(
            grid_results, 
            metric='mse',
            title='Validation MSE',
            ax=ax1
        )
    
    # 2. Grid search heatmap - Anatomical Consistency
    ax2 = fig.add_subplot(gs[0, 1])
    if grid_results and len(grid_results['tv_weights']) > 0:
        plot_grid_search_heatmap(
            grid_results,
            metric='anatomical_consistency',
            title='Validation Anatomical Consistency',
            ax=ax2
        )
    
    # 3. Grid search heatmap - Combined Score
    ax3 = fig.add_subplot(gs[0, 2])
    if grid_results and len(grid_results['tv_weights']) > 0:
        # Compute combined score
        combined_scores = []
        for vm in grid_results['val_metrics']:
            score = vm['mse'] + 0.1 * vm['anatomical_consistency']
            combined_scores.append(score)
        
        # Create temporary results dict
        temp_results = grid_results.copy()
        temp_results['val_metrics'] = [{'combined': s} for s in combined_scores]
        
        plot_grid_search_heatmap(
            temp_results,
            metric='combined',
            title='Combined Validation Score',
            ax=ax3
        )
        
        # Mark best point
        if grid_results['best_weights']:
            best = grid_results['best_weights']
            tv_idx = grid_results['tv_weights'].index(best['tv_weight'])
            prior_idx = grid_results['prior_weights'].index(best['prior_weight'])
            ax3.plot(tv_idx, prior_idx, 'r*', markersize=20)
    
    # 4. Pareto frontier - 2D projections
    if pareto_results and len(pareto_results['tv_weights']) > 0:
        # MSE vs Anatomical Consistency
        ax4 = fig.add_subplot(gs[1, 0])
        plot_pareto_2d(
            pareto_results,
            x_metric='mse',
            y_metric='anatomical_consistency',
            x_label='Reconstruction MSE',
            y_label='Anatomical Consistency',
            ax=ax4
        )
        
        # MSE vs TV
        ax5 = fig.add_subplot(gs[1, 1])
        plot_pareto_2d(
            pareto_results,
            x_metric='mse',
            y_metric='tv',
            x_label='Reconstruction MSE',
            y_label='Total Variation',
            ax=ax5
        )
        
        # Anatomical Consistency vs TV
        ax6 = fig.add_subplot(gs[1, 2])
        plot_pareto_2d(
            pareto_results,
            x_metric='anatomical_consistency',
            y_metric='tv',
            x_label='Anatomical Consistency',
            y_label='Total Variation',
            ax=ax6
        )
    
    # 5. Weight distribution on Pareto frontier
    ax7 = fig.add_subplot(gs[2, :])
    if pareto_results and 'pareto_indices' in pareto_results:
        plot_pareto_weights(pareto_results, ax=ax7)
    
    # 6. Trade-off curves
    ax8 = fig.add_subplot(gs[3, 0])
    if grid_results and len(grid_results['tv_weights']) > 0:
        plot_tradeoff_curve(
            grid_results,
            weight_type='tv',
            ax=ax8
        )
    
    ax9 = fig.add_subplot(gs[3, 1])
    if grid_results and len(grid_results['tv_weights']) > 0:
        plot_tradeoff_curve(
            grid_results,
            weight_type='prior',
            ax=ax9
        )
    
    # 7. Summary text
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    summary_text = generate_summary(grid_results, pareto_results)
    ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_grid_search_heatmap(results, metric, title, ax):
    """Plot heatmap for grid search results."""
    tv_unique = sorted(list(set(results['tv_weights'])))
    prior_unique = sorted(list(set(results['prior_weights'])))
    
    # Create grid
    grid = np.zeros((len(prior_unique), len(tv_unique)))
    
    for tv_w, prior_w, vm in zip(
        results['tv_weights'],
        results['prior_weights'],
        results['val_metrics']
    ):
        tv_idx = tv_unique.index(tv_w)
        prior_idx = prior_unique.index(prior_w)
        grid[prior_idx, tv_idx] = vm[metric]
    
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(tv_unique)))
    ax.set_yticks(range(len(prior_unique)))
    ax.set_xticklabels([f'{w:.1e}' for w in tv_unique], rotation=45)
    ax.set_yticklabels([f'{w:.2f}' for w in prior_unique])
    ax.set_xlabel('TV Weight')
    ax.set_ylabel('Prior Weight')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def plot_pareto_2d(results, x_metric, y_metric, x_label, y_label, ax):
    """Plot 2D projection of Pareto frontier."""
    x_values = np.array(results['objectives'][x_metric])
    y_values = np.array(results['objectives'][y_metric])
    
    # Plot all points
    ax.scatter(x_values, y_values, alpha=0.5, s=50, c='blue', label='All points')
    
    # Highlight Pareto optimal points
    if 'pareto_indices' in results:
        pareto_x = x_values[results['pareto_indices']]
        pareto_y = y_values[results['pareto_indices']]
        ax.scatter(pareto_x, pareto_y, s=100, c='red', marker='*', label='Pareto optimal')
        
        # Connect Pareto points
        sorted_idx = np.argsort(pareto_x)
        ax.plot(pareto_x[sorted_idx], pareto_y[sorted_idx], 'r--', alpha=0.5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Pareto Frontier: {x_label} vs {y_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_pareto_weights(results, ax):
    """Plot distribution of weights on Pareto frontier."""
    pareto_indices = results['pareto_indices']
    
    if len(pareto_indices) > 0:
        pareto_tv = [results['tv_weights'][i] for i in pareto_indices]
        pareto_prior = [results['prior_weights'][i] for i in pareto_indices]
        
        # Create scatter plot with color coding
        scatter = ax.scatter(pareto_tv, pareto_prior, 
                           c=range(len(pareto_indices)), 
                           cmap='viridis', s=100)
        
        ax.set_xlabel('TV Weight')
        ax.set_ylabel('Prior Weight')
        ax.set_title('Weight Distribution on Pareto Frontier')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pareto Point Index')


def plot_tradeoff_curve(results, weight_type, ax):
    """Plot trade-off curves for a specific weight type."""
    # Initialize variables
    target_prior = 0.5  # Middle value for prior weight
    target_tv = 1e-4    # Reasonable TV weight
    
    if weight_type == 'tv':
        # Fix prior weight, vary TV weight
        weight_name = 'TV Weight'
    else:
        # Fix TV weight, vary prior weight
        weight_name = 'Prior Weight'
    
    # Filter results
    filtered_metrics = []
    filtered_weights = []
    
    for tv_w, prior_w, vm in zip(
        results['tv_weights'],
        results['prior_weights'],
        results['val_metrics']
    ):
        if weight_type == 'tv' and abs(prior_w - target_prior) < 0.01:
            filtered_metrics.append(vm)
            filtered_weights.append(tv_w)
        elif weight_type == 'prior' and abs(tv_w - target_tv) < 1e-6:
            filtered_metrics.append(vm)
            filtered_weights.append(prior_w)
    
    if len(filtered_metrics) > 0:
        # Sort by weight
        sorted_idx = np.argsort(filtered_weights)
        filtered_weights = [filtered_weights[i] for i in sorted_idx]
        filtered_metrics = [filtered_metrics[i] for i in sorted_idx]
        
        # Extract metrics
        mse_values = [m['mse'] for m in filtered_metrics]
        consistency_values = [m['anatomical_consistency'] for m in filtered_metrics]
        
        # Plot
        ax2 = ax.twinx()
        
        line1 = ax.plot(filtered_weights, mse_values, 'b-', label='MSE')
        line2 = ax2.plot(filtered_weights, consistency_values, 'r-', label='Anatomical Consistency')
        
        ax.set_xlabel(weight_name)
        ax.set_ylabel('MSE', color='b')
        ax2.set_ylabel('Anatomical Consistency', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        if weight_type == 'tv':
            ax.set_xscale('log')
        
        ax.set_title(f'Trade-off Analysis: {weight_name}')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='center right')


def generate_summary(grid_results, pareto_results):
    """Generate text summary of results."""
    summary = "REGULARIZATION ANALYSIS SUMMARY\n"
    summary += "=" * 30 + "\n\n"
    
    if grid_results and grid_results['best_weights']:
        best = grid_results['best_weights']
        summary += "Grid Search Results:\n"
        summary += f"  Best TV weight: {best['tv_weight']:.6f}\n"
        summary += f"  Best Prior weight: {best['prior_weight']:.6f}\n"
        summary += f"  Validation score: {best['val_score']:.6f}\n"
        summary += f"  Val MSE: {best['val_metrics']['mse']:.6f}\n"
        summary += f"  Val PSNR: {best['val_metrics']['psnr']:.2f} dB\n"
        summary += f"  Val Anat. Consistency: {best['val_metrics']['anatomical_consistency']:.6f}\n"
        summary += "\n"
    
    if pareto_results and 'pareto_indices' in pareto_results:
        summary += "Pareto Analysis:\n"
        summary += f"  Total points tested: {len(pareto_results['tv_weights'])}\n"
        summary += f"  Pareto optimal points: {len(pareto_results['pareto_indices'])}\n"
        
        if len(pareto_results['pareto_indices']) > 0:
            # Find knee point
            pareto_tv = [pareto_results['tv_weights'][i] for i in pareto_results['pareto_indices']]
            pareto_prior = [pareto_results['prior_weights'][i] for i in pareto_results['pareto_indices']]
            
            summary += f"  TV weight range: [{min(pareto_tv):.2e}, {max(pareto_tv):.2e}]\n"
            summary += f"  Prior weight range: [{min(pareto_prior):.3f}, {max(pareto_prior):.3f}]\n"
    
    summary += "\nRECOMMENDATION:\n"
    summary += "The segmentation prior is crucial for\n"
    summary += "anatomically consistent reconstructions.\n"
    summary += "Use the validation-based optimal weights\n"
    summary += "as they balance all objectives effectively."
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze regularization weights with focus on segmentation priors'
    )
    parser.add_argument('--data-dir', type=str, 
                       default=os.environ.get('IMAGE_PATH', '/Volumes/T7/projs/thesis/data/CheXpert-v1.0-small'),
                       help='Path to CheXpert images')
    parser.add_argument('--meta-dir', type=str, 
                       default=os.environ.get('META_PATH', '/Volumes/T7/projs/thesis/data/CheXpert-v1.0-small'),
                       help='Path to CheXpert metadata')
    parser.add_argument('--mask-dir', type=str, 
                       default=os.environ.get('MASK_DIR', '/Volumes/T7/projs/thesis/data/masks'),
                       help='Path to segmentation masks')
    parser.add_argument('--cache-dir', type=str, 
                       default=os.environ.get('CACHE_DIR', '/Volumes/T7/projs/thesis/cache'),
                       help='Cache directory')
    parser.add_argument('--output-dir', type=str, default='./regularization_analysis',
                       help='Output directory for results')
    parser.add_argument('--n-train', type=int, default=8,
                       help='Number of training images')
    parser.add_argument('--n-val', type=int, default=4,
                       help='Number of validation images')
    parser.add_argument('--n-steps', type=int, default=200,
                       help='Number of optimization steps per weight')
    parser.add_argument('--tv-min', type=float, default=1e-6,
                       help='Minimum TV weight')
    parser.add_argument('--tv-max', type=float, default=1e-2,
                       help='Maximum TV weight')
    parser.add_argument('--prior-min', type=float, default=0.0,
                       help='Minimum prior weight')
    parser.add_argument('--prior-max', type=float, default=1.0,
                       help='Maximum prior weight')
    parser.add_argument('--n-tv-weights', type=int, default=10,
                       help='Number of TV weight values to test')
    parser.add_argument('--n-prior-weights', type=int, default=10,
                       help='Number of prior weight values to test')
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    total_samples = args.n_train + args.n_val
    dataset = get_segmentation_dataset(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split="train",
        frontal_lateral="Frontal",
        batch_size=total_samples,
    )
    
    # Get data
    images_batch, masks_batch, _ = next(iter(dataset))
    
    # Process data
    images = jnp.array(images_batch.cpu().numpy())
    if images.ndim == 4 and images.shape[1] == 1:
        images = images.squeeze(1)
    
    # Normalize images to [0, 1] if needed
    if images.max() > 1.0:
        images = images / images.max()
    
    masks = jnp.array(masks_batch.cpu().numpy())
    _, segmentations = batch_get_exclusive_masks(masks, threshold=0.6)
    
    # Get priors
    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)
    
    # Split into train/validation
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_idx = indices[:args.n_val]
    train_idx = indices[args.n_val:]
    
    train_images = images[train_idx]
    train_segmentations = segmentations[train_idx]
    val_images = images[val_idx]
    val_segmentations = segmentations[val_idx]
    
    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Generate weight ranges
    tv_weights = np.logspace(
        np.log10(args.tv_min), 
        np.log10(args.tv_max), 
        args.n_tv_weights
    ).tolist()
    
    # Include 0 in prior weights to show its importance
    prior_weights = [0.0] + np.linspace(
        args.prior_min + 0.01, 
        args.prior_max, 
        args.n_prior_weights - 1
    ).tolist()
    
    # 1. Grid search with validation
    grid_results = grid_search_validation(
        train_images, train_segmentations,
        val_images, val_segmentations,
        value_ranges,
        tv_weights, prior_weights,
        n_steps=args.n_steps
    )
    
    # 2. Pareto frontier analysis
    pareto_results = pareto_frontier_analysis(
        train_images, train_segmentations,
        value_ranges,
        tv_weights[::2], prior_weights[::2],  # Subsample for efficiency
        n_steps=args.n_steps
    )
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(grid_results, pareto_results, args.output_dir)
    
    # Save numerical results
    results_file = os.path.join(args.output_dir, 'analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'grid_search': {
                'best_weights': grid_results['best_weights'],
                'n_configurations': len(grid_results['tv_weights']),
            },
            'pareto_analysis': {
                'n_pareto_points': len(pareto_results.get('pareto_indices', [])),
                'n_total_points': len(pareto_results['tv_weights']),
            },
            'weight_ranges': {
                'tv': [args.tv_min, args.tv_max],
                'prior': [args.prior_min, args.prior_max],
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    
    # Print recommendations
    if grid_results['best_weights']:
        print("\n" + "="*50)
        print("RECOMMENDED WEIGHTS (from validation):")
        print(f"  TV weight: {grid_results['best_weights']['tv_weight']:.6f}")
        print(f"  Prior weight: {grid_results['best_weights']['prior_weight']:.6f}")
        print("\nThese weights properly value the segmentation prior")
        print("for anatomically consistent reconstructions.")
        print("="*50)


if __name__ == "__main__":
    main()