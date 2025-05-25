#!/usr/bin/env python3
"""
Analyze regularization weights using the L-curve method and other techniques.

The L-curve method plots the reconstruction error vs regularization penalty
for different regularization weights. The optimal weight is typically at the
"corner" of the L-shaped curve, balancing data fidelity and regularization.
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.data.priors import get_priors
from chest_xray_sim.inverse import metrics
from chest_xray_sim.inverse.core import Optimizer
import chest_xray_sim.inverse.operators as ops


class SimpleOptimizer(Optimizer):
    """Simple optimizer for regularization analysis."""
    
    def __init__(self, tv_weight: float = 0.0, prior_weight: float = 0.0, value_ranges=None, **kwargs):
        super().__init__(**kwargs)
        self.tv_weight = tv_weight
        self.prior_weight = prior_weight
        self.value_ranges = value_ranges
    
    def _forward_single(self, txm, weights):
        """Forward model for a single image."""
        x = ops.negative_log(txm[jnp.newaxis, ...])  # Add batch dimension
        x = ops.window(x, weights["window_center"], weights["window_width"], 0.0, "linear")
        x = ops.range_normalize(x)
        x = ops.unsharp_masking(x, 2.0, weights["enhance_factor"])
        return ops.clipping(x)[0]  # Remove batch dimension
    
    def forward(self, txm, weights):
        """Simple forward model with batch support."""
        # Ensure txm is in valid range
        txm = jnp.clip(txm, 0.01, 0.99)
        
        # Simple forward model that works well
        x = -jnp.log(txm)
        x = (x - weights["window_center"]) / weights["window_width"]
        x = jnp.clip(x + weights["window_center"], 0.0, 1.0)
        
        # Optional: add unsharp masking
        # x = ops.unsharp_masking(x, 2.0, weights["enhance_factor"])
        
        return x
    
    def loss_fn(self, txm, weights, pred, target, segmentation=None):
        """Loss function with configurable regularization weights."""
        # Data fidelity term (MSE)
        mse = 0.5 * jnp.mean((pred - target) ** 2)
        
        # Regularization terms
        tv = metrics.total_variation(txm, reduction="mean")
        
        if segmentation is not None and self.value_ranges is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm, segmentation, self.value_ranges
            ).sum(axis=-1).mean()
        else:
            seg_penalty = 0.0
        
        # Total loss
        total_loss = mse + self.tv_weight * tv + self.prior_weight * seg_penalty
        
        return total_loss
    
    def compute_metrics(self, txm, weights, target, segmentation=None):
        """Compute metrics outside of JAX tracing."""
        pred = self.forward(txm, weights)
        
        # Data fidelity term (MSE)
        mse = 0.5 * jnp.mean((pred - target) ** 2)
        
        # Regularization terms
        tv = metrics.total_variation(txm, reduction="mean")
        
        if segmentation is not None and self.value_ranges is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm, segmentation, self.value_ranges
            ).sum(axis=-1).mean()
        else:
            seg_penalty = 0.0
            
        return mse, tv, seg_penalty


def compute_l_curve_points(
    images: jnp.ndarray,
    segmentations: jnp.ndarray,
    value_ranges: jnp.ndarray,
    tv_weights: List[float],
    prior_weights: List[float],
    n_steps: int = 200,
    batch_size: int = 4,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute L-curve data points for different regularization weights.
    
    Returns:
        Dictionary containing:
        - 'tv_curve': MSE and TV values for different TV weights
        - 'prior_curve': MSE and segmentation penalty for different prior weights
        - 'combined_curve': Results for combined weight variations
    """
    
    # Select a small batch for faster computation
    images = images[:batch_size]
    segmentations = segmentations[:batch_size]
    
    # Ensure images are in [0, 1] range
    images = images.astype(jnp.float32)
    if images.max() > 1.0:
        images = images / images.max()
    
    print(f"Image stats - min: {images.min():.3f}, max: {images.max():.3f}, mean: {images.mean():.3f}")
    
    # Initialize weights (keep as scalars for L-curve analysis)
    w0 = {
        "enhance_factor": 0.5,
        "window_center": 0.3,
        "window_width": 0.4,
    }
    
    # Initialize transmission maps closer to expected values
    txm0 = jnp.ones_like(images) * 0.8  # Most of the image should be high transmission
    
    results = {
        'tv_curve': {'weights': [], 'mse': [], 'tv': []},
        'prior_curve': {'weights': [], 'mse': [], 'prior': []},
        'combined_curve': {'tv_weights': [], 'prior_weights': [], 'mse': [], 'tv': [], 'prior': []},
    }
    
    # 1. TV weight analysis (prior_weight = 0)
    print("\nAnalyzing TV regularization weights...")
    for tv_weight in tqdm(tv_weights):
        opt = SimpleOptimizer(
            tv_weight=tv_weight,
            prior_weight=0.0,
            value_ranges=value_ranges,
            n_steps=n_steps,
            lr=0.001,
            optimizer=optax.chain(
                optax.clip_by_global_norm(1.0),  # Gradient clipping
                optax.adam(learning_rate=0.001)
            ),
        )
        
        state, _ = opt.optimize(images, txm0, w0, segmentations)
        
        if state is not None:
            # Compute metrics after optimization
            final_txm, final_weights = state
            mse, tv, _ = opt.compute_metrics(final_txm, final_weights, images, segmentations)
            
            # Extract values from JAX arrays
            mse_val = float(mse)
            tv_val = float(tv)
            
            results['tv_curve']['weights'].append(tv_weight)
            results['tv_curve']['mse'].append(mse_val)
            results['tv_curve']['tv'].append(tv_val)
    
    # 2. Prior weight analysis (tv_weight = optimal from above)
    # Find elbow point using maximum curvature
    if len(results['tv_curve']['mse']) > 2:
        mse_vals = np.array(results['tv_curve']['mse'])
        tv_vals = np.array(results['tv_curve']['tv'])
        
        # Normalize values
        mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min() + 1e-8)
        tv_norm = (tv_vals - tv_vals.min()) / (tv_vals.max() - tv_vals.min() + 1e-8)
        
        # Compute curvature
        curvature = compute_curvature(tv_norm, mse_norm)
        optimal_tv_idx = np.argmax(curvature)
        optimal_tv_weight = tv_weights[optimal_tv_idx]
    else:
        optimal_tv_weight = 1e-4
    
    print(f"\nOptimal TV weight: {optimal_tv_weight:.6f}")
    print("\nAnalyzing prior regularization weights...")
    
    for prior_weight in tqdm(prior_weights):
        opt = SimpleOptimizer(
            tv_weight=optimal_tv_weight,
            prior_weight=prior_weight,
            value_ranges=value_ranges,
            n_steps=n_steps,
            lr=0.001,
            optimizer=optax.chain(
                optax.clip_by_global_norm(1.0),  # Gradient clipping
                optax.adam(learning_rate=0.001)
            ),
        )
        
        state, _ = opt.optimize(images, txm0, w0, segmentations)
        
        if state is not None:
            # Compute metrics after optimization
            final_txm, final_weights = state
            mse, _, seg_penalty = opt.compute_metrics(final_txm, final_weights, images, segmentations)
            
            # Extract values from JAX arrays
            mse_val = float(mse)
            prior_val = float(seg_penalty)
            
            results['prior_curve']['weights'].append(prior_weight)
            results['prior_curve']['mse'].append(mse_val)
            results['prior_curve']['prior'].append(prior_val)
    
    # 3. Combined analysis (grid search)
    print("\nAnalyzing combined regularization weights...")
    tv_grid = tv_weights[::2]  # Sample every other point for efficiency
    prior_grid = prior_weights[::2]
    
    for tv_weight in tqdm(tv_grid):
        for prior_weight in prior_grid:
            opt = SimpleOptimizer(
                tv_weight=tv_weight,
                prior_weight=prior_weight,
                value_ranges=value_ranges,
                n_steps=n_steps,
                lr=0.001,
                optimizer=optax.chain(
                    optax.clip_by_global_norm(1.0),  # Gradient clipping
                    optax.adam(learning_rate=0.001)
                ),
            )
            
            state, _ = opt.optimize(images, txm0, w0, segmentations)
            
            if state is not None:
                # Compute metrics after optimization
                final_txm, final_weights = state
                mse, tv, seg_penalty = opt.compute_metrics(final_txm, final_weights, images, segmentations)
                
                # Extract values from JAX arrays
                mse_val = float(mse)
                tv_val = float(tv)
                prior_val = float(seg_penalty)
                
                results['combined_curve']['tv_weights'].append(tv_weight)
                results['combined_curve']['prior_weights'].append(prior_weight)
                results['combined_curve']['mse'].append(mse_val)
                results['combined_curve']['tv'].append(tv_val)
                results['combined_curve']['prior'].append(prior_val)
    
    # Convert lists to arrays
    for curve_type in results:
        for key in results[curve_type]:
            results[curve_type][key] = np.array(results[curve_type][key])
    
    return results


def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute curvature of a parametric curve.
    
    The curvature κ at each point is given by:
    κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    """
    # Compute derivatives using finite differences
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Compute curvature
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # Avoid division by zero
    curvature = np.zeros_like(numerator)
    mask = denominator > 1e-8
    curvature[mask] = numerator[mask] / denominator[mask]
    
    return curvature


def plot_l_curves(results: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """Plot L-curves and analysis results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. TV L-curve
    ax1 = fig.add_subplot(gs[0, 0])
    tv_data = results['tv_curve']
    if len(tv_data['mse']) > 0:
        ax1.plot(tv_data['tv'], tv_data['mse'], 'b.-', markersize=8)
        ax1.set_xlabel('Total Variation')
        ax1.set_ylabel('MSE')
        ax1.set_title('TV Regularization L-curve')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal point
        curvature = compute_curvature(
            np.log10(tv_data['tv']), 
            np.log10(tv_data['mse'])
        )
        if len(curvature) > 0:
            optimal_idx = np.argmax(curvature)
            ax1.plot(tv_data['tv'][optimal_idx], tv_data['mse'][optimal_idx], 
                    'ro', markersize=12, label=f'Optimal: {tv_data["weights"][optimal_idx]:.2e}')
            ax1.legend()
    
    # 2. Prior L-curve
    ax2 = fig.add_subplot(gs[0, 1])
    prior_data = results['prior_curve']
    if len(prior_data['mse']) > 0:
        ax2.plot(prior_data['prior'], prior_data['mse'], 'g.-', markersize=8)
        ax2.set_xlabel('Segmentation Penalty')
        ax2.set_ylabel('MSE')
        ax2.set_title('Prior Regularization L-curve')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal point
        curvature = compute_curvature(
            np.log10(prior_data['prior'] + 1e-10), 
            np.log10(prior_data['mse'])
        )
        if len(curvature) > 0:
            optimal_idx = np.argmax(curvature)
            ax2.plot(prior_data['prior'][optimal_idx], prior_data['mse'][optimal_idx], 
                    'ro', markersize=12, label=f'Optimal: {prior_data["weights"][optimal_idx]:.2e}')
            ax2.legend()
    
    # 3. Weight vs MSE plot
    ax3 = fig.add_subplot(gs[0, 2])
    if len(tv_data['mse']) > 0:
        ax3.plot(tv_data['weights'], tv_data['mse'], 'b.-', label='TV weight')
    if len(prior_data['mse']) > 0:
        ax3.plot(prior_data['weights'], prior_data['mse'], 'g.-', label='Prior weight')
    ax3.set_xlabel('Regularization Weight')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE vs Regularization Weights')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Combined heatmap (MSE)
    ax4 = fig.add_subplot(gs[1, :])
    combined_data = results['combined_curve']
    if len(combined_data['mse']) > 0:
        # Reshape data for heatmap
        tv_unique = np.unique(combined_data['tv_weights'])
        prior_unique = np.unique(combined_data['prior_weights'])
        mse_grid = np.zeros((len(prior_unique), len(tv_unique)))
        
        for i, (tv_w, prior_w, mse) in enumerate(zip(
            combined_data['tv_weights'], 
            combined_data['prior_weights'],
            combined_data['mse']
        )):
            tv_idx = np.where(tv_unique == tv_w)[0][0]
            prior_idx = np.where(prior_unique == prior_w)[0][0]
            mse_grid[prior_idx, tv_idx] = mse
        
        im = ax4.imshow(mse_grid, aspect='auto', origin='lower', cmap='viridis')
        ax4.set_xticks(range(len(tv_unique)))
        ax4.set_yticks(range(len(prior_unique)))
        ax4.set_xticklabels([f'{w:.2e}' for w in tv_unique], rotation=45)
        ax4.set_yticklabels([f'{w:.2e}' for w in prior_unique])
        ax4.set_xlabel('TV Weight')
        ax4.set_ylabel('Prior Weight')
        ax4.set_title('MSE Heatmap')
        plt.colorbar(im, ax=ax4, label='MSE')
    
    # 5. Pareto frontier analysis
    ax5 = fig.add_subplot(gs[2, :2])
    if len(combined_data['mse']) > 0:
        # Plot all points
        sc = ax5.scatter(combined_data['mse'], 
                        combined_data['tv'] + combined_data['prior'],
                        c=np.log10(combined_data['tv_weights']), 
                        cmap='coolwarm', alpha=0.6)
        
        # Find Pareto optimal points
        pareto_mask = compute_pareto_frontier(
            combined_data['mse'],
            combined_data['tv'] + combined_data['prior']
        )
        
        # Plot Pareto frontier
        pareto_mse = combined_data['mse'][pareto_mask]
        pareto_reg = (combined_data['tv'] + combined_data['prior'])[pareto_mask]
        sort_idx = np.argsort(pareto_mse)
        ax5.plot(pareto_mse[sort_idx], pareto_reg[sort_idx], 'k-', linewidth=2, label='Pareto Frontier')
        
        ax5.set_xlabel('MSE (Data Fidelity)')
        ax5.set_ylabel('Total Regularization')
        ax5.set_title('Pareto Frontier Analysis')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax5, label='log10(TV weight)')
    
    # 6. Curvature analysis
    ax6 = fig.add_subplot(gs[2, 2])
    if len(tv_data['mse']) > 0:
        curvature = compute_curvature(
            np.log10(tv_data['tv']), 
            np.log10(tv_data['mse'])
        )
        ax6.plot(tv_data['weights'], curvature, 'b.-')
        ax6.set_xlabel('TV Weight')
        ax6.set_ylabel('Curvature')
        ax6.set_title('L-curve Curvature')
        ax6.set_xscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Mark maximum
        if len(curvature) > 0:
            max_idx = np.argmax(curvature)
            ax6.axvline(tv_data['weights'][max_idx], color='r', linestyle='--', 
                       label=f'Max at {tv_data["weights"][max_idx]:.2e}')
            ax6.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l_curve_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    optimal_weights = {}
    
    if len(tv_data['mse']) > 0:
        curvature = compute_curvature(np.log10(tv_data['tv']), np.log10(tv_data['mse']))
        if len(curvature) > 0:
            optimal_tv_idx = np.argmax(curvature)
            optimal_weights['tv'] = float(tv_data['weights'][optimal_tv_idx])
    
    if len(prior_data['mse']) > 0:
        curvature = compute_curvature(
            np.log10(prior_data['prior'] + 1e-10), 
            np.log10(prior_data['mse'])
        )
        if len(curvature) > 0:
            optimal_prior_idx = np.argmax(curvature)
            optimal_weights['prior'] = float(prior_data['weights'][optimal_prior_idx])
    
    # Find optimal combined weights from Pareto frontier
    if len(combined_data['mse']) > 0:
        pareto_mask = compute_pareto_frontier(
            combined_data['mse'],
            combined_data['tv'] + combined_data['prior']
        )
        
        if np.any(pareto_mask):
            # Choose knee point on Pareto frontier
            pareto_points = np.column_stack([
                combined_data['mse'][pareto_mask],
                (combined_data['tv'] + combined_data['prior'])[pareto_mask]
            ])
            
            knee_idx = find_knee_point(pareto_points)
            pareto_indices = np.where(pareto_mask)[0]
            optimal_idx = pareto_indices[knee_idx]
            
            optimal_weights['combined_tv'] = float(combined_data['tv_weights'][optimal_idx])
            optimal_weights['combined_prior'] = float(combined_data['prior_weights'][optimal_idx])
    
    # Save results
    import json
    with open(os.path.join(output_dir, 'optimal_weights.json'), 'w') as f:
        json.dump(optimal_weights, f, indent=2)
    
    print("\nOptimal weights found:")
    for key, value in optimal_weights.items():
        print(f"  {key}: {value:.6f}")
    
    return optimal_weights


def compute_pareto_frontier(obj1: np.ndarray, obj2: np.ndarray) -> np.ndarray:
    """
    Compute Pareto frontier for two objectives (both to be minimized).
    
    Returns boolean mask of Pareto optimal points.
    """
    n_points = len(obj1)
    pareto_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                if obj1[j] <= obj1[i] and obj2[j] <= obj2[i] and \
                   (obj1[j] < obj1[i] or obj2[j] < obj2[i]):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask


def find_knee_point(points: np.ndarray) -> int:
    """
    Find knee point in a curve using the kneedle algorithm.
    
    Args:
        points: Array of shape (n_points, 2) with x and y coordinates
        
    Returns:
        Index of the knee point
    """
    # Normalize points
    x = points[:, 0]
    y = points[:, 1]
    
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    
    # Sort by x coordinate
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_sorted = y_norm[sort_idx]
    
    # Compute distances from line connecting first and last points
    p1 = np.array([x_sorted[0], y_sorted[0]])
    p2 = np.array([x_sorted[-1], y_sorted[-1]])
    
    distances = []
    for i in range(len(x_sorted)):
        p = np.array([x_sorted[i], y_sorted[i]])
        # Distance from point to line
        dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
    
    # Find maximum distance
    knee_idx_sorted = np.argmax(distances)
    knee_idx = sort_idx[knee_idx_sorted]
    
    return knee_idx


def main():
    parser = argparse.ArgumentParser(description='Analyze regularization weights using L-curve method')
    parser.add_argument('--data-dir', type=str, default=os.environ.get('IMAGE_PATH', '/Volumes/T7/projs/thesis/data/CheXpert-v1.0-small'),
                       help='Path to CheXpert images')
    parser.add_argument('--meta-dir', type=str, default=os.environ.get('META_PATH', '/Volumes/T7/projs/thesis/data/CheXpert-v1.0-small'),
                       help='Path to CheXpert metadata')
    parser.add_argument('--mask-dir', type=str, default=os.environ.get('MASK_DIR', '/Volumes/T7/projs/thesis/data/masks'),
                       help='Path to segmentation masks')
    parser.add_argument('--cache-dir', type=str, default=os.environ.get('CACHE_DIR', '/Volumes/T7/projs/thesis/cache'),
                       help='Cache directory')
    parser.add_argument('--output-dir', type=str, default='./regularization_analysis',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for analysis')
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
    parser.add_argument('--n-weights', type=int, default=20,
                       help='Number of weight values to test')
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = get_segmentation_dataset(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split="train",
        frontal_lateral="Frontal",
        batch_size=args.batch_size,
    )
    
    # Get a batch
    images_batch, masks_batch, _ = next(iter(dataset))
    
    # Process data
    images = jnp.array(images_batch.cpu().numpy())
    if images.ndim == 4 and images.shape[1] == 1:
        images = images.squeeze(1)
    
    masks = jnp.array(masks_batch.cpu().numpy())
    seg_labels, segmentations = batch_get_exclusive_masks(masks, threshold=0.6)
    
    # Get priors
    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)
    
    # Generate weight ranges
    tv_weights = np.logspace(np.log10(args.tv_min), np.log10(args.tv_max), args.n_weights)
    prior_weights = np.linspace(args.prior_min, args.prior_max, args.n_weights)
    
    # Compute L-curve points
    print("\nComputing L-curve points...")
    results = compute_l_curve_points(
        images, segmentations, value_ranges,
        tv_weights.tolist(), prior_weights.tolist(),
        n_steps=args.n_steps,
        batch_size=args.batch_size
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_l_curves(results, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()