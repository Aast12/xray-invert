#!/usr/bin/env python3
"""
Simple L-curve analysis for regularization weights without segmentation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

from chest_xray_sim.inverse import metrics
from chest_xray_sim.inverse.core import Optimizer


class SimpleOptimizer(Optimizer):
    """Simple optimizer for regularization analysis."""
    
    def __init__(self, tv_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.tv_weight = tv_weight
        self.last_mse = 0.0
        self.last_tv = 0.0
    
    def forward(self, txm, weights):
        """Simple forward model."""
        # Clip txm to avoid log(0)
        txm = jnp.clip(txm, 0.01, 0.99)
        
        # Simple forward model
        x = -jnp.log(txm)
        x = (x - weights["window_center"]) / weights["window_width"]
        x = jnp.clip(x + weights["window_center"], 0.0, 1.0)
        return x
    
    def loss_fn(self, txm, weights, pred, target, segmentation=None):
        """Loss function with TV regularization."""
        # Data fidelity term
        mse = 0.5 * jnp.mean((pred - target) ** 2)
        
        # Total variation
        tv = metrics.total_variation(txm, reduction="mean")
        
        # Store for analysis
        self.last_mse = mse
        self.last_tv = tv
        
        loss = mse + self.tv_weight * tv
        
        return loss


def create_synthetic_data(batch_size=2, image_size=256):
    """Create synthetic chest X-ray data for testing."""
    # Create simple synthetic transmission maps
    x = jnp.linspace(-1, 1, image_size)
    y = jnp.linspace(-1, 1, image_size)
    xx, yy = jnp.meshgrid(x, y)
    
    # Create circular "lung" regions with higher transmission
    lung1 = jnp.exp(-((xx + 0.3)**2 + yy**2) / 0.3)
    lung2 = jnp.exp(-((xx - 0.3)**2 + yy**2) / 0.3)
    
    # Background tissue
    background = 0.3 * jnp.ones((image_size, image_size))
    
    # Combine
    txm_true = jnp.clip(background + 0.5 * lung1 + 0.5 * lung2, 0.1, 0.9)
    
    # Create batch
    txm_true = jnp.stack([txm_true] * batch_size)
    
    # Generate target images using known forward model
    weights_true = {
        "window_center": 0.4,
        "window_width": 0.3,
        "enhance_factor": 0.6,
    }
    
    # Simple forward model without optimizer
    txm_clipped = jnp.clip(txm_true, 0.01, 0.99)
    x = -jnp.log(txm_clipped)
    x = (x - weights_true["window_center"]) / weights_true["window_width"]
    x = jnp.clip(x + weights_true["window_center"], 0.0, 1.0)  # Linear windowing
    # Skip unsharp masking for simplicity
    target = x
    
    # Add noise to target
    key = jax.random.PRNGKey(42)
    noise = 0.01 * jax.random.normal(key, target.shape)
    target = jnp.clip(target + noise, 0.0, 1.0)
    
    return target, txm_true, weights_true


def compute_l_curve(tv_weights, n_steps=100, batch_size=2):
    """Compute L-curve for TV regularization."""
    # Generate synthetic data
    target, txm_true, _ = create_synthetic_data(batch_size)
    
    print(f"Data created - target shape: {target.shape}, range: [{float(target.min()):.3f}, {float(target.max()):.3f}]")
    
    # Initial guess
    txm0 = jnp.ones_like(target) * 0.5
    w0 = {
        "window_center": 0.5,
        "window_width": 0.4,
        "enhance_factor": 0.5,
    }
    
    # Test forward pass
    test_opt = SimpleOptimizer(tv_weight=0.0, n_steps=1)
    test_pred = test_opt.forward(txm0, w0)
    print(f"Test forward - pred shape: {test_pred.shape}, range: [{float(test_pred.min()):.3f}, {float(test_pred.max()):.3f}]")
    
    mse_values = []
    tv_values = []
    
    print("Computing L-curve points...")
    for tv_weight in tv_weights:
        print(f"  TV weight: {tv_weight:.6f}")
        
        opt = SimpleOptimizer(
            tv_weight=tv_weight,
            n_steps=n_steps,
            optimizer=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=0.001)
            ),
        )
        
        state, losses = opt.optimize(target, txm0, w0)
        
        if state is not None and len(losses) > 0 and not jnp.isnan(losses[-1]):
            # Get final values by running forward pass
            final_txm, final_weights = state
            final_pred = opt.forward(final_txm, final_weights)
            
            # Compute metrics outside of JAX tracing
            mse_val = float(0.5 * jnp.mean((final_pred - target) ** 2))
            tv_val = float(metrics.total_variation(final_txm, reduction="mean"))
            
            mse_values.append(mse_val)
            tv_values.append(tv_val)
            print(f"    MSE: {mse_val:.6f}, TV: {tv_val:.6f}")
        else:
            print("    Failed!")
    
    return np.array(mse_values), np.array(tv_values)


def plot_l_curve(mse_values, tv_values, tv_weights, output_path):
    """Plot the L-curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L-curve
    ax1.plot(tv_values, mse_values, 'b.-', markersize=8)
    ax1.set_xlabel('Total Variation')
    ax1.set_ylabel('MSE')
    ax1.set_title('L-curve')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # MSE vs weight
    ax2.plot(tv_weights[:len(mse_values)], mse_values, 'b.-', markersize=8)
    ax2.set_xlabel('TV Weight')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE vs TV Weight')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Simple L-curve analysis')
    parser.add_argument('--n-weights', type=int, default=10, help='Number of weights to test')
    parser.add_argument('--n-steps', type=int, default=100, help='Optimization steps')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--output', type=str, default='l_curve_simple.png', help='Output plot path')
    
    args = parser.parse_args()
    
    # Generate TV weights
    tv_weights = np.logspace(-6, -2, args.n_weights)
    
    # Compute L-curve
    mse_values, tv_values = compute_l_curve(tv_weights, args.n_steps, args.batch_size)
    
    if len(mse_values) > 0:
        # Plot results
        plot_l_curve(mse_values, tv_values, tv_weights, args.output)
        
        # Find optimal weight (simple elbow method)
        if len(mse_values) > 2:
            # Normalize values
            mse_norm = (mse_values - mse_values.min()) / (mse_values.max() - mse_values.min() + 1e-8)
            tv_norm = (tv_values - tv_values.min()) / (tv_values.max() - tv_values.min() + 1e-8)
            
            # Find point farthest from line connecting first and last points
            p1 = np.array([tv_norm[0], mse_norm[0]])
            p2 = np.array([tv_norm[-1], mse_norm[-1]])
            
            distances = []
            for i in range(len(mse_norm)):
                p = np.array([tv_norm[i], mse_norm[i]])
                # Distance from point to line
                dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
                distances.append(dist)
            
            optimal_idx = np.argmax(distances)
            optimal_weight = tv_weights[optimal_idx]
            
            print(f"\nOptimal TV weight: {optimal_weight:.6f}")
            print(f"  MSE: {mse_values[optimal_idx]:.6f}")
            print(f"  TV: {tv_values[optimal_idx]:.6f}")
    else:
        print("No successful optimizations!")


if __name__ == "__main__":
    main()