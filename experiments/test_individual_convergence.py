"""Test script for individual gradient optimizer with per-image convergence tracking."""

import os
import jax
import jax.numpy as jnp
import optax
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.inverse.individual_gradient_optimizer import IndividualGradientOptimizer
from experiments.models import ExperimentInputs
from experiments.segmentation_utils import get_priors
import chest_xray_sim.inverse.operators as ops
from chest_xray_sim.inverse import metrics


class TestIndividualOptimizer(IndividualGradientOptimizer):
    """Test implementation of individual gradient optimizer."""
    
    def __init__(self, value_ranges, **kwargs):
        super().__init__(**kwargs)
        self.value_ranges = value_ranges
    
    def loss_call_individual(self, weights, txm, target, segmentation=None):
        """Compute loss for a single image."""
        # Forward pass
        x = ops.negative_log(txm)
        x = ops.window(x, weights["window_center"], weights["window_width"], 5.0, "sigmoid")
        x = ops.range_normalize(x)
        x = ops.unsharp_masking(x, 2.0, weights["enhance_factor"])
        x = ops.clipping(x)
        
        # MSE loss
        mse = 0.5 * jnp.mean((x - target) ** 2)
        
        # Total variation
        tv = metrics.total_variation(txm[jnp.newaxis, ...], reduction="mean")
        
        # Segmentation penalty if provided
        if segmentation is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm[jnp.newaxis, ...], 
                segmentation[jnp.newaxis, ...], 
                self.value_ranges
            ).sum()
        else:
            seg_penalty = 0.0
        
        # Combined loss
        loss = mse + 0.01 * tv + 0.1 * seg_penalty
        
        return loss


def main():
    # Setup
    data_dir = os.environ.get("IMAGE_PATH")
    meta_dir = os.environ.get("META_PATH")
    mask_dir = os.environ.get("MASK_DIR")
    cache_dir = os.environ.get("CACHE_DIR")
    
    # Load dataset
    dataset = get_segmentation_dataset(
        data_dir=data_dir,
        meta_dir=meta_dir,
        mask_dir=mask_dir,
        cache_dir=cache_dir,
        split="train",
        frontal_lateral="Frontal",
        batch_size=8,  # Small batch for testing
    )
    
    # Get priors
    _, value_ranges = get_priors(cache_dir, collimated_region_bound=0.4)
    
    # Get a batch
    batch = next(iter(dataset))
    images_batch, masks_batch, meta_batch = batch
    
    # Process data
    segmentations = jnp.array(masks_batch.cpu().numpy())
    seg_labels, segmentations = batch_get_exclusive_masks(segmentations, 0.6)
    
    # Convert images
    images = jnp.array(images_batch.cpu().numpy())
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0, :, :]
    
    # Create inputs
    inputs = ExperimentInputs(
        images=images,
        segmentations=segmentations,
        prior_labels=seg_labels,
        priors=value_ranges,
    )
    
    # Initialize optimizer with different gradient thresholds
    for grad_eps in [1e-6, 1e-7, 1e-8]:
        print(f"\n\nTesting with grad_norm_eps={grad_eps}")
        print("="*60)
        
        optimizer = TestIndividualOptimizer(
            value_ranges=value_ranges,
            optimizer=optax.adam(learning_rate=1e-3),
            grad_norm_eps=grad_eps,
            n_steps=500,
            log_interval=50,
        )
        
        # Initialize states
        batch_size = images.shape[0]
        txm0 = jnp.ones_like(images) * 0.5
        
        # Per-image weights
        w0 = jax.tree_map(
            lambda x: jnp.array([x] * batch_size),
            {
                "enhance_factor": 0.5,
                "window_center": 0.3,
                "window_width": 0.4,
            }
        )
        
        # Run optimization
        (final_txm, final_weights), losses = optimizer.optimize(
            images, txm0, w0, segmentations
        )
        
        print(f"\nFinal loss: {losses[-1]:.6f}")
        print(f"Total steps: {len(losses)}")
        
        # Check weight diversity
        for key in final_weights:
            values = final_weights[key]
            print(f"\n{key} final values:")
            print(f"  Min: {jnp.min(values):.4f}")
            print(f"  Max: {jnp.max(values):.4f}")
            print(f"  Std: {jnp.std(values):.4f}")


if __name__ == "__main__":
    main()