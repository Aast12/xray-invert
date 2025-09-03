"""Shared weights optimizer that uses the same weights across all images.

This optimizer computes gradients on shared weights by aggregating losses
across all images in the batch. Each image still has its own transmission map,
but all images share the same set of weights.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, PyTree

from chest_xray_sim.inverse.core import (
    BatchT,
    Optimizer,
    OptimizationRetT,
    SegmentationT,
    WeightsT,
)


class SharedWeightsOptimizer(Optimizer):
    """Optimizer that shares weights across all images while keeping individual txms.
    
    This optimizer:
    1. Uses the same weight values for all images in the batch
    2. Computes individual losses for each (txm_i, shared_weights) pair
    3. Aggregates losses across the batch (e.g., mean)
    4. Computes gradients of the aggregated loss w.r.t. shared weights and individual txms
    
    Subclasses must implement loss_call_individual() to define the forward pass
    and loss computation for a single image.
    """
    
    def __init__(
        self,
        optimizer: Optional[optax.GradientTransformation] = None,
        **kwargs
    ):
        """Initialize with separate optimizers for txm and weights."""
        # Use txm_optimizer as the main optimizer for parent class compatibility
        super().__init__(optimizer=optimizer, **kwargs)
    
    def loss_call_individual(
        self,
        weights: PyTree,
        txm: Float[jax.Array, "rows cols"],
        target: Float[jax.Array, "rows cols"],
        segmentation: Optional[Float[jax.Array, "labels rows cols"]] = None,
    ) -> Float[jax.Array, ""]:
        """Loss computation for a single image with shared weights.
        
        Subclasses must implement this method to define the loss computation
        for individual images, including the forward pass.
        
        Args:
            weights: Shared weights used for all images
            txm: Single transmission map
            target: Target image
            segmentation: Optional segmentation mask for this image
            
        Returns:
            Scalar loss value
        """
        raise NotImplementedError("Subclasses must implement loss_call_individual")
    
    def loss_call(
        self,
        weights: PyTree,
        txm: BatchT,
        target: BatchT,
        segmentation: Optional[SegmentationT] = None,
    ) -> Float[jax.Array, ""]:
        """Aggregate loss computation across all images with shared weights.
        
        Args:
            weights: Single set of weights shared across all images
            txm: Batch of transmission maps
            target: Batch of target images
            segmentation: Optional batch of segmentation masks
            
        Returns:
            Aggregated scalar loss value
        """
        # Vmap the individual loss computation over the batch dimension
        # Note: weights is NOT vmapped as it's shared
        loss_fn_vmapped = jax.vmap(
            self.loss_call_individual,
            in_axes=(None, 0, 0, 0 if segmentation is not None else None)
        )
        
        # Compute individual losses
        individual_losses = loss_fn_vmapped(
            weights,
            txm,
            target,
            segmentation if segmentation is not None else None
        )
        
        # Aggregate losses (mean by default)
        return jnp.mean(individual_losses)
    
    def optimize(
        self,
        target: BatchT,
        txm0: BatchT,
        w0: WeightsT,
        segmentation: Optional[SegmentationT] = None,
    ) -> OptimizationRetT:
        """Run optimization with shared weights across all images.
        
        Args:
            target: Batch of target images
            txm0: Initial batch of transmission maps
            w0: Initial weights (can be a single set or batch - will use first if batch)
            segmentation: Optional batch of segmentation masks
            
        Returns:
            Tuple of (optimized_txm_batch, shared_weights), losses
        """
        self.segmentation = segmentation
        
        # Initialize states
        txm_state = txm0
        
        # If w0 is a batch, take the first set as the shared weights
        # This allows compatibility with existing code that might pass batched weights
        weights_state = w0
        
        # Initialize optimizer states
        # For txm: we need separate states for each image
        init_txm_vmapped = jax.vmap(self.optimizer.init, in_axes=0)
        txm_opt_states = init_txm_vmapped(txm_state)
        
        # For weights: single state for shared weights
        weights_opt_state = self.optimizer.init(weights_state)
        
        self.losses = []
        
        # Define gradient function for the aggregated loss
        grad_fn = jax.value_and_grad(self.loss_call, argnums=(0, 1))
        
        # Create update functions
        def update_txm(txm_grad, txm_opt_state, txm_params):
            updates, new_opt_state = self.optimizer.update(txm_grad, txm_opt_state, txm_params)
            new_txm = optax.apply_updates(txm_params, updates)
            return new_txm, new_opt_state
        
        update_txm_vmapped = jax.jit(jax.vmap(update_txm, in_axes=(0, 0, 0)))
        
        # Main optimization loop
        for step in range(self.n_steps):
            self.step = step
            
            # Compute aggregated loss and gradients
            loss_val, grads = grad_fn(
                weights_state,
                txm_state,
                target,
                segmentation if segmentation is not None else None
            )
            
            # Process gradients if needed
            grads = self.process_grads(grads)
            weights_grads, txm_grads = grads
            
            # Update transmission maps (individually)
            new_txm_state, new_txm_opt_states = update_txm_vmapped(
                txm_grads, txm_opt_states, txm_state
            )
            
            # Update shared weights
            if not self.constant_weights:
                weights_updates, new_weights_opt_state = self.optimizer.update(
                    weights_grads, weights_opt_state, weights_state
                )
                new_weights_state = optax.apply_updates(weights_state, weights_updates)
                weights_opt_state = new_weights_opt_state
            else:
                new_weights_state = weights_state
            
            # Apply projections
            # Note: we need to broadcast weights for projection compatibility
            # weights_broadcast = jax.tree_map(
            #     lambda x: jnp.broadcast_to(x, (txm_state.shape[0],) + x.shape),
            #     new_weights_state
            # )
            new_txm_state, new_weights_state = self.project(
                new_txm_state, new_weights_state, segmentation
            )
            # Extract the shared weights back (they should all be the same after projection)
            # new_weights_state = jax.tree_map(lambda x: x[0], weights_broadcast)
            
            # Check for NaN
            if jnp.isnan(loss_val):
                print(f"Loss is NaN at step {step}")
                break
            
            # Update states
            txm_state = new_txm_state
            weights_state = new_weights_state
            txm_opt_states = new_txm_opt_states
            
            self.losses.append(loss_val)
            
            # Check for convergence
            if step > 2 and jnp.abs(self.losses[-1] - self.losses[-2]) < self.eps:
                self.summary({"convergence_steps": step})
                print(f"Converged after {step} steps")
                break
            
            # Logging
            if step % self.log_interval == 0:
                log_info = {"loss": loss_val, "step": step}
                print(f"\nStep {step}, Loss: {loss_val:.6f}")
                self.log(log_info)
        
        
        self.summary({
            "params": weights_state
        })

        return (txm_state, weights_state), self.losses