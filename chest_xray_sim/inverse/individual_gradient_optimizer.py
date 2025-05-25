"""Individual gradient optimizer that computes gradients per image-weight pair.

This optimizer computes gradients for each (txm_i, weights_i) pair based only on
its corresponding loss, using JAX's vmap for efficient parallelization.
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


class IndividualGradientOptimizer(Optimizer):
    """Optimizer that computes gradients individually for each (txm, weights) pair.
    
    Unlike the base optimizer which aggregates losses across all images before
    computing gradients, this optimizer:
    1. Computes the loss for each (txm_i, weights_i) pair individually
    2. Computes gradients for (txm_i, weights_i) based only on loss_i
    3. Updates each pair independently
    
    This is efficiently implemented using JAX's vmap to parallelize the
    individual gradient computations.
    
    Subclasses must implement loss_call_individual() instead of loss_call().
    """
    
    def __init__(
        self,
        txm_optimizer: Optional[optax.GradientTransformation] = None,
        weights_optimizer: Optional[optax.GradientTransformation] = None,
        **kwargs
    ):
        """Initialize with separate optimizers for txm and weights."""
        # Use txm_optimizer as the main optimizer for parent class compatibility
        super().__init__(optimizer=txm_optimizer, **kwargs)
        
        # Store both optimizers
        self.txm_optimizer = self.optimizer if txm_optimizer is None else txm_optimizer
        self.weights_optimizer = self.optimizer if weights_optimizer is None else weights_optimizer
    
    def loss_call_individual(
        self,
        weights: PyTree,
        txm: Float[jax.Array, "rows cols"],
        target: Float[jax.Array, "rows cols"],
        segmentation: Optional[Float[jax.Array, "labels rows cols"]] = None,
    ) -> Float[jax.Array, ""]:
        """Loss computation for a single image-weight pair.
        
        Subclasses must implement this method to define the loss computation
        for individual images, including the forward pass.
        
        Args:
            weights: Weights for this specific image
            txm: Single transmission map
            target: Target image
            segmentation: Optional segmentation mask for this image
            
        Returns:
            Scalar loss value
        """
        raise NotImplementedError("Subclasses must implement loss_call_individual")
    
    def optimize(
        self,
        target: BatchT,
        txm0: BatchT,
        w0: WeightsT,
        segmentation: Optional[SegmentationT] = None,
    ) -> OptimizationRetT:
        """Run optimization with individual gradient computation.
        
        This method computes gradients individually for each (txm, weights) pair
        using vmap for efficiency.
        """
        self.segmentation = segmentation
        
        # Initialize states
        txm_state = txm0
        weights_state = w0
        
        # Initialize optimizer states separately for txm and weights
        init_txm_vmapped = jax.vmap(self.txm_optimizer.init, in_axes=0)
        init_weights_vmapped = jax.vmap(self.weights_optimizer.init, in_axes=0)
        
        txm_opt_states = init_txm_vmapped(txm_state)
        weights_opt_states = init_weights_vmapped(weights_state)
        
        self.losses = []
        
        # Vmap gradient computation
        grad_fn_vmapped = jax.vmap(
            jax.value_and_grad(self.loss_call_individual, argnums=(0, 1)),
            in_axes=(0, 0, 0, 0 if segmentation is not None else None)
        )
        
        # Create separate update functions for txm and weights
        def update_txm(txm_grad, txm_opt_state, txm_params):
            updates, new_opt_state = self.txm_optimizer.update(txm_grad, txm_opt_state, txm_params)
            new_txm = optax.apply_updates(txm_params, updates)
            return new_txm, new_opt_state
        
        def update_weights(weights_grad, weights_opt_state, weights_params):
            updates, new_opt_state = self.weights_optimizer.update(weights_grad, weights_opt_state, weights_params)
            new_weights = optax.apply_updates(weights_params, updates)
            return new_weights, new_opt_state
        
        update_txm_vmapped = jax.vmap(update_txm, in_axes=(0, 0, 0))
        update_weights_vmapped = jax.vmap(update_weights, in_axes=(0, 0, 0))
        
        # Main optimization loop
        for step in range(self.n_steps):
            self.step = step
            
            # Compute individual losses and gradients
            losses_and_grads = grad_fn_vmapped(
                weights_state,
                txm_state,
                target,
                segmentation if segmentation is not None else None
            )
            individual_losses, grads = losses_and_grads
            
            # Average loss for logging
            avg_loss = jnp.mean(individual_losses)
            
            # Process gradients if needed
            grads = self.process_grads(grads)
            weights_grads, txm_grads = grads
            
            # Apply updates separately for txm and weights
            new_txm_state, new_txm_opt_states = update_txm_vmapped(
                txm_grads, txm_opt_states, txm_state
            )
            new_weights_state, new_weights_opt_states = update_weights_vmapped(
                weights_grads, weights_opt_states, weights_state
            )
            
            # Handle constant weights
            if self.constant_weights:
                new_weights_state = weights_state
            
            # Apply projections
            new_txm_state, new_weights_state = self.project(
                new_txm_state, new_weights_state, segmentation
            )
            
            # Check for NaN
            if jnp.isnan(avg_loss):
                print(f"Loss is NaN at step {step}")
                break
            
            # Update states
            txm_state = new_txm_state
            weights_state = new_weights_state
            txm_opt_states = new_txm_opt_states
            weights_opt_states = new_weights_opt_states
            
            self.losses.append(avg_loss)
            
            # Check for convergence
            if step > 2 and jnp.abs(self.losses[-1] - self.losses[-2]) < self.eps:
                self.summary({"convergence_steps": step})
                print(f"Converged after {step} steps")
                break
            
            # Logging
            if step % self.log_interval == 0:
                log_info = {"loss": avg_loss, "step": step}
                print(f"\nStep {step}, Loss: {avg_loss:.6f}")
                self.log(log_info)
        
        return (txm_state, weights_state), self.losses