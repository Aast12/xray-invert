"""Individual gradient optimizer that computes gradients per image-weight pair.

This optimizer computes gradients for each (txm_i, weights_i) pair based only on
its corresponding loss, using JAX's vmap for efficient parallelization.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, PyTree, Bool

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
        optimizer: Optional[optax.GradientTransformation] = None,
        grad_norm_eps: float = 1e-7,
        global_eps: float = 1e-7,
        **kwargs,
    ):
        """Initialize with separate optimizers for txm and weights.

        Args:
            optimizer: Optimizer to use
            grad_norm_eps: Threshold for gradient norm below which updates are stopped
            **kwargs: Additional arguments passed to parent class
        """
        # Use txm_optimizer as the main optimizer for parent class compatibility
        super().__init__(optimizer=optimizer, **kwargs)
        self.grad_norm_eps = grad_norm_eps
        self.global_eps = global_eps

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
        init_state = jax.vmap(self.optimizer.init, in_axes=0)

        txm_opt_states = init_state(txm_state)
        weights_opt_states = init_state(weights_state)

        # Initialize convergence tracking
        batch_size = txm0.shape[0]
        # converged_mask = jnp.zeros(batch_size, dtype=bool)

        self.losses = []
        self.last_loss = jnp.full(batch_size, jnp.inf)
        converged_mask = jnp.zeros(batch_size, dtype=bool)

        # Vmap gradient computation
        grad_fn_vmapped = jax.jit(
            jax.vmap(
                jax.value_and_grad(self.loss_call_individual, argnums=(0, 1)),
                in_axes=(0, 0, 0, 0 if segmentation is not None else None),
            )
        )

        def update_fn(grads, state, params):
            updates, new_opt_state = self.optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        update_vmapped = jax.jit(jax.vmap(update_fn, in_axes=(0, 0, 0)))

        zero_if_converged = jax.jit(
            jax.vmap(
                lambda grad, is_active: jax.tree_map(lambda g: g * is_active, grad),
                in_axes=(0, 0),
            )
        )

        # Main optimization loop
        for step in range(self.n_steps):
            self.step = step

            # Compute individual losses and gradients
            losses_and_grads = grad_fn_vmapped(
                weights_state,
                txm_state,
                target,
                segmentation if segmentation is not None else None,
            )
            individual_losses, grads = losses_and_grads

            # Average loss for logging
            avg_loss = jnp.mean(individual_losses)

            # Process gradients if needed
            grads = self.process_grads(grads)
            weights_grads, txm_grads = grads

            diff = jnp.abs(self.last_loss - individual_losses)
            min_grad = jnp.min(diff)

            # if step > 0:
            #     newly_converged = diff < 1e-5
            #     converged_mask = converged_mask | newly_converged
            
            self.last_loss = individual_losses


            # Update convergence mask - images with small gradients are marked as converged
            # diff = jnp.abs(self.last_loss - individual_losses)
            # min_grad = jnp.min(diff)

            # if step > 0:
            #     newly_converged = diff < 1e-5
            #     converged_mask = converged_mask | newly_converged
            
            # self.last_loss = individual_losses
            # active_mask = ~converged_mask

            # txm_grads = zero_if_converged(txm_grads, active_mask)
            # weights_grads = zero_if_converged(weights_grads, active_mask)

            # Apply updates separately for txm and weights
            new_txm_state, new_txm_opt_states = update_vmapped(
                txm_grads, txm_opt_states, txm_state
            )
            new_weights_state, new_weights_opt_states = update_vmapped(
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
            if step > 2 and jnp.abs(self.losses[-1] - self.losses[-2]) < self.eps: #     
            # if step > 2 and jnp.all(diff < self.eps):
                print(f"Converged after {step} steps")
                self.summary({"convergence_steps": step})
                break

            # Logging
            if step % self.log_interval == 0:
                num_converged = jnp.sum(converged_mask).item()
                num_active = batch_size - num_converged
                log_info = {
                    "loss": avg_loss,
                    "step": step,
                    "num_converged": num_converged,
                    "num_active": num_active,
                }
                print(
                    f"\nStep {step}, Loss: {avg_loss:.6f}, min loss: {jnp.min(individual_losses):.6f} converged: {num_converged}, active: {num_active}"
                )
                self.log(log_info)

            # Early stopping if all images have converged
            # if jnp.all(converged_mask):
            #     print(f"All images converged after {step} steps")
            #     self.summary({"converged_steps": step})
            #     break


        self.summary({
            'params': weights_state 
        })

        return (txm_state, weights_state), self.losses
