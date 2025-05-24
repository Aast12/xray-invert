import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

WeightsT = PyTree
BatchT = Float[Array, "batch rows cols"]
SegmentationT = Float[Array, "batch labels rows cols"]
OptimizationRetT = Union[
    tuple[tuple[PyTree, PyTree], list[float]], tuple[None, list[float]]
]


class Optimizer(ABC):
    """Unified optimizer for inverse problems with or without segmentation.

    This class provides optimization functionality for recovering
    transmission maps and model parameters from images, with optional
    segmentation information.
    
    Available extensions:
    - AdaptiveOptimizer: Per-image gradient stopping for all parameters
    - SelectiveFreezeOptimizer: Freezes only transmission maps while preserving weights
    """

    segmentation: Optional[SegmentationT] = None
    step = 0 
    losses: list[float] = []

    def __init__(
        self,
        optimizer: Optional[optax.GradientTransformation] = None,
        constant_weights: bool = False,
        lr: float = 0.001,
        n_steps: int = 500,
        eps: float = 1e-8,
        log_interval: int = 100,
        track_time: bool = False,
    ):
        """Initialize the optimizer.

        Args:
            optimizer: Pre-configured optimizer (if None, will use adam with lr)
            constant_weights: Whether to freeze weights during optimization
            lr: Learning rate (used only if optimizer is None)
            n_steps: Maximum number of optimization steps
            eps: Convergence threshold
            log_interval: How often to log progress (steps)
            track_time: Whether to track and log time metrics
        """
        self.optimizer = (
            optimizer if optimizer is not None else optax.adam(learning_rate=lr)
        )
        self.constant_weights = constant_weights
        self.n_steps = n_steps
        self.eps = eps
        self.log_interval = log_interval
        self.track_time = track_time

    @abstractmethod
    def forward(self, txm: BatchT, weights: WeightsT) -> BatchT:
        """Forward model that converts transmission maps to processed images."""
        pass

    def loss_fn(
        self,
        txm: BatchT,
        weights: WeightsT,
        pred: BatchT,
        target: BatchT,
        segmentation: Optional[SegmentationT] = None,
    ) -> Float[Array, ""]:
        """Loss function to evaluate the quality of predictions.

        This method handles both segmentation and non-segmentation cases.
        Override this method to implement your custom loss function.
        """
        raise NotImplementedError("Subclasses must implement loss_fn")

    def project(
        self,
        txm: PyTree,
        weights: WeightsT,
        segmentation: Optional[SegmentationT] = None,
    ) -> tuple[PyTree, WeightsT]:
        """Project states to valid ranges.

        Override this method to implement custom projections.
        """
        return txm, weights

    def process_grads(self, grads):
        """Process gradients before applying updates.

        Override this method to apply gradient masks or transformations.
        """
        return grads

    def log(self, metrics: Dict[str, Any]):
        """Log metrics during optimization.

        Override this method for custom logging behavior.
        """
        pass
        for k, v in metrics.items():
            print(f"{k}: {v}")

    def summary(self, metrics: Dict[str, Any]):
        """Log summary metrics at the end of optimization.

        Override this method for custom summary logging.
        """
        pass
        for k, v in metrics.items():
            print(f"Summary - {k}: {v}")

    def loss_call(self, weights, txm, target):
        pred = self.forward(txm, weights)
        loss = self.loss_fn(txm, weights, pred, target, self.segmentation)

        return loss

    def optimize(
        self,
        target: BatchT,
        txm0: BatchT,
        w0: WeightsT,
        segmentation: Optional[SegmentationT] = None,
    ) -> OptimizationRetT:
        """Run the optimization process.

        Args:
            target: The target processed X-ray images
            txm0: Initial transmission map estimate
            w0: Initial weights for the forward model
            segmentation: Optional segmentation masks for anatomical structures

        Returns:
            Tuple of (state, losses) where state is (transmission_map, weights)
        """

        self.segmentation = segmentation

        def loss_call(weights, tx_maps, target):
            return self.loss_call(weights, tx_maps, target) 

        # Initialize state
        state = (txm0, w0)
        opt_state = self.optimizer.init(state)

        self.losses = []
        prev_state = (None, None)

        # Time tracking variables
        it_times = []
        long_step = time.time()

        # Main optimization loop
        for step in range(self.n_steps):
            self.step = step
            st = time.time()

            # Check for convergence
            if step > 2 and jnp.abs(self.losses[-1] - self.losses[-2]) < self.eps:
                self.summary({"convergence_steps": step})
                print(f"Converged after {step} steps")
                break

            # Update step
            original_state = (state, opt_state)
            tx_maps, weights = state

            # Calculate loss and gradients
            loss_value_and_grad = jax.value_and_grad(loss_call, argnums=(0, 1))
            loss, (weight_grads, tx_grads) = loss_value_and_grad(
                weights, tx_maps, target
            )
            grads = (tx_grads, weight_grads)

            # Process gradients (customization point)
            grads = self.process_grads(grads)

            # Apply optimizer update
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            updates_txm, updates_weights = updates

            # Apply updates
            txm_new_state = optax.apply_updates(tx_maps, updates_txm)
            weights_new_state = weights
            if not self.constant_weights:
                weights_new_state = optax.apply_updates(
                    weights, updates_weights
                )

            # Create new state
            new_state = (txm_new_state, weights_new_state)

            # Apply projections
            new_state = self.project(
                txm_new_state, weights_new_state, segmentation
            )

            # Handle NaN loss
            if jnp.isnan(loss):
                state, opt_state = original_state
            else:
                state = new_state
                opt_state = new_opt_state

            self.losses.append(loss)

            if jnp.isnan(loss):
                state, loss = prev_state
                print("Loss is NaN. Last loss:", loss)
                break

            # Logging
            if step % self.log_interval == 0:
                log_info = {"loss": loss, "step": step}

                if self.track_time:
                    loop_time = (time.time() - long_step) / 60
                    log_info["mins_per_interval"] = loop_time
                    print(
                        f"\nStep {step}, Loss: {loss:.6f} ({loop_time:.2f} mins)"
                    )
                    long_step = time.time()
                else:
                    print(f"\nStep {step}, Loss: {loss:.6f}")

                self.log(log_info)

            prev_state = (state, loss)

            # Time tracking
            if self.track_time:
                it_time = time.time() - st
                it_times.append(it_time)

        # Log time metrics
        if self.track_time:
            self.summary(
                {
                    "it_time": it_times
                }
            )

        if state is None:
            return None, self.losses

        return state, self.losses


# Legacy function wrappers for backward compatibility
def base_optimize(
    target: BatchT,
    txm0: BatchT,
    w0: WeightsT,
    loss_fn: Callable,
    forward_fn: Callable,
    project_fn: Optional[Callable] = None,
    optimizer_builder=optax.adam,
    constant_weights=False,
    lr=0.001,
    n_steps=500,
    loss_logger=None,
    logger=None,
    eps=1e-8,
) -> OptimizationRetT:
    """Legacy function wrapper for compatibility."""

    class LegacyOptimizer(Optimizer):
        def forward(self, txm, weights):
            return forward_fn(txm, weights)

        def loss_fn(self, txm, weights, pred, target, segmentation=None):
            return loss_fn(txm, weights, pred, target)

        def project(self, txm, weights, segmentation=None):
            if project_fn:
                return project_fn(txm, weights)
            return txm, weights

        def log(self, metrics):
            if logger:
                logger(metrics)
            else:
                super().log(metrics)

    optimizer = LegacyOptimizer(
        optimizer=optimizer_builder(learning_rate=lr),
        constant_weights=constant_weights,
        n_steps=n_steps,
        eps=eps,
    )

    return optimizer.optimize(target, txm0, w0)


def segmentation_optimize(
    target: BatchT,
    txm0: BatchT,
    w0: WeightsT,
    segmentation: SegmentationT,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    forward_fn: Callable,
    project_fn: Optional[Callable] = None,
    constant_weights=False,
    n_steps=500,
    loss_logger=None,
    summary=None,
    logger=None,
    eps=1e-8,
) -> OptimizationRetT:
    """
    Legacy function wrapper for compatibility with existing code.

    Args:
        target: The target processed X-ray images
        txm0: Initial transmission map estimate
        w0: Initial weights for the forward model
        segmentation: Segmentation masks for anatomical structures
        optimizer: Optimizer instance to use
        loss_fn: Loss function that incorporates segmentation data
        forward_fn: Forward model function
        project_fn: Projection function that can use segmentation masks
        constant_weights: Whether to freeze weights during optimization
        n_steps: Maximum number of optimization steps
        loss_logger: Optional function to log losses
        summary: Function for logging summary metrics
        logger: Function for logging metrics
        eps: Convergence threshold

    Returns:
        Tuple of (state, losses) where state is (transmission_map, weights)
    """

    class LegacySegOptimizer(Optimizer):
        def forward(self, txm, weights):
            return forward_fn(txm, weights)

        def loss_fn(self, txm, weights, pred, target, segmentation=None):
            return loss_fn(txm, weights, pred, target, segmentation)

        def project(self, txm, weights, segmentation=None):
            if project_fn and segmentation is not None:
                return project_fn(txm, weights, segmentation)
            return txm, weights

        def log(self, metrics):
            if logger:
                logger(metrics)
            else:
                super().log(metrics)

        def summary(self, metrics):
            if summary:
                summary(metrics)
            else:
                super().summary(metrics)

    opt = LegacySegOptimizer(
        optimizer=optimizer,
        constant_weights=constant_weights,
        n_steps=n_steps,
        eps=eps,
        track_time=True,
    )

    return opt.optimize(target, txm0, w0, segmentation)
