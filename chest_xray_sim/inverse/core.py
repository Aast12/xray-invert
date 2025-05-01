import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

import wandb

WeightsT = PyTree
BatchT = Float[Array, "batch rows cols"]
SegmentationT = Float[Array, "batch labels rows cols"]
# SegmentationT = PyTree  # dict of mask ids - batch channels rows cols
LossFnT = Callable[[BatchT, WeightsT, BatchT, BatchT], Float[Array, ""]]
SegLossFnT = Callable[
    [BatchT, WeightsT, BatchT, BatchT, SegmentationT], Float[Array, ""]
]
ForwardFnT = Callable[[BatchT, WeightsT], BatchT]
ProjectFnT = Callable[[PyTree, WeightsT], tuple[PyTree, WeightsT]]
SegProjectFnT = Callable[[PyTree, WeightsT, SegmentationT], tuple[PyTree, WeightsT]]

OptimizationRetT = Union[
    tuple[tuple[PyTree, PyTree], list[float]], tuple[None, list[float]]
]


def base_optimize(
    target: BatchT,
    txm0: BatchT,
    w0: WeightsT,
    loss_fn: LossFnT,
    forward_fn: ForwardFnT,
    project_fn: ProjectFnT | None = None,
    optimizer_builder=optax.adam,
    constant_weights=False,
    lr=0.001,
    n_steps=500,
    loss_logger=None,
    eps=1e-8,
) -> OptimizationRetT:
    def loss_call(weights, tx_maps, target):
        pred = forward_fn(tx_maps, weights)
        loss = loss_fn(tx_maps, weights, pred, target)

        if loss_logger:
            loss_logger(loss.item(), tx_maps, weights, pred, target)

        assert pred.shape == target.shape, (
            f"Shapes do not match: {pred.shape} != {target.shape}"
        )

        return loss

    optimizer = optimizer_builder(learning_rate=lr)

    state = (txm0, w0)

    opt_state = optimizer.init(state)

    losses = []
    prev_state = (None, None)

    def update(state, opt_state, target):
        original_state = (state, opt_state)
        tx_maps, weights = state

        loss_value_and_grad = jax.value_and_grad(loss_call, argnums=(0, 1))
        loss, (weight_grads, tx_grads) = loss_value_and_grad(weights, tx_maps, target)
        grads = (tx_grads, weight_grads)
        # jax.debug.print('weight grads= {w}', w=weight_grads)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        updates_txm, updates_weights = updates

        # jax.debug.print('weight updates = {w}', w=updates_weights)

        txm_new_state = optax.apply_updates(tx_maps, updates_txm)
        weights_new_state = weights
        if not constant_weights:
            weights_new_state = optax.apply_updates(weights, updates_weights)

        new_state = (txm_new_state, weights_new_state)
        if project_fn:
            new_state = project_fn(txm_new_state, weights_new_state)

        if jnp.isnan(loss):
            new_state, new_opt_state = original_state

        return loss, new_state, new_opt_state

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            wandb.log({"convergence_steps": step})
            print(f"Converged after {step} steps")
            break

        loss, state, opt_state = update(state, opt_state, target)
        losses.append(loss)

        if jnp.isnan(loss):
            state, loss = prev_state
            print("Loss is NaN. Last loss:", loss)
            break

        if step % 100 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")

        prev_state = (state, loss)

    if state is None:
        return None, losses

    return state, losses


def segmentation_optimize(
    target: BatchT,
    txm0: BatchT,
    w0: WeightsT,
    segmentation: SegmentationT,
    optimizer: optax.GradientTransformation,
    loss_fn: SegLossFnT,
    forward_fn: ForwardFnT,
    project_fn: SegProjectFnT | None = None,
    constant_weights=False,
    n_steps=500,
    loss_logger=None,
    eps=1e-8,
) -> OptimizationRetT:
    """
    Optimization function that incorporates segmentation data for improved transmission map recovery.

    Args:
        target: The target processed X-ray images
        txm0: Initial transmission map estimate
        w0: Initial weights for the forward model
        segmentation: Segmentation masks for anatomical structures
        loss_fn: Loss function that incorporates segmentation data
        forward_fn: Forward model function
        project_fn: Projection function that can use segmentation masks
        optimizer_builder: Function to build the optimizer
        constant_weights: Whether to freeze weights during optimization
        lr: Learning rate
        n_steps: Maximum number of optimization steps
        loss_logger: Optional function to log losses
        eps: Convergence threshold
        segmentation_weights: Optional dict with weights for different segmentation components

    Returns:
        Tuple of (state, losses) where state is (transmission_map, weights)
    """

    def loss_call(weights, tx_maps, target, seg):
        pred = forward_fn(tx_maps, weights)
        loss = loss_fn(tx_maps, weights, pred, target, seg)

        if loss_logger:
            loss_logger(loss.item(), tx_maps, weights, pred, target, seg)

        assert pred.shape == target.shape, (
            f"Shapes do not match: {pred.shape} != {target.shape}"
        )

        return loss

    state = (txm0, w0)

    opt_state = optimizer.init(state)

    losses = []
    prev_state = (None, None)

    def update(state, opt_state, target, seg):
        original_state = (state, opt_state)
        tx_maps, weights = state

        loss_value_and_grad = jax.value_and_grad(loss_call, argnums=(0, 1))
        loss, (weight_grads, tx_grads) = loss_value_and_grad(
            weights, tx_maps, target, seg
        )
        grads = (tx_grads, weight_grads)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        updates_txm, updates_weights = updates

        txm_new_state = optax.apply_updates(tx_maps, updates_txm)
        weights_new_state = weights
        if not constant_weights:
            weights_new_state = optax.apply_updates(weights, updates_weights)

        new_state = (txm_new_state, weights_new_state)
        if project_fn:
            new_state = project_fn(txm_new_state, weights_new_state, seg)

        if jnp.isnan(loss):
            new_state, new_opt_state = original_state

        return loss, new_state, new_opt_state

    avg_it_time = 0.0
    max_it_time = 0.0
    min_it_time = float("inf")

    long_step = time.time()

    for step in range(n_steps):
        st = time.time()
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            wandb.log({"convergence_steps": step})
            print(f"Converged after {step} steps")
            break

        loss, state, opt_state = update(state, opt_state, target, segmentation)
        losses.append(loss)

        if jnp.isnan(loss):
            state, loss = prev_state
            print("Loss is NaN. Last loss:", loss)
            break

        if step % 100 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")
            wandb.log(
                {
                    "mins_per_hundred_steps": (time.time() - long_step) / 60,
                }
            )
            long_step = time.time()
        prev_state = (state, loss)

        it_time = time.time() - st
        avg_it_time = (avg_it_time * step + it_time) / (step + 1)
        max_it_time = max(max_it_time, it_time)
        min_it_time = min(min_it_time, it_time)

    # TODO: abstract out
    wandb.log(
        {
            "avg_it_time": avg_it_time,
            "max_it_time": max_it_time,
            "min_it_time": min_it_time,
        }
    )

    if state is None:
        return None, losses

    return state, losses
