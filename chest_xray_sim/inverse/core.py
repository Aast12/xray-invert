import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike


WeightsT = dict[str, ArrayLike]


def base_optimize(
    target: ArrayLike,
    txm0: ArrayLike,
    w0: WeightsT,
    loss_fn,
    forward_fn,
    optimizer_builder=optax.adam,
    lr=0.001,
    n_steps=500,
    loss_logger=None,
    eps=1e-8,
) -> tuple[tuple[ArrayLike, WeightsT], list[float]]:
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

    print("txm0 ", txm0.shape)
    print("txm0_0", txm0[0].shape)
    print("w0", w0)
    print("target", target.shape)
    print("target0", target[0].shape)

    opt_state = optimizer.init(state)

    losses = []
    prev_state = (None, None)

    def update(state, opt_state, target):
        original_state = (state, opt_state)
        tx_maps, weights = state

        loss_value_and_grad = jax.value_and_grad(loss_call, argnums=(0, 1))
        loss, (weight_grads, tx_grads) = loss_value_and_grad(weights, tx_maps, target)
        grads = (tx_grads, weight_grads)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        updates_txm, updates_weights = updates

        txm_new_state = optax.apply_updates(tx_maps, updates_txm)
        txm_new_state = optax.projections.projection_box(txm_new_state, 0.0, 1.0)

        new_state = (
            txm_new_state,
            optax.apply_updates(weights, updates_weights),
        )

        if jnp.isnan(loss):
            new_state, new_opt_state = original_state

        return loss, new_state, new_opt_state

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
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

    return state, losses
