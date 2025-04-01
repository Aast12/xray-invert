import jax
import jax.numpy as jnp
import optax


def base_optimize(
    target,
    txm0,
    w0,
    loss_fn,
    forward_fn,
    optimizer_builder=optax.adam,
    lr=0.001,
    n_steps=500,
    loss_logger=None,
    eps=1e-8,
):
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
        loss, grads = loss_value_and_grad(weights, tx_maps, target)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        updates_txm, updates_weights = updates

        new_state = (
            optax.apply_updates(tx_maps, updates_txm),
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
