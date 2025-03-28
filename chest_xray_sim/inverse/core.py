import jax
import jax.numpy as jnp
import optax


def zero_grads():
    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


def make_state_constant(optimizer, params, constants):
    mask = {
        **{key: "zero" for key in constants},
        **{key: "default" for key in params.keys() if key not in constants},
    }
    op = optax.multi_transform({"default": optimizer, "zero": zero_grads()}, mask)
    return op, params


def base_optimize(
    target,
    w0,
    loss_fn,
    forward_fn,
    optimizer_builder=optax.adam,
    constant_state=None,
    preprocess_state=None,
    lr=0.001,
    n_steps=500,
    loss_logger=None,
    eps=1e-8,
):
    def loss_call(params, target):
        pred = forward_fn(params)
        loss = loss_fn(params, pred, target)

        assert (
            pred.shape == target.shape
        ), f"Shapes do not match: {pred.shape} != {target.shape}"

        return loss

    def update(optimizer, params, opt_state, target):
        original_state = (params, opt_state)

        loss_value_and_grad = jax.value_and_grad(loss_call)
        loss, grads = loss_value_and_grad(params, target)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if jnp.isnan(loss):
            print("Loss is NaN")
            params, opt_state = original_state

        return loss, grads, params, opt_state

    params = w0
    optimizer = optimizer_builder(learning_rate=lr)

    if constant_state is not None:
        optimizer, params = make_state_constant(optimizer, params, constant_state)

    if preprocess_state is not None:
        optimizer, params = preprocess_state(optimizer, params)

    opt_state = optimizer.init(params)

    losses = []

    prev_state = None

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            print(f"Converged after {step} steps")
            break

        loss, _, params, opt_state = update(optimizer, params, opt_state, target)
        losses.append(loss)

        if loss_logger is not None:
            loss_logger(loss)

        if jnp.isnan(loss):
            params, loss = prev_state
            print("Loss is NaN. Last loss:", loss)
            break

        if step % 100 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")

        prev_state = (params, loss)

    return params, losses
