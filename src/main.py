import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jaxtyping import ArrayLike, Array, Num, Int, Float
from jax import jit
from jax.tree_util import register_pytree_node, register_dataclass
from dataclasses import dataclass
from processings import apply_lut, multiscale_processing


def forward(og_image, weights, filter_sizes):
    image = 1.0 - og_image

    breakpoints = weights[:4]
    values = weights[4:8]
    partitions = weights[8]
    unsharp_weights = weights[9:]

    image = apply_lut(
        image, breakpoints=breakpoints, values=values, partitions=int(partitions * 1000)
    )

    result = multiscale_processing(
        image, jnp.array(filter_sizes, dtype=jnp.int16), unsharp_weights
    )

    # blurred_pyramid = [image]
    # for size in filter_sizes:
    #     blurred = mean_filter(blurred_pyramid[-1], size)
    #     blurred_pyramid.append(blurred)

    # result = image.copy()
    # for i in range(len(unsharp_weights)):
    #     detail = blurred_pyramid[i] - blurred_pyramid[i + 1]
    #     result = result + unsharp_weights[i] * detail

    return jnp.clip(result, 0, 1)


def loss_fn(params, target, filter_sizes):
    image, weights = params
    pred = forward(image, weights, filter_sizes)
    mse = jnp.mean((pred - target) ** 2)

    # normalized regularization
    reg = jnp.sum(jnp.abs(weights))

    loss = mse + 0.01 * reg
    print(f"MSE in loss_fn: {loss}")

    return loss


def update_step(params, target, filter_sizes, lr):
    old_image, old_weights = params
    print("\nBefore update:")
    print(f"Image range: [{old_image.min():.2e}, {old_image.max():.2e}]")
    print(f"Weights: {old_weights}")

    loss_value_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = loss_value_and_grad(params, target, filter_sizes)

    image_grad, weights_grad = grads

    print(f"\nGradients:")
    print(f"Image grad range: [{image_grad.min():.2e}, {image_grad.max():.2e}]")
    print(f"Weights grad: {weights_grad}")

    new_image = old_image - lr * image_grad
    new_weights = old_weights - lr * weights_grad

    print(f"\nBefore projection:")
    print(f"New image range: [{new_image.min():.2e}, {new_image.max():.2e}]")
    print(f"New weights: {new_weights}")

    # Project
    new_weights = jnp.maximum(new_weights, 0)
    # new_weights = new_weights / (jnp.sum(new_weights) + 1e-8)
    new_image = new_image / (jnp.sum(new_image) + 1e-8)
    # new_image = jnp.clip(new_image, 0, 1)

    print(f"\nAfter projection:")
    print(f"Final image range: [{new_image.min():.2e}, {new_image.max():.2e}]")
    print(f"Final weights: {new_weights}")
    print(f"Weight change: {jnp.abs(new_weights - old_weights).max():.2e}")

    return loss, grads, (new_image, new_weights)


def optimize(target, filter_sizes=[3, 9, 27], n_steps=1000, lr=0.1, eps=1e-8):
    filter_sizes = tuple(filter_sizes)

    x0 = jnp.ones_like(target, dtype=jnp.float32)
    # w0 = jnp.ones(len(filter_sizes)) / len(filter_sizes)

    w0 = jnp.concat(
        [
            jnp.array(
                [
                    0.0,
                    0.25,
                    0.75,
                    1.0,
                    0.0,
                    0.1,
                    0.9,
                    1.0,
                    1.0,
                ]
            ),
            jnp.ones(len(filter_sizes), dtype=jnp.float32) / len(filter_sizes),
        ]
    )
    params = (x0, w0)

    print("dtypes:", x0.dtype, w0.dtype)

    losses = []

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            print(f"Converged after {step} steps")
            break

        loss, grads, params = update_step(params, target, filter_sizes, lr)
        losses.append(loss)

        if step % 10 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")

    return params, losses
