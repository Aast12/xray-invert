import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from dataclasses import dataclass
from .processings import (
    apply_lut,
    multiscale_processing,
    ImageType,
    LookupTableWeights,
    MultiscaleProcessingWeights,
)


@register_dataclass
@dataclass
class ModelWeights:
    image: ImageType
    w_multiscale_processing: MultiscaleProcessingWeights
    w_lookup_table: LookupTableWeights


def forward(
    og_image: ImageType,
    weights: tuple[LookupTableWeights, MultiscaleProcessingWeights],
):
    image = -jnp.log(og_image)

    lut_weights, mup_weights = weights

    breakpoints = lut_weights.breakpoints
    values = lut_weights.values
    partitions = lut_weights.partitions
    unsharp_weights = mup_weights.unsharp_weights

    image = apply_lut(
        image, breakpoints=breakpoints, values=values, partitions=int(partitions * 1000)
    )

    filter_sizes = mup_weights.filter_sizes

    result = multiscale_processing(
        image, unsharp_weights, filter_sizes, filter_fn=lut_weights.filter_fn
    )

    return jnp.clip(result, 0.0, 1.0)


def loss_fn(params: ModelWeights, target: ImageType):
    image = params.image
    weights = (params.w_lookup_table, params.w_multiscale_processing)

    # image, weights = params
    pred = forward(image, weights)
    mse = jnp.mean((pred - target) ** 2)

    reg = jnp.sum(jnp.abs(params.image))

    loss = mse + 0.01 * reg
    print(f"MSE in loss_fn: {loss}")

    return loss


def update_step(params: ModelWeights, target, lr):
    loss_value_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = loss_value_and_grad(params, target)
    grads: ModelWeights = grads

    new_grads = ModelWeights(
        image=params.image,
        w_lookup_table=params.w_lookup_table,
        w_multiscale_processing=params.w_multiscale_processing,
    )

    new_grads.image = params.image - lr * grads.image
    new_grads.w_lookup_table.breakpoints = (
        params.w_lookup_table.breakpoints - lr * grads.w_lookup_table.breakpoints
    )
    new_grads.w_lookup_table.values = (
        params.w_lookup_table.values - lr * grads.w_lookup_table.values
    )
    new_grads.w_lookup_table.partitions = (
        params.w_lookup_table.partitions - lr * grads.w_lookup_table.partitions
    )

    new_grads.w_multiscale_processing.unsharp_weights = (
        params.w_multiscale_processing.unsharp_weights
        - lr * grads.w_multiscale_processing.unsharp_weights
    )

    return loss, grads, new_grads


def optimize(target, filter_sizes=[3, 9, 27], n_steps=1000, lr=0.1, eps=1e-8):
    filter_sizes = jnp.array(filter_sizes, dtype=jnp.int16)

    x0 = jnp.ones_like(target, dtype=jnp.float32)
    # x0 = x0 / x0.sum()
    # x0 = jnp.copy(target)

    breakpoints = jnp.array([0, 0.25, 0.75, 1.0])
    values = jnp.array([0, 0.1, 0.9, 1.0])
    partitions = 0.5
    unsharp_weights = jnp.ones(len(filter_sizes), dtype=jnp.float32) / len(filter_sizes)

    mup_weights = MultiscaleProcessingWeights(
        filter_sizes=filter_sizes,
        unsharp_weights=unsharp_weights,
    )
    lut_weights = LookupTableWeights(
        breakpoints=breakpoints,
        values=values,
        partitions=partitions,
    )
    model_weights = ModelWeights(
        image=x0, w_multiscale_processing=mup_weights, w_lookup_table=lut_weights
    )

    params = model_weights

    losses = []

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            print(f"Converged after {step} steps")
            break

        loss, grads, params = update_step(params, target, lr)
        losses.append(loss)

        if step % 10 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")

    return params, losses
