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
import optax


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

    filter_sizes = jax.lax.stop_gradient(mup_weights.filter_sizes)

    result = multiscale_processing(
        image, unsharp_weights, filter_sizes, filter_fn=mup_weights.filter_fn
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


def update_step(optimizer, params: ModelWeights, opt_state, target):
    loss_value_and_grad = jax.value_and_grad(loss_fn)
    loss, grads = loss_value_and_grad(params, target)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return loss, grads, params, opt_state


def optimize(
    target,
    filter_sizes=[3, 9, 27],
    n_steps=1000,
    lr=0.1,
    eps=1e-8,
    optimizer_builder=optax.adam,
):
    filter_sizes = jnp.array(filter_sizes, dtype=jnp.int16)

    x0 = jnp.ones_like(target, dtype=jnp.float32)
    breakpoints = jnp.array([0, 0.5, 0.75, 1.0])
    values = jnp.array([0, 0.0, 0.0, 0.0])
    partitions = 0.5
    unsharp_weights = jnp.ones(len(filter_sizes), dtype=jnp.float32) / len(filter_sizes)

    mup_weights = MultiscaleProcessingWeights(
        filter_sizes=jax.lax.stop_gradient(filter_sizes),
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
    optimizer = optimizer_builder(learning_rate=lr)
    opt_state = optimizer.init(params)

    losses = []

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            print(f"Converged after {step} steps")
            break

        loss, _, params, opt_state = update_step(optimizer, params, opt_state, target)
        losses.append(loss)

        if step % 10 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")

    return opt_state, losses
