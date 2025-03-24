import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from dataclasses import dataclass

from jaxtyping import Array, Float
from .processings import (
    PipelineWeights,
    anisotropic_diffusion,
    apply_display_lut,
    apply_lut,
    dynamic_range_compression,
    initialize_weights,
    mean_filter,
    multiscale_enhance,
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


# def forward(
#     og_image: ImageType,
#     weights: tuple[LookupTableWeights, MultiscaleProcessingWeights],
#     min_value=1e-6,
# ):
#     image = jnp.clip(og_image, min_value, None)
#     image = -jnp.log(image)

#     lut_weights, mup_weights = weights

#     breakpoints = lut_weights.breakpoints
#     values = lut_weights.values
#     partitions = lut_weights.partitions
#     unsharp_weights = mup_weights.unsharp_weights

#     # image = apply_lut(
#     #     image, breakpoints=breakpoints, values=values, partitions=int(partitions * 1000)
#     # )

#     filter_sizes = jax.lax.stop_gradient(mup_weights.filter_sizes)

#     result = multiscale_processing(
#         image, unsharp_weights, filter_sizes, filter_fn=mup_weights.filter_fn
#     )

#     return jnp.clip(result, 0.0, 1.0)


def forward(weights: PipelineWeights) -> Float[Array, "height width"]:
    """Complete pipeline forward pass"""
    image = weights.image

    # Start with -log transform
    x = -jnp.log(jnp.maximum(image, 1e-6))

    x = dynamic_range_compression(x, weights.dynamic_range)
    # x = anisotropic_diffusion(x, weights.noise)
    # x = multiscale_enhance(x, weights.multiscale)
    # x = apply_display_lut(x, weights.display)

    return jnp.clip(x, 0.0, 1.0)


def ssim(x, y, L=1.0):
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_x = jnp.mean(x)
    mu_y = jnp.mean(y)

    sigma_x = jnp.mean((x - mu_x) ** 2)
    sigma_y = jnp.mean((y - mu_y) ** 2)
    sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return jnp.clip(num / den, 0.0, 1.0)


def loss_fn(params: PipelineWeights, target: ImageType):
    # image = params.image

    # image, weights = params
    pred = forward(params)
    # loss = 1 - ssim(pred, target)
    loss = jnp.mean((pred - target) ** 2)

    return loss


def update_step(optimizer, params: PipelineWeights, opt_state, target: ImageType):
    original_state = (params, opt_state)

    loss_value_and_grad = jax.value_and_grad(loss_fn)
    loss, grads = loss_value_and_grad(params, target)

    print("GRADS", grads.image)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if jnp.isnan(loss):
        print("Loss is NaN")
        params, opt_state = original_state

    return loss, grads, params, opt_state


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

        print(f"MSE in loss_fn: {loss}")

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


def optimize(
    target,
    filter_sizes=[3, 9, 27],
    n_steps=1000,
    lr=0.1,
    eps=1e-8,
    optimizer_builder=optax.adam,
    state_init=None,
    masking=None,
    x0_init=None,
):
    filter_sizes = jnp.array(filter_sizes, dtype=jnp.int16)

    x0 = (
        jnp.zeros_like(target, dtype=jnp.float32) + 1e-6 if x0_init is None else x0_init
    )
    weights = initialize_weights(x0)

    if state_init is not None:
        weights = state_init(weights)

    params = weights
    print(params)
    optimizer = optimizer_builder(learning_rate=lr)

    # param_masks = jax.tree_map(lambda x: True, params)  # Default: update all parameters

    if masking is not None:
        param_masks = masking(params)
    # param_masks = param_masks.replace(noise=False)  # Don't update noise parameters

    masked_optimizer = optax.masked(optimizer, param_masks)

    opt_state = optimizer.init(params)

    losses = []

    prev_state = None

    for step in range(n_steps):
        if step > 2 and jnp.abs(losses[-1] - losses[-2]) < eps:
            print(f"Converged after {step} steps")
            break

        loss, _, params, opt_state = update_step(
            masked_optimizer, params, opt_state, target
        )
        losses.append(loss)

        if jnp.isnan(loss):
            params, loss = prev_state
            print("Loss is NaN. Last loss:", loss)
            break

        if step % 100 == 0:
            print(f"\nStep {step}, Loss: {loss:.6f}")

        prev_state = (params, loss)

    return params, losses
