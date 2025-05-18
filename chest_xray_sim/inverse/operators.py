from functools import partial

import jax
import jax.numpy as jnp
from dm_pix import gaussian_blur
from jaxtyping import Array, Float, Scalar

Image = Float[Array, "*batch rows cols"]


@partial(jax.jit, static_argnums=(1,))
def negative_log(image: Image, eps=1e-6):
    x = -jnp.log(jnp.maximum(image, eps))
    return x / (-jnp.log(eps))


@jax.jit
def windowing(
    image: Image, window_center: Scalar, window_width: Scalar, gamma: Scalar
):
    x = image
    x = (x - window_center) / window_width
    x = jax.nn.sigmoid(x) ** gamma

    return x


# @jax.jit
def window(
    image: Image, window_center: Scalar, window_width: Scalar, gamma: Scalar
):
    x = image
    x = (x - window_center) / window_width
    return jax.nn.sigmoid(x * gamma)


@jax.jit
def low_pass(image: Image, sigma: float):
    x = jnp.expand_dims(image, axis=2)
    kernel_size = 2 * sigma
    blurred = gaussian_blur(x, sigma, kernel_size, padding="same")

    return x - blurred


def unsharp_masking_alt(image: Image, sigma: float, enhance_factor: float):
    x = jnp.expand_dims(image, axis=2)
    kernel_size = 2 * sigma
    blurred = gaussian_blur(x, sigma, kernel_size, padding="same")

    factor = 1 / (1 - enhance_factor)
    x = factor * (x - blurred) + blurred

    return x.squeeze()


def unsharp_masking(image: Image, sigma: float, enhance_factor: float):
    x = jnp.expand_dims(image, axis=2)
    kernel_size = 2 * sigma
    blurred = gaussian_blur(x, sigma, kernel_size, padding="SAME")

    # blurred = dmp.pad_to_size(blurred, x.shape[0], x.shape[1], mode="reflect")

    x = (x - enhance_factor * blurred) / (1.0 - enhance_factor)

    return x.squeeze()


@jax.jit
def clipping(image):
    return jnp.clip(image, 0.0, 1.0)


@jax.jit
def range_normalize(image: Image):
    x = image
    return (x - x.min(axis=(-2, -1))) / (
        x.max(axis=(-2, -1)) - x.min(axis=(-2, -1))
    )


@jax.jit
def max_normalize(image: Image):
    x = image
    return x / x.max(axis=(-2, -1))
