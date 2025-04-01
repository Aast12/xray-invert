import jax
from functools import partial
import jax.numpy as jnp
from dm_pix import gaussian_blur


@partial(jax.jit, static_argnums=(1,))
def negative_log(image, eps=1e-6):
    return -jnp.log(jnp.maximum(image, eps))


def windowing(image, window_center, window_width, gamma):
    x = image
    x = (x - window_center) / window_width
    x = jax.nn.sigmoid(x) ** gamma

    return x


def unsharp_masking(image, sigma, enhance_factor):
    x = jnp.expand_dims(image, axis=2)
    kernel_size = int(4 * sigma) + 1
    blurred = gaussian_blur(x, sigma, kernel_size, padding="SAME")
    x = x + enhance_factor * (x - blurred)
    return x.squeeze()


@jax.jit
def clipping(image):
    return jnp.clip(image, 0.0, 1.0)


@jax.jit
def range_normalize(image):
    x = image
    return (x - x.min()) / (x.max() - x.min())


@jax.jit
def max_normalize(image):
    x = image
    return x / x.max()
