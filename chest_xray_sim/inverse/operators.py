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


@jax.jit
def window(
    image: Image, window_center: Scalar, window_width: Scalar, gamma: Scalar
):
    x = image
    x = (x - window_center) / window_width
    return jax.nn.sigmoid(x * gamma)


@jax.jit
def low_pass(image: Image, sigma: float):
    x = jnp.expand_dims(image, axis=-3)
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


@partial(jax.jit, static_argnums=(1,))
def unsharp_masking(image: Image, sigma: float, enhance_factor: float):
    x = jnp.expand_dims(image, axis=-3)
    kernel_size = 2 * sigma
    blurred = gaussian_blur(x, sigma, kernel_size, padding="SAME")

    x = (x - enhance_factor * blurred) / (1.0 - enhance_factor)

    return x.squeeze()

def unsharp_masking_fft(image, sigma, enhance_factor):
    """FFT-based gaussian blur that doesn't need kernel size."""
    
    # Create frequency domain Gaussian
    h, w = image.shape
    y, x = jnp.ogrid[-h//2:h//2, -w//2:w//2]
    
    # Gaussian in frequency domain
    gaussian_freq = jnp.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_freq = jnp.fft.fftshift(gaussian_freq)
    
    # Apply filter in frequency domain
    image_freq = jnp.fft.fft2(image)
    blurred_freq = image_freq * gaussian_freq
    blurred = jnp.real(jnp.fft.ifft2(blurred_freq))
    
    # Unsharp mask
    sharpened = (image - enhance_factor * blurred) / (1.0 - enhance_factor)
    
    return sharpened

@jax.jit
def clipping(image):
    return jnp.clip(image, 0.0, 1.0)


@jax.jit
def range_normalize(image: Image):
    x = image
    return (x - x.min(axis=(-2, -1), keepdims=True)) / (
        x.max(axis=(-2, -1), keepdims=True)
        - x.min(axis=(-2, -1), keepdims=True)
    )


@jax.jit
def max_normalize(image: Image):
    x = image
    return x / x.max(axis=(-2, -1), keepdims=True)
