import typing
from functools import partial

import dm_pix as dmp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def dmp_metric(fn, a, b, **kwargs):
    return fn(
        jnp.expand_dims(a, axis=-3), jnp.expand_dims(b, axis=-3), **kwargs
    )


@jax.jit
def mse(pred: Float[Array, "*dims"], target: Float[Array, "*dims"]):
    return jnp.mean((pred - target) ** 2)


@partial(jax.jit, static_argnames=["reduction"])
def total_variation(
    image: Float[Array, "batch rows cols"],
    reduction: typing.Literal["sum", "mean", "max"] = "mean",
):
    """
    :math:`\sum_{i, j} |y_{i + 1, j} - y_{i, j}| + |y_{i + 1, j} - y_{i, j}| `
    """
    d1 = jnp.diff(image, axis=-2)
    d2 = jnp.diff(image, axis=-1)

    tv = (d1**2).sum(axis=(-2, -1)) + (d2**2).sum(axis=(-2, -1))

    if reduction == "max":
        return tv.max()
    elif reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()

    raise


@partial(jax.jit, static_argnames=["alpha0", "alpha1", "num_iterations"])
def _tgv_regularizer(image, alpha0=1.0, alpha1=2.0, num_iterations=5):
    """
    Implements second-order Total Generalized Variation (TGV) regularization

    Args:
        image: Input image tensor
        alpha0: Weight for the second-order term
        alpha1: Weight for the first-order term
        num_iterations: Number of iterations for the alternating minimization

    Returns:
        TGV regularization value
    """
    # Initialize auxiliary variable v (vector field)
    vx = jnp.zeros_like(image)
    vy = jnp.zeros_like(image)

    # Initialize auxiliary variables for alternating minimization
    px = jnp.zeros_like(image)
    py = jnp.zeros_like(image)
    pxx = jnp.zeros_like(image)
    pxy = jnp.zeros_like(image)
    pyx = jnp.zeros_like(image)
    pyy = jnp.zeros_like(image)

    # Compute gradients of the image
    dx = jnp.diff(image, axis=0, append=0)
    dy = jnp.diff(image, axis=1, append=0)

    # Precompute constants for numerical stability
    tau = 0.2  # Step size
    sigma = 0.5  # Dual step size

    # Alternating minimization iterations
    for _ in range(num_iterations):
        # Update dual variables p
        px_new = px + sigma * (dx - vx)
        py_new = py + sigma * (dy - vy)

        # Projection step for first-order term
        norm_p = jnp.sqrt(px_new**2 + py_new**2)
        scale_p = jnp.minimum(1.0, alpha1 / (norm_p + 1e-8))
        px = px_new * scale_p
        py = py_new * scale_p

        # Update dual variables for second-order term
        pxx_new = pxx + sigma * jnp.diff(vx, axis=0, append=0)
        pxy_new = pxy + sigma * jnp.diff(vx, axis=1, append=0)
        pyx_new = pyx + sigma * jnp.diff(vy, axis=0, append=0)
        pyy_new = pyy + sigma * jnp.diff(vy, axis=1, append=0)

        # Projection step for second-order term
        norm_p2 = jnp.sqrt(pxx_new**2 + pxy_new**2 + pyx_new**2 + pyy_new**2)
        scale_p2 = jnp.minimum(1.0, alpha0 / (norm_p2 + 1e-8))
        pxx = pxx_new * scale_p2
        pxy = pxy_new * scale_p2
        pyx = pyx_new * scale_p2
        pyy = pyy_new * scale_p2

        # Update primal variables v
        div_p = jnp.diff(px, axis=0, prepend=0) + jnp.diff(
            py, axis=1, prepend=0
        )
        div_p2x = jnp.diff(pxx, axis=0, prepend=0) + jnp.diff(
            pxy, axis=1, prepend=0
        )
        div_p2y = jnp.diff(pyx, axis=0, prepend=0) + jnp.diff(
            pyy, axis=1, prepend=0
        )

        vx = vx + tau * (div_p - div_p2x)
        vy = vy + tau * (div_p - div_p2y)

    # Compute the TGV value
    first_order_term = alpha1 * jnp.sum(
        jnp.sqrt((dx - vx) ** 2 + (dy - vy) ** 2)
    )
    second_order_term = alpha0 * jnp.sum(
        jnp.sqrt(
            jnp.diff(vx, axis=0, append=0) ** 2
            + jnp.diff(vx, axis=1, append=0) ** 2
            + jnp.diff(vy, axis=0, append=0) ** 2
            + jnp.diff(vy, axis=1, append=0) ** 2
        )
    )

    return first_order_term + second_order_term


tgv_regularizer = jax.vmap(_tgv_regularizer, in_axes=(0,))

# def pnsr(pred, target):
#     mse_value = mse(pred, target)
#     max_pixel = jnp.max(target)
#     psnr = 20 * jnp.log10(max_pixel / jnp.sqrt(mse_value))
#     return psnr
#


@partial(jax.jit, static_argnames=["max_val"])
def ssim(
    pred: Float[Array, "*dims"], target: Float[Array, "*dims"], max_val=1.0
):
    return dmp_metric(dmp.ssim, pred, target, max_val=max_val)


@jax.jit
def psnr(pred: Float[Array, "*dims"], target: Float[Array, "*dims"]):
    return dmp_metric(dmp.psnr, pred, target)
