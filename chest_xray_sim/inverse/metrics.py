import jax
import jax.numpy as jnp


# @jax.jit
def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


# @jax.jit
def total_variation(image):
    reg = (jnp.diff(image, axis=0) ** 2).mean() + (jnp.diff(image, axis=1) ** 2).mean()
    return 0.5 * reg
