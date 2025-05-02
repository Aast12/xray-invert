import jax
import jax.numpy as jnp
import dm_pix as dmp


def dmp_metric(fn, a, b, **kwargs):
    return fn(jnp.expand_dims(a, axis=2), jnp.expand_dims(b, axis=2), **kwargs)


@jax.jit
def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


@jax.jit
def total_variation(image):
    reg = (jnp.diff(image, axis=0) ** 2).mean() + (jnp.diff(image, axis=1) ** 2).mean()
    return 0.5 * reg


# def pnsr(pred, target):
#     mse_value = mse(pred, target)
#     max_pixel = jnp.max(target)
#     psnr = 20 * jnp.log10(max_pixel / jnp.sqrt(mse_value))
#     return psnr
#


def ssim(pred, target, max_val=1.0):
    return dmp_metric(dmp.ssim, pred, target, max_val=max_val)


def psnr(pred, target):
    return dmp_metric(dmp.psnr, pred, target)
