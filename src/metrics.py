import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def naive_inversion(image):
    naive = -jnp.exp(image)
    return (naive - naive.min()) / (naive.max() - naive.min())


def total_variation_2d(image):
    val = (jnp.diff(image, axis=0) ** 2).mean() + (jnp.diff(image, axis=1) ** 2).mean()
    return val * 0.5


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


def sqerr(pred, target):
    err = (pred - target) ** 2
    return err, err.flatten()
