import jax.numpy as jnp
import matplotlib.pyplot as plt
from .metrics import naive_inversion, sqerr, ssim
from jax.typing import ArrayLike, Array


def run_metrics(
    true_raw: ArrayLike, true_processed: ArrayLike, recovered_raw: ArrayLike
) -> dict[str, Array | tuple[Array, Array]]:
    naive = naive_inversion(true_processed)

    sqerr_naive, sqerr_naive_flat = sqerr(naive, true_raw)
    sqerr_opt, sqerr_opt_flat = sqerr(recovered_raw, true_raw)

    return {
        "sqerr_naive": jnp.histogram(sqerr_naive_flat, bins=100, range=(0, 1)),
        "sqerr_opt": jnp.histogram(sqerr_opt_flat, bins=100, range=(0, 1)),
        "mse_naive": jnp.mean(sqerr_naive),
        "mse_opt": jnp.mean(sqerr_opt),
        "ssim_naive": ssim(naive, true_raw),
        "ssim_opt": ssim(recovered_raw, true_raw),
    }


def plot_result(loss, true_raw, true_processed, recovered_raw):
    naive = naive_inversion(true_processed)

    sqerr_naive, sqerr_naive_flat = sqerr(naive, true_raw)
    sqerr_opt, sqerr_opt_flat = sqerr(recovered_raw, true_raw)
    vmax = max(jnp.percentile(sqerr_naive_flat, 99), jnp.percentile(sqerr_opt_flat, 99))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()

    ax[0].imshow(true_raw, cmap="gray")
    ax[0].set_title("raw")

    ax[1].imshow(recovered_raw, cmap="gray")
    ax[1].set_title("recovered")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(loss)
    ax[0].set_title("loss")

    ax[1].set_title("Sq. Error distribution")
    ax[1].hist(
        sqerr_naive_flat,
        bins=50,
        alpha=0.5,
        range=(0, vmax),
        density=True,
        label="naive",
    )
    ax[1].hist(
        sqerr_opt_flat, bins=50, alpha=0.5, range=(0, vmax), density=True, label="opt"
    )

    mse_naive = jnp.mean(sqerr_naive)
    mse_opt = jnp.mean(sqerr_opt)
    ssim_naive = ssim(naive, true_raw)
    ssim_opt = ssim(recovered_raw, true_raw)

    ax[1].annotate(
        f"MSE naive: {mse_naive:.4f}",
        (0.5, 0.5),
        xycoords="axes fraction",
        color="red" if mse_naive > mse_opt else "green",
    )
    ax[1].annotate(
        f"MSE opt: {mse_opt:.4f}",
        (0.5, 0.45),
        xycoords="axes fraction",
        color="red" if mse_naive < mse_opt else "green",
    )

    ax[1].annotate(
        f"SSIM naive: {ssim_naive:.4f}",
        (0.5, 0.4),
        xycoords="axes fraction",
        color="red" if ssim_naive < ssim_opt else "green",
    )
    ax[1].annotate(
        f"SSIM opt: {ssim_opt:.4f}",
        (0.5, 0.35),
        xycoords="axes fraction",
        color="red" if ssim_naive > ssim_opt else "green",
    )

    ax[1].legend()


def plot_result_latent(target, recovered):
    naive = -jnp.exp(target)
    naive = (naive - naive.min()) / (naive.max() - naive.min())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()

    ax[0].imshow(naive, cmap="gray")
    ax[0].set_title("naive")

    ax[1].imshow(recovered, cmap="gray")
    ax[1].set_title("recovered")
