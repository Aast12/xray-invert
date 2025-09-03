import dm_pix as dmp
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.inverse.metrics import ms_ssim, ssim
import jax

# import torchmetrics
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)


@jax.jit
def ms_ssim_l1(
    image1,
    image2,
    sigmas=(0.5, 1.0, 2.0, 4.0, 8.0),
    filter_size=11,
    C1=0.01,
    C2=0.03,
    alpha=0.84,
):
    # image1 and image2 are the same shape Nx1xHxWxC and are floats in the range [-1.0, 1.0]
    C1 = C1**2
    C2 = C2**2
    num_scale = len(sigmas)

    image1 = image1[..., None]
    image2 = image2[..., None]
    diff = jnp.abs(image2 - image1)
    width = filter_size

    w = jnp.exp(
        (-1.0 * jnp.arange(-(width // 2), width // 2 + 1) ** 2)[None, :]
        / (2 * jnp.array(sigmas) ** 2)[:, None]
    )
    w = w[:, :, None] @ w[:, None, :]
    w = w / jnp.sum(w, axis=(1, 2))[:, None, None]
    w = jnp.reshape(w, (num_scale, 1, width, width))
    w = jnp.transpose(w, axes=(3, 2, 1, 0))

    w = jnp.transpose(w, [3, 2, 0, 1])
    image1 = jnp.transpose(image1, [0, 3, 1, 2])
    image2 = jnp.transpose(image2, [0, 3, 1, 2])

    mux = jax.lax.conv(image1, w, (1, 1), "SAME")
    muy = jax.lax.conv(image2, w, (1, 1), "SAME")
    sigmax2 = jax.lax.conv(image1**2, w, (1, 1), "SAME") - mux**2
    sigmay2 = jax.lax.conv(image2**2, w, (1, 1), "SAME") - muy**2
    sigmaxy = jax.lax.conv(image1 * image2, w, (1, 1), "SAME") - mux * muy
    l = (2 * mux * muy + C1) / (mux**2 + muy**2 + C1)
    cs = (2 * sigmaxy + C2) / (sigmax2 + sigmay2 + C2)

    Pcs = jnp.prod(cs, axis=1)
    l1_loss = jnp.mean(
        jax.lax.conv(
            jnp.transpose(diff, [0, 3, 1, 2]),
            w[-1, :, :, :][None, :, :, :],
            (1, 1),
            "SAME",
        )
    )

    return alpha * (1 - jnp.mean(l[:, -1, :, :] * Pcs)) + (1 - alpha) * l1_loss


if __name__ == "__main__":
    # Example usage of SSIM and MS-SSIM
    import jax.numpy as jnp

    p_rec = (
        "/Volumes/T7/projs/thesis/outputs/crg2tms5/patient00007_view1_frontal_proc.png"
    )
    p_og = "/Volumes/T7/datasets/chexpert_plus/PNG/PNG/train/patient00007/study1/view1_frontal.png"

    _img1 = read_image(p_rec)
    _img2 = read_image(p_og)
    img1 = jnp.array(_img1)
    img2 = jnp.array(_img2)

    print("Image shapes:", img1.shape, img2.shape)

    val = msssim(_img1[None, ...], _img2[None, ...])

    print(f"MS-SSIM value: {val}")
    mssim = 1 - ms_ssim_l1(img1, img2)

    print(f"MSSSIML1 loss: {mssim}")

    # Compute SSIM
    ssim_value = ssim(img1, img2)
    print(f"SSIM: {ssim_value}")

    # Compute MS-SSIM
    ms_ssim_value = ms_ssim(img1, img2)
    print(f"MS-SSIM: {ms_ssim_value}")
