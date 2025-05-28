import typing
from functools import partial

import dm_pix as dmp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..types import TransmissionMapT, SegmentationT, ValueRangeT


def dmp_metric(fn, a, b, **kwargs):
    return fn(jnp.expand_dims(a, axis=-3), jnp.expand_dims(b, axis=-3), **kwargs)


@jax.jit
def mse(pred: Float[Array, "*dims"], target: Float[Array, "*dims"]):
    return jnp.mean((pred - target) ** 2)


@partial(jax.jit, static_argnames=["reduction", "alpha"])
def total_generalized_variation(
    image: Float[Array, "batch rows cols"],
    reduction: typing.Literal["sum", "mean", "max", "none"] = "mean",
    alpha: float = 2.0,  # Ratio between second and first order
) -> Float[Array, "batch"] | Float[Array, ""]:
    """
    Total Generalized Variation (TGV) regularization for JAX/Optax optimization.

    This is a differentiable approximation of TGV that penalizes both first-order
    (gradients) and second-order (curvature) variations to reduce blocky artifacts
    while preserving smooth gradients.

    Args:
        image: Input images with shape (batch, height, width)
        reduction: How to reduce the result
        alpha: Balance between first and second order terms (default 2.0)

    Returns:
        TGV regularization values
    """
    # First-order differences (gradients)
    dx = jnp.diff(image, axis=-1, append=image[..., -1:])
    dy = jnp.diff(image, axis=-2, append=image[..., -1:, :])

    # Second-order differences (Hessian approximation)
    dxx = jnp.diff(dx, axis=-1, prepend=dx[..., :1])
    dxy = jnp.diff(dx, axis=-2, prepend=dx[..., :1, :])
    # dyx = jnp.diff(dy, axis=-1, prepend=dy[..., :, :1])
    dyy = jnp.diff(dy, axis=-2, prepend=dy[..., :1, :])

    # First-order term: Huber-like smooth approximation of L1 norm
    eps = 1e-8
    first_order = jnp.sqrt(dx**2 + dy**2 + eps)

    # Second-order term: Frobenius norm of Hessian
    second_order = jnp.sqrt(dxx**2 + dyy**2 + 2 * dxy**2 + eps)

    tgv = first_order + alpha * second_order

    tgv = tgv.sum(axis=(-2, -1))

    if reduction == "max":
        return tgv.max()
    elif reduction == "mean":
        return tgv.mean()
    elif reduction == "sum":
        return tgv.sum()
    else:  # "none"
        return tgv


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
    tv = jnp.sqrt(tv + 1e-8)

    if reduction == "max":
        return tv.max()
    elif reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()
    else:
        return tv


@partial(jax.jit, static_argnames=["reduction"])
def tikhonov(
    image: Float[Array, "*batch rows cols"],
    reduction: typing.Literal["sum", "mean", "max"] = "mean",
):
    """
    Tikhonov regularization, also known as Tikhonov smoothing.
    :math:`\sum_{i, j} (y_{i + 1, j} - y_{i, j})^2 + (y_{i, j + 1} - y_{i, j})^2`
    """
    d1 = jnp.diff(image, axis=-2)
    d2 = jnp.diff(image, axis=-1)

    tikhonov = (d1**2).sum(axis=(-2, -1)) + (d2**2).sum(axis=(-2, -1))

    if reduction == "max":
        return tikhonov.max()
    elif reduction == "mean":
        return tikhonov.mean()
    elif reduction == "sum":
        return tikhonov.sum()
    else:
        return tikhonov


@partial(jax.jit, static_argnames=["max_val"])
def ssim(pred: Float[Array, "*dims"], target: Float[Array, "*dims"], max_val=1.0):
    return dmp_metric(dmp.ssim, pred, target, max_val=max_val)


@partial(jax.jit, static_argnames=["sigma"])
def downsample(image: jnp.ndarray, sigma: float = 1.5) -> jnp.ndarray:
    """Downsample image by factor of 2 using Gaussian blur and subsampling."""
    # Apply Gaussian blur using dm_pix
    kernel_size = int(2 * int(3 * sigma) + 1)
    blurred = dmp.gaussian_blur(
        image, sigma=sigma, kernel_size=kernel_size, padding="SAME"
    )

    return blurred[..., ::2, ::2]


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def compute_ssim_components(
    img1: Float[Array, "*dims channels rows cols"],
    img2: Float[Array, "*dims channels rows cols"],
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> tuple[
    Float[Array, "*dims channels rows cols"],
    Float[Array, "*dims channels rows cols"],
    Float[Array, "*dims channels rows cols"],
]:
    """
    Compute SSIM components: luminance, contrast, and structure.

    Args:
        img1, img2: Input images (batch_size, height, width) or (height, width)
        sigma: Standard deviation of Gaussian kernel
        k1, k2: SSIM constants
        data_range: Dynamic range of the images (255 for uint8, 1.0 for normalized)
        alpha, beta, gamma: Exponents for luminance, contrast, and structure
        return_components: Whether to return individual components

    Returns:
        ssim_map: SSIM values
        luminance: Luminance comparison
        contrast: Contrast comparison
        structure: Structure comparison
    """
    kernel_size = 2 * int(3 * sigma) + 1

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c3 = c2 / 2

    mu1 = dmp.gaussian_blur(img1, sigma=sigma, kernel_size=kernel_size)
    mu2 = dmp.gaussian_blur(img2, sigma=sigma, kernel_size=kernel_size)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        dmp.gaussian_blur(img1**2, sigma=sigma, kernel_size=kernel_size) - mu1_sq
    )
    sigma2_sq = (
        dmp.gaussian_blur(img2**2, sigma=sigma, kernel_size=kernel_size) - mu2_sq
    )
    sigma12 = (
        dmp.gaussian_blur(img1 * img2, sigma=sigma, kernel_size=kernel_size) - mu1_mu2
    )

    luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)

    sigma1 = jnp.sqrt(jnp.maximum(sigma1_sq, 1e-10))
    sigma2 = jnp.sqrt(jnp.maximum(sigma2_sq, 1e-10))
    contrast = (2 * sigma1 * sigma2 + c2) / (sigma1_sq + sigma2_sq + c2)
    structure = (sigma12 + c3) / (sigma1 * sigma2 + c3)

    return (
        luminance.clip(1e-6, 1.0),
        contrast.clip(1e-6, 1.0),
        structure.clip(-1.0, 1.0),
    )


MS_SSIM_SCALES = 5


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10))
def ms_ssim(
    img01: jnp.ndarray,
    img02: jnp.ndarray,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
    weights: typing.Optional[typing.Sequence[float]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Compute Multi-Scale Structural Similarity Index (MS-SSIM).

    Args:
        img1, img2: Input images (batch_size, height, width) or (height, width)
        num_scales: Number of scales to use
        sigma: Standard deviation of Gaussian kernel
        k1, k2: SSIM constants
        data_range: Dynamic range of the images (255 for uint8, 1.0 for normalized)
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        alpha, beta, gamma: Exponents for luminance, contrast, and structure

    Returns:
        MS-SSIM value(s)
    """
    num_scales = MS_SSIM_SCALES
    # Default weights from the MS-SSIM paper
    if weights is None:
        weights = [
            0.0448,
            0.2856,
            0.3001,
            0.2363,
            0.1333,
        ]  # Normalize weights if we use fewer scales
        weights = jnp.array(weights)
        weights = weights / weights.sum()
    else:
        weights = jnp.array(weights)

    # Ensure 3D tensors
    img1 = jnp.expand_dims(img01, axis=-3)
    img2 = jnp.expand_dims(img02, axis=-3)

    # Lists to store contrast and structure at each scale
    mcs_list = []

    # Process each scale
    for scale in range(num_scales):
        # Compute SSIM components
        luminance, contrast, structure = compute_ssim_components(
            img1, img2, sigma, k1, k2, data_range
        )

        if scale < num_scales - 1:
            # Store contrast * structure for all scales except the last
            # Clip values to avoid extreme exponentiation
            contrast_clipped = jnp.clip(contrast, 1e-6, 1.0)
            structure_clipped = jnp.clip(structure, -1.0, 1.0)
            mcs = (contrast_clipped**beta) * (structure_clipped**gamma)

            mcs_list.append(jnp.mean(mcs, axis=(-2, -1)).squeeze())

            # Downsample for next scale
            img1 = downsample(img1, sigma)
            img2 = downsample(img2, sigma)

            # Check if images are too small (need at least 11x11 for default gaussian kernel)
            min_size = 11
            if img1.shape[-2] < min_size or img1.shape[-1] < min_size:
                # If we can't process all scales, the last processed scale needs luminance
                luminance_clipped = jnp.clip(luminance, 1e-6, 1.0)
                contrast_clipped = jnp.clip(contrast, 1e-6, 1.0)
                structure_clipped = jnp.clip(structure, -1.0, 1.0)
                lcs = (
                    (luminance_clipped**alpha)
                    * (contrast_clipped**beta)
                    * (structure_clipped**gamma)
                )
                mcs_list[-1] = jnp.mean(lcs, axis=(-2, -1)).squeeze()
                # Pad remaining scales with 1.0
                for _ in range(scale + 1, num_scales):
                    mcs_list.append(jnp.ones_like(mcs_list[-1]).squeeze())
                break
        else:
            # At coarsest scale (final scale), include luminance
            luminance_clipped = jnp.clip(luminance, 1e-6, 1.0)
            contrast_clipped = jnp.clip(contrast, 1e-6, 1.0)
            structure_clipped = jnp.clip(structure, -1.0, 1.0)
            lcs = (
                (luminance_clipped**alpha)
                * (contrast_clipped**beta)
                * (structure_clipped**gamma)
            )
            y = jnp.mean(lcs, axis=(-2, -1)).squeeze()
            mcs_list.append(y)

    # Stack all scales
    mcs_stack = jnp.stack(mcs_list)
    # Ensure values are positive before exponentiation to avoid NaN
    mcs_stack_clipped = jnp.clip(mcs_stack, 1e-10, 1.0)

    # Compute weighted product with numerical stability
    res = jnp.prod(mcs_stack_clipped ** weights[None, :])

    # Apply weights and compute final MS-SSIM
    return res


@jax.jit
def psnr(pred: Float[Array, "*dims"], target: Float[Array, "*dims"]):
    return dmp_metric(dmp.psnr, pred, target)


@partial(jax.jit, static_argnums=(2))
def unsharp_mask_similarity(
    pred: Float[Array, "batch height width"],
    target: Float[Array, "batch height width"],
    sigma=3.0,
    reduction: typing.Literal["mean", "sum", "max"] = "mean",
):
    x_detail = (
        pred
        - dmp.gaussian_blur(
            jnp.expand_dims(pred, axis=-3),
            sigma,
            kernel_size=int(2 * sigma),
            padding="SAME",
        ).squeeze()
    )
    y_detail = (
        target
        - dmp.gaussian_blur(
            jnp.expand_dims(target, axis=-3),
            sigma,
            kernel_size=int(2 * sigma),
            padding="SAME",
        ).squeeze()
    )

    detail_diff = (x_detail - y_detail) ** 2

    if reduction == "max":
        return detail_diff.max()
    elif reduction == "mean":
        return detail_diff.mean()
    elif reduction == "sum":
        return detail_diff.sum()
    else:
        return detail_diff


@jax.jit
def compute_single_mask_penalty(
    mask_id: int,
    mask: TransmissionMapT,
    value_range: Float[Array, " 2"],
    txm: TransmissionMapT,
) -> Float[Array, " batch"]:
    min_val, max_val = value_range

    region_values = txm * mask

    region_size = jnp.sum(mask, axis=(-2, -1))

    below_min_capped = jnp.maximum(0.0, min_val - region_values)
    above_max_capped = jnp.maximum(0.0, region_values - max_val)

    below_min = jnp.where(mask > 0, below_min_capped, 0.0) ** 2
    above_max = jnp.where(mask > 0, above_max_capped, 0.0) ** 2

    region_penalty = jnp.sum(below_min + above_max, axis=(-2, -1)) / region_size
    return region_penalty


@jax.jit
def batch_segmentation_sq_penalty(
    txm: TransmissionMapT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
):
    penalties = jnp.ones((value_ranges.shape[0], txm.shape[0]))

    # TODO: possibly improve by making broadcast operations
    for mask_id, val_range in enumerate(value_ranges):
        penalty = compute_single_mask_penalty(
            mask_id, segmentation[:, mask_id], val_range, txm
        )
        penalties = penalties.at[mask_id].set(penalty)

    return penalties


@jax.jit
def segmentation_sq_penalty(
    txm: TransmissionMapT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
):
    return jnp.sum(batch_segmentation_sq_penalty(txm, segmentation, value_ranges))
