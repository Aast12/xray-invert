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


@jax.jit
def psnr(pred: Float[Array, "*dims"], target: Float[Array, "*dims"]):
    return dmp_metric(dmp.psnr, pred, target)

@partial(jax.jit, static_argnums=(2))
def unsharp_mask_similarity(
    pred: Float[Array, "batch height width"],
    target: Float[Array, "batch height width"],
    sigma=3.0,
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

    detail_mse = jnp.mean((x_detail - y_detail) ** 2)

    return detail_mse

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
