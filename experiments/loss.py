import functools

import dm_pix as dmp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from chest_xray_sim.inverse import metrics
from chest_xray_sim.types import (
    ForwardT,
    SegmentationT,
    TransmissionMapT,
    ValueRangeT,
    WeightsT,
)


def mse(pred: ForwardT, target: ForwardT):
    return metrics.mse(pred, target)


def total_variation(pred: ForwardT):
    return metrics.total_variation(pred)


@functools.partial(jax.jit, static_argnums=(2))
def unsharp_mask_similarity(pred, target, sigma=3.0):
    x_detail = (
        pred
        - dmp.gaussian_blur(
            jnp.expand_dims(pred, axis=2),
            sigma,
            kernel_size=int(2 * sigma),
            padding="SAME",
        ).squeeze()
    )
    y_detail = (
        target
        - dmp.gaussian_blur(
            jnp.expand_dims(target, axis=2),
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
    penalties: Float[Array, " reduced_labels"],
) -> Float[Array, " reduced_labels"]:
    min_val, max_val = value_range

    # region_values = txm[mask.astype(bool)]
    region_values = txm * mask

    region_size = jnp.sum(mask, axis=(-2, -1))

    below_min_capped = jnp.maximum(0.0, min_val - region_values)
    above_max_capped = jnp.maximum(0.0, region_values - max_val)

    # jax.debug.print('below_min_capped mean={x} std={xi}, ', x=below_min_capped.mean(), xi=below_min_capped.std())
    # jax.debug.print('above_max_capped mean={x} std={xi}', x=above_max_capped.mean(), xi=above_max_capped.std())

    below_min = jnp.where(mask > 0, below_min_capped, 0.0) ** 2
    above_max = jnp.where(mask > 0, above_max_capped, 0.0) ** 2

    # below_min = jnp.maximum(0.0, min_val - region_values) ** 2
    # above_max = jnp.maximum(0.0, region_values - max_val) ** 2
    # jax.debug.print("region size {x}", x=jnp.expand_dims(region_size, (1, 2)))
    # jax.debug.print('sum mean={x} std={xi}', x=sum.mean(), xi=sum.std())
    # jax.debug.print('region size shape = {x}', x=jnp.expand_dims(region_size, (1, 2)).shape)

    sum = jnp.sum(below_min + above_max, axis=(-2, -1))
    region_penalty = jnp.mean(sum / region_size)
    return penalties.at[mask_id].set(region_penalty)


@jax.jit
def segmentation_sq_penalty(
    txm: TransmissionMapT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
):
    """
    Assumes batch arrays
    """
    penalties = jnp.ones(value_ranges.shape[0])

    # TODO: possibly improve by making broadcast operations
    for mask_id, val_range in enumerate(value_ranges):
        penalties = compute_single_mask_penalty(
            mask_id, segmentation[:, mask_id], val_range, txm, penalties
        )

    return jnp.sum(penalties)


def segmentation_loss(
    txm: TransmissionMapT,
    weights: WeightsT,
    pred: ForwardT,
    target: ForwardT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
    tv_factor=0.1,
    prior_weight=0.5,
):
    """
    Loss function that incorporates segmentation information using probabilistic priors.
    The loss consists of:
    1. MSE between prediction and target (data fidelity)
    2. Total variation regularization (spatial smoothness)
    3. Anatomical region range penalty

    args:
        txm: Transmission map
        weights: Forward model parameters
        pred: Predicted image
        target: Target image
        segmentation: Segmentation map
        tv_factor: Weight for total variation regularization
        prior_weight: Weight for anatomical priors
    """
    mse = metrics.mse(pred, target)
    tv = metrics.total_variation(txm)

    segmentation_penalty = segmentation_sq_penalty(
        txm, segmentation, value_ranges
    )

    return mse + tv_factor * tv + prior_weight * segmentation_penalty


def detached_segmentation_sq_penalty(
    txm: TransmissionMapT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
):
    penalties = jnp.ones((value_ranges.shape[0], txm.shape[0]))

    # TODO: possibly improve by making broadcast operations
    for mask_id, val_range in enumerate(value_ranges):
        mask = segmentation[:, mask_id]
        min_val, max_val = val_range

        region_values = txm * mask
        region_size = jnp.sum(mask, axis=(-2, -1))

        below_min = jnp.maximum(0.0, min_val - region_values) ** 2
        above_max = jnp.maximum(0.0, region_values - max_val) ** 2

        region_penalty = (below_min + above_max) / (
            jnp.expand_dims(region_size, (1, 2)) + 1e6
        )
        penalties = penalties.at[mask_id].set(
            jnp.sum(region_penalty, axis=(-2, -1))
        )

    return penalties
