from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from chest_xray_sim.inverse import metrics
from chest_xray_sim.types import (
    ForwardT,
    SegmentationT,
    TransmissionMapT,
    ValueRangeT,
)
from experiments.loss import compute_single_mask_penalty

DEBUG = True


def ssim(pred: ForwardT, target: ForwardT):
    """Structural Similarity Index Measure (SSIM)"""
    return metrics.ssim(pred, target)


def psnr(pred: ForwardT, target: ForwardT):
    """Peak Signal-to-Noise Ratio (PSNR)"""
    return metrics.psnr(pred, target)


def bound_compliance(
    pred: ForwardT, segmentations: SegmentationT, value_ranges: ValueRangeT
) -> Float[Array, "reduced_labels batch"]:
    """Measure a ratio of how many pixels are within the bounds of the value ranges"""
    compliance = jnp.ones((value_ranges.shape[0], pred.shape[0]))

    for i, val_range in enumerate(value_ranges):
        mask = segmentations[:, i]
        minv, maxv = val_range

        masked = pred * mask
        mask_size = jnp.sum(mask, axis=(-2, -1))

        compliancev = jnp.logical_and(masked >= minv, masked <= maxv)
        compliance_count = jnp.where(mask > 0, compliancev.astype(jnp.int8), 0)
        compliance_count = jnp.sum(compliance_count, axis=(-2, -1))

        compliance = compliance.at[i].set(compliance_count / mask_size)

    print("compliance shape:", compliance.shape)
    return compliance


def bound_violations(
    pred: ForwardT, segmentations: SegmentationT, value_ranges: ValueRangeT
) -> tuple[
    Float[Array, "reduced_labels batch"], Float[Array, "reduced_labels batch"]
]:
    """Max and min violations, only when the pixel is outside the bounds of the value ranges"""
    # For max violations: Only count values that exceed the upper bound
    max_violations = jnp.ones((value_ranges.shape[0], pred.shape[0]))
    min_violations = jnp.ones((value_ranges.shape[0], pred.shape[0]))

    for i, val_range in enumerate(value_ranges):
        mask = segmentations[:, i]

        minv, maxv = val_range
        region_values = pred * mask

        # TODO: masks dont preserve shape
        max_violation = jnp.where(
            mask > 0, jnp.maximum(0.0, region_values - maxv), 0.0
        )
        min_violation = jnp.where(
            mask > 0, jnp.maximum(0.0, minv - region_values), 0.0
        )

        max_violations = max_violations.at[i].set(
            max_violation.max(axis=(-2, -1))
        )
        min_violations = min_violations.at[i].set(
            min_violation.max(axis=(-2, -1))
        )

    print("max violations shape:", max_violations.shape)
    print("min violations shape:", min_violations.shape)

    return max_violations, min_violations


@dataclass
class EvalMetrics:
    ssim: Array
    psnr: Array
    penalties: Float[Array, " reduced_labels"]
    bound_compliance: Float[Array, "reduced_labels batch"]
    min_violations: Float[Array, "reduced_labels batch"]
    max_violations: Float[Array, "reduced_labels batch"]


def batch_evaluation(
    images: ForwardT,
    txm: TransmissionMapT,
    pred: ForwardT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
) -> EvalMetrics:
    """
    Evaluate the model on a batch of images and segmentations.
    """

    psnr_val = psnr(pred, images)
    ssim_val = ssim(pred, images)
    penalties = jnp.ones(value_ranges.shape[0])

    for mask_id, val_range in enumerate(value_ranges):
        penalties = compute_single_mask_penalty(
            mask_id, segmentation[:, mask_id], val_range, txm, penalties
        )

    compliance = bound_compliance(pred, segmentation, value_ranges)
    max_violations, min_violations = bound_violations(
        pred, segmentation, value_ranges
    )

    return EvalMetrics(
        ssim=ssim_val,
        psnr=psnr_val,
        penalties=penalties,
        bound_compliance=compliance,
        min_violations=min_violations,
        max_violations=max_violations,
    )
