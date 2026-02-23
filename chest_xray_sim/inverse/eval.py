from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..types import (
    ForwardT,
    SegmentationT,
    TransmissionMapT,
    ValueRangeT,
)

from .metrics import (
    batch_segmentation_sq_penalty,
    ms_ssim,
    ms_ssim_l1,
    ssim,
    psnr,
    tikhonov,
    total_variation,
    unsharp_mask_similarity,
)


@dataclass
class EvalMetrics:
    ssim: Array
    ms_ssim: Array
    psnr: Array
    mse: Float[Array, " batch"]
    total_variation: Float[Array, " batch"]
    tikonov: Float[Array, " batch"]
    band_similarity: Float[Array, " batch"]
    penalties: Float[Array, " reduced_labels"]
    bound_compliance: Float[Array, "reduced_labels batch"]
    min_violations: Float[Array, "reduced_labels batch"]
    max_violations: Float[Array, "reduced_labels batch"]

    def concat(self, other: "EvalMetrics") -> "EvalMetrics":
        """Concatenate two EvalMetrics objects along the batch dimension."""
        _1d = jnp.atleast_1d
        return EvalMetrics(
            ssim=jnp.concatenate((_1d(self.ssim), _1d(other.ssim))),
            ms_ssim=jnp.concatenate((_1d(self.ms_ssim), _1d(other.ms_ssim))),
            psnr=jnp.concatenate((_1d(self.psnr), _1d(other.psnr))),
            mse=jnp.concatenate((_1d(self.mse), _1d(other.mse))),
            total_variation=jnp.concatenate(
                (_1d(self.total_variation), _1d(other.total_variation))
            ),
            tikonov=jnp.concatenate((_1d(self.tikonov), _1d(other.tikonov))),
            band_similarity=jnp.concatenate(
                (_1d(self.band_similarity), _1d(other.band_similarity))
            ),
            penalties=jnp.concatenate((_1d(self.penalties), _1d(other.penalties)), axis=0),
            bound_compliance=jnp.concatenate(
                (self.bound_compliance, other.bound_compliance), axis=1
            ),
            min_violations=jnp.concatenate(
                (self.min_violations, other.min_violations), axis=1
            ),
            max_violations=jnp.concatenate(
                (self.max_violations, other.max_violations), axis=1
            ),
        )

    @classmethod
    def merge(cls, metrics_list: list["EvalMetrics"]) -> "EvalMetrics":
        """Merge a list of EvalMetrics into one by concatenating along the batch dimension."""
        _1d = jnp.atleast_1d
        return cls(
            ssim=jnp.concatenate([_1d(m.ssim) for m in metrics_list]),
            ms_ssim=jnp.concatenate([_1d(m.ms_ssim) for m in metrics_list]),
            psnr=jnp.concatenate([_1d(m.psnr) for m in metrics_list]),
            mse=jnp.concatenate([_1d(m.mse) for m in metrics_list]),
            total_variation=jnp.concatenate([_1d(m.total_variation) for m in metrics_list]),
            tikonov=jnp.concatenate([_1d(m.tikonov) for m in metrics_list]),
            band_similarity=jnp.concatenate([_1d(m.band_similarity) for m in metrics_list]),
            penalties=jnp.concatenate([_1d(m.penalties) for m in metrics_list], axis=0),
            bound_compliance=jnp.concatenate([m.bound_compliance for m in metrics_list], axis=1),
            min_violations=jnp.concatenate([m.min_violations for m in metrics_list], axis=1),
            max_violations=jnp.concatenate([m.max_violations for m in metrics_list], axis=1),
        )


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

        compliance = compliance.at[i].set(
            jnp.where(mask_size > 0, compliance_count / mask_size, 0.0)
        )

    return compliance


def bound_violations(
    pred: ForwardT, segmentations: SegmentationT, value_ranges: ValueRangeT
) -> tuple[Float[Array, "reduced_labels batch"], Float[Array, "reduced_labels batch"]]:
    """Max and min violations, only when the pixel is outside the bounds of the value ranges"""
    # For max violations: Only count values that exceed the upper bound
    max_violations = jnp.ones((value_ranges.shape[0], pred.shape[0]))
    min_violations = jnp.ones((value_ranges.shape[0], pred.shape[0]))

    for i, val_range in enumerate(value_ranges):
        mask = segmentations[:, i]

        minv, maxv = val_range
        region_values = pred * mask

        # TODO: masks dont preserve shape
        max_violation = jnp.where(mask > 0, jnp.maximum(0.0, region_values - maxv), 0.0)
        min_violation = jnp.where(mask > 0, jnp.maximum(0.0, minv - region_values), 0.0)

        max_violations = max_violations.at[i].set(max_violation.max(axis=(-2, -1)))
        min_violations = min_violations.at[i].set(min_violation.max(axis=(-2, -1)))

    return max_violations, min_violations


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

    # ms_ssim_val = jax.vmap(ms_ssim, in_axes=(0, 0))(pred, images)
    ms_ssim_val = jax.vmap(ms_ssim_l1, in_axes=(0, 0))(pred, images)
    mse_val = jnp.mean((pred - images) ** 2, axis=(-2, -1))
    penalties = batch_segmentation_sq_penalty(txm, segmentation, value_ranges)
    penalties = penalties.sum(axis=0)

    compliance = bound_compliance(pred, segmentation, value_ranges)
    max_violations, min_violations = bound_violations(pred, segmentation, value_ranges)

    low_similarity = unsharp_mask_similarity(pred, images, sigma=3.0, reduction="none")
    high_similarity = unsharp_mask_similarity(pred, images, sigma=10.0, reduction="none")

    return EvalMetrics(
        ssim=ssim_val,
        psnr=psnr_val,
        penalties=penalties,
        bound_compliance=compliance,
        min_violations=min_violations,
        max_violations=max_violations,
        mse=mse_val,
        total_variation=total_variation(txm, "none"),
        tikonov=tikhonov(txm, "none"),
        band_similarity=low_similarity + high_similarity,
        ms_ssim=ms_ssim_val,
    )
