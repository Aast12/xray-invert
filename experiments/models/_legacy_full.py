"""Legacy single-band segmentation experiment (from the original full.py).

This uses a different forward pipeline than the dual-band experiments:
  negative_log -> windowing -> range_normalize -> unsharp_masking -> clipping
"""

from dataclasses import dataclass
from typing import Any, Callable, TypedDict
import typing

import jax
import jax.numpy as jnp
import optax
import chest_xray_sim.utils.projections as proj
from jaxtyping import Array, Float, PyTree
from chest_xray_sim.inverse.shared_weights_optimizer import SharedWeightsOptimizer
from chest_xray_sim.inverse import metrics
from .inputs import ExperimentInputs
from utils import projection_with_spec
from utils.tracking import build_segmentation_model_inputs
from chest_xray_sim.types import TransmissionMapT

import chest_xray_sim.inverse.operators as ops


@dataclass
class ExperimentArgs:
    lr: float
    n_steps: int
    total_variation: float
    prior_weight: float
    gmse_weight: float
    PRNGKey: int
    tm_init_params: tuple[str, tuple[float, float] | None]
    constant_weights: bool
    eps: float
    max_sigma: float
    max_enhancement: float
    smooth_metric: typing.Literal["tikonov", "tv"]
    use_band_similarity: bool
    windowing_type: typing.Literal["sigmoid", "linear"]
    sub_batch_size: int = 32
    segmentation_th: float = 0.6


class ForwardParams(TypedDict):
    enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]


class SegmentationExperiment(SharedWeightsOptimizer):
    hyperparams: ExperimentArgs

    projection_fn: Callable[
        [TransmissionMapT, WeightsT], tuple[TransmissionMapT, WeightsT]
    ]

    common_weights: bool = False
    w0: dict

    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentArgs,
        w0: dict,
        common_weights: bool = True,  # Default to True for shared weights
        **args,
    ):
        # Initialize optimizer
        optimizer = optax.adam(learning_rate=hyperparams.lr)

        super().__init__(
            optimizer=optimizer,
            constant_weights=hyperparams.constant_weights,
            n_steps=hyperparams.n_steps,
            eps=hyperparams.eps,
            track_time=True,
            log_interval=100,
            **args,
        )

        self.inputs = inputs
        self.hyperparams = hyperparams
        self.common_weights = common_weights
        self.w0 = w0

        projection_spec = {
            # "low_sigma": proj.box(0.5, self.hyperparams.max_sigma),
            "enhance_factor": proj.box(0.05, self.hyperparams.max_enhancement),
            "window_center": proj.box(0.1, 0.8),
            "window_width": proj.box(0.1, 1.0),
        }

        if hyperparams.windowing_type == "sigmoid":
            projection_spec["gamma"] = proj.box(1, 20)

        self.projection_fn = projection_with_spec(projection_spec)

        # For shared weights, we don't vmap over weights dimension
        self.forward_fn = jax.vmap(
            self._forward,
            in_axes=(0, None),  # None for shared weights
        )

    def _forward(self, txm, weights):
        """Forward processing for a single image."""
        x = ops.negative_log(txm)
        x = ops.window(
            x,
            weights["window_center"],
            weights["window_width"],
            weights.get("gamma", 0.0),
            self.hyperparams.windowing_type,
        )
        x = ops.range_normalize(x)
        x = ops.unsharp_masking(x, 2.0, weights["enhance_factor"])

        x = ops.clipping(x)

        return x

    def forward(self, txm, weights):
        return self.forward_fn(txm, weights)

    def init_state(self):
        txm0, w0 = build_segmentation_model_inputs(
            self.inputs,
            self.hyperparams,
            w0_state=self.w0,
            common_weights=True,  # Always use shared weights
        )

        jax.debug.print("w0: {w0}", w0=w0)

        return txm0, w0

    def project(self, txm, weights, segmentation=None):
        return self.projection_fn(txm, weights)

    def loss_call_individual(self, weights, txm, target, segmentation=None):
        """Loss computation for a single image with shared weights."""
        # Expand dimensions for single image processing
        txm_single = txm[jnp.newaxis, ...]
        target_single = target[jnp.newaxis, ...]
        seg_single = segmentation[jnp.newaxis, ...] if segmentation is not None else None

        # Forward pass for single image
        pred = self.forward(txm_single, weights)

        # Compute loss for single image
        loss = self.loss_fn(
            txm_single,
            weights,
            pred[0],  # Remove batch dimension
            target_single[0],  # Remove batch dimension
            seg_single[0] if seg_single is not None else None  # Remove batch dimension
        )

        return loss

    def loss_fn(
        self,
        txm: TransmissionMapT,
        weights: WeightsT,
        pred: Float[Array, "height width"],  # Single image
        target: Float[Array, "height width"],  # Single image
        segmentation: Float[Array, "reduced_labels height width"] | None = None,  # Single image
    ):
        value_ranges = self.inputs.priors
        tv_factor = self.hyperparams.total_variation
        gmse_weight = self.hyperparams.gmse_weight
        prior_weight = self.hyperparams.prior_weight

        band_similarity = 1.0 if self.hyperparams.use_band_similarity else 0.0

        # l2 loss
        mse_per_image = 0.5 * jnp.mean((pred - target) ** 2)

        # Total variation for single image
        tv_value = (
            metrics.tikhonov(txm, reduction="mean")
            if self.hyperparams.smooth_metric == "tikonov"
            else metrics.total_variation(txm, reduction="mean")
        )

        # Segmentation penalty for single image
        if segmentation is not None:
            seg_penalty = (
                metrics.batch_segmentation_sq_penalty(txm, segmentation[jnp.newaxis, ...], value_ranges)
                .sum(axis=-1)
                .mean()
            )
        else:
            seg_penalty = 0.0

        gms = metrics.unsharp_mask_similarity(
            pred, target, sigma=3.0
        ) + metrics.unsharp_mask_similarity(pred, target, sigma=10.0)

        loss = (
            mse_per_image
            + tv_factor * tv_value
            + prior_weight * seg_penalty
            + gmse_weight * gms * band_similarity
        )

        return loss
