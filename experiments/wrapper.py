import os
from dataclasses import dataclass
from utils.logging import (
    log,
    summary as log_summary,
)
from typing import Any, Callable, Generic, TypeVar, TypedDict

import jax
import jax.numpy as jnp
import optax
from chest_xray_sim.inverse.core import Optimizer
from jaxtyping import Array, Float, PyTree
from chest_xray_sim.inverse import metrics
from models import ExperimentInputs
from utils.tracking import (
    ExperimentProtocol,
)
from chest_xray_sim.inverse.eval import bound_compliance
from chest_xray_sim.types import TransmissionMapT, SegmentationT

DEBUG = True
DTYPE = jnp.float32


@dataclass(frozen=True)
class ExperimentArgs(ExperimentProtocol):
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


class ForwardParams(TypedDict):
    low_sigma: float
    low_enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]


H = TypeVar("H", bound=ExperimentArgs)


class SegmentationOptimizer(Optimizer, Generic[H]):
    hyperparams: H 
    segmentation: SegmentationT 

    projection_fn: Callable[
        [TransmissionMapT, WeightsT], tuple[TransmissionMapT, WeightsT]
    ]

    def __init__(self, inputs: ExperimentInputs, hyperparams: H, **args):
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
        self.hyperparams.log_samples = 10

    def init_state(self) -> tuple[TransmissionMapT, WeightsT]:
        raise NotImplementedError(
            "init_state method must be implemented in the subclass"
        ) 


    def log(self, metrics: dict[str, Any]):
        jax.debug.callback(log, metrics)

    def summary(self, metrics: dict[str, Any]):
        jax.debug.callback(log_summary, metrics)

    def loss_fn_unaggregated(
        self,
        txm: TransmissionMapT,
        weights: WeightsT,
        pred: TransmissionMapT,
        target: TransmissionMapT,
        segmentation: SegmentationT | None = None,
    ) -> Float[Array, " batch"]:
        raise NotImplementedError(
            "loss_fn_unaggregated method must be implemented in the subclass"
        )

    def loss_call(self, weights, txm, target):
        pred = self.forward(txm, weights)
        per_image_losses = self.loss_fn_unaggregated(
            txm, weights, pred, target, self.segmentation
        )

        loss = jnp.mean(per_image_losses)

        ssim = metrics.ssim(pred, target)
        psnr = metrics.psnr(pred, target)
        compliance = bound_compliance(pred, self.segmentation, self.inputs.priors)
        compliance_metrics = {
            label: compliance[i] for i, label in enumerate(self.inputs.prior_labels)
        }

        self.log(
            {
                **{
                    "batch_loss": per_image_losses,
                    "loss": loss.item(),
                    "ssim": ssim,
                    "psnr": psnr,
                },
                **compliance_metrics,
            }
        )

        return loss

    def run(self) -> tuple[TransmissionMapT, WeightsT, TransmissionMapT, SegmentationT] | None:
        """Run the experiment."""
        # Initialize state
        txm0, w0 = self.init_state()

        # Run optimization
        state, losses = self.optimize(
            self.inputs.images, txm0, w0, self.inputs.segmentations
        )

        if state is not None:
            pred = self.forward(*state)
            return (*state, pred, self.inputs.segmentations)
        return None

