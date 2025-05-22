import functools
from abc import ABC, abstractmethod
from typing import Any, Protocol, Tuple

import initialization as init
import jax.numpy as jnp
import optax
import yaml
from jaxtyping import Array, Float, PyTree, Scalar
from logging_utils import (
    log,
    log_image_histograms,
    log_priors_table,
    log_txm_histograms,
    summary,
)
from models import ExperimentInputs

from chest_xray_sim.inverse.core import (
    BatchT,
    SegmentationT,
    segmentation_optimize,
)
from chest_xray_sim.types import WeightsT


def load_config(path: str):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def build_segmentation_model_inputs(
    images: Float[Array, "batch rows cols"],
    segmentations: Float[Array, "labels batch rows cols"],
    seg_labels,
    value_ranges: Float[Array, "labels 2"],
    hyperparams: Any,
    common_weights: bool = False,
    segmentation_th=0.6,
):
    log_priors_table(seg_labels, value_ranges)
    init_mode, init_params = init.parse_init_params(hyperparams, images)
    summary({"init_mode": init_mode})

    txm0 = init.initialize(hyperparams.PRNGKey, images.shape, **init_params)

    w0 = full_pytree(
        {
            "low_sigma": 4.0,
            "low_enhance_factor": 0.5,
            "window_center": 0.2,
            "window_width": 0.2,
            "gamma": 5,
        },
        images.shape[0] if not common_weights else (),
        jnp.float32,
    )
    w0["low_sigma"] = 4.0

    return txm0, w0


def full_pytree(w0, shape, dtype=jnp.float32):
    return {k: jnp.full(shape, v, dtype=dtype) for k, v in w0.items()}


class ExperimentProtocol(Protocol):
    segmentation_th: float
    log_samples: int
    constant_weights: bool
    n_steps: int
    lr: float
    eps: float


class AbstractSegmentationExperiment(ABC):
    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentProtocol,
        optimizer: optax.GradientTransformation,
    ):
        self.inputs = inputs
        self.hyperparams = hyperparams
        self.optax = optimizer

    @abstractmethod
    def init_state(self) -> Tuple[BatchT, WeightsT]:
        pass

    @abstractmethod
    def forward(self, txm: BatchT, weights: WeightsT) -> BatchT:
        pass

    @abstractmethod
    def loss_fn(
        self,
        txm: BatchT,
        weights: WeightsT,
        pred: BatchT,
        target: BatchT,
        segmentation: SegmentationT,
    ) -> Scalar:
        pass

    def loss_logger(self, loss, *args):
        pass

    def logger(self, body):
        log(body)

    def summary(self, data):
        summary(data)

    @abstractmethod
    def projection(
        self, txm: PyTree, weights: WeightsT, segmentation: SegmentationT
    ) -> Tuple[PyTree, WeightsT]:
        pass

    def run(self):
        txm0, w0 = self.init_state()

        images = self.inputs.images
        segmentations = self.inputs.segmentations

        self.summary(
            {
                "segmentation_th": self.hyperparams.segmentation_th,
                "viz_samples": self.hyperparams.log_samples,
            }
        )

        log_image_histograms(self.inputs)

        forward_adapter = functools.partial(self.forward)
        loss_fn_adapter = functools.partial(self.loss_fn)
        projection_adapter = functools.partial(self.projection)
        loss_logger_adapter = functools.partial(self.loss_logger)
        logger_adapter = functools.partial(self.logger)
        summary_adapter = functools.partial(self.summary)

        # TODO: any processing with losses output?
        state, _ = segmentation_optimize(
            target=images,
            txm0=txm0,
            w0=w0,
            segmentation=segmentations,
            loss_fn=loss_fn_adapter,
            optimizer=self.optax,
            forward_fn=forward_adapter,
            loss_logger=loss_logger_adapter,
            summary=summary_adapter,
            logger=logger_adapter,
            project_fn=projection_adapter,
            constant_weights=self.hyperparams.constant_weights,
            n_steps=self.hyperparams.n_steps,
            eps=self.hyperparams.eps,
        )

        if state is None:
            raise RuntimeError("Optimization failed")

        txm, weights = state
        pred = self.forward(txm, weights)

        # Log recovered parameters
        self.logger({"recovered_params": weights})

        log_txm_histograms(self.inputs, txm, pred, self.hyperparams.log_samples)

        return txm, weights, pred, segmentations
