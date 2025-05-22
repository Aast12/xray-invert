# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple

import initialization as init

# Third-party imports
import jax.numpy as jnp
import optax
import yaml
from jaxtyping import Array, Float, PyTree
from logging_utils import (
    log,
    log_image_histograms,
    log_priors_table,
    log_txm_histograms,
)
from logging_utils import (
    summary as log_summary,
)
from models import ExperimentInputs

# Local imports
from chest_xray_sim.inverse.core import (
    BatchT,
    Optimizer,
    SegmentationT,
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
    log_summary({"init_mode": init_mode})

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

    w0["low_sigma"] = 4

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


class AbstractSegmentationExperiment(Optimizer, ABC):
    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentProtocol,
        optimizer: optax.GradientTransformation,
    ):
        super().__init__(
            optimizer=optimizer,
            constant_weights=hyperparams.constant_weights,
            n_steps=hyperparams.n_steps,
            eps=hyperparams.eps,
            track_time=True,
            log_interval=100,
        )
        self.inputs = inputs
        self.hyperparams = hyperparams

    @abstractmethod
    def init_state(self) -> Tuple[BatchT, WeightsT]:
        """Initialize the optimization state."""
        pass

    def loss_logger(self, loss, *args):
        """Legacy method for loss logging."""
        pass

    def logger(self, body):
        """Log information using the logging utility."""
        log(body)

    def custom_summary(self, data):
        """Log summary information using the logging utility."""
        log_summary(data)

    # Implement abstract methods from Optimizer
    @abstractmethod
    def forward(self, txm: BatchT, weights: WeightsT) -> BatchT:
        """Forward model that converts transmission maps to processed images."""
        pass

    @abstractmethod
    def loss_fn(
        self,
        txm: BatchT,
        weights: WeightsT,
        pred: BatchT,
        target: BatchT,
        segmentation: Optional[SegmentationT] = None,
    ) -> Float[Array, ""]:
        """Loss function to evaluate the quality of predictions."""
        pass

    def log(self, metrics: Dict[str, Any]):
        """Log metrics during optimization."""
        self.logger(metrics)

    def summary(self, metrics: Dict[str, Any]):
        """Log summary metrics at the end of optimization."""
        self.custom_summary(metrics)
        # super().summary(metrics)

    def project(
        self,
        txm: PyTree,
        weights: WeightsT,
        segmentation: Optional[SegmentationT] = None,
    ) -> tuple[PyTree, WeightsT]:
        """Project states to valid ranges based on segmentation information."""
        return self.projection(txm, weights, segmentation)

    # Additional abstract methods specific to this class
    @abstractmethod
    def projection(
        self,
        txm: PyTree,
        weights: WeightsT,
        segmentation: Optional[SegmentationT] = None,
    ) -> Tuple[PyTree, WeightsT]:
        """Project states to valid ranges based on segmentation information."""
        pass

    def run(self):
        txm0, w0 = self.init_state()

        images = self.inputs.images
        segmentations = self.inputs.segmentations

        self.custom_summary(
            {
                "segmentation_th": self.hyperparams.segmentation_th,
                "viz_samples": self.hyperparams.log_samples,
            }
        )

        log_image_histograms(self.inputs)

        # Run the optimization (self is now an Optimizer)
        state, _ = self.optimize(
            target=images,
            txm0=txm0,
            w0=w0,
            segmentation=segmentations,
        )

        if state is None:
            raise RuntimeError("Optimization failed")

        txm, weights = state
        pred = self.forward(txm, weights)

        # Log recovered parameters
        self.logger({"recovered_params": weights})

        log_txm_histograms(self.inputs, txm, pred, self.hyperparams.log_samples)

        return txm, weights, pred, segmentations
