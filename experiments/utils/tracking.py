# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Tuple 

import chest_xray_sim.utils.initialization as init

# Third-party imports
import jax
import jax.numpy as jnp
import optax
import yaml
from utils.logging import (
    log,
    log_priors_table,
)
from utils.logging import (
    summary as log_summary,
)
from models import ExperimentInputs

# Local imports
from chest_xray_sim.inverse.core import (
    BatchT,
    Optimizer,
)
from chest_xray_sim.types import WeightsT


def load_config(path: str):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def build_segmentation_model_inputs(
    inputs: ExperimentInputs,
    hyperparams: Any,
    w0_state,
    common_weights: bool = False,
):
    images = inputs.images
    seg_labels = inputs.prior_labels
    value_ranges = inputs.priors

    log_priors_table(seg_labels, value_ranges)
    init_mode, init_params = init.parse_init_params(hyperparams, images)
    log_summary({"init_mode": init_mode})

    txm0 = init.initialize(hyperparams.PRNGKey, images.shape, **init_params)

    w0 = full_pytree(
        w0_state,
        images.shape[0] if not common_weights else (),
        jnp.float32,
    )

    return txm0, w0


def full_pytree(w0, shape, dtype=jnp.float32):
    return jax.tree.map(lambda v: jnp.full(shape, v, dtype=dtype), w0)


class ExperimentProtocol(Protocol):
    segmentation_th: float
    log_samples: int
    constant_weights: bool
    n_steps: int
    lr: float
    eps: float


class AbstractSegmentationExperiment(Optimizer, ABC):
    """Base class for segmentation experiments using adaptive optimization.

    This class properly extends AdaptiveOptimizer without reimplementing
    its core logic. It provides a clean interface for segmentation experiments.
    """

    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentProtocol,
        optimizer: optax.GradientTransformation,
        # Additional adaptive optimization parameters
        # per_image_eps: float = 1e-8,
        # convergence_window: int = 5,
        # min_steps_before_stopping: int = 50,
        # shared_parameters: Optional[List[str]] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            constant_weights=hyperparams.constant_weights,
            n_steps=hyperparams.n_steps,
            eps=hyperparams.eps,
            track_time=True,
            log_interval=100,
            # Adaptive optimization parameters
            # per_image_eps=per_image_eps,
            # convergence_window=convergence_window,
            # min_steps_before_stopping=min_steps_before_stopping,
            # shared_parameters=shared_parameters,
        )
        self.inputs = inputs
        self.hyperparams = hyperparams

    @abstractmethod
    def init_state(self) -> Tuple[BatchT, WeightsT]:
        """Initialize the optimization state."""
        pass

    # Override log to use wandb logging
    def log(self, metrics: Dict[str, Any]):
        """Log metrics using wandb."""
        log(metrics)

    # Override summary to use wandb summary
    def summary(self, metrics: Dict[str, Any]):
        """Log summary using wandb."""
        log_summary(metrics)

    def run(self):
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
