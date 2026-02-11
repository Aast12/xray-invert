"""Concrete experiment class wiring forward, loss, and projections together."""

from typing import Any

from jaxtyping import Array, Float, PyTree

import optax

from chest_xray_sim.inverse.individual_gradient_optimizer import (
    IndividualGradientOptimizer,
)
from chest_xray_sim.types import ValueRangeT
from .config import PipelineConfig
from .forward import forward_batch
from .loss import compute_loss
from .projections import build_projection_fn


def _create_optimizer(config: PipelineConfig) -> optax.GradientTransformation:
    """Create an optax optimizer from config."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate=config.lr)
    elif config.optimizer == "sgd":
        return optax.sgd(learning_rate=config.lr, momentum=config.momentum)
    elif config.optimizer == "amsgrad":
        return optax.amsgrad(learning_rate=config.lr)
    elif config.optimizer == "rmsprop":
        return optax.rmsprop(learning_rate=config.lr)
    elif config.optimizer == "adamw":
        return optax.adamw(
            learning_rate=config.lr, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer}")


class InversePipelineExperiment(IndividualGradientOptimizer):
    """Pipeline experiment with no wandb dependency.

    Wires forward_single, compute_loss, and build_projection_fn together
    into an IndividualGradientOptimizer subclass.
    """

    def __init__(self, config: PipelineConfig, priors: ValueRangeT, **kwargs):
        optimizer = _create_optimizer(config)
        super().__init__(
            optimizer=optimizer,
            constant_weights=config.constant_weights,
            n_steps=config.n_steps,
            eps=config.eps,
            track_time=True,
            log_interval=config.log_interval,
            **kwargs,
        )
        self.config = config
        self.priors = priors
        self._projection_fn = build_projection_fn(config)

    def forward(self, txm, weights):
        """Batch forward processing with per-image weights."""
        result = forward_batch(txm, weights, self.config)  # pyright: ignore[reportArgumentType]
        return result

    def loss_call_individual(
        self,
        weights: PyTree,
        txm: Float[Array, "rows cols"],
        target: Float[Array, "rows cols"],
        segmentation: Float[Array, "labels rows cols"] | None = None,
    ) -> Float[Array, ""]:
        return compute_loss(weights, txm, target, self.config, self.priors, segmentation)

    def project(self, txm, weights, segmentation=None):
        return self._projection_fn(txm, weights)

    def log(self, metrics: dict[str, Any]):
        pass

    def summary(self, metrics: dict[str, Any]):
        pass
