from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import optax
import chest_xray_sim.utils.projections as proj
from jaxtyping import PyTree
from chest_xray_sim.inverse import metrics
from chest_xray_sim.pipeline.config import PipelineConfig
from chest_xray_sim.pipeline.forward import forward_single
from chest_xray_sim.pipeline.loss import compute_loss
from .inputs import ExperimentInputs
from .hyperparams import ExperimentArgs
from utils import projection_with_spec
from utils.logging import (
    log,
    log_txm_histograms,
    summary as log_summary,
)

from chest_xray_sim.inverse.eval import bound_compliance


class ForwardParams(TypedDict):
    enhance_factor: float
    high_enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]


def def_projection_spec(hyperparams: ExperimentArgs):
    projection_spec = {
        "enhance_factor": proj.box(
            hyperparams.low_min_enhancement, hyperparams.max_enhancement
        ),
        "high_enhance_factor": proj.box(
            hyperparams.high_min_enhancement, hyperparams.max_enhancement
        ),
        "window_center": proj.box(0.1, 0.5),
        "window_width": proj.box(0.1, 1.0),
    }

    if hyperparams.windowing_type == "sigmoid":
        projection_spec["gamma"] = proj.box(1, 20)

    return projection_spec


def default_params(hp):
    params = {
        "enhance_factor": 0.5,
        "window_center": 0.2,
        "window_width": 0.2,
        "high_enhance_factor": 0.2,
    }

    if hp.windowing_type == "sigmoid":
        params["gamma"] = 5.0
    return params


def create_optimizer(
    opt_type: str, lr: float, hyperparams
) -> optax.GradientTransformation:
    if opt_type == "adam":
        return optax.adam(learning_rate=lr)
    elif opt_type == "sgd":
        return optax.sgd(learning_rate=lr, momentum=hyperparams.momentum)
    elif opt_type == "amsgrad":
        return optax.amsgrad(learning_rate=lr)
    elif opt_type == "rmsprop":
        return optax.rmsprop(learning_rate=lr)
    elif opt_type == "adamw":
        return optax.adamw(
            learning_rate=lr, weight_decay=hyperparams.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


class DualBandForwardMixin:
    """Mixin providing the dual-band forward pipeline and shared optimization logic.

    Delegates forward and loss computation to chest_xray_sim.pipeline modules.

    Provides: __init__, _forward_single, loss_call_individual, project, log, summary, optimize.
    Subclasses must define forward() with the appropriate vmap strategy.

    Must be mixed with an Optimizer subclass (e.g. IndividualGradientOptimizer,
    SharedWeightsOptimizer) that provides optimize() and the optimizer __init__ params.
    """

    inputs: ExperimentInputs
    hyperparams: ExperimentArgs
    common_weights: bool
    w0: dict

    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentArgs,
        w0: dict,
        common_weights: bool = False,
        **kwargs,
    ):
        optimizer = create_optimizer(hyperparams.optimizer, hyperparams.lr, hyperparams)

        init_kwargs = dict(
            optimizer=optimizer,
            constant_weights=hyperparams.constant_weights,
            n_steps=hyperparams.n_steps,
            eps=hyperparams.eps,
            track_time=True,
            log_interval=100,
            **kwargs,
        )
        super().__init__(**init_kwargs)  # pyright: ignore[reportCallIssue]

        self.inputs = inputs
        self.hyperparams = hyperparams
        self.common_weights = common_weights
        self.w0 = w0
        if hasattr(hyperparams, 'to_pipeline_config'):
            self._pipeline_config = hyperparams.to_pipeline_config()
        else:
            self._pipeline_config = PipelineConfig(
                n_steps=hyperparams.n_steps,
                lr=hyperparams.lr,
                optimizer=hyperparams.optimizer,
                eps=hyperparams.eps,
                sigma=hyperparams.sigma,
                high_sigma=hyperparams.high_sigma,
                windowing_type=hyperparams.windowing_type,
                total_variation=hyperparams.total_variation,
                prior_weight=hyperparams.prior_weight,
                smooth_metric=hyperparams.smooth_metric,
                data_fidelity=hyperparams.data_fidelity,
                max_enhancement=hyperparams.max_enhancement,
                low_min_enhancement=hyperparams.low_min_enhancement,
                high_min_enhancement=hyperparams.high_min_enhancement,
                PRNGKey=hyperparams.PRNGKey,
                tm_init_params=hyperparams.tm_init_params,
                constant_weights=hyperparams.constant_weights,
                momentum=hyperparams.momentum,
                weight_decay=hyperparams.weight_decay,
                segmentation_th=getattr(hyperparams, 'segmentation_th', 0.6),
            )

        self.projection_fn = projection_with_spec(def_projection_spec(hyperparams))

    def _forward_single(self, txm, weights):
        """Forward processing for a single image: dual-band pipeline."""
        return forward_single(txm, weights, self._pipeline_config)

    def loss_call_individual(self, weights, txm, target, segmentation=None):
        """Individual loss computation including forward pass."""
        return compute_loss(
            weights, txm, target, self._pipeline_config,
            self.inputs.priors, segmentation,
        )

    def project(self, txm, weights, segmentation=None):
        """Apply projections to parameters."""
        return self.projection_fn(txm, weights)

    def log(self, metrics: dict[str, Any]):
        """Custom logging for experiments."""
        jax.debug.callback(log, metrics)

    def summary(self, metrics: dict[str, Any]):
        """Custom summary logging."""
        jax.debug.callback(log_summary, metrics)

    def optimize(self, target, txm0, w0, segmentation=None):
        """Override optimize to add custom logging."""
        self.segmentation = segmentation

        result = super().optimize(target, txm0, w0, segmentation)  # pyright: ignore[reportAttributeAccessIssue]

        if result[0] is not None:
            final_txm, final_weights = result[0]

            final_pred = self.forward(final_txm, final_weights)  # pyright: ignore[reportAttributeAccessIssue]

            log_txm_histograms(self.inputs, final_txm, final_pred, log_samples=30)

            # Ensure target has the same shape as final_pred
            if target.ndim == 4 and target.shape[1] == 1:
                target_squeezed = target.squeeze(1)
            else:
                target_squeezed = target

            ssim = metrics.ssim(final_pred, target_squeezed)
            psnr = metrics.psnr(final_pred, target_squeezed)

            if segmentation is not None:
                compliance = bound_compliance(
                    final_pred, segmentation, self.inputs.priors
                )
                compliance_metrics = {
                    label: compliance[i]
                    for i, label in enumerate(self.inputs.prior_labels)
                }
            else:
                compliance_metrics = {}

            self.summary(
                {
                    "final_loss": result[1][-1],
                    "final_ssim": ssim,
                    "final_psnr": psnr,
                    **compliance_metrics,
                }
            )

        return result
