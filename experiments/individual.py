from dataclasses import dataclass
from typing import Any, TypedDict
import typing

import jax
import jax.numpy as jnp
import optax
import chest_xray_sim.utils.projections as proj
from jaxtyping import  PyTree
from chest_xray_sim.inverse.individual_gradient_optimizer import IndividualGradientOptimizer
from chest_xray_sim.inverse import metrics
from models import ExperimentInputs
from utils import (
    projection_with_spec,
)
from utils.tracking import (
    ExperimentProtocol,
)
from utils.logging import (
    log,
    log_txm_histograms,
    summary as log_summary,
)

import chest_xray_sim.inverse.operators as ops
from chest_xray_sim.inverse.eval import bound_compliance

DEBUG = True
DTYPE = jnp.float32

@dataclass()
class ExperimentArgs(ExperimentProtocol):
    variant: str
    lr: float
    segmentation_th: float
    log_samples: int
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
    data_fidelity: typing.Literal["l2", "ssim"]
    # Individual optimizer specific params
    optimizer: str  # Optimizer type for transmission maps
    lr: float  # factor to multiply base lr for txm optimizer
    momentum: float  # Momentum for SGD
    weight_decay: float  # Weight decay for AdamW
    sigma: float
    high_sigma: float
    low_min_enhancement: float
    high_min_enhancement: float


class ForwardParams(TypedDict):
    enhance_factor: float
    high_enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]

def def_projection_spec(hyperparams: ExperimentArgs):
    projection_spec = {
        "enhance_factor": proj.box(hyperparams.low_min_enhancement, hyperparams.max_enhancement),
        "high_enhance_factor": proj.box(hyperparams.high_min_enhancement, hyperparams.max_enhancement),
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


class IndividualSegmentationExperiment(IndividualGradientOptimizer):
    """Segmentation experiment using individual gradient optimization."""
    
    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentArgs,
        w0: dict,
        common_weights: bool = False,
        **kwargs,
    ):
        # Create optimizers based on hyperparameter choices
        def create_optimizer(opt_type: str, lr: float) -> optax.GradientTransformation:
            if opt_type == "adam":
                return optax.adam(learning_rate=lr)
            elif opt_type == "sgd":
                return optax.sgd(learning_rate=lr, momentum=hyperparams.momentum)
            elif opt_type == "amsgrad":
                return optax.amsgrad(learning_rate=lr)
            elif opt_type == "rmsprop":
                return optax.rmsprop(learning_rate=lr)
            elif opt_type == "adamw":
                return optax.adamw(learning_rate=lr, weight_decay=hyperparams.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        optimizer = create_optimizer(hyperparams.optimizer, hyperparams.lr)
        
        super().__init__(
            optimizer=optimizer,
            constant_weights=hyperparams.constant_weights,
            n_steps=hyperparams.n_steps,
            eps=hyperparams.eps,
            track_time=True,
            log_interval=100,
            **kwargs,
        )
        
        self.inputs = inputs
        self.hyperparams = hyperparams
        self.common_weights = common_weights
        self.w0 = w0
        
        # Setup projection function
        
        self.projection_fn = projection_with_spec(def_projection_spec(hyperparams))
    
    def _forward_single(self, txm, weights):
        """Forward processing for a single image."""
        # txm should be 2D (H, W) for a single image
        # if txm.ndim != 2:
        #     raise ValueError(f"Expected 2D txm for single image, got shape {txm.shape}")
            
        # Add batch dimension for operators that expect it
        # txm_batched = txm[jnp.newaxis, ...]
        x = ops.window(
            txm,
            weights["window_center"],
            weights["window_width"],
            weights.get("gamma", 0.0),
            self.hyperparams.windowing_type,
        )
        x = ops.negative_log(x)
        x = ops.unsharp_masking(x, self.hyperparams.high_sigma, weights["high_enhance_factor"])
        x = ops.clipping(x)
        x = ops.unsharp_masking(x, self.hyperparams.sigma, weights["enhance_factor"])
        x = ops.clipping(x)

        return x
        
        # Remove batch dimension to return single image
        # result = x[0]
        # assert result.ndim == 2, f"Expected 2D output, got shape {result.shape}"
        # return result
    
    def forward(self, txm, weights):
        """Batch forward processing."""
        # Always vmap over both since we're using per-image weights
        result = jax.vmap(self._forward_single, in_axes=(0, 0))(txm, weights)
        assert result.ndim == 3, f"Expected 3D batch output, got shape {result.shape}"
        return result
    
    def loss_call_individual(self, weights, txm, target, segmentation=None):
        """Individual loss computation including forward pass."""
        # Forward pass for single image
        pred = self._forward_single(txm, weights)
        
        # Compute individual loss components
        value_ranges = self.inputs.priors
        tv_factor = self.hyperparams.total_variation
        prior_weight = self.hyperparams.prior_weight
        
        # l2 loss 
        mse = 0.0
        if self.hyperparams.data_fidelity == "l2":
            mse = 0.5 * jnp.mean((pred - target) ** 2)
        elif self.hyperparams.data_fidelity == "ssim":
            mse =  metrics.ms_ssim_l1(pred, target)
        
        tv = 0.0

        if self.hyperparams.smooth_metric == "tikonov":
            tv = metrics.tikhonov(txm, reduction="mean")
        elif self.hyperparams.smooth_metric == "tv2":
            tv = metrics.total_generalized_variation(txm, reduction="mean")
        elif self.hyperparams.smooth_metric == "tv":
            tv = metrics.total_variation(txm[jnp.newaxis, ...], reduction="mean")
        
        # Segmentation penalty
        if segmentation is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm[jnp.newaxis, ...], 
                segmentation[jnp.newaxis, ...], 
                value_ranges
            ).sum()
        else:
            seg_penalty = 0.0
        
        
        # Total loss
        loss = (
            mse
            + tv_factor * tv
            + prior_weight * seg_penalty
        )
        
        return loss
    
    def project(self, txm, weights, segmentation=None):
        """Apply projections to parameters."""
        return self.projection_fn(txm, weights)
    
    def log(self, metrics: dict[str, Any]):
        """Custom logging for experiments."""
        # pass
        jax.debug.callback(log, metrics)
    
    def summary(self, metrics: dict[str, Any]):
        """Custom summary logging."""
        # pass
        jax.debug.callback(log_summary, metrics)
    
    def optimize(self, target, txm0, w0, segmentation=None):
        """Override optimize to add custom logging."""
        # Store segmentation
        self.segmentation = segmentation

        # Call parent optimize
        result = super().optimize(target, txm0, w0, segmentation)
        
        if result[0] is not None:
            # Compute final metrics
            final_txm, final_weights = result[0]

            final_pred = self.forward(final_txm, final_weights)
            
            log_txm_histograms(self.inputs, final_txm, final_pred, log_samples = 30)

            # Compute metrics
            # Ensure target has the same shape as final_pred
            if target.ndim == 4 and target.shape[1] == 1:
                target_squeezed = target.squeeze(1)
            else:
                target_squeezed = target
                
            ssim = metrics.ssim(final_pred, target_squeezed)
            psnr = metrics.psnr(final_pred, target_squeezed)
            
            if segmentation is not None:
                compliance = bound_compliance(final_pred, segmentation, self.inputs.priors)
                compliance_metrics = {
                    label: compliance[i] for i, label in enumerate(self.inputs.prior_labels)
                }
            else:
                compliance_metrics = {}
            
            # Log final metrics
            self.summary({
                "final_loss": result[1][-1],
                "final_ssim": ssim,
                "final_psnr": psnr,
                **compliance_metrics,
            })
        
        return result

