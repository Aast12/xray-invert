import os
from dataclasses import dataclass
from typing import Any, TypedDict
import typing

import jax
import jax.numpy as jnp
import optax
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
import chest_xray_sim.utils.projections as proj
from jaxtyping import Array, Float, PyTree
from chest_xray_sim.inverse.shared_weights_optimizer import SharedWeightsOptimizer
from chest_xray_sim.inverse import metrics
from models import ExperimentInputs
from segmentation_utils import get_priors
from torch import Tensor
from utils import (
    experiment_args,
    process_results,
    projection_with_spec,
)
from utils.tracking import (
    ExperimentProtocol,
    build_segmentation_model_inputs,
    load_config,
)
from utils.logging import (
    log,
    log_txm_histograms,
    summary as log_summary,
)

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.inverse.eval import bound_compliance

DEBUG = True
DTYPE = jnp.float32


args_spec = experiment_args(
    batch_size=32,
    frontal_lateral="Frontal",
    split="train",
    data_dir=os.environ.get("IMAGE_PATH"),
    meta_dir=os.environ.get("META_PATH"),
    mask_dir=os.environ.get("MASK_DIR"),
    cache_dir=os.environ.get("CACHE_DIR"),
    save_dir=os.environ.get("OUTPUT_DIR"),
    sweep_conf="",
)


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
    smooth_metric: typing.Literal["tikonov", "tv"]
    use_band_similarity: bool
    windowing_type: typing.Literal["sigmoid", "linear"]
    data_fidelity: typing.Literal["l2", "ssim"]
    # Shared weights optimizer specific params
    optimizer: str  # Optimizer type for weights
    momentum: float  # Momentum for SGD
    weight_decay: float  # Weight decay for AdamW
    sigma: float  # Sigma for unsharp masking
    high_sigma: float  # High sigma for unsharp masking
    low_min_enhancement: float  # Minimum enhancement factor for low enhancement
    high_min_enhancement: float  # Minimum enhancement factor for high enhancement


class ForwardParams(TypedDict):
    enhance_factor: float
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


class SharedWeightsSegmentationExperiment(SharedWeightsOptimizer):
    """Segmentation experiment using shared weights optimization."""
    
    def __init__(
        self,
        inputs: ExperimentInputs,
        hyperparams: ExperimentArgs,
        w0: dict,
        common_weights: bool = True,  # Should always be True for shared weights
        **kwargs,
    ):
        # Use different learning rates for txm and weights
        lr = hyperparams.lr
        
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
        
        optimizer = create_optimizer(hyperparams.optimizer, lr)
        
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
        projection_spec =  def_projection_spec(hyperparams)
        
        self.projection_fn = projection_with_spec(projection_spec)
    
    def _forward_single(self, txm, weights):
        """Forward processing for a single image."""
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
    
    def forward(self, txm, weights):
        """Batch forward processing."""
        # For shared weights, we vmap only over txm (weights are shared)
        result = jax.vmap(self._forward_single, in_axes=(0, None))(txm, weights)
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
        jax.debug.callback(log, metrics)
    
    def summary(self, metrics: dict[str, Any]):
        """Custom summary logging."""
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
            
            shared_weights = final_weights

            final_pred = self.forward(final_txm, shared_weights)
            
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


def run_processing(
    images_batch: Float[Tensor, "batch channels height width"],  # Fix type annotation
    masks_batch: Float[Tensor, "batch labels height width"],
    meta_batch: list[ChexpertMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams: Any,
    save_dir=None,
    segmentation_th=0.6,
):
    # Get model inputs
    segmentations = jnp.array(masks_batch.cpu().numpy())
    seg_labels, segmentations = batch_get_exclusive_masks(
        segmentations, segmentation_th
    )
    
    # Convert images and ensure they have shape (batch, H, W)
    images = jnp.array(images_batch.cpu().numpy())
    
    # The dataloader returns (batch, channels, H, W), we need (batch, H, W)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0, :, :]  # Take first channel explicitly
    elif images.ndim != 3:
        raise ValueError(f"Unexpected image shape: {images.shape}")
    
    inputs = ExperimentInputs(
        images=images,
        segmentations=segmentations,
        prior_labels=seg_labels,
        priors=value_ranges,
    )
    
    # Verify the shape is correct
    assert inputs.images.ndim == 3, f"Expected 3D images, got shape {inputs.images.shape}"
    
    hyperparams.segmentation_th = segmentation_th
    
    w0 = default_params(hyperparams)

    # Build initial states (always shared weights for this optimizer)
    txm0, w0_shared = build_segmentation_model_inputs(
        inputs,
        hyperparams,
        w0_state=w0,
        common_weights=True,  # Always use shared weights
    )
    
    # Create experiment
    exp = SharedWeightsSegmentationExperiment(
        inputs,
        hyperparams,
        w0=w0_shared,
        common_weights=True,  # Always use shared weights
    )
    
    # Run optimization
    state, _ = exp.optimize(
        inputs.images, txm0, w0_shared, inputs.segmentations
    )
    
    if state is not None:
        final_txm, shared_weights = state

        final_pred = exp.forward(final_txm, shared_weights)
        
        # For compatibility with process_results, broadcast weights back
        # final_weights_broadcast = jax.tree_map(
        #     lambda x: jnp.broadcast_to(x, (final_txm.shape[0],) + x.shape),
        #     shared_weights
        # )
        
        results = (final_txm, shared_weights, final_pred)
        
        process_results(
            inputs.images,
            inputs.segmentations,
            meta_batch,
            value_ranges,
            results,
            save_dir=save_dir,
        )


def sweep_based_exec(dataset, project, tags, sweep_config):
    _ = wandb.login()
    
    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)
    
    run_init = dict(
        project=project,
        notes=f"Shared weights optimization with {FWD_DESC}",
        tags=tags,
    )
    
    # Define hyperparameter sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )
    
    batch = next(iter(dataset))
    g_images, g_masks, g_meta = batch
    
    def sweep_runner():
        run = wandb.init(
            **run_init,  # pyright: ignore
        )
        
        hyperparams = run.config
        batch_size = len(batch)
        sub_batch_size = hyperparams.sub_batch_size
        
        run.tags = run.tags + (
            f"batch_size={batch_size}",
            f"sub_batch_size={sub_batch_size}",
            "fix-windowing",
            "fix-order",
        )
        
        for i in range(0, len(g_images), sub_batch_size):
            try:
                batch_size = hyperparams.batch_size
            except AttributeError:
                pass
            
            offset = i + sub_batch_size
            
            images, masks, meta = (
                g_images[i:offset],
                g_masks[i:offset],
                g_meta[i:offset],
            )
            
            save_dir = (
                os.path.join(args.save_dir, run.id)
                if args.save_dir is not None
                else None
            )
            
            run_processing(
                images,
                masks,
                meta,
                hyperparams=hyperparams,
                value_ranges=value_ranges,
                save_dir=save_dir,
            )
    
    wandb.agent(
        sweep_id,
        function=sweep_runner,
        count=200,
    )


if __name__ == "__main__":
    args = args_spec()
    
    # Sweep based config
    PROJECT = "shared-weights-search"
    FWD_DESC = "normalized negative log, windowing, range normalization, unsharp masking, clipping"
    
    TAGS = [
        "shared-weights",
        "segmentation-guided",
        "square-penalty",
        "fixed",
        "gmse",
        "valid",
        "sqrt-tv",
        *[f.strip() for f in FWD_DESC.split(",")],
    ]
    
    config = load_config(args.sweep_conf)
    
    print("Sweep config:", config)
    
    params = config.get("parameters", {})
    batch_size_d = params.get("batch_size", {})
    batch_size = max(
        batch_size_d.get("values", [batch_size_d.get("value", args.batch_size)])
    )
    
    dataset = get_segmentation_dataset(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        frontal_lateral=args.frontal_lateral,
        batch_size=batch_size,
    )
    
    # Execute sweep
    sweep_based_exec(
        dataset,
        PROJECT,
        tags=TAGS,
        sweep_config=config,
    )