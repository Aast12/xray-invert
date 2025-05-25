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
from chest_xray_sim.inverse.individual_gradient_optimizer import IndividualGradientOptimizer
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
    # Individual optimizer specific params
    txm_optimizer: str  # Optimizer type for transmission maps
    weights_optimizer: str  # Optimizer type for weights
    txm_lr_factor: float  # Factor to multiply base lr for txm optimizer
    weights_lr_factor: float  # Factor to multiply base lr for weights optimizer
    momentum: float  # Momentum for SGD
    weight_decay: float  # Weight decay for AdamW


class ForwardParams(TypedDict):
    enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]


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
        # Determine optimizers based on sweep parameters
        base_lr = hyperparams.lr
        
        # Use different learning rates for txm and weights
        txm_lr = base_lr * hyperparams.txm_lr_factor
        weights_lr = base_lr * hyperparams.weights_lr_factor
        
        # Create optimizers based on hyperparameter choices
        def create_optimizer(opt_type: str, lr: float) -> optax.GradientTransformation:
            if opt_type == "adam":
                return optax.adam(learning_rate=lr)
            elif opt_type == "sgd":
                return optax.sgd(learning_rate=lr, momentum=hyperparams.momentum)
            elif opt_type == "rmsprop":
                return optax.rmsprop(learning_rate=lr)
            elif opt_type == "adamw":
                return optax.adamw(learning_rate=lr, weight_decay=hyperparams.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        txm_optimizer = create_optimizer(hyperparams.txm_optimizer, txm_lr)
        weights_optimizer = create_optimizer(hyperparams.weights_optimizer, weights_lr)
        
        super().__init__(
            txm_optimizer=txm_optimizer,
            weights_optimizer=weights_optimizer,
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
        projection_spec = {
            "enhance_factor": proj.box(0.05, self.hyperparams.max_enhancement),
            "window_center": proj.box(0.1, 0.8),
            "window_width": proj.box(0.1, 1.0),
        }
        
        if hyperparams.windowing_type == "sigmoid":
            projection_spec["gamma"] = proj.box(1, 20)
        
        self.projection_fn = projection_with_spec(projection_spec)
    
    def _forward_single(self, txm, weights):
        """Forward processing for a single image."""
        # txm should be 2D (H, W) for a single image
        # if txm.ndim != 2:
        #     raise ValueError(f"Expected 2D txm for single image, got shape {txm.shape}")
            
        # Add batch dimension for operators that expect it
        # txm_batched = txm[jnp.newaxis, ...]

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
        gmse_weight = self.hyperparams.gmse_weight
        prior_weight = self.hyperparams.prior_weight
        band_similarity = 1.0 if self.hyperparams.use_band_similarity else 0.0
        
        # MSE loss
        mse = 0.5 * jnp.mean((pred - target) ** 2)
        
        # Total variation (for single image)
        tv = (
            metrics.tikhonov(txm[jnp.newaxis, ...], reduction="mean")
            if self.hyperparams.smooth_metric == "tikonov"
            else metrics.total_variation(txm[jnp.newaxis, ...], reduction="mean")
        )
        
        # Segmentation penalty
        if segmentation is not None:
            seg_penalty = metrics.batch_segmentation_sq_penalty(
                txm[jnp.newaxis, ...], 
                segmentation[jnp.newaxis, ...], 
                value_ranges
            ).sum()
        else:
            seg_penalty = 0.0
        
        # Unsharp mask similarity
        gms = metrics.unsharp_mask_similarity(
            pred[jnp.newaxis, ...], target[jnp.newaxis, ...], sigma=3.0
        ) + metrics.unsharp_mask_similarity(
            pred[jnp.newaxis, ...], target[jnp.newaxis, ...], sigma=10.0
        )
        
        # Total loss
        loss = (
            mse
            + tv_factor * tv
            + prior_weight * seg_penalty
            + gmse_weight * gms * band_similarity
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

        print('shapes: ')
        print('target:', target.shape, target.dtype)
        print('txm0:', txm0.shape, txm0.dtype)
        print('w0:', w0)
        
        # Call parent optimize
        result = super().optimize(target, txm0, w0, segmentation)

        print('optimize result:', result)
        print('result[0]:', result[0][0].shape if result[0] is not None else None)
        print('result[1]:', result[1][1] if result[1] is not None else None)
        
        if result[0] is not None:
            # Compute final metrics
            final_txm, final_weights = result[0]
            final_pred = self.forward(final_txm, final_weights)
            
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
    
    print("inputs:")
    print("images", inputs.images.shape, inputs.images.dtype)
    print("segmentations", inputs.segmentations.shape, inputs.segmentations.dtype)
    print("prior_labels", inputs.prior_labels)
    print("priors", inputs.priors.shape, inputs.priors.dtype)
    
    hyperparams.segmentation_th = segmentation_th
    
    # Initialize weights
    w0 = {
        "enhance_factor": 0.5,
        "window_center": 0.2,
        "window_width": 0.2,
    }
    
    if hyperparams.windowing_type == "sigmoid":
        w0["gamma"] = 5.0
    
    # Build initial states (always per-image weights for this optimizer)
    txm0, w0_batched = build_segmentation_model_inputs(
        inputs,
        hyperparams,
        w0_state=w0,
        common_weights=False,  # Always use per-image weights
    )
    
    # Create experiment
    exp = IndividualSegmentationExperiment(
        inputs,
        hyperparams,
        w0=w0,
        common_weights=False,  # Always use per-image weights
    )
    
    # Run optimization
    state, _ = exp.optimize(
        inputs.images, txm0, w0_batched, inputs.segmentations
    )
    
    if state is not None:
        final_txm, final_weights = state
        final_pred = exp.forward(final_txm, final_weights)
        
        results = (final_txm, final_weights, final_pred)
        
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
        notes=f"Individual gradient optimization with {FWD_DESC}",
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
    PROJECT = "individual-gradient-search"
    FWD_DESC = "normalized negative log, windowing, range normalization, unsharp masking, clipping"
    
    TAGS = [
        "individual-gradient",
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