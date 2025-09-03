import os
from dataclasses import dataclass
from typing import Any, Callable, TypedDict
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
    build_segmentation_model_inputs,
    load_config,
)

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.types import TransmissionMapT

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


def run_processing(
    images_batch: Float[Tensor, "batch height width"],
    masks_batch: Float[Tensor, "batch labels height width"],
    meta_batch: list[ChexpertMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams: Any,
    save_dir=None,
    segmentation_th=0.6,
):
    # get model inputs
    results = None
    segmentations = jnp.array(masks_batch.cpu().numpy())
    seg_labels, segmentations = batch_get_exclusive_masks(
        segmentations, segmentation_th
    )

    inputs = ExperimentInputs(
        images=jnp.array(images_batch.cpu().numpy()).squeeze(1),
        segmentations=segmentations,
        prior_labels=seg_labels,
        priors=value_ranges,
    )

    print("inputs:")
    print("images", inputs.images.shape, inputs.images.dtype)
    print("segmentations", inputs.segmentations.shape, inputs.segmentations.dtype)
    print("prior_labels", inputs.prior_labels)
    print("priors", inputs.priors.shape, inputs.priors.dtype)

    hyperparams.segmentation_th = segmentation_th

    w0 = {
        "enhance_factor": 0.5,
        "window_center": 0.2,
        "window_width": 0.2,
    }

    if hyperparams.windowing_type == "sigmoid":
        w0["gamma"] = 5.0

    exp = SegmentationExperiment(
        inputs,
        hyperparams,
        common_weights=True,
        w0=w0,
    )
    
    # Initialize states
    txm0, w0_shared = exp.init_state()
    
    # Run optimization
    state, _ = exp.optimize(inputs.images, txm0, w0_shared, inputs.segmentations)
    
    if state is not None:
        final_txm, final_weights = state
        final_pred = exp.forward(final_txm, final_weights)
        results = (final_txm, final_weights, final_pred, inputs.segmentations)
    else:
        results = None

    assert results is not None

    results, segmentations = results[:-1], results[-1]

    process_results(
        inputs.images,
        segmentations,
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
        notes=f"Segmentation-guided optimization with {FWD_DESC}",
        tags=tags,
    )

    # Define hyperparameter sweep search space
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

    # sweep based config
    PROJECT = "full-search"
    SWEEP_NAME = "full-sweep"
    FWD_DESC = "normalized negative log, windowing, range normalization, unsharp masking, clipping"

    TAGS = [
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

    # execute sweep
    sweep_based_exec(
        dataset,
        PROJECT,
        tags=TAGS,
        sweep_config=config,
    )
