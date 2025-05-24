import os
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import jax
import jax.numpy as jnp
import optax
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
import chest_xray_sim.utils.projections as proj
from jaxtyping import Array, Float, PyTree
from wrapper import SegmentationOptimizer
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
    enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]


class SegmentationExperiment(SegmentationOptimizer):
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
        common_weights: bool = False,
        **args,
    ):
        super().__init__(
            inputs,
            hyperparams,
            **args,
        )

        self.common_weights = common_weights
        self.w0 = w0

        self.projection_fn = projection_with_spec(
            {
                # "low_sigma": proj.box(0.5, self.hyperparams.max_sigma),
                "enhance_factor": proj.box(0.05, self.hyperparams.max_enhancement),
                "gamma": proj.box(1, 20),
                "window_center": proj.box(0.1, 0.8),
                "window_width": proj.box(0.1, 1.0),
            }
        )

        weights_ax = jax.tree.map(lambda _: None if self.common_weights else 0, self.w0)

        self.forward_fn = jax.jit(
            jax.vmap(
                self._forward,
                in_axes=(
                    0,
                    weights_ax,
                ),
            )
        )

    def _forward(self, txm, weights):
        """Forward processing for a single image."""
        x = ops.negative_log(txm)
        x = ops.window(
            x,
            weights["window_center"],
            weights["window_width"],
            weights["gamma"],
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
            common_weights=self.common_weights,
        )

        jax.debug.print("w0: {w0}", w0=w0)

        return txm0, w0

    def project(self, txm, weights, segmentation=None):
        return self.projection_fn(txm, weights)

    def loss_fn_unaggregated(
        self,
        txm: TransmissionMapT,
        weights: WeightsT,
        pred: Float[Array, "batch height width"],
        target: Float[Array, "batch height width"],
        segmentation: Float[Array, "batch reduced_labels height width"],
    ) -> Float[Array, " batch"]:
        """
        Compute per-image losses without aggregation.

        Returns a vector of losses, one per image in the batch.
        """
        value_ranges = self.inputs.priors
        tv_factor = self.hyperparams.total_variation
        gmse_weight = self.hyperparams.gmse_weight
        prior_weight = self.hyperparams.prior_weight

        # Per-image MSE
        mse_per_image = jnp.mean((pred - target) ** 2, axis=(-2, -1))

        # gradient_loss = (
        #     (jnp.diff(pred, axis=-2) - jnp.diff(target, axis=-2)) ** 2
        # ).sum(axis=(-2, -1)) + (
        #     (jnp.diff(pred, axis=-1) - jnp.diff(target, axis=-1)) ** 2
        # ).sum(
        #     axis=(-2, -1)
        # )

        # Per-image total variation
        # Use the metrics module directly to get per-image TV
        tv_per_image = metrics.total_variation(txm, reduction=None)

        seg_penalty_per_image = metrics.batch_segmentation_sq_penalty(
            txm, segmentation, value_ranges
        )


        # Per-image unsharp mask similarity
        def per_image_gms(p, t):
            return metrics.unsharp_mask_similarity(
                p[None, ...], t[None, ...], 3.0
            ) + metrics.unsharp_mask_similarity(p[None, ...], t[None, ...], 10.0)

        gms_per_image = jax.vmap(per_image_gms)(pred, target)

        per_image_losses = (
            mse_per_image
            # + gradient_loss
            + tv_factor * tv_per_image
            + prior_weight * seg_penalty_per_image
            + gmse_weight * gms_per_image
        )

        return per_image_losses

    def loss_fn(
        self,
        txm: TransmissionMapT,
        weights: WeightsT,
        pred: Float[Array, "batch height width"],
        target: Float[Array, "batch height width"],
        segmentation: Float[Array, "batch reduced_labels height width"],
    ):
        per_image_losses = self.loss_fn_unaggregated(
            txm, weights, pred, target, segmentation
        )
        return jnp.mean(per_image_losses)


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

    exp = SegmentationExperiment(
        inputs,
        hyperparams,
        common_weights=True,
        w0={
            "enhance_factor": 0.5,
            "window_center": 0.2,
            "window_width": 0.2,
            "gamma": 5,
        },
    )
    results = exp.run()

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


def sweep_based_exec(dataset, project, sweep_name, desc, tags, sweep_config):
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
        SWEEP_NAME,
        desc="Segmentation-guided optimization with square penalty",
        tags=TAGS,
        sweep_config=config,
    )
