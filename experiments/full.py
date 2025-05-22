import os
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import jax
import jax.numpy as jnp
import optax
import projections as proj
from jaxtyping import Array, Float, PyTree
from loss import (
    mse,
    segmentation_sq_penalty,
    total_variation,
    unsharp_mask_similarity,
)
from models import ExperimentInputs
from segmentation_utils import get_priors
from torch import Tensor
from utils import (
    experiment_args,
    process_results,
    projection_with_spec,
)
from wandb_utils import (
    AbstractSegmentationExperiment,
    ExperimentProtocol,
    build_segmentation_model_inputs,
    load_config,
)

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.data.segmentation_dataset import (
    get_segmentation_dataset,
)
from chest_xray_sim.types import TransmissionMapT

DEBUG = True
DTYPE = jnp.float32


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
    low_sigma: float
    low_enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]


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


class SegmentationExperiment(AbstractSegmentationExperiment):
    hyperparams: ExperimentArgs

    projection_fn: Callable[
        [TransmissionMapT, WeightsT], tuple[TransmissionMapT, WeightsT]
    ]

    def __init__(self, inputs, hyperparams: ExperimentArgs):
        optimizer = optax.adam(learning_rate=hyperparams.lr)
        # todo
        super().__init__(inputs, hyperparams, optimizer)

        self.hyperparams.log_samples = 10
        self.projection_fn = projection_with_spec(
            {
                "low_sigma": proj.box(0.5, self.hyperparams.max_sigma),
                "low_enhance_factor": proj.box(
                    0.3, self.hyperparams.max_enhancement
                ),
                "gamma": proj.box(1, 20),
                "window_center": proj.box(0.1, 0.8),
                "window_width": proj.box(0.1, 1.0),
            }
        )

    def _forward(self, txm, weights):
        """Forward processing for batches of images with individual weights."""
        x = ops.negative_log(txm)
        x = ops.window(
            x,
            weights["window_center"],
            weights["window_width"],
            weights["gamma"],
        )
        x = ops.range_normalize(x)

        x = ops.unsharp_masking(
            x, weights["low_sigma"], weights["low_enhance_factor"]
        )

        x = ops.clipping(x)

        return x

    def forward(self, txm, weights):
        """Forward processing for batches of images with individual weights."""
        # jax.debug.print("weights: {w}", w=weights)
        return jax.vmap(
            self._forward,
            in_axes=(
                0,
                {
                    "window_center": 0,
                    "window_width": 0,
                    "gamma": 0,
                    "low_sigma": None,
                    "low_enhance_factor": 0,
                },
            ),
        )(txm, weights)

    def init_state(self):
        txm0, w0 = build_segmentation_model_inputs(
            self.inputs.images,
            self.inputs.segmentations,
            self.inputs.prior_labels,
            self.inputs.priors,
            self.hyperparams,
            common_weights=False,
        )

        return txm0, w0

    def projection(self, txm, weights, segmentation):
        return self.projection_fn(txm, weights)

    def loss_fn(
        self,
        txm: TransmissionMapT,
        weights: WeightsT,
        pred: Float[Array, "batch height width"],
        target: Float[Array, "batch height width"],
        segmentation: Float[Array, "batch reduced_labels height width"],
    ):
        """
        Loss function that incorporates segmentation information using probabilistic priors.

        The loss consists of:
        1. MSE between prediction and target (data fidelity)
        2. Total variation regularization (spatial smoothness)
        3. Anatomical region range penalty

        args:
            txm: Transmission map
            weights: Forward model parameters
            pred: Predicted image
            target: Target image
            segmentation: Segmentation map
            tv_factor: Weight for total variation regularization
            prior_weight: Weight for anatomical priors
        """
        value_ranges = self.inputs.priors
        tv_factor = self.hyperparams.total_variation
        gmse_weight = self.hyperparams.gmse_weight
        prior_weight = self.hyperparams.prior_weight

        mse_value = mse(pred, target)
        tv = total_variation(txm)

        segmentation_penalty = segmentation_sq_penalty(
            txm, segmentation, value_ranges
        )

        gms = unsharp_mask_similarity(
            pred, target, 3.0
        ) + unsharp_mask_similarity(pred, target, 10.0)

        return (
            mse_value
            + tv_factor * tv
            + prior_weight * segmentation_penalty
            + gmse_weight * gms
        )


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
    print(
        "segmentations", inputs.segmentations.shape, inputs.segmentations.dtype
    )
    print("prior_labels", inputs.prior_labels)
    print("priors", inputs.priors.shape, inputs.priors.dtype)

    hyperparams.segmentation_th = segmentation_th

    exp = SegmentationExperiment(inputs, hyperparams)
    results = exp.run()

    results, segmentations = results[:-1], results[-1]

    assert results is not None

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

    def sweep_runner():
        # new batch for each sweep run to increase diversity
        batch = next(iter(dataset))
        images, masks, meta = batch

        run = wandb.init(**run_init)  # pyright: ignore

        hyperparams = run.config
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
        f"batch_size={args.batch_size}",
        *[f.strip() for f in FWD_DESC.split(",")],
    ]

    dataset = get_segmentation_dataset(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        frontal_lateral=args.frontal_lateral,
        batch_size=args.batch_size,
    )

    config = load_config(args.sweep_conf)

    # execute sweep
    sweep_based_exec(
        dataset,
        PROJECT,
        SWEEP_NAME,
        desc="Segmentation-guided optimization with square penalty",
        tags=TAGS,
        sweep_config=config,
    )
