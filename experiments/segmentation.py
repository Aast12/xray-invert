import itertools
import os
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import initialization as init
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import projections as proj
import torch
from eval import batch_evaluation
from jaxtyping import Array, Float, PyTree
from loss import (
    mse,
    segmentation_sq_penalty,
    total_variation,
    unsharp_mask_similarity,
)
from segmentation_utils import get_priors
from torch import Tensor
from utils import (
    basic_loss_logger,
    empty_loss_logger,
    experiment_args,
    process_results,
    pull_image,
)

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import (
    ChestSegmentation,
    batch_get_exclusive_masks,
)
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.inverse.core import segmentation_optimize
from chest_xray_sim.types import (
    ForwardT,
    SegmentationT,
    TransmissionMapT,
    ValueRangeT,
)

DEBUG = True
DTYPE = jnp.float32


@dataclass(frozen=True)
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
)


def forward(
    image: TransmissionMapT, weights: WeightsT
) -> Float[Array, "*batch rows cols"]:
    """Forward processing function that converts transmission maps to processed X-rays"""
    x = ops.negative_log(image)
    x = ops.windowing(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)
    x = ops.unsharp_masking(
        x, weights["low_sigma"], weights["low_enhance_factor"]
    )
    x = ops.clipping(x)

    return x


def segmentation_loss(
    txm: TransmissionMapT,
    weights: WeightsT,
    pred: ForwardT,
    target: ForwardT,
    segmentation: SegmentationT,
    value_ranges: ValueRangeT,
    tv_factor=0.1,
    prior_weight=0.5,
    gmse_weight=0.1,
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
    mse_val = mse(pred, target)
    tv_val = total_variation(txm)

    segmentation_penalty = segmentation_sq_penalty(
        txm, segmentation, value_ranges
    )

    # TODO: rename gms
    gms = unsharp_mask_similarity(pred, target)

    return (
        mse_val
        + tv_factor * tv_val
        + prior_weight * segmentation_penalty
        + gmse_weight * gms
    )


def segmentation_projection(
    txm_state: TransmissionMapT, weights_state: WeightsT, _
) -> tuple[TransmissionMapT, WeightsT]:
    """
    Project transmission map values based on segmentation information.
    Uses softer constraints with confidence-weighted projections.

    Args:
        txm_state: Transmission mp state
        weights_state: Weights state - parameters of the forward model
    """
    # General constraints
    new_txm_state = optax.projections.projection_hypercube(txm_state)
    new_weights_state = optax.projections.projection_non_negative(weights_state)

    # Apply constraints on image processing parameters
    new_weights_state = proj.projection_spec(
        new_weights_state,
        {
            "low_sigma": proj.box(0.2, 10),
            "low_enhance_factor": proj.box(0.2, 1.0),
        },
    )

    return new_txm_state, new_weights_state


def empty_logger(body):
    pass


def wandb_experiment(
    images: Float[Array, "batch height width"],
    masks_batch: Float[Array, "batch labels height width"],
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams: ExperimentArgs,
    logger: Callable[[dict], None] = empty_logger,
    loss_logger=empty_loss_logger,
    summary=empty_logger,
    segmentation_th=0.6,
    log_samples=5,
):
    """Main processing function to recover transmission maps with segmentation guidance"""

    log_samples = min(log_samples, images.shape[0])

    seg_labels, segmentations = batch_get_exclusive_masks(
        masks_batch, segmentation_th
    )

    priors_table = wandb.Table(
        columns=["region", "min", "max"],
        data=[
            (region, min_val, max_val)
            for region, (min_val, max_val) in zip(seg_labels, value_ranges)
        ],
    )
    # priors_table = wandb.Table(columns=["region", "min", "max"])
    # for region, value_range in zip(seg_labels, value_ranges):
    #     min_val, max_val = value_range
    #     priors_table.add_data(region, min_val, max_val)

    wandb.log({"priors": priors_table})

    summary(
        {
            "segmentation_th": segmentation_th,
            "viz_samples": log_samples,
        }
    )
    rand_samples = np.random.randint(
        0, images.shape[0], size=log_samples
    ).tolist()
    rand_samples = sorted(rand_samples)

    samples_tables = wandb.Table(columns=["index", "Image"])

    mask_labels = {i + 1: label for i, label in enumerate(seg_labels)}
    for idx in range(images.shape[0]):
        curr_image = images[idx]
        curr_masks = segmentations[idx]

        image_histogram = wandb.Histogram(curr_image.flatten())
        logger({"image histogram": image_histogram})
        for i, label in mask_labels.items():
            mask_idx = i - 1
            mask = curr_masks[mask_idx]
            logger(
                {
                    f"image ({label}) histogram": wandb.Histogram(
                        curr_image[mask.astype(jnp.bool)].flatten()
                    )
                }
            )

    for idx in rand_samples:
        curr_image = images[idx]
        curr_masks = segmentations[idx]
        general_mask = jnp.zeros_like(curr_masks[idx])

        for i, label in mask_labels.items():
            mask_idx = i - 1
            mask = curr_masks[mask_idx]
            general_mask = general_mask + mask * i
        # Only log full images for random samples
        masked_image = wandb.Image(
            pull_image(curr_image),
            masks={
                "predictions": {
                    "mask_data": np.array(general_mask),
                    "class_labels": mask_labels,
                },
            },
        )
        row_data = [idx, masked_image]
        samples_tables.add_data(*row_data)

    # Initialize transmission map with appropriate range
    init_mode, init_config = hyperparams.tm_init_params

    init_params: dict[str, Any] = dict(mode=init_mode)
    if init_mode in ["uniform", "normal"]:
        init_params["val_range"] = init_config
    elif init_mode == "target":
        init_params["target"] = images

    summary({"init_mode": init_mode})

    txm0 = init.initialize(hyperparams.PRNGKey, images.shape, **init_params)

    # Initial parameters for the forward model, should yield a proper image processing
    # for constant weights
    # TODO: long-term: automatic DIP parameter selection
    w0 = {
        "low_sigma": 4.0,
        "low_enhance_factor": 0.5,
        "window_center": 0.5,
        "window_width": 0.8,
        "gamma": 1.2,
    }

    summary({"initial_weights": w0})

    # TODO: gaussian blur enforces a conversion to float32, requires parameters to match the type
    w0 = {k: jnp.array(v, dtype=jnp.float32) for k, v in w0.items()}

    def loss_fn(*args):
        return segmentation_loss(
            *args,
            value_ranges=value_ranges,
            tv_factor=hyperparams.total_variation,
            prior_weight=hyperparams.prior_weight,
            gmse_weight=hyperparams.gmse_weight,
        )

    optimizer = optax.adam(learning_rate=hyperparams.lr)

    # todo: any processing with losses?
    state, _ = segmentation_optimize(
        target=images,
        txm0=txm0,
        w0=w0,
        segmentation=segmentations,
        loss_fn=loss_fn,
        optimizer=optimizer,
        forward_fn=forward,
        loss_logger=loss_logger,
        summary=summary,
        logger=logger,
        project_fn=segmentation_projection,
        constant_weights=hyperparams.constant_weights or False,
        n_steps=hyperparams.n_steps,
        eps=hyperparams.eps,
    )

    if state is None:
        raise RuntimeError("Optimization failed")

    txm, weights = state
    pred = forward(txm, weights)

    # Log recovered parameters
    summary({"recovered_params": weights})

    for idx in range(images.shape[0]):
        curr_image = txm[idx]
        curr_masks = segmentations[idx]
        # Log example images and segmentations

        for idx, label in enumerate(seg_labels):
            mask = curr_masks[idx]
            logger(
                {
                    f"transmission map ({label}) histogram": wandb.Histogram(
                        curr_image[mask.astype(jnp.bool)].flatten()
                    )
                }
            )

        image_histogram = wandb.Histogram(curr_image.flatten())
        logger({"transmission map histogram": image_histogram})

    # Log example recovered images
    fwd_data = [pull_image(pred[i]) for i in rand_samples]
    recovered_data = [pull_image(txm[i]) for i in rand_samples]

    samples_tables.add_column("Recovered", recovered_data)
    samples_tables.add_column("Forward", fwd_data)

    wandb.log({"samples": samples_tables})

    return txm, weights, pred, segmentations


def run_processing(
    images_batch: Float[Tensor, "batch height width"],
    masks_batch: Float[Tensor, "batch labels height width"],
    meta_batch: list[ChexpertMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    run_init={},
    save_dir=None,
):
    """Main processingfunction to recover transmission maps with segmentation guidance"""
    run = wandb.init(**run_init)
    hyperparams = run.config

    results = None
    images = jnp.array(images_batch.cpu().numpy()).squeeze(1)
    segmentations = jnp.array(masks_batch.cpu().numpy())

    def summary(body):
        for k, v in body.items():
            run.summary[k] = v

    try:
        results = wandb_experiment(
            images,
            segmentations,
            value_ranges=value_ranges,
            hyperparams=hyperparams,
            logger=wandb.log,
            summary=summary,
            loss_logger=basic_loss_logger,
        )
    except Exception as e:
        raise e
        print("Error during experiment:", e)
        return

    results, segmentations = results[:-1], results[-1]

    assert results is not None
    process_results(images, segmentations, meta_batch, value_ranges, results)


def sweep_based_exec(
    dataset, project, sweep_name, desc, tags, sweep_config=None, args=None
):
    _ = wandb.login()

    _, value_ranges = get_priors(args.cache_dir)

    run_init = dict(
        project=project,
        notes=f"Segmentation-guided optimization with {FWD_DESC}",
        tags=tags,
    )

    # Define hyperparameter sweep search space
    if sweep_config is None:
        sweep_config = {
            "name": sweep_name,
            "method": "bayes",
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {
                "lr": {"min": 5e-3, "max": 5e-2},
                "n_steps": {"values": [300, 600, 1200]},
                # regularization params, ensure they have some influence in loss
                "total_variation": {"min": 0.1, "max": 0.5},
                "prior_weight": {"min": 0.2, "max": 1.0},
                "gmse_weight": {"min": 0.0, "max": 1.0},
                "PRNGKey": {"values": [0, 42]},
                "tm_init_params": {
                    "values": [
                        *list(
                            itertools.product(
                                ["uniform", "normal"],
                                [(0.01, 0.99), (0.2, 0.8)],
                            )
                        ),
                        # ("zeros", None),
                        ("target", None),
                    ]
                },
                "constant_weights": {"values": [False]},
                # Fixed parameters
                "eps": {"value": 1e-6},
            },
        }

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )

    batch = next(iter(dataset))
    images, masks, meta = batch

    def sweep_runner():
        # new batch for each sweep run to increase diversity

        run_processing(
            images,
            masks,
            meta,
            value_ranges=value_ranges,
            save_dir=args.save_dir,
            run_init=run_init,
        )

    wandb.agent(
        sweep_id,
        function=sweep_runner,
        count=100,
    )


memory = joblib.Memory("cache", verbose=3)


# @memory.cache
def single_experiment(
    images,
    masks,
    value_ranges,
    hyperparams: ExperimentArgs,
    sample_size: int,
    logger=empty_logger,
    summary=empty_logger,
    loss_logger=empty_loss_logger,
):
    print("starting single experiment exec")
    segmentations = jnp.array(masks, dtype=DTYPE)
    value_ranges = jnp.array(value_ranges, dtype=DTYPE)
    images = jnp.array(images, dtype=DTYPE).squeeze(1)

    txm, weights, pred, segmentations = wandb_experiment(
        images,
        segmentations,
        value_ranges,
        hyperparams,
        logger=logger,
        summary=summary,
        loss_logger=loss_logger,
    )

    # seg_labels, segmentations = batch_get_exclusive_masks(segmentations, 0.6)
    metrics = batch_evaluation(images, txm, pred, segmentations, value_ranges)
    ssim = metrics.ssim
    psnr = metrics.psnr
    penalties = metrics.penalties

    sample_size = min(sample_size, images.shape[0])

    return (
        (
            txm[:sample_size],
            pred[:sample_size],
            images[:sample_size],
            segmentations[:sample_size],
        ),
        weights,
        (
            ssim,
            psnr,
            penalties,
        ),
    )


if __name__ == "__main__":
    args = args_spec()

    model = ChestSegmentation(cache_dir=args.cache_dir)

    def get_experiment_input_from_files(
        filepaths: list[str],
    ):
        images = []
        masks = []
        meta = []

        for filepath in filepaths:
            image = read_image(filepath)

            images.append(image)
            masks.append(model(image))
            patient_id, _ = os.path.splitext(filepath)

            meta.append(
                ChexpertMeta(
                    abs_img_path=os.path.abspath(filepath),
                    deid_patient_id=os.path.basename(patient_id),
                )
            )

        yield torch.stack(images), torch.stack(masks).squeeze(1), meta

    print("Using args:", args)

    # sweep based config
    PROJECT = "segmentation-shared-operator"
    SWEEP_NAME = "include-gmse"
    FWD_DESC = "negative log, windowing, range normalization, unsharp masking, clipping"

    TAGS = [
        "segmentation-guided",
        "square-penalty",
        "gmse",
        "fixed",
        *[f.strip() for f in FWD_DESC.split(",")],
    ]

    # Initialize W&B
    # Load dataset with segmentations
    dataset = get_segmentation_dataset(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        frontal_lateral=args.frontal_lateral,
        batch_size=args.batch_size,
    )

    # hyperparams = ExperimentArgs(
    #     lr=0.02,
    #     n_steps=400,
    #     total_variation=0.1,
    #     prior_weight=0.15,
    #     PRNGKey=0,
    #     tm_init_params=("normal", (0.2, 0.8)),
    #     constant_weights=False,
    #     eps=1e-6,
    # )

    # dataset = iter_as_numpy(dataset)
    # batch = next(iter(dataset))
    # images, masks, meta = batch
    # images, masks, meta = get_experiment_input_from_files(
    #     [
    #         "/Volumes/T7/projs/thesis/data/00000001_000.png",
    #         "/Volumes/T7/projs/thesis/data/00000001_001.png",
    #         "/Volumes/T7/projs/thesis/data/00000001_002.png",
    #     ]
    # )

    seg_labels, value_ranges = get_priors(args.cache_dir, as_numpy=True)

    # dataset = get_experiment_input_from_files(
    #     [
    #         "/Volumes/T7/projs/thesis/data/00000001_000.png",
    #         "/Volumes/T7/projs/thesis/data/00000001_001.png",
    #         "/Volumes/T7/projs/thesis/data/00000001_002.png",
    #     ]
    # )
    sweep_based_exec(
        dataset,
        project=PROJECT,
        sweep_name=SWEEP_NAME,
        desc="Segmentation-guided optimization with square penalty",
        tags=TAGS,
        args=args,
    )

    import sys

    sys.exit()

    # a = plt.imshow(masks[0][0])
    # plt.colorbar(a)

    # run = wandb.init(
    #     project="headless-runs",
    #     tags=[f"batch_size={args.batch_size}"],
    #     config=asdict(hyperparams),
    # )

    # sample_size = args.batch_size

    # def summary(body):
    #     for k, v in body.items():
    #         run.summary[k] = v

    # batch_data, weights, metrics_data = single_experiment(
    #     images,
    #     masks,
    #     value_ranges,
    #     hyperparams,
    #     sample_size,
    #     logger=run.log,
    #     summary=summary,
    #     loss_logger=basic_loss_logger,
    # )

    # save_dir = os.path.join("./outputs", run.id)

    # (txm, pred, images, segmentations), (ssim, psnr, penalties) = (
    #     batch_data,
    #     metrics_data,
    # )
    # # sync returned sample size
    # meta = meta[:sample_size]

    # i = jnp.argmax(ssim).item()

    # sample_fwd, sample_pred = images[i], pred[i]

    # run.log(
    #     {
    #         "SSIM": wandb.Histogram(ssim.flatten()),
    #         "PSNR": wandb.Histogram(psnr.flatten()),
    #     }
    # )

    # # TODO: fix segmentation labels forwarding
    # for i, label in enumerate(MASK_GROUPS):
    #     run.log({f"{label} penalty": wandb.Histogram(penalties[i].flatten())})

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # run_save_path = save_dir
    # os.makedirs(run_save_path, exist_ok=True)

    # joblib.dump(hyperparams, os.path.join(save_dir, "hyperparams.joblib"))

    # save_results(
    #     save_dir,
    #     txm,
    #     weights,
    #     meta,
    # )

    # run.finish()
