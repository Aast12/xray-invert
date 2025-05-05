import itertools
import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch import Tensor
from jaxtyping import Array, Float, PyTree
from typing import TypedDict
import wandb

from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
import chest_xray_sim.inverse.metrics as metrics
import chest_xray_sim.inverse.operators as ops
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from chest_xray_sim.data.segmentation import (
    MASK_GROUPS,
    ChestSegmentation,
    batch_get_exclusive_masks,
    MaskGroupsT,
)
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.inverse.core import segmentation_optimize
from utils import basic_loss_logger, log_image, experiment_args, map_range
import initialization as init
import projections as proj


DEBUG = True


class ForwardParams(TypedDict):
    low_sigma: float
    low_enhance_factor: float
    window_center: float
    window_width: float
    gamma: float

TransmissionMapT = Float[Array, "batch height width"]
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


def forward(image: TransmissionMapT, weights: WeightsT) -> Float[Array, "*batch rows cols"]:
    """Forward processing function that converts transmission maps to processed X-rays"""
    x = ops.negative_log(image)
    x = ops.windowing(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)
    x = ops.unsharp_masking(x, weights["low_sigma"], weights["low_enhance_factor"])
    x = ops.clipping(x)

    return x


def extract_region_value_ranges(
    images: Array,  
    segmentation: Array,
    regions: list[MaskGroupsT] | MaskGroupsT,
    threshold: float = 0.6,
    collimated_area_bound: float = 0.8,
) -> tuple[list[MaskGroupsT], Float[Array, "mask_groups 2"]]:
    """
    Idea:

        Different transmission maps (maybe due to dose or different BMIs) will have different ranges
        in the collimated area (largest consecutive value area in the histogram). A way to normalize it is
        to take the range from 0 to the max value found in the lungs, that way we can stretch the transmission
        range to a common value, say 60% transmission.
    """
    if isinstance(regions, str):
        regions = [regions]

    region_ranges = {}

    exclusive_mask_labels, exclusive_masks = batch_get_exclusive_masks(
        segmentation, threshold
    )
    merged_masks = jnp.sum(exclusive_masks, axis=1)

    # TODO: assumes hard masks (not-None threshold)
    merged_masks = jnp.clip(merged_masks, 0, 1)

    max_masked_values = jnp.max(images * merged_masks, axis=(1, 2))
    min_masked_values = jnp.min(images * merged_masks, axis=(1, 2))

    for region_id in regions:
        mask = exclusive_masks[:, exclusive_mask_labels.index(region_id)]

        min_values = jnp.min(images, axis=(1, 2), where=mask > 0, initial=jnp.inf)
        max_values = jnp.max(images, axis=(1, 2), where=mask > 0, initial=0.0)

        # map the values to the range of the merged masks
        min_values = map_range(
            min_values, min_masked_values, max_masked_values, 0.0, collimated_area_bound
        )
        max_values = map_range(
            max_values, min_masked_values, max_masked_values, 0.0, collimated_area_bound
        )

        region_ranges[region_id] = jnp.array([min_values.mean(), max_values.mean()])

    return (regions, jnp.stack(list(region_ranges.values())))


@jax.jit
def compute_single_mask_penalty(
    mask_id: int,
    mask: TransmissionMapT,
    value_range: Float[Array, " 2"],
    txm: TransmissionMapT,
    penalties: Float[Array, " reduced_labels"],
) -> Float[Array, " reduced_labels"]:
    min_val, max_val = value_range

    region_values = txm * mask
    region_size = jnp.sum(mask, axis=(-2, -1))

    below_min = jnp.maximum(0.0, min_val - region_values) ** 2
    above_max = jnp.maximum(0.0, region_values - max_val) ** 2

    region_penalty = jnp.sum(
        (below_min + above_max) / (jnp.expand_dims(region_size, (1, 2)) + 1e6)
    )
    return penalties.at[mask_id].set(region_penalty)


@jax.jit
def segmentation_sq_penalty(
    txm: Float[Array, "batch height width"],
    segmentation: Float[Array, "batch reduced_labels height width"],
    value_ranges: Float[Array, "reduced_labels 2"],
):
    penalties = jnp.ones(value_ranges.shape[0])

    # TODO: possibly improve by making broadcast operations
    for mask_id, val_range in enumerate(value_ranges):
        penalties = compute_single_mask_penalty(
            mask_id, segmentation[:, mask_id], val_range, txm, penalties
        )

    return jnp.sum(penalties)


def segmentation_loss(
    txm: TransmissionMapT,
    weights: WeightsT,
    pred: Float[Array, "batch height width"],
    target: Float[Array, "batch height width"],
    segmentation: Float[Array, "batch reduced_labels height width"],
    value_ranges: Float[Array, "reduced_labels 2"],
    tv_factor=0.1,
    prior_weight=0.5,
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
    mse = metrics.mse(pred, target)
    tv = metrics.total_variation(txm)

    segmentation_penalty = segmentation_sq_penalty(txm, segmentation, value_ranges)

    return mse + tv_factor * tv + prior_weight * segmentation_penalty


def batch_evaluation(
    images: Float[Array, "batch height width"],
    txm: Float[Array, "batch height width"],
    pred: Float[Array, "batch height width"],
    segmentation: Float[Array, "batch labels height width"],
):
    """
    Evaluate the model on a batch of images and segmentations.
    """

    psnr = metrics.psnr(pred, images)
    ssim = metrics.ssim(pred, images)
    penalties = segmentation_sq_penalty(txm, segmentation, value_ranges)

    wandb.log(
        {
            "psnr": {
                "mean": psnr.mean(),
                "std": psnr.std(),
                "min": psnr.min(),
                "max": psnr.max(),
            },
            "ssim": {
                "mean": ssim.mean(),
                "std": ssim.std(),
                "min": ssim.min(),
                "max": ssim.max(),
            },
            "penalies": {
                "sum": penalties.sum(),
                "mean": penalties.mean(),
                "data": penalties,
            },
        }
    )


def segmentation_projection(txm_state: TransmissionMapT,
                            weights_state: WeightsT, _) -> tuple[TransmissionMapT, WeightsT]:
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
            "low_sigma": proj.box(0.1, 10),
            "low_enhance_factor": proj.box(0.1, 1.0),
        },
    )

    return new_txm_state, new_weights_state


def save_image(img, path: str, bits=8):
    """Save normalized image to file"""
    x = ops.range_normalize(img)
    max_val = 2**bits - 1
    cv2.imwrite(path, np.array(x * max_val, dtype=np.uint8 if bits == 8 else np.uint16))


def run_processing(
    images_batch: Float[Tensor, "batch height width"],
    masks_batch: Float[Tensor, "batch labels height width"],
    meta_batch: list[ChexpertMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    run_init={},
    save_dir=None,
):
    """Main processing function to recover transmission maps with segmentation guidance"""
    run = wandb.init(**run_init)
    hyperparams = run.config

    wandb.log({"priors": value_ranges})

    images = jnp.array(images_batch.cpu().numpy()).squeeze(1)
    segmentation_th = 0.6
    seg_labels, segmentations = batch_get_exclusive_masks(
        jnp.array(masks_batch.cpu().numpy()), segmentation_th
    )

    rand_index = np.random.randint(0, images.shape[0])

    # Log example images and segmentations
    wandb.log({"image_histogram": wandb.Histogram(images[rand_index].flatten())})
    for i, label in enumerate(seg_labels):
        mask = segmentations[rand_index, i]
        log_image("input_image", images[rand_index])
        log_image(f"{label}_seg", mask)
        wandb.log(
            {
                "segmentation_histogram": wandb.Histogram(
                    images[rand_index][mask].flatten()
                ),
            }
        )

    # Initialize transmission map with appropriate range
    init_mode = hyperparams["tm_init_params"][0]
    init_params = dict(mode=init_mode)
    if init_mode in ["uniform", "normal"]:
        init_params["val_range"] = hyperparams["tm_init_params"][1]
    if init_mode == "target":
        init_params["target"] = images

    wandb.log({"init_mode": init_mode})

    txm0 = init.initialize(hyperparams["PRNGKey"], images.shape, **init_params)

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

    def loss_fn(*args):
        return segmentation_loss(
            *args,
            value_ranges=value_ranges,
            tv_factor=hyperparams["total_variation"],
            prior_weight=hyperparams["prior_weight"],
        )

    optimizer = optax.adam(learning_rate=hyperparams["lr"])

    # todo: any processing with losses?
    state, _ = segmentation_optimize(
        target=images,
        txm0=txm0,
        w0=w0,
        segmentation=segmentations,
        loss_fn=loss_fn,
        optimizer=optimizer,
        forward_fn=forward,
        loss_logger=basic_loss_logger,
        project_fn=segmentation_projection,
        constant_weights=hyperparams.get("constant_weights", False),
        n_steps=hyperparams["n_steps"],
        eps=hyperparams["eps"],
    )

    if state is None:
        print("Optimization failed")
        return

    txm, weights = state

    # Log recovered parameters
    wandb.log({"recovered_params": weights})

    # Log example recovered images
    recovered_image = forward(txm[rand_index], weights)
    log_image(f"original_{rand_index}", images[i])
    log_image(f"recovered_txm_{rand_index}", txm[rand_index])
    log_image(f"recovered_processed_{rand_index}", recovered_image)

    try:
        pred = forward(txm, weights)
        batch_evaluation(images, txm, pred, segmentations)
    except Exception as e:
        print("Error during batch evaluation:", e)
        # wandb.log({"batch_evaluation_error": str(e)})

    # Save results if needed
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        run_save_path = os.path.join(save_dir, run.id)
        os.makedirs(run_save_path, exist_ok=True)

        # Create file names from metadata
        file_names = [
            f"{m['deid_patient_id']}_{os.path.basename(m['abs_img_path'])}"
            for m in meta_batch
        ]

        # Save transmission maps and their processed versions
        for i, (img, name) in enumerate(zip(txm, file_names)):
            save_path = os.path.join(run_save_path, f"{name}")
            save_image(img, save_path)

            # Also save the processed version
            processed = forward(img, weights)
            proc_path = save_path.replace(".png", "_proc.png")
            save_image(processed, proc_path)

    return state


def get_priors(segmentation_cache_dir: str):
    # read prior images
    real_tm_paths = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
        "/Volumes/T7/projs/thesis/data/conventional_transmissionmap.tif",
    ]
    fwd_img_paths = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
        "/Volumes/T7/projs/thesis/data/conventional_processed.tif",
    ]

    segmentation_model = ChestSegmentation(cache_dir=segmentation_cache_dir)

    real_tm_data = torch.stack([read_image(path) for path in real_tm_paths])
    processed_data = torch.stack([read_image(path) for path in fwd_img_paths])

    print("prior data size:", len(real_tm_data))

    # segmentation model was trained on processed images, segmentations have to be
    # pulled from applying a forward image processing to the real transmission maps
    real_tm_segmentations = segmentation_model(processed_data)
    real_tm_images = jnp.array(real_tm_data.cpu().numpy()).squeeze(1)
    real_tm_segmentations = jnp.array(real_tm_segmentations.cpu().numpy())

    # TODO: incomplete label forwarding
    _, value_ranges = extract_region_value_ranges(
        real_tm_images, real_tm_segmentations, list(MASK_GROUPS)
    )
    print("value ranges:", value_ranges)

    del segmentation_model

    return value_ranges


if __name__ == "__main__":
    args = args_spec()

    from pprint import pprint

    print("args: ")
    pprint(args)

    # Initialize W&B
    _ = wandb.login()

    PROJECT = "segmentation-guided-transmission"
    SWEEP_NAME = "seg-guided-optimization"
    FWD_DESC = "negative log, windowing, range normalization, unsharp masking, clipping"

    TAGS = [
        "segmentation-guided",
        "square-penalty",
        *[f.strip() for f in FWD_DESC.split(",")],
    ]

    print("Using args:", args)

    value_ranges = get_priors(args.cache_dir)

    print("value ranges:", value_ranges)

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

    run_init = dict(
        project=PROJECT,
        notes=f"Segmentation-guided optimization with {FWD_DESC}",
        tags=TAGS,
    )

    # Define hyperparameter sweep search space
    sweep_config = {
        "name": SWEEP_NAME,
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 5e-3, "max": 5e-2},
            "n_steps": {"values": [300, 600, 1200]},
            # regularization params, ensure they have some influence in loss
            "total_variation": {"min": 0.1, "max": 0.5},
            "prior_weight": {"min": 0.1, "max": 0.5},
            "PRNGKey": {"values": [0, 42]},
            "tm_init_params": {
                "values": [
                    *list(
                        itertools.product(
                            ["uniform", "normal"], [(0.01, 0.99), (0.2, 0.8)]
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
        project=PROJECT,
    )

    def sweep_runner():
        # new batch for each sweep run to increase diversity
        batch = next(iter(dataset))
        images, masks, meta = batch

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
        count=20,
    )
