from dataclasses import field, dataclass, MISSING
import itertools
import os
import argparse

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jaxtyping import Array, Float
import wandb

from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
import chest_xray_sim.inverse.metrics as metrics
import chest_xray_sim.inverse.operators as ops
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from chest_xray_sim.data.segmentation import (
    ChestSegmentation,
    get_group_mask,
    MaskGroupsT,
)
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.inverse.core import segmentation_optimize
from utils import basic_loss_logger, log_image, experiment_args
import initialization as init
import projections as proj
import matplotlib.pyplot as plt


b_get_group_mask = jax.vmap(get_group_mask, in_axes=(0, None, None))
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


def forward(image, weights):
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
    images: jax.Array,
    segmentation: jax.Array,
    regions: list[MaskGroupsT] | MaskGroupsT,
    threshold: float = 0.6,
) -> dict[str, tuple[float, float]]:
    if isinstance(regions, str):
        regions = [regions]

    region_ranges: dict[str, tuple[float, float]] = {}

    for region_id in regions:
        mask = b_get_group_mask(segmentation, region_id, threshold)

        region_values = images * mask
        min_val = jnp.percentile(region_values[mask > 0], 5).item()
        max_val = jnp.percentile(region_values[mask > 0], 95).item()

        region_ranges[region_id] = (min_val, max_val)

    # plt.show()

    return region_ranges


def segmentation_loss(
    txm, weights, pred, target, segmentation, tv_factor=0.1, prior_weight=0.5
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

    bone_mask = b_get_group_mask(segmentation, "bone", 0.6)
    lung_mask = b_get_group_mask(segmentation, "lung", 0.6)

    segmentation_penalty = 0.0

    for mask, val_range in zip([bone_mask, lung_mask], [(0.0, 0.4), (0.6, 1.0)]):
        min_val, max_val = val_range

        region_values = txm * mask
        region_size = jnp.sum(mask)

        below_min = jnp.maximum(0.0, min_val - region_values) ** 2
        above_max = jnp.maximum(0.0, region_values - max_val) ** 2

        region_penalty = jnp.sum(below_min + above_max)

        segmentation_penalty += region_penalty / (region_size + 1e-6)

    return mse + tv_factor * tv + prior_weight * segmentation_penalty


def segmentation_projection(txm_state, weights_state, segmentation):
    """
    Project transmission map values based on segmentation information.
    Uses softer constraints with confidence-weighted projections.
    """
    # Basic projections first
    new_txm_state = optax.projections.projection_hypercube(txm_state)
    new_weights_state = optax.projections.projection_non_negative(weights_state)

    # Apply parameter constraints
    new_weights_state = proj.project_spec(
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
    images_batch,
    masks_batch,
    meta_batch: list[ChexpertMeta],
    run_init={},
    save_dir=None,
):
    """Main processing function to recover transmission maps with segmentation guidance"""
    run = wandb.init(**run_init)
    hyperparams = run.config

    images = jnp.array(images_batch.cpu().numpy()).squeeze(1)
    segmentations = jnp.array(masks_batch.cpu().numpy())

    # Log example images and segmentations
    for i in range(min(2, images.shape[0])):
        bone_mask = get_group_mask(segmentations[i], "bone", 0.6)
        lung_mask = get_group_mask(segmentations[i], "lung", 0.6)
        log_image(f"input_image_{i}", images[i])
        log_image(f"bone_seg_{i}", bone_mask)
        log_image(f"input_image_{i}", images[i])
        log_image(f"lung_seg_{i}", lung_mask)
        wandb.log(
            {
                "image_histogram": wandb.Histogram(images[i].flatten()),
                "bone_histogram": wandb.Histogram(images[i][bone_mask].flatten()),
                "lung_histogram": wandb.Histogram(images[i][lung_mask].flatten()),
            }
        )

    # Initialize transmission map with appropriate range
    key = jax.random.PRNGKey(hyperparams["PRNGKey"])

    init_mode = hyperparams["tm_init_params"][0]
    init_params = dict(mode=init_mode)
    if init_mode in ["uniform", "normal"]:
        init_params["val_range"] = hyperparams["tm_init_params"][1]
    if init_mode == "target":
        init_params["target"] = images

    txm0 = init.initialize(hyperparams["PRNGKey"], images.shape, **init_params)

    # Initial parameters for the forward model
    w0 = {
        "low_sigma": 4.0,
        "low_enhance_factor": 0.5,
        "window_center": 0.5,
        "window_width": 0.8,
        "gamma": 1.2,
    }

    # Create loss function with configurable weights
    def loss_fn(*args):
        return segmentation_loss(
            *args,
            tv_factor=hyperparams["total_variation"],
            prior_weight=hyperparams["prior_weight"],
        )

    # Run optimization with segmentation guidance
    optimizer = optax.adam(learning_rate=hyperparams["lr"])
    state, losses = segmentation_optimize(
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
    for i in range(min(2, images.shape[0])):
        recovered_image = forward(txm[i], weights)
        log_image(f"original_{i}", images[i])
        log_image(f"recovered_txm_{i}", txm[i])
        log_image(f"recovered_processed_{i}", recovered_image)

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

    # read prior images
    real_tm_paths = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
        "/Volumes/T7/projs/thesis/data/conventional_transmissionmap.tif",
    ]
    fwd_img_paths = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
        "/Volumes/T7/projs/thesis/data/conventional_processed.tif",
    ]
    print("len paths", len(real_tm_paths))

    model = ChestSegmentation(cache_dir=args.cache_dir)
    im = [read_image(path) for path in real_tm_paths]
    print("len images", len(im))

    real_tm_images = torch.stack(im)
    real_tm_segmentations = model(
        torch.stack([read_image(path) for path in fwd_img_paths])
    )
    real_tm_images = jnp.array(real_tm_images.cpu().numpy()).squeeze(1)
    real_tm_segmentations = jnp.array(real_tm_segmentations.cpu().numpy())

    print("tm shape:", real_tm_images[0].shape)
    print("seg shape:", real_tm_segmentations[0].shape)

    value_ranges = extract_region_value_ranges(
        real_tm_images, real_tm_segmentations, ["bone", "lung"]
    )
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

    # Set up W&B configuration
    run_init = dict(
        project=PROJECT,
        notes=f"Segmentation-guided optimization with {FWD_DESC}",
        tags=TAGS,
    )

    # Define hyperparameter sweep
    sweep_config = {
        "name": SWEEP_NAME,
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 5e-3, "max": 5e-2},
            "n_steps": {"values": [300, 600, 1200]},
            "total_variation": {"min": 0.0, "max": 0.5},
            "prior_weight": {"min": 0.0, "max": 0.2},
            "PRNGKey": {"values": [0, 42]},
            "tm_init_params": {
                "values": [
                    *list(
                        itertools.product(
                            ["uniform", "normal"], [(0.01, 0.99), (0.2, 0.8)]
                        )
                    ),
                    ("zeros", None),
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

        run_processing(images, masks, meta, save_dir=args.save_dir)

    wandb.agent(
        sweep_id,
        function=sweep_runner,
        count=20,
    )
