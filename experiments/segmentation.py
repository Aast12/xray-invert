from dataclasses import field, dataclass, MISSING
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
from chest_xray_sim.data.segmentation import ChestSegmentation, get_group_mask
from chest_xray_sim.inverse.core import segmentation_optimize
from utils import basic_loss_logger, log_image, experiment_args


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


def segmentation_loss(
    txm, weights, pred, target, segmentation, tv_factor=0.1, prior_weight=0.5
):
    """
    Loss function that incorporates segmentation information using probabilistic priors.

    The loss consists of:
    1. MSE between prediction and target (data fidelity)
    2. Total variation regularization (spatial smoothness)
    3. Log-likelihood of the transmission map given anatomical priors (anatomical plausibility)
    """
    # Basic MSE loss (data fidelity term)
    mse = metrics.mse(pred, target)

    # Total variation regularization (spatial smoothness)
    tv = metrics.total_variation(txm)

    # Extract bone and lung segmentation masks (probabilities)
    bone_mask = b_get_group_mask(segmentation, "bone", None)  # Use raw probabilities
    lung_mask = b_get_group_mask(segmentation, "lung", None)  # Use raw probabilities

    # Anatomical priors based on known physical properties
    # For bones (high attenuation = low transmission), we model as beta distribution with mode near 0.2
    # For lungs (low attenuation = high transmission), we model as beta distribution with mode near 0.8
    # We use log probabilities to convert to a loss term

    # Log-likelihood for bone regions - penalize high transmission values in bone regions
    # Higher probability = lower transmission (negative correlation)
    bone_prior = -jnp.sum(jnp.log(jnp.maximum(1.0 - txm, 1e-6)) * bone_mask)

    # Log-likelihood for lung regions - penalize low transmission values in lung regions
    # Higher probability = higher transmission (positive correlation)
    lung_prior = -jnp.sum(jnp.log(jnp.maximum(txm, 1e-6)) * lung_mask)

    # Normalize by area of masks to keep consistent scale regardless of mask size
    bone_area = jnp.sum(bone_mask) + 1e-6
    lung_area = jnp.sum(lung_mask) + 1e-6

    prior_loss = (bone_prior / bone_area + lung_prior / lung_area) * prior_weight

    # Combined loss with weighted components
    return mse + tv_factor * tv + prior_loss


def segmentation_projection(txm_state, weights_state, segmentation):
    """
    Project transmission map values based on segmentation information.
    Uses softer constraints with confidence-weighted projections.
    """
    # Basic projections first
    new_txm_state = optax.projections.projection_hypercube(txm_state)
    new_weights_state = optax.projections.projection_non_negative(weights_state)

    # Apply parameter constraints
    new_weights_state["low_sigma"] = optax.projections.projection_box(
        weights_state["low_sigma"], 0.1, 10
    )
    new_weights_state["low_enhance_factor"] = optax.projections.projection_box(
        new_weights_state["low_enhance_factor"], 0.1, 1.0
    )

    # Get raw segmentation probabilities (confidence values)
    bone_prob = b_get_group_mask(segmentation, "bone", None)
    lung_prob = b_get_group_mask(segmentation, "lung", None)

    # Normalize probabilities to [0,1] if needed
    bone_prob = jnp.clip(bone_prob, 0.0, 1.0)
    lung_prob = jnp.clip(lung_prob, 0.0, 1.0)

    # Define ideal values for each anatomical region
    ideal_bone_value = 0.2  # Lower transmission for bones (higher density)
    ideal_lung_value = 0.8  # Higher transmission for lungs (lower density)

    # Apply soft anatomical constraints - weighted combination of current value and ideal value
    # The higher the segmentation confidence, the stronger the pull toward the ideal value
    bone_strength = 0.5  # Controls how strongly we enforce bone constraints
    lung_strength = 0.5  # Controls how strongly we enforce lung constraints

    # Apply bone constraints with confidence weighting
    bone_influenced = (
        1.0 - bone_strength * bone_prob
    ) * new_txm_state + bone_strength * bone_prob * ideal_bone_value

    # Apply lung constraints with confidence weighting, respecting the bone constraints already applied
    lung_influenced = (
        1.0 - lung_strength * lung_prob
    ) * bone_influenced + lung_strength * lung_prob * ideal_lung_value

    # Ensure we're still in valid range
    constrained_txm = jnp.clip(lung_influenced, 0.0, 1.0)

    return constrained_txm, new_weights_state


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
    batch_shape = images.shape

    print("using shapes (imgs, segmentation):", images.shape, segmentations.shape)

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

    if hyperparams["tm_init_strategy"] == "target":
        txm0 = images.copy()
    elif hyperparams["tm_init_strategy"] == "segmentation_guided":
        # Create initial guess based on segmentation
        txm0 = jnp.ones(batch_shape)
        bone_mask = b_get_group_mask(segmentations, "bone", 0.6)
        lung_mask = b_get_group_mask(segmentations, "lung", 0.6)

        # Set initial values based on anatomy
        txm0 = jnp.where(bone_mask, 0.2, txm0)  # Bones have high attenuation
        txm0 = jnp.where(lung_mask, 0.8, txm0)  # Lungs have low attenuation

        # Add small random noise
        noise = jax.random.uniform(key, shape=batch_shape, minval=-0.05, maxval=0.05)
        txm0 = jnp.clip(txm0 + noise, 0.01, 0.99)
    else:
        # Random initialization
        txm_min, txm_max = hyperparams["tm_init_range"]
        if hyperparams["tm_distribution"] == "normal":
            txm0 = jax.random.normal(key, shape=batch_shape)
            txm0 = (txm0 - txm0.min()) / (txm0.max() - txm0.min())
            txm0 = txm0 * (txm_max - txm_min) + txm_min
        else:
            txm0 = jax.random.uniform(
                key, minval=txm_min, maxval=txm_max, shape=batch_shape
            )

    # Initial parameters for the forward model
    w0 = {
        "low_sigma": 4.0,
        "low_enhance_factor": 0.5,
        "window_center": 0.5,
        "window_width": 0.8,
        "gamma": 1.2,
    }

    # Create loss function with configurable weights
    def loss_fn(txm, weights, pred, target, seg):
        return segmentation_loss(
            txm,
            weights,
            pred,
            target,
            seg,
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
        project_fn=segmentation_projection
        if hyperparams["use_seg_projection"]
        else None,
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
        *[f.strip() for f in FWD_DESC.split(",")],
    ]

    print("Using args:", args)

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
            "total_variation": {"min": 0.001, "max": 0.5},
            "prior_weight": {"min": 0.1, "max": 2.0},  # Weight for anatomical priors
            "PRNGKey": {"values": [0, 42]},
            "tm_distribution": {"values": ["uniform", "normal"]},
            "tm_init_strategy": {"values": ["random", "segmentation_guided"]},
            "tm_init_range": {
                "values": [
                    (0.01, 0.99),
                    (0.2, 0.8),
                ]
            },
            "use_seg_projection": {"values": [True, False]},
            "constant_weights": {"values": [True, False]},
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
