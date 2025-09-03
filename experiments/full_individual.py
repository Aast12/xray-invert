import os
from typing import Any

import jax.numpy as jnp
import joblib
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from jaxtyping import Array, Float
from utils.logging import summary
from individual import IndividualSegmentationExperiment, default_params
from models import ExperimentInputs
from segmentation_utils import get_priors
from torch import Tensor
from utils import (
    experiment_args,
    process_results,
)
from utils.tracking import (
    build_segmentation_model_inputs,
    load_config,
)

import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks

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


def run_processing(
    images_batch: Float[Tensor, "batch channels height width"],  # Fix type annotation
    masks_batch: Float[Tensor, "batch labels height width"],
    meta_batch: list[ChexpertMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams: Any,
    save_dir=None,
    segmentation_th=0.6,
    label=None,
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
    assert (
        inputs.images.ndim == 3
    ), f"Expected 3D images, got shape {inputs.images.shape}"

    hyperparams.segmentation_th = segmentation_th

    # Initialize weights
    # w0 = default_params(hyperparams)
    w0 = {
        "enhance_factor": 0.5285595655441284,
        "gamma": 4.998322486877441,
        "high_enhance_factor": 0.23372367024421692,
        "window_center": 0.29324406385421753,
        "window_width": 0.23960231244564056,
    }

    summary({'w0': w0})
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
    state, _ = exp.optimize(inputs.images, txm0, w0_batched, inputs.segmentations)


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
            label=label
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

    real_im, real_mask, real_meta = joblib.load('/Volumes/T7/projs/thesis/data/real_tm.joblib')

    def sweep_runner():
        run = wandb.init(
            **run_init,  # pyright: ignore
        )

        hyperparams = run.config
        batch_size = len(batch)
        sub_batch_size = hyperparams.sub_batch_size

        # sub_batch_size = 

        run.tags = run.tags + (
            f"batch_size={batch_size}",
            f"sub_batch_size={sub_batch_size}",
            "fix-windowing",
            "fix-order",
        )

        save_dir = (
            os.path.join(args.save_dir, run.id)
            if args.save_dir is not None
            else None
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

            run_processing(
                images,
                masks,
                meta,
                hyperparams=hyperparams,
                value_ranges=value_ranges,
                save_dir=save_dir,
            )
        

        run_processing(
            real_im,
            real_mask,
            real_meta,
            hyperparams=hyperparams,
            value_ranges=value_ranges,
            save_dir=save_dir,
            label='bench'
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
