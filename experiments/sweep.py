"""Unified W&B sweep runner for segmentation experiments.

Replaces full_individual.py and full_shared.py.

Usage:
    uv run python experiments/sweep.py --sweep_conf experiments/sweep_configs/individual_gradient.yaml --mode individual
    uv run python experiments/sweep.py --sweep_conf experiments/sweep_configs/shared_gradient.yaml --mode shared
"""

import os
from typing import Any

import jax.numpy as jnp
import joblib
from chest_xray_sim.data.dataset import Loader
from chest_xray_sim.data.segmentation_dataset import SegmentationChexpertDataset
from chest_xray_sim.pipeline.inputs import prepare_torch_inputs
from jaxtyping import Array, Float
from models import (
    ExperimentInputs,
    IndividualSegmentationExperiment,
    SharedWeightsSegmentationExperiment,
    default_params,
)
from segmentation_utils import get_priors
from torch import Tensor
from utils import experiment_args, process_results
from utils.logging import summary
from utils.tracking import load_config

import wandb
from chest_xray_sim.data.chexpert import to_xray_meta
from chest_xray_sim.data.meta import XRayImageMeta

EXPERIMENT_CLASSES = {
    "individual": IndividualSegmentationExperiment,
    "shared": SharedWeightsSegmentationExperiment,
}

PROJECT_NAMES = {
    "individual": "individual-gradient-search",
    "shared": "shared-weights-search",
}


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
    mode="individual",
    bench_data="",
)


def run_processing(
    images_batch: Tensor,
    masks_batch: Tensor,
    meta_batch: list[XRayImageMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams: Any,
    mode: str,
    w0: dict | None = None,
    save_dir: str | None = None,
    segmentation_th: float = 0.6,
    label: str | None = None,
):
    common_weights = mode == "shared"
    exp_class = EXPERIMENT_CLASSES[mode]

    if w0 is None:
        w0 = default_params(hyperparams)

    images, segmentations, seg_labels, txm0, w0_batched = prepare_torch_inputs(
        images_batch,
        masks_batch,
        value_ranges,
        hyperparams,
        w0,
        common_weights=common_weights,
        segmentation_th=segmentation_th,
    )

    inputs = ExperimentInputs(
        images=images,
        segmentations=segmentations,
        prior_labels=seg_labels,
        priors=value_ranges,
    )

    assert inputs.images.ndim == 3, (
        f"Expected 3D images, got shape {inputs.images.shape}"
    )

    hyperparams.segmentation_th = segmentation_th
    summary({"w0": w0})

    exp = exp_class(
        inputs,
        hyperparams,
        w0=w0,
        common_weights=common_weights,
    )

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
            label=label,
        )


def sweep_based_exec(dataset, project, tags, sweep_config, mode, bench_data=None):
    _ = wandb.login()

    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)

    run_init = dict(
        project=project,
        notes=f"{mode} optimization sweep",
        tags=tags,
    )

    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    batch = dataset.next()
    g_images, g_masks, g_meta = batch

    # Load optional benchmark data
    real_im, real_mask, real_meta = None, None, None
    if bench_data and os.path.exists(bench_data):
        real_im, real_mask, real_meta = joblib.load(bench_data)
        real_meta = [to_xray_meta(m) for m in real_meta]

    def sweep_runner():
        run = wandb.init(**run_init)  # pyright: ignore

        hyperparams = run.config
        batch_size = len(batch)
        sub_batch_size = hyperparams.sub_batch_size

        run.tags = (run.tags or ()) + (
            f"batch_size={batch_size}",
            f"sub_batch_size={sub_batch_size}",
            f"mode={mode}",
        )

        save_dir = (
            os.path.join(args.save_dir, run.id)
            if args.save_dir is not None
            else None
        )

        for i in range(0, len(g_images), sub_batch_size):
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
                mode=mode,
                save_dir=save_dir,
            )

        # Run benchmark evaluation if data is available
        if real_im is not None and real_mask is not None and real_meta is not None:
            run_processing(
                real_im,
                real_mask,
                real_meta,
                hyperparams=hyperparams,
                value_ranges=value_ranges,
                mode=mode,
                save_dir=save_dir,
                label="bench",
            )

    wandb.agent(sweep_id, function=sweep_runner, count=200)


if __name__ == "__main__":
    args = args_spec()

    mode = args.mode
    assert mode in EXPERIMENT_CLASSES, f"Unknown mode: {mode}. Use 'individual' or 'shared'."

    project = PROJECT_NAMES[mode]

    FWD_DESC = "normalized negative log, windowing, range normalization, unsharp masking, clipping"
    TAGS = [
        f"{mode}-weights",
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

    dataset = SegmentationChexpertDataset(
        root=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        frontal_lateral=args.frontal_lateral,
    )

    loader = Loader(
        dataset,
        batch_size=batch_size,
        collate_fn=SegmentationChexpertDataset.collate,
        shuffle=False,
    )

    bench_data = args.bench_data if args.bench_data else None

    sweep_based_exec(
        loader,
        project,
        tags=TAGS,
        sweep_config=config,
        mode=mode,
        bench_data=bench_data,
    )
