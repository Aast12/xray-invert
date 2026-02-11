import os
from typing import Any

import jax.numpy as jnp
import numpy as np
import joblib
import torch
from chest_xray_sim.data.segmentation_dataset import SegmentationChexpertDataset
from chest_xray_sim.data.dataset import Loader
from jaxtyping import Array, Float
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.inverse.eval import batch_evaluation
from models import ExperimentArgs, IndividualSegmentationExperiment, default_params, ExperimentInputs
from segmentation_utils import get_priors
from torch import Tensor
from utils import (
    chexpert_filenames,
    experiment_args,
    save_results,
)
from utils.tracking import (
    build_segmentation_model_inputs,
    load_config,
)

import wandb
from chest_xray_sim.data.chexpert import ChexpertMeta, to_xray_meta
from chest_xray_sim.segmentation import batch_get_exclusive_masks
from chest_xray_sim.segmentation.model import ChestSegmentation, MASK_GROUPS

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
)


def process_results(
    images,
    segmentations,
    meta_batch,
    value_ranges,
    results,
    save_dir=None,
):
    txm, weights, pred = results

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # run_save_path = os.path.join(save_dir, run.id)
        # os.makedirs(run_save_path, exist_ok=True)
        joblib.dump(weights, os.path.join(save_dir, "weights.joblib"))
        joblib.dump(meta_batch, os.path.join(save_dir, "meta.joblib"))

        save_results(save_dir, txm, weights, pred, chexpert_filenames(meta_batch))

    m = batch_evaluation(images, txm, pred, segmentations, value_ranges)

    return m


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
    assert inputs.images.ndim == 3, (
        f"Expected 3D images, got shape {inputs.images.shape}"
    )

    hyperparams.segmentation_th = segmentation_th

    # Initialize weights
    w0 = default_params(hyperparams)
    # Build initial states (always per-image weights for this optimizer)
    txm0, w0_batched = build_segmentation_model_inputs(
        inputs,
        hyperparams,
        w0_state=w0,
        common_weights=False,  # Always use per-image weights
        log=False,
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

        metrics = process_results(
            inputs.images,
            inputs.segmentations,
            meta_batch,
            value_ranges,
            results,
            save_dir=save_dir,
        )

        return metrics, final_txm, final_weights, final_pred

    return None


def execute(dataset, hyperparams, save_dir, batches=3):
    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)

    metrics = []
    txm = []
    weights = []
    pred = []

    for i in range(batches):
        try:
            batch = next(iter(dataset))
        except StopIteration:
            print("No more batches available in the dataset.")
            break
        images, masks, meta = batch

        res = run_processing(
            images,
            masks,
            meta,
            hyperparams=hyperparams,
            value_ranges=value_ranges,
            save_dir=save_dir,
        )
        if res is not None:
            pmetrics, final_txm, final_weights, final_pred = res
            metrics.append(pmetrics)
            txm.append(final_txm)
            weights.append(final_weights)
            pred.append(final_pred)

    return metrics, txm, weights, pred


# from plot_replicate_results import plot_metrics_from_replicate


def segmentation_histogram(
    ax, data, mask_groups, labels=MASK_GROUPS, bins=0.02, small_bins=0.01
):
    small_bins_d = np.arange(data.min(), data.max() + small_bins, small_bins)
    bins_d = np.arange(data.min(), data.max() + bins, bins)

    _, priors = get_priors(args.cache_dir)
    colors = ["red", "blue", "green"]
    ax.hist(data.ravel(), bins=bins_d, alpha=0.5, label="total")

    for i, m in enumerate(mask_groups):
        label = labels[i]
        ax.axvline(priors[i, 0], color=colors[i], linestyle="--")
        ax.axvline(priors[i, 1], color=colors[i], linestyle="--")
        ax.hist(
            data[m.squeeze() > 0].ravel(),
            bins=small_bins_d,
            alpha=0.3,
            color=colors[i],
            label=f"{label}",
        )
    ax.legend()

    return ax


if __name__ == "__main__":
    args = args_spec()

    import matplotlib.pyplot as plt
    import wandb
    import seaborn as sns

    sns.set_theme()

    _ = wandb.login()

    path = "alamst-kth-royal-institute-of-technology/individual-gradient-search/runs/"
    # /Volumes/T7/projs/thesis/outputs/
    # run = 'ma2kaqjp'
    # run = 'cuhsitz3'
    # run = 'gcrl1gha'
    run = "yiutl75n"

    api = wandb.Api()

    run = api.run(f"{path}{run}")

    conf = run.config
    del conf["batch_size"]
    del conf["sub_batch_size"]

    conf["log_samples"] = 10
    conf["gmse_weight"] = 0.0

    config = ExperimentArgs(**conf)

    import pathlib

    model = ChestSegmentation(cache_dir=args.cache_dir)

    def dataset_from_files(file_paths):
        images = torch.stack([read_image(fp) for fp in file_paths]).squeeze(0)
        masks = torch.stack([model(im) for im in images]).squeeze(0)
        print(images.shape, masks.shape)
        metas = [
            to_xray_meta(
                ChexpertMeta(
                    abs_img_path=fp,
                    deid_patient_id=pathlib.Path(fp).stem,
                    frontal_lateral="frontal",
                )
            )
            for fp in file_paths
        ]  # Placeholder for metadata

        yield images, masks, metas

    # dataset = get_segmentation_dataset(
    #     data_dir=args.data_dir,
    #     meta_dir=args.meta_dir,
    #     mask_dir=args.mask_dir,
    #     cache_dir=args.cache_dir,
    #     split=args.split,
    #     frontal_lateral=args.frontal_lateral,
    #     batch_size=32,
    # )

    path = "/Volumes/T7/projs/thesis/replicates_og/"
    tm_paths = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif"
    ]
    proc_paths = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif"
    ]

    def f(path):
        dir = f"/Volumes/T7/projs/thesis/data/gt_cache/{run.id}"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        return os.path.join(dir, path)

    tm = jnp.array(read_image(tm_paths[0])).squeeze()
    proc = jnp.array(read_image(proc_paths[0])).squeeze()

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex="col", sharey="col")

    ax[0, 0].imshow(tm, cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("Real Transmission Map")
    ax[0, 0].axis("off")
    ax[0, 1].hist(tm.ravel(), bins=50, color="blue", alpha=0.7)
    # ax[0, 1].set_xlabel('Pixel Intensity')
    ax[0, 1].set_ylabel("Frequency")

    ax[1, 0].imshow((1 - proc) / 2.0, cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title("Trivial Transmission Map")
    ax[1, 0].axis("off")
    ax[1, 1].hist(((1 - proc) / 2.0).ravel(), bins=50, color="blue", alpha=0.7)
    # ax[1, 1].set_xlabel('Pixel Intensity')
    ax[1, 1].set_ylabel("Frequency")

    plt.savefig(f("naive_histogram.png"))
    # plt.show()

    # import sys

    # sys.exit()

    # ax[0, 0].imshow(tm, cmap='gray', vmin=0, vmax=1)
    # ax[0, 0].set_title('Transmission Map')
    # ax[0, 1].imshow((1 - proc) / 2.0, cmap='gray', vmin=0, vmax=1)
    # ax[0, 1].set_title('Processed Image')

    # ax[1, 0].hist(tm.ravel(), bins=50, color='blue', alpha=0.7)
    # ax[1, 0].set_title('Histogram of Transmission Map')
    # ax[1, 1].hist((1 - proc).ravel(), bins=50, color='red', alpha=0.7)
    # ax[1, 1].set_title('Histogram of Processed Image')

    # plt.show()

    dataset = dataset_from_files(proc_paths)

    # joblib.dump(batch, '/Volumes/T7/projs/thesis/data/real_tm.joblib')

    tm = jnp.stack([jnp.array(read_image(fp)) for fp in tm_paths]).squeeze(0)
    print(f"Transmission map shape: {tm.shape}")

    # Execute sweep
    metrics, txm, weights, pred = execute(dataset, config, path)

    print("Metrics", metrics)

    from chest_xray_sim.inverse.metrics import ssim, ms_ssim_l1, mse

    print(tm[0], txm[0])

    tm0 = tm[0].squeeze()
    ptm0 = txm[0].squeeze()

    a = mse(tm0, ptm0)
    b = ssim(tm0, ptm0)
    c = ms_ssim_l1(tm0, ptm0)

    tm0 = np.array(tm0)
    ptm0 = np.array(ptm0)

    print(f"recoverd TM = MSE: {a}, SSIM: {b}, MS-SSIM: {c}")

    dataset = dataset_from_files(proc_paths)
    batch = next(iter(dataset))
    images, masks, meta = batch
    images = jnp.array(images.cpu().numpy()).squeeze(0)
    lb, masks = batch_get_exclusive_masks(jnp.array(masks.cpu()), threshold=0.6)
    # masks = jnp.array(masks.cpu().numpy()).squeeze()

    im0 = np.array(images)
    pred0 = pred[0].squeeze()
    pred0 = np.array(pred0)

    naive = (1 - im0) / 2.0

    fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex="col", sharey="col")
    ax[0, 0].imshow(tm0, cmap="gray", vmin=0, vmax=1)
    ax[1, 0].imshow(naive, cmap="gray", vmin=0, vmax=1)
    colors = ["blue", "red", "green"]

    segmentation_histogram(ax[0, 1], tm0, masks.squeeze(), small_bins=20, bins=40)
    segmentation_histogram(ax[1, 1], naive, masks.squeeze(), small_bins=20, bins=40)
    # ax[0, 1].hist(tm0.ravel(), bins=30, color='gray', alpha=0.3)
    # ax[1, 1].hist(naive.ravel(), bins=30, color='gray', alpha=0.3)
    # for i, (mask, color) in enumerate(zip(masks, colors)):
    #     ax[0, 1].hist(tm0[mask > 0].ravel(), bins=30, color=color, alpha=0.3)
    #     ax[1, 1].hist(naive[mask > 0].ravel(), bins=30, color=color, alpha=0.3)

    ax[0, 0].set_title("Original Transmission Map")
    ax[1, 0].set_title("Trivial Transmission Map")
    # ax[0, 1].set_title('Histogram of Original TM')
    # ax[1, 1].set_title('Histogram of Naive TM')
    ax[0, 0].axis("off")
    ax[1, 0].axis("off")

    fig.savefig(f("_histogram_tm.png"))

    a = mse(naive, tm0)
    b = ssim(naive, tm0)
    c = ms_ssim_l1(naive, tm0)
    print(f"naive tm = mse: {a}, ssim: {b}, ms-ssim: {c}")

    a = mse(naive, tm0)
    b = ssim(naive, tm0)
    c = ms_ssim_l1(naive, tm0)
    print(f"naive tm = mse: {a}, ssim: {b}, ms-ssim: {c}")

    print("image shape", im0.shape, pred0.shape)
    a = mse(im0, pred0)
    b = ssim(im0, pred0)
    c = ms_ssim_l1(im0, pred0)

    print(f"Image MSE: {a}, SSIM: {b}, MS-SSIM: {c}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(tm0, cmap="gray")
    ax[0, 0].set_title("Transmission Map")
    ax[0, 1].imshow(ptm0, cmap="gray")
    ax[0, 1].set_title("Recovered Transmission Map")
    ax[1, 0].imshow(im0, cmap="gray")
    ax[1, 0].set_title("Original Image")
    ax[1, 1].imshow(pred0, cmap="gray")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(ptm0, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Recovered Transmission Map")
    ax[1].imshow(tm0, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Original Transmission Map")
    ax[2].imshow(naive, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Trivial Transmission Map")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    plt.tight_layout()

    contrasted_im0 = ptm0.clip(0, 0.6) / 0.6
    # save image
    plt.imsave(
        f("contrasted recovered.png"), contrasted_im0, cmap="gray", vmin=0, vmax=1
    )
    contrasted_tm0 = tm0.clip(0, 0.6) / 0.6
    plt.imsave(f("contrasted tm.png"), contrasted_tm0, cmap="gray")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex="col", sharey="col")

    im = ax[0, 0].imshow(ptm0, cmap="gray", vmin=0, vmax=0.6)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("Recovered Transmission Map")
    # ax[0, 1].hist(ptm0.ravel(), bins=50, color='blue', alpha=0.7)
    segmentation_histogram(
        ax[0, 1], ptm0, masks.squeeze(), small_bins=0.015, bins=0.015
    )

    im = ax[1, 0].imshow(tm0, cmap="gray", vmin=0, vmax=0.6)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title("Original Transmission Map")
    # ax[2, 1].hist(tm0.ravel(), bins=50, color='green', alpha=0.7)
    segmentation_histogram(ax[1, 1], tm0, masks.squeeze(), small_bins=0.015, bins=0.015)
    # ax[2, 1].set_title('Histogram of Original TM')
    ax[0, 0].axis("off")
    ax[1, 0].axis("off")
    ax[0, 1].set_xlabel("Pixel Intensity")
    ax[1, 1].set_xlabel("Pixel Intensity")

    fig.savefig(f("cmp_histogram.pdf"))

    plt.show()
