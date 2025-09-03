import functools
import math
import os
from typing import Any

import time
import jax.numpy as jnp
import joblib
import matplotlib
from chest_xray_sim.data.segmentation_dataset import get_segmentation_dataset
from jaxtyping import Array, Float
from chest_xray_sim.inverse.eval import batch_evaluation
from individual import ExperimentArgs, IndividualSegmentationExperiment, default_params
from models import ExperimentInputs
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
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import MASK_GROUPS, batch_get_exclusive_masks

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

times = []
lens = []
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
    assert (
        inputs.images.ndim == 3
    ), f"Expected 3D images, got shape {inputs.images.shape}"

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

    st = time.time()

    # Run optimization
    state, _ = exp.optimize(inputs.images, txm0, w0_batched, inputs.segmentations)

    times.append(time.time() - st)
    lens.append(len(inputs.images))

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
    imgs = []
    txm = []
    masks_ = []
    weights = []
    pred = []

    it = iter(dataset)

    for i in range(batches):
        batch = next(it)
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

        segmentations = jnp.array(masks.cpu().numpy())
        seg_labels, segmentations = batch_get_exclusive_masks(segmentations)
        imgs.append(jnp.array(images.cpu().numpy()))

        masks_.append(segmentations)

    return metrics, imgs, masks_, txm, weights, pred


# from plot_replicate_results import plot_metrics_from_replicate


memory = joblib.Memory()

def segmentation_histogram(ax, data, mask_groups,labels = MASK_GROUPS, bins=50, small_bins=30):
    _, priors = get_priors(args.cache_dir)
    colors = ['red', 'blue', 'green']
    ax.hist(data.ravel(), bins=bins, alpha=0.5, label="total")

    for i, m in enumerate(mask_groups):
        label = labels[i]
        ax.axvline(priors[i, 0], color=colors[i], linestyle="--")
        ax.axvline(priors[i, 1], color=colors[i], linestyle="--")
        ax.hist(
            data[m.squeeze() > 0].ravel(),
            bins=small_bins,
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

    _ = wandb.login()

    path = "alamst-kth-royal-institute-of-technology/individual-gradient-search/runs/"
    # /Volumes/T7/projs/thesis/outputs/
    # run = "ma2kaqjp"
    run = "gcrl1gha"
    # run = 'cuhsitz3'

    config = None
    api = wandb.Api()
    run =  api.run(f"{path}{run}")
    run_id = run.id
    conf = run.config

    del run
    del api
    del conf["batch_size"]
    del conf["sub_batch_size"]

    conf["log_samples"] = 10
    conf["gmse_weight"] = 0.0

    config = ExperimentArgs(**conf)
    batch_size = 64

    cache = True
    path = "/Volumes/T7/projs/thesis/replicates/"
    cache_path = f"/Volumes/T7/projs/thesis/replicates_cache/{run_id}_{batch_size}"
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    # Load config
    imgs, masks, txm, pred, weights, metrics = [], [], [], [], [], []
    try:
        if not cache:
            raise FileNotFoundError("Cache is disabled, forcing recomputation.")

        imgs = joblib.load(os.path.join(cache_path, "imgs.joblib"))
        masks = joblib.load(os.path.join(cache_path, "masks.joblib"))
        txm = joblib.load(os.path.join(cache_path, "txm.joblib"))
        pred = joblib.load(os.path.join(cache_path, "pred.joblib")) 
        weights = joblib.load(os.path.join(cache_path, "weights.joblib"))
        metrics = joblib.load(os.path.join(cache_path, "metrics.joblib"))
        
        # a, b = get_priors(args.cache_dir,)
        # metrics = batch_evaluation(imgs.squeeze(), txm.squeeze(), pred.squeeze(), masks.squeeze(), b)
        # joblib.dump(metrics, os.path.join(cache_path, "metrics.joblib"))
    except Exception as e:
        dataset = get_segmentation_dataset(
            data_dir=args.data_dir,
            meta_dir=args.meta_dir,
            mask_dir=args.mask_dir,
            cache_dir=args.cache_dir,
            split=args.split,
            frontal_lateral=args.frontal_lateral,
            batch_size=batch_size,
        )

        times = []

        metrics_, imgs_, masks_, txm_, weights_, pred_ = execute(dataset, config, path)

        print('times: ', times)
        print('lens: ', lens)

        imgs = jnp.concatenate(imgs_, axis=0)
        masks = jnp.concatenate(masks_, axis=0)
        txm = jnp.concatenate(txm_, axis=0)
        weights = {k: jnp.concatenate([w[k] for w in weights_], axis=0) for k in weights_[0].keys()}
        pred = jnp.concatenate(pred_, axis=0)
        metrics = functools.reduce(
            lambda x, y: x.concat(y), metrics_[1:], metrics_[0]
        )
        # metrics = sum(metrics_, [])

        joblib.dump(imgs, os.path.join(cache_path, "imgs.joblib"))
        joblib.dump(masks, os.path.join(cache_path, "masks.joblib"))
        joblib.dump(txm, os.path.join(cache_path, "txm.joblib"))
        joblib.dump(pred, os.path.join(cache_path, "pred.joblib"))
        joblib.dump(weights, os.path.join(cache_path, "weights.joblib"))
        joblib.dump(metrics, os.path.join(cache_path, "metrics.joblib"))


    def f(path):
        dir = os.path.join(cache_path, "figures")
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        return os.path.join(dir, path)

    import numpy as np
    import random

    metric = metrics
    weighs = weights

    sample_id = random.choice(range(len(pred)))

    mask = masks[sample_id].squeeze()
    txm_sample = txm[sample_id].squeeze()
    pred_sample = pred[sample_id].squeeze()
    img_sample = imgs[sample_id].squeeze()


    fig, ax = plt.subplots(1, len(weighs.keys()), figsize=(15, 5))
    for i, (key, value) in enumerate(weighs.items()):
        ax[i].boxplot(value)
        ax[i].set_title(f"{key}")
    fig.savefig(f("weights_boxplot.svg"))

    num_samples = txm.shape[0]
    num_bins = 50
    hist_matrix = []
    for i in range(num_samples):
        hist, bin_edges = np.histogram(txm[i].ravel(), bins=num_bins, range=(0, 1), density=True)
        hist_matrix.append(hist)

    hist_matrix = np.array(hist_matrix).T  # shape: (num_bins, num_samples)
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    im = ax_hist.imshow(hist_matrix, aspect='auto', origin='lower', cmap='viridis',
                        extent=[0, num_samples, bin_edges[0], bin_edges[-1]])
    ax_hist.set_xlabel('Sample Index')
    ax_hist.set_ylabel('Transmission Map  Value')
    # ax_hist.set_title('TXM Sample Histograms (Heatmap)')
    fig_hist.colorbar(im, ax=ax_hist, label='Density')
    plt.tight_layout()

    fig_hist.savefig(f("histogram_heatmap.png"))


    max_penalty_idx = jnp.argmax(metric.penalties)
    min_penalty_idx = jnp.argmin(metric.penalties)

    max_mse_idx = jnp.argmax(metric.mse)
    min_mse_idx = jnp.argmin(metric.mse)

    max_ssim_idx = jnp.argmax(metric.ssim)
    min_ssim_idx = jnp.argmin(metric.ssim)

    max_ms_ssim_idx = jnp.argmax(metric.ms_ssim)
    min_ms_ssim_idx = jnp.argmin(metric.ms_ssim)

    indices = {
        'penalty': (max_penalty_idx, min_penalty_idx),
        # 'mse': (max_mse_idx, min_mse_idx),
        # 'ssim': (max_ssim_idx, min_ssim_idx),   
        '$\\mathcal{L}^{\\text{Mix}}$': (max_ms_ssim_idx, min_ms_ssim_idx),
    }

    # plot all txms in the extreme cases
    # fig, ax =  plt.subplots(2, len(indices.keys()), figsize=(25, 10))
    # # cmap = matplotlib.cm.get_cmap('gist_heat')
    # for i, (key, (max_idx, min_idx)) in enumerate(indices.items()):
    #     ax[0, i].imshow(txm[max_idx].squeeze(), cmap='gray', vmin=0, vmax=1)
    #     ax[0, i].axis('off')
    #     ax[0, i].set_title(f"Max {key.upper()} TXM")
    #     ax[1, i].imshow(txm[min_idx].squeeze(), cmap='gray', vmin=0, vmax=1)
    #     ax[1, i].axis('off')
    #     ax[1, i].set_title(f"Min {key.upper()} TXM")

    # fig.savefig(f("extreme_txm.svg"))

    # Plot a row of images from pred at uniformly separated percentiles of ms_ssim error
    num_samples_to_plot = 5
    ms_ssim_errors = metric.ms_ssim
    indices_perc = np.argsort(ms_ssim_errors)
    step = math.ceil(ms_ssim_errors.shape[0] / num_samples_to_plot)
    percentile_labels = np.linspace(0, 100, num_samples_to_plot, dtype=int)
    indices_percentiles = indices_perc[::step]
    print('indices_percentiles: ', indices_percentiles)

    fig, ax = plt.subplots(2, num_samples_to_plot, figsize=(3 * num_samples_to_plot, 8))
    for i, idx in enumerate(indices_percentiles):
        ax[0, i].imshow(pred[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title("$\\mathcal{L}^{\\text{Mix}}$:" + f" {ms_ssim_errors[idx]:.3f}\nP{percentile_labels[i]}")
        ax[0, i].axis('off')
        ax[1, i].imshow(txm[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[1, i].axis("off")
    # fig.suptitle("Predicted Images at Uniform $\mathcal{L}^{\\text{Mix}}$ Percentiles")
    fig.savefig(f("pred_ms_ssim_percentiles.png"))
    # plt.close(fig)

    # fig, ax =  plt.subplots(2, len(indices.keys()), figsize=(25, 10))
    # for i, (key, (max_idx, min_idx)) in enumerate(indices.items()):
    #     ax[0, i].imshow(pred[max_idx].squeeze(), cmap="gray", vmin=0, vmax=1)
    #     ax[0, i].axis('off')
    #     ax[0, i].set_title(f"Max {key.upper()} Pred. Processed")
    #     ax[1, i].imshow(pred[min_idx].squeeze(), cmap="gray", vmin=0, vmax=1)
    #     ax[1, i].axis('off')
    #     ax[1, i].set_title(f"Min {key.upper()} Pred. Processed")

    # fig.savefig(f("extreme_proc.svg"))


    fig, ax =  plt.subplots(2, 2, figsize=(20, 10))
    for i, (key, (max_idx, min_idx)) in enumerate(indices.items()):
        segmentation_histogram(
            ax[0, i],
            txm[max_idx].squeeze(),
            masks[max_idx]
        )
        ax[0, i].set_title(f"Max {key} Transmission Map Histogram")
        segmentation_histogram(
            ax[1, i],
            txm[min_idx].squeeze(),
            masks[min_idx]
        )
        ax[1, i].set_title(f"Min {key} Transmission Map Histogram")

    fig.savefig(f("extreme_histograms.png"))


    fig, ax = plt.subplots(1,2, figsize=(10, 10)) 
    ax = ax.flatten()

    ax[0].boxplot(metric.penalties.ravel(), showfliers=False)
    ax[0].set_title("Penalties")
    ax[1].boxplot(metric.ms_ssim.ravel(), showfliers=False)
    # ax[1].set_xscale("log")
    ax[1].set_title("$\\mathcal{L}^{\\text{Mix}}")


    fig.savefig(f("metrics.png"))


    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(img_sample, cmap="gray", vmin=0, vmax=1)
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    ax[1].imshow(pred_sample, cmap="gray", vmin=0, vmax=1)
    ax[1].axis('off')
    ax[1].set_title("Estimated Processed Image")
    plt.tight_layout()

    naive = (1 - img_sample) / 2.0

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(jnp.diff(naive, axis=-1), cmap="jet", vmin=0, vmax=1)
    ax[0].set_title("Trivial Diff")
    ax[1].imshow(jnp.diff(txm_sample, axis=-1), cmap="jet", vmin=0, vmax=1)
    ax[1].set_title("Recovered TM Diff")
    ax[2].imshow(jnp.diff(pred_sample, axis=-1), cmap="jet", vmin=0, vmax=1)
    ax[2].set_title("Predicted Diff")
    plt.tight_layout()

    # show txm histogram on mask
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    # ax[0, 0].hist(txm_sample.ravel(), bins=50, alpha=0.5, label="full")
    ax[0, 0].set_title("Recovered Transmission Map")
    # ax[1, 0].hist(naive.ravel(), bins=50, alpha=0.5, label="naive")
    ax[1, 0].set_title("Trivial Proc. Transmission Map")

    segmentation_histogram(
        ax[0, 0],
        txm_sample,
        mask,
    )
    segmentation_histogram(
        ax[1, 0],
        naive,
        mask,
    )

    fig.savefig(f("trivial_hist_comparison.png"))


    # for i, m in enumerate(mask):
    #     label = MASK_GROUPS[i]
    #     ax[0, 0].hist(
    #         txm_sample[m.squeeze() > 0].ravel(),
    #         bins=50,
    #         alpha=0.3,
    #         label=f"{label}",
    #     )
    #     ax[1, 0].hist(
    #         naive[m.squeeze() > 0].ravel(), bins=50, alpha=0.3, label=f"{label}"
    #     )



    ax[1, 0].legend()
    ax[0, 0].legend()
    ax[1, 1].imshow(naive.squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[0, 1].imshow(txm_sample.squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()

    plt.show()

    # plot_metrics_from_replicate(metrics)
