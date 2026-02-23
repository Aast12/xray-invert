"""Unified replication and analysis script for W&B experiment runs.

Replaces individual_replicate.py and individual_constants.py.

Usage:
    uv run python experiments/replicate.py --run-id yiutl75n --batch-size 64 --batches 5
    uv run python experiments/replicate.py --run-id cuhsitz3 --batch-size 32 --batches 3
"""

import os

import jax.numpy as jnp
import joblib
import seaborn as sns
from chest_xray_sim.data.dataset import Loader
from chest_xray_sim.data.segmentation_dataset import SegmentationChexpertDataset
from chest_xray_sim.inverse.eval import EvalMetrics, batch_evaluation
from chest_xray_sim.pipeline.inputs import prepare_torch_inputs
from jaxtyping import Array, Float
from models import (
    ExperimentArgs,
    ExperimentInputs,
    IndividualSegmentationExperiment,
    default_params,
)
from plotting import (
    load_cache,
    plot_extreme_histograms,
    plot_gt_comparison,
    plot_gt_metrics,
    plot_gt_vs_naive,
    plot_histogram_heatmap,
    plot_metric_distributions,
    plot_metrics_boxplot,
    plot_percentile_samples,
    plot_sample_comparison,
    plot_weights_boxplot,
)
from segmentation_utils import get_priors
from torch import Tensor
from utils import chexpert_filenames, experiment_args, save_results

import wandb

WANDB_ENTITY = "alamst-kth-royal-institute-of-technology"
WANDB_PROJECT = "individual-gradient-search"


args_spec = experiment_args(
    batch_size=64,
    batches=5,
    frontal_lateral="Frontal",
    split="train",
    data_dir=os.environ.get("IMAGE_PATH"),
    meta_dir=os.environ.get("META_PATH"),
    mask_dir=os.environ.get("MASK_DIR"),
    cache_dir=os.environ.get("CACHE_DIR"),
    save_dir=os.environ.get("OUTPUT_DIR"),
    run_id="",
    cache=1,
    output_dir="",
    gt_tm_path="",
    gt_proc_path="",
)


# --- Processing ---


def process_results_local(images, segmentations, meta_batch, value_ranges, results, save_dir=None):
    txm, weights, pred = results

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(weights, os.path.join(save_dir, "weights.joblib"))
        joblib.dump(meta_batch, os.path.join(save_dir, "meta.joblib"))
        save_results(save_dir, txm, weights, pred, chexpert_filenames(meta_batch))

    return batch_evaluation(images, txm, pred, segmentations, value_ranges)


def run_processing(
    images_batch: Tensor,
    masks_batch: Tensor,
    meta_batch,
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams,
    w0: dict,
    save_dir: str | None = None,
    segmentation_th: float = 0.6,
):
    images, segmentations, seg_labels, txm0, w0_batched = prepare_torch_inputs(
        images_batch,
        masks_batch,
        value_ranges,
        hyperparams,
        w0,
        common_weights=False,
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

    exp = IndividualSegmentationExperiment(
        inputs,
        hyperparams,
        w0=w0,
        common_weights=False,
    )

    state, _ = exp.optimize(inputs.images, txm0, w0_batched, inputs.segmentations)

    if state is not None:
        final_txm, final_weights = state
        final_pred = exp.forward(final_txm, final_weights)
        results = (final_txm, final_weights, final_pred)

        metrics = process_results_local(
            inputs.images,
            inputs.segmentations,
            meta_batch,
            value_ranges,
            results,
            save_dir=save_dir,
        )
        return metrics, final_txm, final_weights, final_pred

    return None


def execute(dataset, hyperparams, w0, value_ranges, save_dir, batches=5):
    metrics_list = []
    imgs_list = []
    masks_list = []
    txm_list = []
    weights_list = []
    pred_list = []

    for i, batch in enumerate(dataset):
        if i >= batches:
            break

        images, masks, meta = batch

        res = run_processing(
            images, masks, meta,
            hyperparams=hyperparams,
            w0=w0,
            value_ranges=value_ranges,
            save_dir=save_dir,
        )
        if res is not None:
            pmetrics, final_txm, final_weights, final_pred = res
            metrics_list.append(pmetrics)
            txm_list.append(final_txm)
            weights_list.append(final_weights)
            pred_list.append(final_pred)

        # Store raw images/masks for plotting
        from chest_xray_sim.segmentation.utils import batch_get_exclusive_masks

        seg_jax = jnp.array(masks.cpu().numpy())
        _, seg_exclusive = batch_get_exclusive_masks(seg_jax)
        imgs_list.append(jnp.array(images.cpu().numpy()))
        masks_list.append(seg_exclusive)

    return metrics_list, imgs_list, masks_list, txm_list, weights_list, pred_list


# --- Plotting orchestration ---


def run_all_plots(metric, imgs, masks, txm, pred, weights, priors, save_fn):
    sns.set_theme()
    plot_metric_distributions(metric, save_fn)
    plot_weights_boxplot(weights, save_fn)
    plot_histogram_heatmap(txm, save_fn)
    plot_percentile_samples(pred, txm, metric.ms_ssim, save_fn)
    plot_extreme_histograms(metric, txm, masks, priors, save_fn)
    plot_metrics_boxplot(metric, save_fn)
    plot_sample_comparison(imgs, txm, pred, masks, priors, save_fn)


# --- Main ---


if __name__ == "__main__":
    args = args_spec()

    _ = wandb.login()

    run_id = args.run_id
    assert run_id, "Must provide --run-id"

    batch_size = args.batch_size
    batches = args.batches
    use_cache = bool(args.cache)

    wandb_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}"
    api = wandb.Api()
    run = api.run(wandb_path)
    conf = dict(run.config)
    del run, api

    # Clean up wandb-specific keys
    conf.pop("batch_size", None)
    conf.pop("sub_batch_size", None)
    conf.setdefault("log_samples", 10)
    conf.setdefault("gmse_weight", 0.0)

    config = ExperimentArgs(**conf)
    w0 = default_params(config)

    # Paths
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "replicates"
    )
    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "replicates_cache", f"{run_id}_{batch_size}_{batches}",
    )
    os.makedirs(cache_path, exist_ok=True)

    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)

    # Try loading from cache
    imgs, masks, txm, pred, weights, metrics = None, None, None, None, None, None
    if use_cache:
        try:
            data = load_cache(cache_path)
            imgs = data["imgs"]
            masks = data["masks"]
            txm = data["txm"]
            pred = data["pred"]
            weights = data["weights"]
            metrics = data["metrics"]
            print(f"Loaded cached results from {cache_path}")
        except FileNotFoundError:
            print("Cache miss, will recompute.")

    if imgs is None:
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

        metrics_list, imgs_list, masks_list, txm_list, weights_list, pred_list = execute(
            loader, config, w0, value_ranges, output_dir, batches=batches
        )

        imgs = jnp.concatenate(imgs_list, axis=0)
        masks = jnp.concatenate(masks_list, axis=0)
        txm = jnp.concatenate(txm_list, axis=0)
        weights = {
            k: jnp.concatenate([w[k] for w in weights_list], axis=0)
            for k in weights_list[0].keys()
        }
        pred = jnp.concatenate(pred_list, axis=0)
        metrics = EvalMetrics.merge(metrics_list)

        # Save to cache
        joblib.dump(imgs, os.path.join(cache_path, "imgs.joblib"))
        joblib.dump(masks, os.path.join(cache_path, "masks.joblib"))
        joblib.dump(txm, os.path.join(cache_path, "txm.joblib"))
        joblib.dump(pred, os.path.join(cache_path, "pred.joblib"))
        joblib.dump(weights, os.path.join(cache_path, "weights.joblib"))
        joblib.dump(metrics, os.path.join(cache_path, "metrics.joblib"))
        print(f"Saved results to {cache_path}")

    # Generate plots
    fig_dir = os.path.join(cache_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    def save_fn(filename):
        return os.path.join(fig_dir, filename)

    _, priors = get_priors(args.cache_dir)
    run_all_plots(metrics, imgs, masks, txm, pred, weights, priors, save_fn)
    print(f"Figures saved to {fig_dir}")

    # Ground truth comparison (optional)
    gt_tm_path = args.gt_tm_path
    gt_proc_path = args.gt_proc_path
    if gt_tm_path and gt_proc_path:
        import numpy as np
        from chest_xray_sim.data.utils import read_image

        assert txm is not None and masks is not None

        gt_txm = jnp.array(read_image(gt_tm_path)).squeeze()
        gt_proc = jnp.array(read_image(gt_proc_path)).squeeze()
        naive = (1 - gt_proc) / 2.0

        # Use first recovered sample for comparison
        recovered = txm[0].squeeze()
        sample_masks = np.array(masks[0].squeeze())

        plot_gt_comparison(recovered, gt_txm, sample_masks, priors, save_fn)
        plot_gt_vs_naive(gt_txm, gt_proc, sample_masks, priors, save_fn)
        plot_gt_metrics(recovered, gt_txm, naive, save_fn)
        print("Ground truth comparison plots saved.")
