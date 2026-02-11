"""Ground truth comparison script.

Compares recovered transmission maps against known ground truth
using a real unprocessed/processed image pair.

Usage:
    uv run python experiments/ground_truth_compare.py \\
        --gt-tm-path data/real/Z01-oprocess.tif \\
        --gt-proc-path data/real/Z01-process.tif \\
        --run-id yiutl75n
"""

import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from chest_xray_sim.data.chexpert import ChexpertMeta
from chest_xray_sim.data.utils import read_image
from chest_xray_sim.segmentation import batch_get_exclusive_masks
from chest_xray_sim.segmentation.model import ChestSegmentation
from models import ExperimentArgs, default_params
from plotting import load_cache
from plotting.ground_truth import plot_gt_comparison, plot_gt_metrics, plot_gt_vs_naive
from segmentation_utils import get_priors
from utils import experiment_args, dataset_from_files

import wandb

WANDB_ENTITY = "alamst-kth-royal-institute-of-technology"
WANDB_PROJECT = "individual-gradient-search"


args_spec = experiment_args(
    batch_size=32,
    frontal_lateral="Frontal",
    split="train",
    data_dir=os.environ.get("IMAGE_PATH"),
    meta_dir=os.environ.get("META_PATH"),
    mask_dir=os.environ.get("MASK_DIR"),
    cache_dir=os.environ.get("CACHE_DIR"),
    save_dir=os.environ.get("OUTPUT_DIR"),
    gt_tm_path="",
    gt_proc_path="",
    run_id="",
    cache_dir_results="",
)


if __name__ == "__main__":
    args = args_spec()

    assert args.gt_tm_path, "Must provide --gt-tm-path"
    assert args.gt_proc_path, "Must provide --gt-proc-path"

    sns.set_theme()

    _, priors = get_priors(args.cache_dir)
    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)

    # Load ground truth images
    gt_tm = jnp.array(read_image(args.gt_tm_path)).squeeze()
    gt_proc = jnp.array(read_image(args.gt_proc_path)).squeeze()

    # Compute masks for the GT image
    model = ChestSegmentation(cache_dir=args.cache_dir)
    proc_tensor = torch.tensor(np.array(gt_proc)).unsqueeze(0).unsqueeze(0).float()
    mask_tensor = model(proc_tensor)
    _, masks = batch_get_exclusive_masks(jnp.array(mask_tensor.cpu()), threshold=0.6)
    masks = np.array(masks.squeeze())

    # Output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "gt_cache"
    )
    if args.run_id:
        output_dir = os.path.join(output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    def save_fn(filename):
        return os.path.join(output_dir, filename)

    # Plot GT vs naive
    plot_gt_vs_naive(gt_tm, gt_proc, masks, priors, save_fn)

    # If we have cached results from a run, load and compare
    if args.cache_dir_results:
        data = load_cache(args.cache_dir_results)
        recovered = data["txm"][0].squeeze()
        naive = (1 - np.array(gt_proc)) / 2.0

        plot_gt_comparison(np.array(recovered), np.array(gt_tm), masks, priors, save_fn)
        plot_gt_metrics(recovered, gt_tm, naive, save_fn)
    elif args.run_id:
        # Run optimization on the GT processed image
        from replicate import run_processing

        _ = wandb.login()
        api = wandb.Api()
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{args.run_id}")
        conf = dict(run.config)
        del run, api

        conf.pop("batch_size", None)
        conf.pop("sub_batch_size", None)
        conf.setdefault("log_samples", 10)
        conf.setdefault("gmse_weight", 0.0)

        config = ExperimentArgs(**conf)
        w0 = default_params(config)

        # Create single-image dataset from the processed GT image
        dataset = dataset_from_files([args.gt_proc_path], model)
        batch = next(iter(dataset))
        images, batch_masks, meta = batch

        res = run_processing(
            images, batch_masks, meta,
            hyperparams=config,
            w0=w0,
            value_ranges=value_ranges,
            save_dir=output_dir,
        )

        if res is not None:
            _, recovered_txm, _, _ = res
            recovered = np.array(recovered_txm[0].squeeze())
            naive = (1 - np.array(gt_proc)) / 2.0

            plot_gt_comparison(recovered, np.array(gt_tm), masks, priors, save_fn)
            plot_gt_metrics(jnp.array(recovered), gt_tm, jnp.array(naive), save_fn)

    print(f"Figures saved to {output_dir}")
