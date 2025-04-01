import jax
import jax.numpy as jnp
import numpy as np
import cv2
from chest_xray_sim.inverse.operators import (
    negative_log,
    unsharp_masking,
    windowing,
    range_normalize,
)
import chest_xray_sim.inverse.operators as ops
import chest_xray_sim.inverse.metrics as metrics
import wandb
from chest_xray_sim.inverse.core import base_optimize
from chest_xray_sim.utils.results import run_metrics
import utils
import itertools
import argparse
import dm_pix as dmp
from utils import experiment_args

# parser = argparse.ArgumentParser()
# _ = parser.add_argument(
#     "--raw", type=str, default="data/conventional_transmissionmap.tif"
# )
# _ = parser.add_argument("--target", type=str, default="data/conventional_processed.tif")
#
parser = experiment_args(root_dir=None, meta_path=None, img_dir=None)


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {f"{prefix}_{key}": value for key, value in metrics.items()}
    for key, value in prefixed_metrics.items():
        run.summary[key] = value


def forward(image, weights):
    x = ops.negative_log(image)
    x = ops.unsharp_masking(x, weights["low_sigma"], weights["low_enhance_factor"])
    x = ops.unsharp_masking(x, weights["high_sigma"], weights["high_enhance_factor"])
    x = ops.windowing(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)

    return x


def loss(txm, weights, pred, target, tv_factor=0.1):
    mse = metrics.mse(pred, target)
    tv = metrics.total_variation(txm)

    return mse + tv_factor * tv


def loss_logger(loss, *args):
    wandb.log(
        dict(
            loss=loss,
        )
    )


def optimize_unknown_processing(target, run_init={}):
    run = wandb.init(**run_init)
    hyperparams = run.config

    shape = target[0].shape
    batch_shape = target.shape

    key = jax.random.PRNGKey(hyperparams["PRNGKey"])
    txm0 = jax.random.normal(key, shape=batch_shape)
    w0 = {
        "window_center": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "window_width": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "low_sigma": jax.random.uniform(key, minval=0.1, maxval=min(target.shape) / 2),
        "low_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "high_sigma": jax.random.uniform(key, minval=0.1, maxval=min(target.shape)),
        "high_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "gamma": jax.random.uniform(key, minval=0.1, maxval=3.0),
    }

    def unbatched_loss(*args):
        return loss(*args, tv_factor=hyperparams["total_variation"])

    def loss_fn(*args):
        batched_loss = jax.vmap(unbatched_loss, in_axes=(0, None, 0, 0))
        loss_val = batched_loss(*args)

        return jnp.mean(loss_val)

    batched_forward = jax.vmap(forward, in_axes=(0, None))

    state, losses = base_optimize(
        target,
        txm0,
        w0,
        loss_fn=loss_fn,
        forward_fn=batched_forward,
        eps=hyperparams["eps"],
        lr=hyperparams["lr"],
        loss_logger=loss_logger,
        n_steps=hyperparams["n_steps"],
    )
    (txm, weights) = state

    wandb.log({"recovered_params": weights})
    # vnormalize = jax.vmap(ops.range_normalize, in_axes=(0,))


if __name__ == "__main__":
    from chest_xray_sim.data.loader import load_chexpert

    args = parser()

    chexpert_data = load_chexpert(args.root_dir, args.meta_path, args.img_dir, limit=30)
    images = np.stack([d["image"] for d in chexpert_data])

    target = utils.preprocess_chexpert_batch(images, target_size=(512, 450))

    project = "batch-unsharp-mask"

    wandb.login()

    run_init = dict(
        project=project,
        notes="transformation: negative log, 2-layer unsharp masking, windowing and range normalization",
        tags=[f"max_dim={512}"],
    )
    sweep_config = {
        "name": "unknown-transform-sweep-v2",
        "method": "bayes",
        "metric": {"name": "mse", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 1e-5, "max": 1e-1},
            "n_steps": {"values": [500, 700, 1200]},
            "total_variation": {"values": [0.0, 0.1, 0.01, 0.001]},
            "PRNGKey": {"values": [0, 42]},
            # Fixed/metadata parameters
            "loss": {"value": "MSE + TV"},
            "initialization": {"value": "uniform random"},
            "image_init_range": {"value": [1e-6, 0.1]},
            "window_center_init_range": {"value": [0.1, 0.9]},
            "window_width_init_range": {"value": [0.1, 0.9]},
            "gamma_init_range": {"value": [0.1, 3.0]},
            "eps": {"value": 1e-6},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )
    wandb.agent(
        sweep_id,
        function=lambda: optimize_unknown_processing(target, run_init=run_init),
        count=10,
    )
