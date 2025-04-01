import jax
import jax.numpy as jnp
import cv2
from chest_xray_sim.inverse.operators import (
    negative_log,
    unsharp_masking,
    windowing,
    range_normalize,
)
import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.inverse.core import base_optimize
from chest_xray_sim.utils.results import run_metrics
import utils
import itertools
import argparse
import dm_pix as dmp

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "--raw", type=str, default="data/conventional_transmissionmap.tif"
)
_ = parser.add_argument("--target", type=str, default="data/conventional_processed.tif")


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {f"{prefix}_{key}": value for key, value in metrics.items()}
    for key, value in prefixed_metrics.items():
        run.summary[key] = value


def run_optimization(
    target, hyperparams, operator=None, loss_fn=None, w0=None, **opt_params
):
    assert operator is not None
    assert loss_fn is not None
    assert w0 is not None

    def_opt_params = dict(
        target=target,
        w0=w0,
        lr=hyperparams["lr"],
        loss_fn=loss_fn,
        n_steps=hyperparams["n_steps"],
        forward_fn=operator,
    )

    return base_optimize(**{**def_opt_params, **opt_params})


def optimize_unknown_processing(true_raw, target, operator, run_init={}):
    run = wandb.init(**run_init)

    def dmp_metric(fn, a, b):
        return fn(jnp.expand_dims(a, axis=2), jnp.expand_dims(b, axis=2)).item()

    def loss_logger(loss, params, pred, target):
        mse_tm = dmp_metric(dmp.mse, params["image"], true_raw)
        ssim_tm = dmp_metric(dmp.ssim, params["image"], true_raw)
        psnr_tm = dmp_metric(dmp.psnr, params["image"], true_raw)
        mse_fwd = dmp_metric(dmp.mse, pred, target)
        ssim_fwd = dmp_metric(dmp.ssim, pred, target)
        psnr_fwd = dmp_metric(dmp.psnr, pred, target)

        run.log(
            dict(
                loss=loss,
                mse_tm=mse_tm,
                ssim_tm=ssim_tm,
                psnr_tm=psnr_tm,
                mse_fwd=mse_fwd,
                ssim_fwd=ssim_fwd,
                psnr_fwd=psnr_fwd,
            )
        )

    # loss_logger = lambda loss, params, pred, target: run.log({"loss": loss})

    run.use_artifact("transmission-map-raw:latest")
    run.use_artifact("transmission-map-target:latest")

    key = jax.random.PRNGKey(run.config["PRNGKey"])
    w0 = {
        "image": jax.random.uniform(key, minval=1e-6, maxval=0.1, shape=target.shape),
        "window_center": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "window_width": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "low_sigma": jax.random.uniform(key, minval=0.1, maxval=min(target.shape) / 2),
        "low_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "high_sigma": jax.random.uniform(key, minval=0.1, maxval=min(target.shape)),
        "high_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "gamma": jax.random.uniform(key, minval=0.1, maxval=3.0),
    }
    loss_fn = ops.build_loss(
        ops.mse, ops.total_variation(run.config["total_variation"])
    )

    recovered, loss = run_optimization(
        target,
        run.config,
        operator=operator,
        loss_fn=loss_fn,
        loss_logger=loss_logger,
        w0=w0,
    )

    wandb.log(
        {
            "recovered_params": {
                key: value for key, value in recovered.items() if key != "image"
            }
        }
    )

    normalize = ops.build_forward_fn(range_normalize)
    rec_range_normal = normalize(recovered)
    utils.log_image("recovered", rec_range_normal)

    rec_processed = operator(recovered)
    utils.log_image("recovered_processed", normalize({"image": rec_processed}))

    range_normal_metrics = run_metrics(true_raw, target, rec_range_normal)
    log_run_metrics(
        run,
        range_normal_metrics,
        "summ",
    )

    run.log(
        {
            "mse": range_normal_metrics["mse_opt"],
            "mse_diff": range_normal_metrics["mse_opt"]
            - range_normal_metrics["mse_naive"],
        }
    )


def setup_artifacts(raw_path: str, target_path: str, project=None):
    with wandb.init(project=project, job_type="data-preparation") as run:
        # Create artifacts for input images
        raw_artifact = wandb.Artifact("transmission-map-raw", type="dataset")
        _ = raw_artifact.add_file(raw_path)
        run.log_artifact(raw_artifact)

        target_artifact = wandb.Artifact("transmission-map-target", type="dataset")
        _ = target_artifact.add_file(target_path)
        un.log_artifact(target_artifact)

        return raw_artifact.name, target_artifact.name


if __name__ == "__main__":
    project = "unsharp-mask-unknown-processing"

    args = parser.parse_args()
    true_raw = cv2.imread(args.raw, cv2.IMREAD_UNCHANGED)
    true_raw = true_raw / 255.0

    target = cv2.imread(args.target, cv2.IMREAD_UNCHANGED)
    target = target / 255.0

    MAX_DIM = 512

    target = utils.cap_resize(target, MAX_DIM)
    true_raw = utils.cap_resize(true_raw, MAX_DIM)

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
    operator = ops.build_forward_fn(
        negative_log,
        unsharp_masking("low_"),
        unsharp_masking("high_"),
        windowing,
        range_normalize,
    )
    raw_artifact, target_artifact = setup_artifacts(
        args.raw, args.target, project=project
    )

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )
    wandb.agent(
        sweep_id,
        function=lambda: optimize_unknown_processing(
            true_raw, target, operator=operator, run_init=run_init
        ),
        count=10,
    )
