import jax
import jax.numpy as jnp
import cv2
import numpy as np
from chest_xray_sim.inverse.operators import (
    mse,
    total_variation,
    build_loss,
    build_forward_fn,
    negative_log,
    windowing,
    range_normalize,
)
import wandb
from chest_xray_sim.inverse.core import base_optimize
from chest_xray_sim.utils.results import run_metrics
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=str, default="data/conventional_transmissionmap.tif")


def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {f"{prefix}_{key}": value for key, value in metrics.items()}
    for key, value in prefixed_metrics.items():
        run.summary[key] = value


true_raw = None


def run_optimization(target, hyperparams, operator=None, loss_fn=None, **opt_params):

    if operator is None:
        operator = build_forward_fn(negative_log, windowing, range_normalize)
    if loss_fn is None:
        loss_fn = build_loss(mse, total_variation(hyperparams["total_variation"]))

    key = jax.random.PRNGKey(hyperparams["PRNGKey"])

    def_opt_params = dict(
        target=target,
        w0={
            "image": jax.random.uniform(
                key, minval=1e-6, maxval=0.1, shape=target.shape
            ),
            "window_center": jax.random.uniform(key, minval=0.1, maxval=0.9),
            "window_width": jax.random.uniform(key, minval=0.1, maxval=0.9),
            "gamma": jax.random.uniform(key, minval=0.1, maxval=3.0),
        },
        lr=hyperparams["lr"],
        loss_fn=loss_fn,
        n_steps=hyperparams["n_steps"],
        forward_fn=operator,
    )

    return base_optimize(**{**def_opt_params, **opt_params})


def build_known_processing(raw, params, operator=None):
    if operator is None:
        operator = build_forward_fn(negative_log, windowing, range_normalize)

    target = operator({**params, "image": raw})
    return target


def optimize_known_processing():
    global true_raw

    run = wandb.init(
        project="simple-transform",
        notes="transformation: negative log, windowing and range normalization",
    )

    loss_logger = lambda val: run.log({"loss": val})

    operator = build_forward_fn(negative_log, windowing, range_normalize)

    target = build_known_processing(true_raw, run.config, operator=operator)
    recovered, loss = run_optimization(
        target, run.config, operator=operator, loss_logger=loss_logger
    )

    rec = recovered["image"]
    rec_range_normal = (rec - rec.min()) / (rec.max() - rec.min())
    out = wandb.Image(np.array(rec_range_normal * 255, dtype=np.uint8))
    wandb.log({"recovered": out})

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


if __name__ == "__main__":
    args = parser.parse_args()
    true_raw = cv2.imread(args.raw, cv2.IMREAD_UNCHANGED)
    true_raw = true_raw / 255.0

    wandb.login()

    sweep_config = {
        "name": "simple-transform-sweep-v2",
        "method": "bayes",
        "metric": {"name": "mse", "goal": "minimize"},
        "parameters": {
            "PRNGKey": {"values": [0]},
            "window_center": {"min": 0.1, "max": 0.9},
            "window_width": {"min": 0.1, "max": 0.9},
            "gamma": {"min": 0.1, "max": 3.0},
            "lr": {"min": 1e-5, "max": 1e-1},
            "n_steps": {"values": [500, 1000, 2000]},
            "total_variation": {"values": [0.0, 0.1, 0.01, 0.001]},
            # Fixed/metadata parameters
            "loss": {"value": "MSE + TV"},
            "initialization": {"value": "uniform random"},
            "image_init_range": {"value": [1e-6, 0.1]},
            "window_center_init_range": {"value": [0.1, 0.9]},
            "window_width_init_range": {"value": [0.1, 0.9]},
            "gamma_init_range": {"value": [0.1, 3.0]},
            "eps": {"value": 1e-8},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config, project="simple-transform-known-processing"
    )
    wandb.agent(sweep_id, function=optimize_known_processing, count=200)
