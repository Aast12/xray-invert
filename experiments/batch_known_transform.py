import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float
import utils
from utils import (
    BIT_DTYPES,
    experiment_args,
    save_image,
    basic_loss_logger as loss_logger,
)

import chest_xray_sim.inverse.metrics as metrics
import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.loader import ChexpertMeta
from chest_xray_sim.inverse.core import base_optimize

parser = experiment_args(
    root_dir=os.environ.get("CHEXPERT_ROOT"),
    meta_path=os.environ.get("META_PATH"),
    img_dir=os.environ.get("IMAGE_PATH"),
    save_dir=os.environ.get("OUTPUT_DIR"),
)


def forward(image, weights):
    x = ops.negative_log(image)
    x = ops.unsharp_masking(x, weights["low_sigma"], weights["low_enhance_factor"])
    x = ops.unsharp_masking(x, weights["high_sigma"], weights["high_enhance_factor"])
    x = ops.windowing(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)
    x = ops.clipping(x)

    return x


def loss(txm, weights, pred, target, tv_factor=0.1):
    mse = metrics.mse(pred, target)
    tv = metrics.total_variation(txm)

    return mse + tv_factor * tv


def projection(txm_state, weights_state):
    new_txm_state = optax.projections.projection_hypercube(txm_state)
    new_weights_state = optax.projections.projection_non_negative(weights_state)
    new_weights_state["low_sigma"] = optax.projections.projection_box(
        weights_state["low_sigma"], 0.1, 10
    )
    new_weights_state["high_sigma"] = optax.projections.projection_box(
        weights_state["high_sigma"], 0.1, 10
    )
    new_weights_state["low_enhance_factor"] = optax.projections.projection_box(
        new_weights_state["low_enhance_factor"], 0.1, 1.0
    )
    new_weights_state["high_enhance_factor"] = optax.projections.projection_box(
        new_weights_state["high_enhance_factor"], 0.0, 1.0
    )

    return new_txm_state, new_weights_state


def optimize_unknown_processing(
    target: Float[Array, "batch rows cols"],
    run_init={},
    save_dir=None,
    target_meta=None,
):
    run = wandb.init(**run_init)
    hyperparams = run.config

    batch_shape = target.shape

    key = jax.random.PRNGKey(hyperparams["PRNGKey"])
    txm0 = None

    txm_min, txm_max = hyperparams["tm_init_range"]
    if hyperparams["tm_distribution"] == "normal":
        txm0 = jax.random.normal(key, shape=batch_shape)
        ratio = (txm0 - txm0.min(axis=0)) / (txm0.max(axis=0) - txm0.min(axis=0))
        txm0 = ratio * (txm_max - txm_min) + txm_min
    else:
        txm0 = jax.random.uniform(
            key, minval=txm_min, maxval=txm_max, shape=batch_shape
        )

    w0 = {
        "window_center": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "window_width": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "gamma": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "low_sigma": jax.random.uniform(key, minval=0.1, maxval=min(target.shape) / 2),
        "low_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "high_sigma": jax.random.uniform(key, minval=0.1, maxval=min(target.shape)),
        "high_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=3.0),
    }

    def loss_fn(*args):
        return loss(*args, tv_factor=hyperparams["total_variation"])

    state, _ = base_optimize(
        target,
        txm0,
        w0,
        loss_fn=utils.get_batch_mean_loss(loss_fn),
        forward_fn=utils.get_batch_fwd(forward),
        project_fn=projection,
        eps=hyperparams["eps"],
        lr=hyperparams["lr"],
        loss_logger=loss_logger,
        n_steps=hyperparams["n_steps"],
    )

    if state is None:
        print("run failed")
        return

    (txm, weights) = state

    wandb.log({"recovered_params": weights})

    if save_dir is not None and target_meta is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        run_save_path = os.path.join(save_dir, run.id)
        os.mkdir(run_save_path)

        labels = [f"{d['patient_id']}_{d['study']}.tif" for d in target_meta]
        labels = [os.path.join(run_save_path, lb) for lb in labels]

        for img, label in zip(txm, labels):
            save_image(img, label)
            save_image(
                ops.range_normalize(forward(img, weights)),
                label.replace(".tif", "_proc.tif"),
            )


if __name__ == "__main__":
    from chest_xray_sim.data.loader import load_chexpert

    wandb.login()

    PROJECT = "batch-unsharp-mask"
    SWEEP_NAME = "unknown-transform-sweep-v2"
    FWD_DESC = (
        "negative log, 2-layer unsharp masking, windowing and range normalization"
    )

    TARGET_SIZE = (512, 450)
    SWEEP_COUNT = 100

    args = parser()

    print("using args:", args)

    # input data
    chexpert_data: list[ChexpertMeta] = load_chexpert(
        args.root_dir, args.meta_path, args.img_dir, limit=20
    )
    images = [d["image"] for d in chexpert_data]
    image_meta = [{k: v for k, v in d.items() if k != "image"} for d in chexpert_data]

    target = np.stack(utils.preprocess_chexpert_batch(images, target_size=TARGET_SIZE))
    run_init = dict(
        project=PROJECT,
        notes=f"transformation: {FWD_DESC}",
        tags=[f"dims={TARGET_SIZE}"],
    )
    sweep_config = {
        "name": SWEEP_NAME,
        "method": "bayes",
        "metric": {"name": "mse", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 1e-5, "max": 1e-1},
            "n_steps": {"values": [500, 700, 1200]},
            "total_variation": {"values": [0.0, 0.1, 0.01, 0.001]},
            "PRNGKey": {"values": [0, 42]},
            "tm_distribution": {"values": ["uniform", "normal"]},
            "tm_init_range": {
                "values": [
                    (1e-6, 0.1),
                    (1e-6, 0.5),
                    (1e-6, 1.0),
                    (0.5, 1.0),
                ]
            },
            # Fixed/metadata parameters
            "loss": {"value": "MSE + TV"},
            "initialization": {"value": "uniform random"},
            "window_center_init_range": {"value": [0.1, 0.9]},
            "window_width_init_range": {"value": [0.1, 0.9]},
            "gamma_init_range": {"value": [0.1, 3.0]},
            "eps": {"value": 1e-6},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=PROJECT,
    )
    wandb.agent(
        sweep_id,
        function=lambda: optimize_unknown_processing(
            target, run_init=run_init, target_meta=image_meta, save_dir=args.save_dir
        ),
        count=SWEEP_COUNT,
    )
