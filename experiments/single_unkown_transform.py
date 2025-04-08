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
    tm_path=os.environ.get('INPUT_PATH'),
    target_path=os.environ.get('TARGET_PATH'),
    save_dir=os.environ.get("OUTPUT_DIR"),
)


def forward(image, weights):
    # jax.debug.print('weights= {w}', w=weights)
    x = ops.negative_log(image)
    x = ops.windowing(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)
    x = ops.unsharp_masking(x, weights["low_sigma"], weights["low_enhance_factor"])
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
        weights_state["low_sigma"], 1.0, 10.0
    )
    new_weights_state["low_enhance_factor"] = optax.projections.projection_box(
        new_weights_state["low_enhance_factor"], 0.1, 1.0
    )

    return new_txm_state, new_weights_state


def run_processing(
    target: Float[Array, "batch rows cols"],
    run_init={},
    save_dir: str | None = None,
    image_labels = None
):
    run = wandb.init(**run_init)
    hyperparams = run.config
    batch_shape = target.shape

    key = jax.random.PRNGKey(hyperparams["PRNGKey"])
    txm0 = utils.get_random(key, batch_shape, distribution=hyperparams["tm_init_range"])
    # jax.debug.print('TXM0 mean= {m} ', m=txm0.mean())

    w0 = {
        "window_center": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "window_width": jax.random.uniform(key, minval=0.1, maxval=0.9),
        "gamma": jax.random.uniform(key, minval=0.1, maxval=3.0),
        "low_sigma": jax.random.uniform(key, minval=1., maxval=10.0),
        "low_enhance_factor": jax.random.uniform(key, minval=0.1, maxval=1.0),
    }

    def loss_fn(*args):
        return loss(*args, tv_factor=hyperparams["total_variation"])

    utils.base_wandb_batch_optim(
        target,
        txm0,
        w0,
        forward_fn=forward,
        loss_fn=loss_fn,
        run=run,
        projection=projection,
        save_dir=save_dir,
        image_labels=image_labels
    )


if __name__ == "__main__":
    from chest_xray_sim.data.loader import load_chexpert

    _ = wandb.login()

    PROJECT = "single-unknown-unsharp-mask"
    SWEEP_NAME = "fixed-unsharp-operator"
    FWD_DESC = (
        "negative log, windowing, range normalization, unsharp masking, clipping"
    )

    TARGET_SIZE = (512, 450)
    SWEEP_COUNT = 100

    TAGS = [f"dims={TARGET_SIZE}", *[f.strip() for f in FWD_DESC.split(',')], 'single', 'unknown']

    args = parser()

    print("using args:", args)

    # input data
    images = [
        cv2.imread(args.target_path, cv2.IMREAD_UNCHANGED) / 255.0
    ]
    image_labels = ['recovered.tif']

    target = np.stack(utils.preprocess_chexpert_batch(images, target_size=TARGET_SIZE))

    run_init = dict(
        project=PROJECT,
        notes=f"transformation: {FWD_DESC}",
        tags=TAGS
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
        function=lambda: run_processing(
            target, run_init=run_init, image_labels=image_labels, save_dir=args.save_dir
        ),
        count=SWEEP_COUNT,
    )
