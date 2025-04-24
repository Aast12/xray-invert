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
        weights_state["low_sigma"], 0.1, 10
    )
    new_weights_state["low_enhance_factor"] = optax.projections.projection_box(
        new_weights_state["low_enhance_factor"], 0.1, 1.0
    )

    return new_txm_state, new_weights_state


def run_processing(
    target: Float[Array, "batch rows cols"],
    run_init={},
    save_dir: str | None = None,
    target_meta: list[ChexpertMeta] | None = None,
):
    run = wandb.init(**run_init)

    run.config["tm_range"] = ",".join(map(str, run.config["tm_init_range"]))
    hyperparams = run.config
    batch_shape = target.shape

    key = jax.random.PRNGKey(hyperparams["PRNGKey"])

    if hyperparams["tm_init_range"] == "target":
        txm0 = target.copy()
    else:
        txm0 = utils.get_random(
            key, batch_shape, distribution=hyperparams["tm_init_range"]
        )

    w0 = {
        "low_sigma": 10.0,
        "low_enhance_factor": 0.75,
        "high_sigma": 0.1,
        "high_enhance_factor": 0.0,
        "window_center": 0.5,
        "window_width": 0.8,
        "gamma": 1.2,
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
        constant_weights=True,
        image_labels=[f"{d['patient_id']}_{d['study']}.tif" for d in target_meta]
        if target_meta
        else None,
    )


if __name__ == "__main__":
    from chest_xray_sim.data.loader import load_chexpert

    _ = wandb.login()

    PROJECT = "batch-known-transform"
    SWEEP_NAME = "fixed-unsharp"
    FWD_DESC = "negative log, windowing, range normalization, unsharp masking, clipping"

    TARGET_SIZE = (512, 450)
    SWEEP_COUNT = 100
    BATCH_LIMIT = None

    TAGS = [
        f"dims={TARGET_SIZE}",
        *[f.strip() for f in FWD_DESC.split(",")],
        "batch",
        "known",
        f"limit={BATCH_LIMIT}",
    ]

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
        tags=TAGS,
    )
    sweep_config = {
        "name": SWEEP_NAME,
        "method": "bayes",
        "metric": {"name": "mse", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 2e-2, "max": 1e-1},
            "n_steps": {"values": [300, 600, 1200]},
            "total_variation": {"min": 0.1, "max": 2.0},
            "PRNGKey": {"values": [0, 42]},
            "tm_distribution": {"values": ["target", "uniform", "normal"]},
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
            target, run_init=run_init, target_meta=image_meta, save_dir=args.save_dir
        ),
        count=SWEEP_COUNT,
    )
