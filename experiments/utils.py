import argparse
import functools
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from cv2.typing import MatLike
from jaxtyping import Float, Array
from skimage.transform import resize

import wandb
from chest_xray_sim.inverse.core import base_optimize

BIT_DTYPES = {8: np.uint8, 16: np.uint16}


def center_crop_with_aspect_ratio(image, target_size=(512, 512)):
    """
    Center crops an image to maintain the given aspect ratio and size.

    Args:
        image: Input NumPy array with shape (height, width) or (height, width, channels)
        target_size: Tuple of (height, width) for the target size

    Returns:
        Center-cropped image with dimensions matching target_size
    """
    # Get current dimensions
    if image.ndim == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = None

    target_h, target_w = target_size
    target_aspect = target_w / target_h

    # Calculate current aspect ratio
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Image is wider than target aspect ratio
        # Need to crop width
        new_w = int(h * target_aspect)
        offset_w = (w - new_w) // 2
        if c is not None:
            cropped = image[:, offset_w : offset_w + new_w, :]
        else:
            cropped = image[:, offset_w : offset_w + new_w]
    else:
        # Image is taller than target aspect ratio
        # Need to crop height
        new_h = int(w / target_aspect)
        offset_h = (h - new_h) // 2
        if c is not None:
            cropped = image[offset_h : offset_h + new_h, :, :]
        else:
            cropped = image[offset_h : offset_h + new_h, :]

    # Resize to target dimensions using scikit-image
    if c is not None:
        resized = resize(
            cropped, target_size + (c,), preserve_range=True, anti_aliasing=True
        )
    else:
        resized = resize(cropped, target_size, preserve_range=True, anti_aliasing=True)

    return resized.astype(image.dtype)


def preprocess_chexpert_batch(
    images, target_size=(512, 512)
) -> Float[Array, "batch rows cols"]:
    """
    Preprocess a batch of CheXpert images with NumPy.

    Args:
        images: Batch of images with shape (batch_size, height, width) or (batch_size, height, width, channels)
        target_size: Target size (height, width) for all images

    Returns:
        Batch of preprocessed images with consistent dimensions
    """
    return np.array([center_crop_with_aspect_ratio(img, target_size) for img in images])


def cap_resize(img: MatLike, max_dim: int) -> MatLike:
    og_rows, og_cols = int(img.shape[0]), int(img.shape[1])
    og_ratio = min(og_rows, og_cols) / max(og_rows, og_cols)

    low_dim = int(og_ratio * max_dim)
    # reverse order (cols, rows) to feed cv2.resize
    new_shape = (low_dim, max_dim) if og_rows > og_cols else (max_dim, low_dim)

    return cv2.resize(img, new_shape)


def dmp_metric(fn, a, b):
    return fn(jnp.expand_dims(a, axis=2), jnp.expand_dims(b, axis=2)).item()


def experiment_args(**arguments):
    parser = argparse.ArgumentParser()
    for arg, default in arguments.items():
        parser.add_argument(f"--{arg}", type=str, default=default)

    def parse_args():
        return parser.parse_args()

    return parse_args


def log_image(label, img, bits=8):
    rng = 2**bits - 1
    dtype = BIT_DTYPES[bits]
    wandb.log({label: wandb.Image(np.array(img * rng, dtype=dtype))})


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
    for key, value in prefixed_metrics.items():
        run.summary[key] = value


def random_initialization(key, **params):
    state = {
        k: jax.random.uniform(
            key, minval=v["min"], maxval=v["min"], shape=v.get("shape", ())
        )
        for k, v in params.items()
    }

    return state


def dict_merge(*dicts: dict[Any, Any]):
    return functools.reduce(lambda a, b: {**a, **b}, dicts)


def build_optim_runner(w0_builder, operator, loss_fn, **opt_params_high):
    def run_optimization(target, hyperparams, **opt_params):
        w0 = w0_builder(hyperparams)

        def_opt_params = dict(
            target=target,
            w0=w0,
            lr=hyperparams["lr"],
            n_steps=hyperparams["n_steps"],
            forward_fn=operator,
            loss_fn=loss_fn,
        )

        operator_params = operator.keys
        missing_params = set(operator_params) - set(w0.keys())
        assert len(missing_params) == 0, (
            "Initial weights missing for operator params: " + ", ".join(missing_params)
        )

        opt_params = dict_merge(def_opt_params, opt_params_high, opt_params)

        print(
            "Optimization params: ",
            {k: v for k, v in opt_params.items() if k not in ["target", "w0"]},
        )
        print("initial weights:", {k: v for k, v in w0.items() if k not in ["image"]})
        print("image stats:", w0["image"].mean())
        print("targetstats:", target.mean())
        return base_optimize(**opt_params)

    return run_optimization


def get_sweep_step(
    true_raw,
    target,
    weight_init,
    forward_op_builder,
    loss_fn_builder,
    end_cb=None,
    run_config={},
    **opt_params,
):
    def optimize():
        with wandb.init(**run_config) as run:
            forward_op = forward_op_builder(run.config)
            loss_fn = loss_fn_builder(run.config)

            # run.tags.append(forward_op.desc)

            run_optimization = build_optim_runner(
                weight_init,
                forward_op,
                loss_fn,
                **opt_params,
            )

            recovered, _ = run_optimization(target, run.config)
            # recovere

            if end_cb:
                end_cb(true_raw, target, recovered, forward=forward_op, loss_fn=loss_fn)

    return optimize
