import argparse
import functools
import os
from typing import Any, Literal

import cv2
import initialization as init
import jax
import jax.numpy as jnp
import joblib
import numpy as np
from cv2.typing import MatLike
from eval import batch_evaluation
from jaxtyping import Array, Float, Scalar
from logging_utils import log_priors_table, summary
from skimage.transform import resize

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import batch_get_exclusive_masks
from chest_xray_sim.inverse.core import base_optimize
from chest_xray_sim.types import (
    ForwardT,
    SegmentationT,
    TransmissionMapT,
    ValueRangeT,
    WeightsT,
)

BIT_DTYPES = {8: np.uint8, 16: np.uint16}


def chexpert_filenames(meta_batch):
    return [
        f"{m['deid_patient_id']}_{os.path.basename(m['abs_img_path'])}"
        for m in meta_batch
    ]


def save_results(
    save_dir: str,
    txm: TransmissionMapT,
    weights: WeightsT,
    pred: TransmissionMapT,
    file_names: list[str],
):
    joblib.dump(weights, os.path.join(save_dir, "weights.joblib"))

    # Save transmission maps and their processed versions
    for i, (img, name) in enumerate(zip(txm, file_names)):
        save_path = os.path.join(save_dir, f"{name}")
        save_image(img, save_path)

        # Also save the processed version
        processed = pred[i]
        proc_path = save_path.replace(".png", "_proc.png")
        save_image(processed, proc_path)


def build_segmentation_model_inputs(
    images: Float[Array, "batch rows cols"],
    segmentations: Float[Array, "batch rows cols"],
    value_ranges: Float[Array, "labels 2"],
    hyperparams: Any,
    segmentation_th=0.6,
):
    seg_labels, segmentations = batch_get_exclusive_masks(
        segmentations, segmentation_th
    )
    log_priors_table(seg_labels, value_ranges)
    init_mode, init_params = init.parse_init_params(hyperparams, images)
    summary({"init_mode": init_mode})

    txm0 = init.initialize(hyperparams.PRNGKey, images.shape, **init_params)

    w0 = full_pytree(
        {
            "low_sigma": 4.0,
            "low_enhance_factor": 0.5,
            "window_center": 0.2,
            "window_width": 0.2,
            "gamma": 5,
        },
        images.shape[0],
        jnp.float32,
    )

    return txm0, w0


def full_pytree(w0, shape, dtype=jnp.float32):
    return {k: jnp.full(shape, v, dtype=dtype) for k, v in w0.items()}


def sample_random(size: int, log_samples: int | None = None):
    if log_samples is None:
        log_samples = size

    log_samples = min(log_samples, size)

    rand_samples = np.random.randint(0, size, size=log_samples).tolist()
    rand_samples = sorted(rand_samples)

    return rand_samples


def wandb_vec_metric(id: str, data: Array):
    return {
        f"{id}": wandb.Histogram(data),
        f"{id}_max": data.max(),
        f"{id}_min": data.min(),
        f"{id}_mean": data.mean(),
    }


def save_image(img, path: str, bits=8):
    """save normalized image to file"""
    x = ops.range_normalize(img)
    max_val = 2**bits - 1
    cv2.imwrite(
        path, np.array(x * max_val, dtype=np.uint8 if bits == 8 else np.uint16)
    )


def process_results(
    images: ForwardT,
    segmentations: SegmentationT,
    meta_batch: list[ChexpertMeta],
    value_ranges: ValueRangeT,
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

        save_results(
            save_dir, txm, weights, pred, chexpert_filenames(meta_batch)
        )

    try:
        metrics = batch_evaluation(
            images, txm, pred, segmentations, value_ranges
        )
        wandb.log(
            {
                **wandb_vec_metric("max_violations", metrics.max_violations),
                **wandb_vec_metric("min_violations", metrics.min_violations),
                **wandb_vec_metric("compliance", metrics.bound_compliance),
                **wandb_vec_metric("psnr", metrics.psnr),
                **wandb_vec_metric("ssim", metrics.ssim),
                **wandb_vec_metric("penalties", metrics.penalties),
            }
        )
    except Exception as e:
        raise e
        print("Error during batch evaluation:", e)


@jax.jit
def map_range(
    x: Float[Array, "*dims"],
    src_min: Scalar,
    src_max: Scalar,
    dst_min: Scalar,
    dst_max: Scalar,
):
    "Maps the range of x from [src_min, src_max] to [dst_min, dst_max]."
    return (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min


def get_random(
    key,
    shape,
    distribution: Literal["normal", "uniform"] = "uniform",
    val_range=(0.0, 1.0),
    axis: int | None = 0,
):
    arr = None
    minval, maxval = val_range
    if distribution == "normal":
        arr = jax.random.normal(key, shape=shape)
        ratio = (arr - arr.min(axis=axis)) / (
            arr.max(axis=axis) - arr.min(axis=axis)
        )
        arr = ratio * (maxval - minval) + minval
    else:
        arr = jax.random.uniform(key, minval=minval, maxval=maxval, shape=shape)

    return arr


def get_batch_mean_loss(unbatched_loss, in_axes=(0, None, 0, 0), **vmap_args):
    def loss_fn(*args):
        batched_loss = jax.vmap(unbatched_loss, in_axes=in_axes, **vmap_args)
        loss_val = batched_loss(*args)
        return jnp.mean(loss_val)

    return loss_fn


def get_batch_fwd(forward, in_axes=(0, None), **vmap_args):
    return jax.vmap(forward, in_axes=in_axes, **vmap_args)


def empty_loss_logger(loss, *args):
    pass


def basic_loss_logger(loss, *args):
    wandb.log(
        dict(
            loss=loss,
        )
    )


def base_wandb_optim_generic(
    optim_params,
    run_init={},
    save_dir: str | None = None,
    image_labels=None,
    run=None,
):
    if run is None:
        run = wandb.init(**run_init)

    hyperparams = run.config

    forward_fn = optim_params["forward_fn"]
    params = {
        **optim_params,
        **{
            "loss_fn": get_batch_mean_loss(optim_params["loss_fn"]),
            "forward_fn": get_batch_fwd(forward_fn),
            "eps": hyperparams["eps"],
            "n_steps": hyperparams["n_steps"],
        },
    }
    state, _ = base_optimize(
        **params,
    )

    if state is None:
        print("run failed")
        return

    (txm, weights) = state

    wandb.log({"recovered_params": weights})

    if save_dir is not None and image_labels is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        run_save_path = os.path.join(save_dir, run.id)
        os.mkdir(run_save_path)

        labels = [os.path.join(run_save_path, lb) for lb in image_labels]

        for img, label in zip(txm, labels):
            save_image(img, label)
            save_image(
                ops.range_normalize(forward_fn(img, weights)),
                label.replace(".tif", "_proc.tif"),
            )


def base_wandb_batch_optim(
    target: Float[Array, "batch rows cols"],
    txm0,
    w0,
    loss_fn,
    forward_fn,
    projection,
    run_init={},
    save_dir: str | None = None,
    image_labels=None,
    run=None,
    **optim_args,
):
    if run is None:
        run = wandb.init(**run_init)

    hyperparams = run.config

    state, _ = base_optimize(
        target,
        txm0,
        w0,
        loss_fn=get_batch_mean_loss(loss_fn),
        forward_fn=get_batch_fwd(forward_fn),
        project_fn=projection,
        eps=hyperparams["eps"],
        lr=hyperparams["lr"],
        loss_logger=basic_loss_logger,
        n_steps=hyperparams["n_steps"],
        **optim_args,
    )

    if state is None:
        print("run failed")
        return

    (txm, weights) = state

    wandb.log({"recovered_params": weights})

    if save_dir is not None and image_labels is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        run_save_path = os.path.join(save_dir, run.id)
        os.mkdir(run_save_path)

        labels = [os.path.join(run_save_path, lb) for lb in image_labels]

        for img, label in zip(txm, labels):
            save_image(img, label)
            save_image(
                ops.range_normalize(forward_fn(img, weights)),
                label.replace(".tif", "_proc.tif"),
            )


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
        resized = resize(
            cropped, target_size, preserve_range=True, anti_aliasing=True
        )

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
    return np.array(
        [center_crop_with_aspect_ratio(img, target_size) for img in images]
    )


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
        parser.add_argument(f"--{arg}", type=type(default), default=default)

    def parse_args():
        return parser.parse_args()

    return parse_args


def pull_image(img, bits=8, logger=wandb.log):
    rng = 2**bits - 1
    dtype = BIT_DTYPES[bits]
    return wandb.Image(np.array(img * rng, dtype=dtype))


def log_image(label: str, img, bits=8, logger=wandb.log):
    """
    Log an image to wandb with a specified label and bit depth.
    Assumes the image is in the range [0, 1] and scales it to the specified bit depth.
    """
    rng = 2**bits - 1
    dtype = BIT_DTYPES[bits]
    logger({label: wandb.Image(np.array(img * rng, dtype=dtype))})


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {
        f"{prefix}{key}": value for key, value in metrics.items()
    }
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
            "Initial weights missing for operator params: "
            + ", ".join(missing_params)
        )

        opt_params = dict_merge(def_opt_params, opt_params_high, opt_params)

        print(
            "Optimization params: ",
            {k: v for k, v in opt_params.items() if k not in ["target", "w0"]},
        )
        print(
            "initial weights:",
            {k: v for k, v in w0.items() if k not in ["image"]},
        )
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
                end_cb(
                    true_raw,
                    target,
                    recovered,
                    forward=forward_op,
                    loss_fn=loss_fn,
                )

    return optimize
