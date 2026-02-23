import argparse
import functools
import os
import pathlib
from typing import Any, Tuple

import cv2
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import torch
from chest_xray_sim.data.meta import XRayImageMeta
from chest_xray_sim.data.utils import read_image
import chest_xray_sim.utils.projections as projections
from chest_xray_sim.inverse.eval import batch_evaluation
from jaxtyping import Array, Float, Scalar

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert import ChexpertMeta
from chest_xray_sim.types import (
    ForwardT,
    SegmentationT,
    TransmissionMapT,
    ValueRangeT,
    WeightsT,
)

BIT_DTYPES = {8: np.uint8, 16: np.uint16}


def dataset_from_files(file_paths, model):
    images = torch.stack([read_image(fp) for fp in file_paths]).squeeze(0)
    masks = torch.stack([model(im) for im in images]).squeeze(0)
    print(images.shape, masks.shape)
    metas = [
        ChexpertMeta(
            abs_img_path=fp,
            deid_patient_id=pathlib.Path(fp).stem,
            frontal_lateral="frontal",
        )
        for fp in file_paths
    ]  # Placeholder for metadata

    yield images, masks, metas


def projection_with_spec(spec):
    def projection_wrapper(
        txm_state: TransmissionMapT,
        weights_state: WeightsT,
    ) -> Tuple[TransmissionMapT, WeightsT]:
        """
        Project transmission map values based on segmentation information.
        Uses softer constraints with confidence-weighted projections.

        Args:
            txm_state: Transmission map state
            weights_state: Weights state - parameters of the forward model
        """

        # General constraints
        new_txm_state = optax.projections.projection_hypercube(txm_state)
        new_weights_state = optax.projections.projection_non_negative(weights_state)

        # Apply constraints on image processing parameters
        new_weights_state = projections.projection_spec(new_weights_state, spec)

        return new_txm_state, new_weights_state

    return projection_wrapper


def chexpert_filenames(meta_batch: list[XRayImageMeta]):
    return [
        f"{m['img_id']}_{os.path.split(os.path.split(m['local_img_path'])[0])[1]}_{os.path.basename(m['local_img_path'])}"
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
    cv2.imwrite(path, np.array(x * max_val, dtype=np.uint8 if bits == 8 else np.uint16))


def process_results(
    images: ForwardT,
    segmentations: SegmentationT,
    meta_batch: list[XRayImageMeta],
    value_ranges: ValueRangeT,
    results,
    save_dir=None,
    label=None,
):
    txm, weights, pred = results

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        filenames = chexpert_filenames(meta_batch)

        meta_batch_ext = [{**m, "filepath": f} for m, f in zip(meta_batch, filenames)]

        joblib.dump(weights, os.path.join(save_dir, "weights.joblib"))
        joblib.dump(meta_batch_ext, os.path.join(save_dir, "meta.joblib"))

        save_results(save_dir, txm, weights, pred, filenames)

    try:
        metrics = batch_evaluation(images, txm, pred, segmentations, value_ranges)
        prefix = f"{label}_" if label else ""

        wandb.log(
            {
                **wandb_vec_metric(f"{prefix}max_violations", metrics.max_violations),
                **wandb_vec_metric(f"{prefix}min_violations", metrics.min_violations),
                **wandb_vec_metric(f"{prefix}compliance", metrics.bound_compliance),
                **wandb_vec_metric(f"{prefix}psnr", metrics.psnr),
                **wandb_vec_metric(f"{prefix}ssim", metrics.ssim),
                **wandb_vec_metric(f"{prefix}ms_ssim_l1", metrics.ms_ssim),
                **wandb_vec_metric(f"{prefix}penalties", metrics.penalties),
                **wandb_vec_metric(f"{prefix}mse", metrics.mse),
                **wandb_vec_metric(f"{prefix}band_similarity", metrics.band_similarity),
                **wandb_vec_metric(f"{prefix}tikhonov", metrics.tikonov),
                **wandb_vec_metric(f"{prefix}total_variation", metrics.total_variation),
            }
        )

    except Exception as e:
        raise e


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


def dmp_metric(fn, a, b):
    return fn(jnp.expand_dims(a, axis=2), jnp.expand_dims(b, axis=2)).item()


def experiment_args(**arguments):
    parser = argparse.ArgumentParser()
    for arg, default in arguments.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)

    def parse_args():
        return parser.parse_args()

    return parse_args


def pull_image(img, bits=8, logger=None):
    rng = 2**bits - 1
    dtype = BIT_DTYPES[bits]
    return wandb.Image(np.array(img * rng, dtype=dtype))


def log_image(label: str, img, bits=8, logger=None):
    """
    Log an image to wandb with a specified label and bit depth.
    Assumes the image is in the range [0, 1] and scales it to the specified bit depth.
    """
    if logger is None:
        logger = wandb.log
    rng = 2**bits - 1
    dtype = BIT_DTYPES[bits]
    logger({label: wandb.Image(np.array(img * rng, dtype=dtype))})


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
    for key, value in prefixed_metrics.items():
        run.summary[key] = value


def dict_merge(*dicts: dict[Any, Any]):
    return functools.reduce(lambda a, b: {**a, **b}, dicts)
