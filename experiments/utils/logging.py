import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from models import ExperimentInputs
from utils import pull_image, sample_random
import torch

import wandb
from chest_xray_sim.data.segmentation import MaskGroupsT

def is_array_like(obj):
    return isinstance(obj, (np.ndarray, jnp.ndarray, torch.Tensor)) and obj.ndim > 0 

def summary(body):
    try:
        for k, v in body.items():
            if is_array_like(v):
                wandb.log({k: wandb.Histogram(v)})
            else:
                wandb.summary[k] = v
    except wandb.Error:
        pass


def log(body):
    try:
        mapped = {
            k: v if not is_array_like(v) else wandb.Histogram(v)
            for k, v in body.items()
        }
        wandb.log(mapped)
    except wandb.Error:
        pass
    except Exception as e:
        raise e


def empty_logger(body):
    pass


def log_priors_table(
    seg_labels: list[MaskGroupsT],
    value_ranges: Float[Array, "labels 2"],
    run=wandb,
):
    priors_table = wandb.Table(columns=["region", "min", "max"])
    for region, value_range in zip(seg_labels, value_ranges):
        min_val, max_val = value_range
        priors_table.add_data(region, min_val, max_val)

    run.log({"priors": priors_table})


def log_image_histograms(inputs: ExperimentInputs, samples: int | None = None):
    images = inputs.images
    segmentations = inputs.segmentations
    seg_labels = inputs.prior_labels

    if samples is None:
        samples = images.shape[0]

    mask_labels = {i + 1: label for i, label in enumerate(seg_labels)}
    for idx in range(samples):
        curr_image = images[idx]
        curr_masks = segmentations[idx]

        image_histogram = wandb.Histogram(curr_image.flatten())
        log({"image histogram": image_histogram})
        for i, label in mask_labels.items():
            mask_idx = i - 1
            mask = curr_masks[mask_idx]
            log(
                {
                    f"image ({label}) histogram": wandb.Histogram(
                        curr_image[mask.astype(jnp.bool)].flatten()
                    )
                }
            )


def log_txm_histograms(inputs: ExperimentInputs, txm, pred, log_samples: int):
    images = inputs.images
    segmentations = inputs.segmentations
    seg_labels = inputs.prior_labels
    log_samples = min(log_samples, images.shape[0])
    rand_samples = sample_random(images.shape[0], log_samples)

    mask_labels = {i + 1: label for i, label in enumerate(seg_labels)}

    samples_tables = wandb.Table(columns=["index", "Image"])
    for idx in rand_samples:
        curr_image = images[idx]
        curr_masks = segmentations[idx]
        general_mask = jnp.zeros_like(curr_masks[idx])

        for i, label in mask_labels.items():
            mask_idx = i - 1
            mask = curr_masks[mask_idx]
            general_mask = general_mask + mask * i
        # Only log full images for random samples
        masked_image = wandb.Image(
            pull_image(curr_image),
            masks={
                "predictions": {
                    "mask_data": np.array(general_mask),
                    "class_labels": mask_labels,
                },
            },
        )
        row_data = [idx, masked_image]
        samples_tables.add_data(*row_data)

    for idx in range(images.shape[0]):
        curr_image = txm[idx]
        curr_masks = segmentations[idx]
        # Log example images and segmentations

        for idx, label in enumerate(seg_labels):
            mask = curr_masks[idx]
            log(
                {
                    f"transmission map ({label}) histogram": wandb.Histogram(
                        curr_image[mask.astype(jnp.bool)].flatten()
                    )
                }
            )

        image_histogram = wandb.Histogram(curr_image.flatten())
        log({"transmission map histogram": image_histogram})

    # Log example recovered images
    fwd_data = [pull_image(pred[i]) for i in rand_samples]
    recovered_data = [pull_image(txm[i]) for i in rand_samples]

    samples_tables.add_column("Recovered", recovered_data)
    samples_tables.add_column("Forward", fwd_data)

    log({"samples": samples_tables})
