import os
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import initialization as init
import jax.numpy as jnp
import joblib
import numpy as np
import optax
import projections as proj
import yaml
from eval import batch_evaluation
from jaxtyping import Array, Float, PyTree
from loss import (
    mse,
    segmentation_sq_penalty,
    total_variation,
    unsharp_mask_similarity,
)
from segmentation_utils import get_priors
from torch import Tensor
from utils import (
    basic_loss_logger,
    empty_loss_logger,
    experiment_args,
    process_results,
    pull_image,
    save_image,
)

import chest_xray_sim.inverse.operators as ops
import wandb
from chest_xray_sim.data.chexpert_dataset import ChexpertMeta
from chest_xray_sim.data.segmentation import (
    batch_get_exclusive_masks,
)
from chest_xray_sim.data.segmentation_dataset import (
    get_segmentation_dataset,
)
from chest_xray_sim.inverse.core import segmentation_optimize
from chest_xray_sim.types import TransmissionMapT

DEBUG = True
DTYPE = jnp.float32


@dataclass(frozen=True)
class ExperimentArgs:
    lr: float
    n_steps: int
    total_variation: float
    prior_weight: float
    gmse_weight: float
    PRNGKey: int
    tm_init_params: tuple[str, tuple[float, float] | None]
    constant_weights: bool
    eps: float
    max_sigma: float
    max_enhancement: float


class ForwardParams(TypedDict):
    low_sigma: float
    low_enhance_factor: float
    window_center: float
    window_width: float
    gamma: float


WeightsT = PyTree[ForwardParams]

args_spec = experiment_args(
    batch_size=32,
    frontal_lateral="Frontal",
    split="train",
    data_dir=os.environ.get("IMAGE_PATH"),
    meta_dir=os.environ.get("META_PATH"),
    mask_dir=os.environ.get("MASK_DIR"),
    cache_dir=os.environ.get("CACHE_DIR"),
    save_dir=os.environ.get("OUTPUT_DIR"),
    sweep_conf="",
)


def forward(
    image: TransmissionMapT, weights: WeightsT
) -> Float[Array, "*batch rows cols"]:
    """Forward processing function that converts transmission maps to processed X-rays"""
    x = ops.negative_log(image)
    x = ops.window(
        x, weights["window_center"], weights["window_width"], weights["gamma"]
    )
    x = ops.range_normalize(x)
    x = ops.unsharp_masking(
        x, weights["low_sigma"], weights["low_enhance_factor"]
    )
    x = ops.clipping(x)

    return x


def segmentation_loss(
    txm: TransmissionMapT,
    weights: WeightsT,
    pred: Float[Array, "batch height width"],
    target: Float[Array, "batch height width"],
    segmentation: Float[Array, "batch reduced_labels height width"],
    value_ranges: Float[Array, "reduced_labels 2"],
    tv_factor=0.1,
    prior_weight=0.5,
    gmse_weight=0.5,
):
    """
    Loss function that incorporates segmentation information using probabilistic priors.

    The loss consists of:
    1. MSE between prediction and target (data fidelity)
    2. Total variation regularization (spatial smoothness)
    3. Anatomical region range penalty

    args:
        txm: Transmission map
        weights: Forward model parameters
        pred: Predicted image
        target: Target image
        segmentation: Segmentation map
        tv_factor: Weight for total variation regularization
        prior_weight: Weight for anatomical priors
    """
    mse_value = mse(pred, target)
    tv = total_variation(txm)

    segmentation_penalty = segmentation_sq_penalty(
        txm, segmentation, value_ranges
    )

    gms = unsharp_mask_similarity(pred, target, 3.0) + unsharp_mask_similarity(
        pred, target, 10.0
    )

    return (
        mse_value
        + tv_factor * tv
        + prior_weight * segmentation_penalty
        + gmse_weight * gms
    )


def make_projection(spec):
    def segmentation_projection(
        txm_state: TransmissionMapT, weights_state: WeightsT, _
    ) -> tuple[TransmissionMapT, WeightsT]:
        """
        Project transmission map values based on segmentation information.
        Uses softer constraints with confidence-weighted projections.

        Args:
            txm_state: Transmission mp state
            weights_state: Weights state - parameters of the forward model
        """
        # General constraints
        new_txm_state = optax.projections.projection_hypercube(txm_state)
        new_weights_state = optax.projections.projection_non_negative(
            weights_state
        )

        # Apply constraints on image processing parameters
        new_weights_state = proj.projection_spec(new_weights_state, spec)

        return new_txm_state, new_weights_state

    return segmentation_projection


def save_results(
    save_dir: str,
    txm: TransmissionMapT,
    weights: WeightsT,
    pred: TransmissionMapT,
    meta_batch: list[ChexpertMeta],
):
    joblib.dump(weights, os.path.join(save_dir, "weights.joblib"))
    joblib.dump(meta_batch, os.path.join(save_dir, "meta.joblib"))

    # Create file names from metadata
    file_names = [
        f"{m['deid_patient_id']}_{os.path.basename(m['abs_img_path'])}"
        for m in meta_batch
    ]

    # Save transmission maps and their processed versions
    for i, (img, name) in enumerate(zip(txm, file_names)):
        save_path = os.path.join(save_dir, f"{name}")
        save_image(img, save_path)

        # Also save the processed version
        processed = pred[i]
        proc_path = save_path.replace(".png", "_proc.png")
        save_image(processed, proc_path)


def empty_logger(body):
    pass


def wandb_experiment(
    images: Float[Array, "batch height width"],
    masks_batch: Float[Array, "batch labels height width"],
    value_ranges: Float[Array, "reduced_labels 2"],
    hyperparams: ExperimentArgs,
    logger: Callable[[dict], None] = empty_logger,
    loss_logger=empty_loss_logger,
    summary=empty_logger,
    segmentation_th=0.6,
    log_samples=5,
):
    """Main processing function to recover transmission maps with segmentation guidance"""

    log_samples = min(log_samples, images.shape[0])

    seg_labels, segmentations = batch_get_exclusive_masks(
        masks_batch, segmentation_th
    )

    priors_table = wandb.Table(columns=["region", "min", "max"])
    for region, value_range in zip(seg_labels, value_ranges):
        min_val, max_val = value_range
        priors_table.add_data(region, min_val, max_val)

    wandb.log({"priors": priors_table})

    summary(
        {
            "segmentation_th": segmentation_th,
            "viz_samples": log_samples,
        }
    )

    rand_samples = np.random.randint(
        0, images.shape[0], size=log_samples
    ).tolist()
    rand_samples = sorted(rand_samples)

    samples_tables = wandb.Table(columns=["index", "Image"])

    mask_labels = {i + 1: label for i, label in enumerate(seg_labels)}
    for idx in range(images.shape[0]):
        curr_image = images[idx]
        curr_masks = segmentations[idx]

        image_histogram = wandb.Histogram(curr_image.flatten())
        logger({"image histogram": image_histogram})
        for i, label in mask_labels.items():
            mask_idx = i - 1
            mask = curr_masks[mask_idx]
            logger(
                {
                    f"image ({label}) histogram": wandb.Histogram(
                        curr_image[mask.astype(jnp.bool)].flatten()
                    )
                }
            )

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

    print("sample size:", log_samples)
    print("logged samples:", len(samples_tables.data))

    # Initialize transmission map with appropriate range
    init_mode, init_config = hyperparams.tm_init_params

    init_params: dict[str, Any] = dict(mode=init_mode)
    if init_mode in ["uniform", "normal"]:
        init_params["val_range"] = init_config
    elif init_mode in ["target", "negative"]:
        init_params["target"] = images

    summary({"init_mode": init_mode})

    print("using init params:", hyperparams.tm_init_params)
    print("using init config:", init_params)

    txm0 = init.initialize(hyperparams.PRNGKey, images.shape, **init_params)

    # Initial parameters for the forward model, should yield a proper image processing
    # for constant weights
    # TODO: long-term: automatic DIP parameter selection
    w0 = {
        "low_sigma": 4.0,
        "low_enhance_factor": 0.5,
        "window_center": 0.2,
        "window_width": 0.2,
        "gamma": 5,
    }
    #     for _ in range(images.shape[0])
    # ]
    # w0 = [{k: jnp.array(v, dtype=jnp.float32) for k, v in w.items()} for w in w0]
    # w0 = {k: jnp.full((txm0.shape[0]), v, dtype=jnp.float32) for k, v in w0.items()}

    # TODO: gaussian blur enforces a conversion to float32, requires parameters to match the type
    # HACK: this is a bit hacky, the most straightforward to broadcast the original single parameter dict
    # is to make each value an array in a broadcastable shape, txm has 3 dimensions [batch rows cols]
    summary({"initial_weights": w0})
    

    # TODO: gaussian blur enforces a conversion to float32, requires parameters to match the type
    w0 = {k: jnp.array(v, dtype=jnp.float32) for k, v in w0.items()}

    def loss_fn(*args):
        return segmentation_loss(
            *args,
            value_ranges=value_ranges,
            tv_factor=hyperparams.total_variation,
            prior_weight=hyperparams.prior_weight,
            gmse_weight=hyperparams.gmse_weight,
        )

    optimizer = optax.adam(learning_rate=hyperparams.lr)

    segmentation_projection = make_projection(
        {
            "low_sigma": proj.box(0.5, hyperparams.max_sigma),
            "low_enhance_factor": proj.box(0.3, hyperparams.max_enhancement),
            "gamma": proj.box(1, 20),
            "window_center": proj.box(0.1, 0.8),
            "window_width": proj.box(0.1, 1.0),
        }
    )

    # TODO: any processing with losses output?
    state, _ = segmentation_optimize(
        target=images,
        txm0=txm0,
        w0=w0,
        segmentation=segmentations,
        loss_fn=loss_fn,
        optimizer=optimizer,
        forward_fn=forward,
        loss_logger=loss_logger,
        summary=summary,
        logger=logger,
        project_fn=segmentation_projection,
        constant_weights=hyperparams.constant_weights or False,
        n_steps=hyperparams.n_steps,
        eps=hyperparams.eps,
    )

    if state is None:
        raise RuntimeError("Optimization failed")

    txm, weights = state
    pred = forward(txm, weights)

    # Log recovered parameters
    logger({"recovered_params": weights})

    for idx in range(images.shape[0]):
        curr_image = txm[idx]
        curr_masks = segmentations[idx]
        # Log example images and segmentations

        for idx, label in enumerate(seg_labels):
            mask = curr_masks[idx]
            logger(
                {
                    f"transmission map ({label}) histogram": wandb.Histogram(
                        curr_image[mask.astype(jnp.bool)].flatten()
                    )
                }
            )

        image_histogram = wandb.Histogram(curr_image.flatten())
        logger({"transmission map histogram": image_histogram})

    # Log example recovered images
    fwd_data = [pull_image(pred[i]) for i in rand_samples]
    recovered_data = [pull_image(txm[i]) for i in rand_samples]

    samples_tables.add_column("Recovered", recovered_data)
    samples_tables.add_column("Forward", fwd_data)

    wandb.log({"samples": samples_tables})

    return txm, weights, pred, segmentations


def run_processing(
    images_batch: Float[Tensor, "batch height width"],
    masks_batch: Float[Tensor, "batch labels height width"],
    meta_batch: list[ChexpertMeta],
    value_ranges: Float[Array, "reduced_labels 2"],
    run_init={},
    save_dir=None,
):
    """Main processing function to recover transmission maps with segmentation guidance"""
    run = wandb.init(**run_init)
    hyperparams = run.config

    results = None
    images = jnp.array(images_batch.cpu().numpy()).squeeze(1)
    segmentations = jnp.array(masks_batch.cpu().numpy())

    def summary(body):
        for k, v in body.items():
            run.summary[k] = v

    try:
        results = wandb_experiment(
            images,
            segmentations,
            value_ranges=value_ranges,
            hyperparams=hyperparams,
            logger=wandb.log,
            summary=summary,
            loss_logger=basic_loss_logger,
        )
    except Exception as e:
        raise e
        print("Error during experiment:", e)
        return

    results, segmentations = results[:-1], results[-1]

    assert results is not None
    save_dir = os.path.join(save_dir, run.id) if save_dir is not None else None

    process_results(
        images,
        segmentations,
        meta_batch,
        value_ranges,
        results,
        save_dir=save_dir,
    )


def sweep_based_exec(dataset, project, sweep_name, desc, tags, sweep_config):
    _ = wandb.login()

    _, value_ranges = get_priors(args.cache_dir, collimated_region_bound=0.4)

    run_init = dict(
        project=project,
        notes=f"Segmentation-guided optimization with {FWD_DESC}",
        tags=tags,
    )

    # Define hyperparameter sweep search space
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )

    def sweep_runner():
        # new batch for each sweep run to increase diversity
        batch = next(iter(dataset))
        images, masks, meta = batch

        run_processing(
            images,
            masks,
            meta,
            value_ranges=value_ranges,
            save_dir=args.save_dir,
            run_init=run_init,
        )

    wandb.agent(
        sweep_id,
        function=sweep_runner,
        count=200,
    )


# @memory.cache
def single_experiment(
    images,
    masks,
    value_ranges,
    hyperparams: ExperimentArgs,
    sample_size: int,
    logger=empty_logger,
    summary=empty_logger,
    loss_logger=empty_loss_logger,
):
    print("starting single experiment exec")
    segmentations = jnp.array(masks, dtype=DTYPE)
    value_ranges = jnp.array(value_ranges, dtype=DTYPE)
    images = jnp.array(images, dtype=DTYPE).squeeze(1)

    txm, weights, pred, segmentations = wandb_experiment(
        images,
        segmentations,
        value_ranges,
        hyperparams,
        logger=logger,
        summary=summary,
        loss_logger=loss_logger,
    )

    seg_labels, segmentations = batch_get_exclusive_masks(segmentations, 0.6)
    eval_metrics = batch_evaluation(
        images, txm, pred, segmentations, value_ranges
    )
    ssim = eval_metrics.ssim
    psnr = eval_metrics.psnr
    penalties = eval_metrics.penalties

    sample_size = min(sample_size, images.shape[0])

    return (
        (
            txm[:sample_size],
            pred[:sample_size],
            images[:sample_size],
            segmentations[:sample_size],
        ),
        weights,
        (
            ssim,
            psnr,
            penalties,
        ),
    )


if __name__ == "__main__":
    args = args_spec()

    # sweep based config
    PROJECT = "full-search"
    SWEEP_NAME = "full-sweep"
    FWD_DESC = "normalized negative log, windowing, range normalization, unsharp masking, clipping"

    TAGS = [
        "segmentation-guided",
        "square-penalty",
        "fixed",
        "gmse",
        "valid",
        "sqrt-tv",
        f"batch_size={args.batch_size}",
        *[f.strip() for f in FWD_DESC.split(",")],
    ]

    dataset = get_segmentation_dataset(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        mask_dir=args.mask_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        frontal_lateral=args.frontal_lateral,
        batch_size=args.batch_size,
    )

    with open(args.sweep_conf) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        from pprint import pprint

        print("using config")
        pprint(config)

    # execute sweep
    sweep_based_exec(
        dataset,
        PROJECT,
        SWEEP_NAME,
        desc="Segmentation-guided optimization with square penalty",
        tags=TAGS,
        sweep_config=config,
    )
