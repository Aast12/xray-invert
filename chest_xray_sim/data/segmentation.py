import typing
from typing import get_args, overload

import jax
import jax.numpy as jnp
import torch
from torch import Tensor
import torchxrayvision as xrv
from torchvision.transforms import v2
from torchxrayvision.baseline_models.chestx_det import PSPNet
from jaxtyping import Array, Float, Integer

ArrayT = typing.Union[Array, Tensor]

SegmentationLabelsT = typing.Literal[
    "Left Clavicle",
    "Right Clavicle",
    "Left Scapula",
    "Right Scapula",
    "Left Lung",
    "Right Lung",
    "Left Hilus Pulmonis",
    "Right Hilus Pulmonis",
    "Heart",
    "Aorta",
    "Facies Diaphragmatica",
    "Mediastinum",
    "Weasand",
    "Spine",
]

bone_labels = [
    "Left Clavicle",
    "Right Clavicle",
    "Left Scapula",
    "Right Scapula",
    "Spine",
]

lung_labels = [
    "Left Lung",
    "Right Lung",
    "Left Hilus Pulmonis",
    "Right Hilus Pulmonis",
]

soft_labels = [
    "Heart",
    "Aorta",
    "Mediastinum",
    "Facies Diaphragmatica",
    "Weasand",
]


# ORDER MATTERS! see get_exclusive_masks
MaskGroupsT = typing.Literal["lung", "soft", "bone"]
MASK_GROUPS: tuple[MaskGroupsT, ...] = get_args(MaskGroupsT)

MASK_GROUPS_LABELS: dict[MaskGroupsT, list[str]] = {
    "bone": bone_labels,
    "lung": lung_labels,
    "soft": soft_labels,
}


def mask_threshold(
    pred: Float[Tensor, "*batch height width"],
    threshold: float,
) -> Integer[Tensor, "*batch height width"]:
    mask = pred.clone()
    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0
    return mask.int()


@overload
def substract_mask(
    mask0: Float[Tensor, "*batch height width"],
    exclude_mask: Float[Tensor, "*batch height width"],
) -> Float[Tensor, "*batch height width"]: ...
@overload
def substract_mask(
    mask0: Float[Array, "*batch height width"],
    exclude_mask: Float[Array, "*batch height width"],
) -> Float[Array, "*batch height width"]: ...
def substract_mask(
    mask0,
    exclude_mask,
):
    if isinstance(mask0, jax.Array):
        return jnp.bitwise_xor(mask0, mask0 * exclude_mask)

    return torch.bitwise_xor(mask0, mask0 * exclude_mask)


@overload
def _join_masks(
    pred: Float[Array, "num_classes height width"],
    threshold: float | None = None,
) -> Float[Array, "height width"]: ...
@overload
def _join_masks(
    pred: Float[Tensor, "num_classes height width"],
    threshold: float | None = None,
) -> Float[Tensor, "height width"]: ...
def _join_masks(
    pred,
    threshold=None,
):
    aggregated = (
        torch.sigmoid(pred.sum(dim=0))
        if isinstance(pred, torch.Tensor)
        else jax.nn.sigmoid(pred.sum(axis=0))
    )

    if threshold is not None:
        if isinstance(pred, jax.Array):
            aggregated = jnp.where(aggregated < threshold, 0, 1)
        else:
            aggregated[aggregated < threshold] = 0
            aggregated[aggregated >= threshold] = 1

    return aggregated


@overload
def _batch_join_masks(
    pred: Float[Array, "batch labels height width"],
    threshold=...,
) -> Float[Array, "batch labels height width"]: ...
@overload
def _batch_join_masks(
    pred: Float[Tensor, "batch labels height width"],
    threshold=...,
) -> Float[Tensor, "batch labels height width"]: ...


def _batch_join_masks(
    pred,
    threshold: float | None = None,
):
    aggregated = (
        torch.sigmoid(pred.sum(dim=1, keepdim=True))
        if isinstance(pred, torch.Tensor)
        else jax.nn.sigmoid(pred.sum(axis=1, keepdims=True))
    )

    if threshold is not None:
        if isinstance(pred, jax.Array):
            aggregated = jnp.where(aggregated < threshold, 0, 1)
        else:
            aggregated[aggregated < threshold] = 0
            aggregated[aggregated >= threshold] = 1

    return aggregated


class ChestSegmentation(xrv.baseline_models.chestx_det.PSPNet):
    # Assumes input images are in the range [0, 1]
    preprocess = v2.Lambda(lambda x: (2 * x - 1.0) * 1024)

    targets_dict: dict[str, int] = {
        PSPNet.targets[i]: i for i in range(len(PSPNet.targets))
    }

    def __init__(self, cache_dir: str | None = None):
        kwargs = {} if cache_dir is None else {"cache_dir": cache_dir}
        super(ChestSegmentation, self).__init__(**kwargs)

    def forward(self, x):
        return super(ChestSegmentation, self).forward(ChestSegmentation.preprocess(x))

    def __call__(self, x):
        with torch.inference_mode():
            return super(ChestSegmentation, self).__call__(x)

    @classmethod
    def get_mask(
        cls,
        pred: torch.Tensor | jax.Array,
        label: SegmentationLabelsT,
        threshold: float | None = None,
    ) -> torch.Tensor | jax.Array:
        if threshold is not None:
            if isinstance(pred, jax.Array):
                pred = jnp.where(jax.nn.sigmoid(pred) < threshold, 0, 1)
            else:
                pred = torch.where(torch.sigmoid(pred) < threshold, 0, 1)

        if isinstance(pred, jax.Array):
            return pred[jnp.array(cls.targets_dict[label])]
        return pred[cls.targets_dict[label]]


@overload
def get_group_mask(
    pred: Float[Array, "batch_size num_clases height width"],
    group,
    threshold=...,
) -> Float[Array, "batch_size height width"]: ...
@overload
def get_group_mask(
    pred: Float[Tensor, "batch_size num_clases height width"],
    group,
    threshold=...,
) -> Float[Tensor, "batch_size height width"]: ...


def get_group_mask(
    pred,
    group: MaskGroupsT,
    threshold=None,
):
    """
    Get the group mask for the given group from the prediction.

    Args:
        pred: The prediction tensor, shaped (batch_size, num_classes, height, width).
        group: The group to get the mask (see MaskGroupsT).
        threshold: The threshold to discretize the mask (confidence to binary). If None, the mask is not thresholded.
    """
    labels = MASK_GROUPS_LABELS[group]

    q = [ChestSegmentation.targets_dict[label] for label in labels]
    if isinstance(pred, jax.Array):
        q = jnp.array(q)

    return _join_masks(
        jax.nn.sigmoid(pred[q])
        if isinstance(pred, jax.Array)
        else torch.sigmoid(pred[q]),
        threshold,
    )


def batched_get_group_mask(
    pred: Float[Array, "batch labels height width"],
    group: MaskGroupsT,
    threshold: float | None = None,
) -> Float[Array, "batch labels height width"]:  # TODO: check shape here
    """
    Get the group mask for the given group from the prediction.

    Args:
        pred: The prediction tensor, shaped (batch_size, num_classes, height, width).
        group: The group to get the mask (see MaskGroupsT).
        threshold: The threshold to discretize the mask (confidence to binary). If None, the mask is not thresholded.
    """
    labels = MASK_GROUPS_LABELS[group]

    q = [ChestSegmentation.targets_dict[label] for label in labels]
    if isinstance(pred, jax.Array):
        q = jnp.array(q)

    return _batch_join_masks(
        jax.nn.sigmoid(pred[:, q])
        if isinstance(pred, jax.Array)
        else torch.sigmoid(pred[:, q]),
        threshold,
    )


def batch_get_exclusive_masks(
    pred: Float[Array, "batch labels height width"],
    threshold: float | None = None,
) -> tuple[list[MaskGroupsT], Float[Array, "batch reduced_labels height width"]]:
    ordered_groups = MASK_GROUPS
    # TODO: rewrite for group of masks?
    complete_masks = [
        batched_get_group_mask(pred, group, threshold) for group in ordered_groups
    ]

    complete_masks = jnp.concat(complete_masks, axis=1)

    # complete_masks: Float[Array, "batch labels height width"] = batched_get_group_mask(
    #     pred, MASK_GROUPS, threshold
    # )

    exclusive_masks = complete_masks.copy()
    for i in range(complete_masks.shape[1]):
        for j in range(complete_masks.shape[1]):
            if i == j:
                continue
            if j < i:
                exclusive_masks = exclusive_masks.at[:, i].set(
                    substract_mask(exclusive_masks[:, i], exclusive_masks[:, j])
                )
            elif j > i:
                exclusive_masks = exclusive_masks.at[:, i].set(
                    substract_mask(exclusive_masks[:, i], complete_masks[:, j])
                )

    return list(ordered_groups), exclusive_masks


def get_exclusive_masks(
    pred: Float[Array, "batch labels height width"],
    threshold: float | None = None,
) -> dict[MaskGroupsT, jax.Array]:
    print("getting exclusive masks", pred.shape)
    complete_masks = [get_group_mask(pred, group, threshold) for group in MASK_GROUPS]

    exclusive_masks = []
    for i, mask in enumerate(complete_masks):
        exclusive_mask = mask.copy()
        for other_mask in exclusive_masks + complete_masks[i + 1 :]:
            exclusive_mask = substract_mask(exclusive_mask, other_mask)

        exclusive_masks.append(exclusive_mask)

    return {group_id: mask for group_id, mask in zip(MASK_GROUPS, exclusive_masks)}


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    from chest_xray_sim.data.chexpert_dataset import get_chexpert_dataset
    from chest_xray_sim.data.utils import read_image

    model_path = "/Volumes/T7/datasets/torchxrayvision"
    samples = [
        (
            "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
            "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
        ),
        (
            "/Volumes/T7/projs/thesis/outputs/0ji4skar/patient02609_study1.tif",
            "/Volumes/T7/projs/thesis/outputs/0ji4skar/patient02609_study1_proc.tif",
        ),
        (
            "/Volumes/T7/projs/thesis/data/conventional_transmissionmap_32bit_[0 1].tif",
            None,
        ),
    ]

    img_path, fwd_path = samples[0]
    base = read_image(fwd_path)

    print("base shape:", base.shape)

    model = ChestSegmentation(cache_dir=model_path)

    base_seg = model(base)

    print("base_seg shape:", base_seg.shape)

    images = get_chexpert_dataset(
        data_dir="/Volumes/T7/datasets/chexpert_plus/PNG/PNG/",
        split="train",
        meta_dir="/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv",
        frontal_lateral="Frontal",
        batch_size=15,
    )
    batch = next(iter(images))

    img, meta = batch
    import os

    path = [os.path.basename(m["abs_img_path"]) for m in meta]

    print("images shape:", img.shape)

    segmentations = model(img)

    print("segmentations shape:", segmentations.shape)

    fig, ax = plt.subplots(3, 4, figsize=(15, 5))

    for i in range(4):
        if i == 3:
            im, sgm = base.squeeze(0), base_seg[0]
        else:
            im, sgm = img[i].squeeze(0), segmentations[i]

        print("sgm shape", sgm.shape)
        print("img shape", im.shape)

        bone = get_group_mask(sgm, "bone", threshold=None)
        lung = get_group_mask(sgm, "lung", threshold=None)

        ax[0, i].imshow(im, cmap="gray")
        ax[0, i].set_title(f"Label: {path[i]}")
        ax[0, i].axis("off")

        a = ax[1, i].imshow(bone)
        plt.colorbar(a)
        a = ax[2, i].imshow(lung)
        plt.colorbar(a)
    #
    plt.show()

    sys.exit(0)

    model_path = "/Volumes/T7/datasets/torchxrayvision"
    samples = [
        (
            "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
            "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
        ),
        (
            "/Volumes/T7/projs/thesis/outputs/0ji4skar/patient02609_study1.tif",
            "/Volumes/T7/projs/thesis/outputs/0ji4skar/patient02609_study1_proc.tif",
        ),
        (
            "/Volumes/T7/projs/thesis/data/conventional_transmissionmap_32bit_[0 1].tif",
            None,
        ),
    ]

    img_path, fwd_path = samples[0]
    base = read_image(img_path)

    model = ChestSegmentation(cache_dir=model_path)

    if fwd_path is not None:
        basefwd = read_image(fwd_path)
    else:
        basefwd = 1 - base

    plt.imshow(basefwd, cmap="gray")

    pred = model(basefwd)
    th = 0.6

    # visualize segmentations
    fig, ax = plt.subplots(1, len(model.targets), figsize=(15, 5))
    for i, label in enumerate(model.targets):
        ax[i].imshow(pred[0, i], cmap="gray")
        ax[i].set_title(label)
        ax[i].axis("off")

    lung_mask = model.get_group_mask(pred, "lung", threshold=None)
    bone_mask = model.get_group_mask(pred, "bone", threshold=None)

    fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    a = ax[0].imshow(lung_mask)
    fig.colorbar(a)
    b = ax[1].imshow(bone_mask)
    fig.colorbar(b)

    d_lung_mask = mask_threshold(lung_mask.clone(), th)
    d_bone_mask = mask_threshold(bone_mask.clone(), th)
    d_only_lung_mask = substract_mask(d_lung_mask, d_bone_mask)
    d_only_bone_mask = substract_mask(d_bone_mask, d_lung_mask)
    d_rest_mask = substract_mask(
        torch.ones_like(d_lung_mask), torch.bitwise_or(d_lung_mask, d_bone_mask)
    )

    fig, ax = plt.subplots(2, 2, figsize=(15, 5), sharey="row")
    ax = ax.flatten()

    ax[0].imshow(base, cmap="gray")
    a = ax[1].imshow(d_only_lung_mask + d_bone_mask * 2 + d_rest_mask * 3)
    plt.colorbar(a)
    ax[3].hist(base.ravel(), bins=50, color="blue", label="Base")
    ax[2].hist(
        base[d_only_lung_mask.bool()].ravel(),
        bins=50,
        color="red",
        alpha=0.2,
        label="Only Lung",
    )
    ax[2].hist(
        base[d_bone_mask.bool()].ravel(),
        bins=50,
        color="green",
        alpha=0.2,
        label="All Bone",
    )
    ax[2].hist(
        base[d_rest_mask.bool()].ravel(),
        bins=50,
        color="yellow",
        alpha=0.2,
        label="Rest",
    )
    plt.legend()

    plt.show()
