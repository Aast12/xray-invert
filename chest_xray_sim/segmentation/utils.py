from typing import overload
import jax
import jax.numpy as jnp
import torch
from jaxtyping import Array, Float, Integer
from torch import Tensor
from chest_xray_sim.types import ArrayT

from chest_xray_sim.segmentation.model import (
    MASK_GROUPS,
    MASK_GROUPS_LABELS,
    SEGMENTATION_LABELS,
    MaskGroupsT,
)

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


@overload
def mask_threshold(
    pred: Float[Array, "*batch height width"],
    threshold: float,
) -> Integer[Array, "*batch height width"]: ...
@overload
def mask_threshold(
    pred: Float[Tensor, "*batch height width"],
    threshold: float,
) -> Integer[Tensor, "*batch height width"]: ...
def mask_threshold(
    pred,
    threshold,
):
    """
    Returns a (integer) binary mask by thresholding a prediction tensor.

    Args:
        pred: The prediction tensor.
        threshold: The threshold to discretize the mask (confidence to binary).

    Returns:
        The binary mask.
    """
    if isinstance(pred, Array):
        mask = pred.copy()
        mask = jnp.where(mask >= threshold, 1, 0)
        return mask.astype(jnp.int8)

    if isinstance(pred, Tensor):
        mask = pred.clone()
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        return mask.int()

    raise ValueError(f"Unsupported type {type(pred)}. Expected Array or Tensor.")


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
    """
    Subtracts the given mask from the input image.

    Parameters:
        image (numpy.ndarray): The input image array.
        mask (numpy.ndarray): The mask to subtract from the image. Must be the same shape as image.

    Returns:
        numpy.ndarray: The resulting image after subtracting the mask.

    Raises:
        ValueError: If the image and mask shapes do not match.
    """
    if isinstance(mask0, Array):
        return jnp.bitwise_xor(mask0, mask0 * exclude_mask)
    if isinstance(mask0, Tensor):
        return torch.bitwise_xor(mask0, mask0 * exclude_mask)

    raise ValueError(f"Unsupported type {type(mask0)}. Expected Array or Tensor.")


@overload
def _sigmoid_mask_sum(
    multimask: Float[Array, "num_classes height width"],
    threshold: float | None,
) -> Float[Array, "height width"]: ...
@overload
def _sigmoid_mask_sum(
    multimask: Float[Tensor, "num_classes height width"],
    threshold: float | None,
) -> Float[Tensor, "height width"]: ...
def _sigmoid_mask_sum(
    multimask,
    threshold=None,
):
    aggregated = (
        jax.nn.sigmoid(multimask)
        if isinstance(multimask, Array)
        else torch.sigmoid(multimask)
    )

    if threshold is not None:
        if isinstance(multimask, Array):
            aggregated = jnp.where(aggregated < threshold, 0, 1)
        elif isinstance(multimask, Tensor):
            aggregated = torch.where(aggregated < threshold, 0, 1)
        else:
            raise TypeError(f"Unsupported type {type(multimask)}")

    if isinstance(aggregated, Array):
        aggregated = aggregated.sum(axis=0).clip(0, 1)
    elif isinstance(aggregated, Tensor):
        aggregated = aggregated.sum(dim=0).clip(0, 1)
    else:
        raise TypeError(f"Unsupported type {type(multimask)}")

    return aggregated


@overload
def _batch_sigmoid_mask_sum(
    pred: Float[Array, "batch labels height width"],
    threshold=...,
) -> Float[Array, "batch labels height width"]: ...
@overload
def _batch_sigmoid_mask_sum(
    pred: Float[Tensor, "batch labels height width"],
    threshold=...,
) -> Float[Tensor, "batch labels height width"]: ...
def _batch_sigmoid_mask_sum(
    pred,
    threshold: float | None = None,
) -> Array | Tensor:
    if isinstance(pred, Array):
        normalized = jax.nn.sigmoid(pred)
        if threshold is not None:
            thresholded = jnp.where(normalized >= threshold, 1, 0)
            return thresholded.sum(axis=1, keepdims=True).clip(0, 1)
        else:
            # TODO: verify
            normalized = normalized.sum(axis=1, keepdims=True)
            return normalized.clip(0, 1)

    elif isinstance(pred, Tensor):
        aggregated = torch.sigmoid(pred)
        if threshold is not None:
            aggregated[aggregated < threshold] = 0
            aggregated[aggregated >= threshold] = 1
            aggregated = aggregated.sum(dim=1, keepdim=True)
            return aggregated.clip(0, 1)
        else:
            # TODO: verify
            return aggregated.sum(dim=1, keepdim=True).clip(0, 1)

    raise ValueError(f"Unsupported type {type(pred)}. Expected Array or Tensor.")


# TODO: refactor get_group_mask and batched versin
@overload
def get_group_mask(
    pred: Float[Array, "batch_size num_clases height width"],
    group: MaskGroupsT,
    threshold=...,
) -> Float[Array, "batch_size height width"]: ...
@overload
def get_group_mask(
    pred: Float[Tensor, "batch_size num_clases height width"],
    group: MaskGroupsT,
    threshold=...,
) -> Float[Tensor, "batch_size height width"]: ...
def get_group_mask(
    pred,
    group,
    threshold: float | None = None,
) -> ArrayT:
    """
    Get the group mask for the given group from the prediction.

    Args:
        pred: The prediction tensor, shaped (batch_size, num_classes, height, width).
        group: The group to get the mask (see MaskGroupsT).
        threshold: The threshold to discretize the mask (confidence to binary). If None, the mask is not thresholded.
    """
    labels = MASK_GROUPS_LABELS[group]

    q = [SEGMENTATION_LABELS[label] for label in labels]
    if isinstance(pred, Array):
        q = jnp.array(q)

    if isinstance(pred, Array):
        return _sigmoid_mask_sum(pred[q], threshold)
    else:
        return _sigmoid_mask_sum(pred[q], threshold)


@overload
def batched_get_group_mask(
    pred: Float[Tensor, "batch labels height width"],
    group,
    threshold=...,
) -> Float[Tensor, "batch labels height width"]: ...
@overload
def batched_get_group_mask(
    pred: Float[Array, "batch labels height width"],
    group,
    threshold=...,
) -> Float[Array, "batch labels height width"]: ...
def batched_get_group_mask(
    pred,
    group: MaskGroupsT,
    threshold: float | None = None,
) -> ArrayT:  # TODO: check shape here
    """
    Get the group mask for the given group from the prediction.

    Args:
        pred: The prediction tensor, shaped (batch_size, num_classes, height, width).
        group: The group to get the mask (see MaskGroupsT).
        threshold: The threshold to discretize the mask (confidence to binary). If None, the mask is not thresholded.
    """
    labels = MASK_GROUPS_LABELS[group]

    q = [SEGMENTATION_LABELS[label] for label in labels]
    if isinstance(pred, Array):
        q = jnp.array(q)

    if isinstance(pred, Array):
        return _batch_sigmoid_mask_sum(pred[:, q], threshold)
    else:
        return _batch_sigmoid_mask_sum(pred[:, q], threshold)


def batch_get_exclusive_masks(
    pred: Float[Array, "batch labels height width"],
    threshold: float | None = None,
) -> tuple[list[MaskGroupsT], Float[Array, "batch reduced_labels height width"]]:
    ordered_groups = MASK_GROUPS
    # TODO: rewrite for group of masks?
    complete_masks = [
        batched_get_group_mask(pred, group, threshold) for group in ordered_groups
    ]

    complete_masks = jnp.concat(complete_masks, axis=1).astype(jnp.int8)

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
) -> dict[MaskGroupsT, Array]:
    print("getting exclusive masks", pred.shape)
    complete_masks = [get_group_mask(pred, group, threshold) for group in MASK_GROUPS]

    exclusive_masks = []
    for i, mask in enumerate(complete_masks):
        exclusive_mask = mask.copy()
        for other_mask in exclusive_masks + complete_masks[i + 1 :]:
            exclusive_mask = substract_mask(exclusive_mask, other_mask)

        exclusive_masks.append(exclusive_mask)

    return {group_id: mask for group_id, mask in zip(MASK_GROUPS, exclusive_masks)}
