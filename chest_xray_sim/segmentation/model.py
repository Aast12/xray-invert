from typing import get_args, overload
import typing
import jax
import jax.numpy as jnp
import torch
import torchxrayvision as xrv
from jaxtyping import Array
from torchvision.transforms import v2
from torchxrayvision.baseline_models.chestx_det import PSPNet

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

SEGMENTATION_LABELS: dict[str, int] = {
    PSPNet.targets[i]: i for i in range(len(PSPNet.targets))
}

# Assumes input images are in the range [0, 1]
_preprocessing = v2.Lambda(lambda x: (2 * x - 1.0) * 1024)


class ChestSegmentation(xrv.baseline_models.chestx_det.PSPNet):
    def __init__(self, cache_dir: str | None = None):
        kwargs = {} if cache_dir is None else {"cache_dir": cache_dir}
        super(ChestSegmentation, self).__init__(**kwargs)

    def forward(self, x):
        return super(ChestSegmentation, self).forward(_preprocessing(x))

    def __call__(self, x):
        with torch.inference_mode():
            return super(ChestSegmentation, self).__call__(x)

    @classmethod
    def get_mask(
        cls,
        pred: torch.Tensor | Array,
        label: SegmentationLabelsT,
        threshold: float | None = None,
    ) -> torch.Tensor | Array:
        if threshold is not None:
            if isinstance(pred, Array):
                pred = jnp.where(jax.nn.sigmoid(pred) < threshold, 0, 1)
            else:
                pred = torch.where(torch.sigmoid(pred) < threshold, 0, 1)

        if isinstance(pred, Array):
            return pred[jnp.array(SEGMENTATION_LABELS[label])]

        return pred[SEGMENTATION_LABELS[label]]
