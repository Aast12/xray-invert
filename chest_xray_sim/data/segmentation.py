import typing

import torch
import torchxrayvision as xrv
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


def mask_threshold(
    pred: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    mask = pred.clone()
    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0
    return mask.int()


def substract_mask(mask0: torch.Tensor, exclude_mask: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_xor(mask0, mask0 * exclude_mask)


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
        with torch.no_grad():
            return super(ChestSegmentation, self).__call__(x)

    @classmethod
    def get_mask(
        cls,
        pred: torch.Tensor,
        label: SegmentationLabelsT,
        threshold: float | None = None,
    ) -> torch.Tensor:
        pred = 1 / (1 + torch.exp(-pred))

        if threshold is not None:
            pred[pred < threshold] = 0
            pred[pred >= threshold] = 1

        return pred[0, ChestSegmentation.targets_dict[label]]

    @classmethod
    def get_group_mask(
        cls,
        pred: torch.Tensor,
        group: typing.Literal["bone", "lung"],
        threshold: float | None = None,
    ) -> torch.Tensor:
        labels = bone_labels if group == "bone" else lung_labels

        q = [ChestSegmentation.targets_dict[label] for label in labels]

        return cls._join_masks(torch.sigmoid(pred[0, q]), threshold)

    @classmethod
    def _join_masks(
        cls,
        pred: torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        aggregated = torch.sigmoid(pred.sum(dim=0))

        if threshold is not None:
            aggregated[aggregated < threshold] = 0
            aggregated[aggregated >= threshold] = 1

        return aggregated


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
