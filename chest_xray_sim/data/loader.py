import os
import typing

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image


class ChexpertMeta(typing.TypedDict):
    abs_img_path: typing.NotRequired[str]
    deid_patient_id: typing.NotRequired[str]
    frontal_lateral: typing.NotRequired[str]
    ap_pa: typing.NotRequired[str]


ItemT = tuple[torch.Tensor, int, str, ChexpertMeta]
CollateItemT = list[ItemT]
BatchItemT = tuple[torch.Tensor, torch.Tensor, list[str], list[ChexpertMeta]]


def remove_color_channels(x: torch.Tensor) -> torch.Tensor:
    # Assuming x is a 3D tensor (C, H, W)
    if x.ndim == 3:
        # Convert to grayscale by averaging the color channels
        x = x.mean(dim=0, keepdim=False)
    return x


def read_image(path: str, size: int = 512) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L")  # Convert to grayscale

    print("pre stats:", img.getextrema())

    t = get_chexpert_transform(size)
    return t(img)


def get_chexpert_transform(
    size: int = 512, dtype: torch.dtype = torch.float32
) -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            # v2.Grayscale(),
            v2.PILToTensor(),
            v2.Resize(size),
            v2.CenterCrop(size),
            v2.ToDtype(dtype, scale=True),
            v2.Lambda(remove_color_channels),
        ]
    )


class ChexpertDataset(torchvision.datasets.ImageFolder):
    metadata: pd.DataFrame

    def __init__(
        self,
        root: str,
        meta_dir: str,
        split: typing.Literal["train", "valid"] | None = None,
        frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = None,
        **kwargs,
    ):
        data_dir = os.path.join(root, split) if split else root
        self.metadata = pd.read_csv(meta_dir)

        is_valid_file: typing.Callable[[str], bool] | None = None
        if frontal_lateral:
            self.metadata = self.metadata[
                self.metadata["frontal_lateral"] == frontal_lateral
            ]
            is_valid_file = lambda path: frontal_lateral.lower() in path.lower()  # noqa: E731

        super(ChexpertDataset, self).__init__(
            root=data_dir, is_valid_file=is_valid_file, allow_empty=True, **kwargs
        )

        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split]

        self.metadata["abs_img_path"] = self.metadata["path_to_image"].apply(
            lambda x: os.path.join(root, x.replace(".jpg", ".png"))
        )
        self.metadata["exists"] = self.metadata["abs_img_path"].apply(
            lambda x: os.path.exists(x)
        )

        self.metadata = self.metadata[self.metadata["exists"]]
        self.metadata = self.metadata[list(ChexpertMeta.__annotations__.keys())]

    def __getitem__(self, index: int) -> ItemT:
        original_tuple = super(ChexpertDataset, self).__getitem__(index)

        img, target = original_tuple

        img_path = self.imgs[index][0]

        item = self.metadata[self.metadata["abs_img_path"] == img_path].head(1)
        meta: ChexpertMeta = (
            ChexpertMeta(**item.iloc[0].to_dict()) if not item.empty else {}
        )

        # make a new tuple that includes original and the path
        out = img, target, img_path, meta
        return out

    @staticmethod
    def collate(batch: CollateItemT) -> BatchItemT:
        images, labels, paths, meta = zip(*batch)

        images = torch.stack(images)

        # For labels, ensure they're all tensors before stacking
        if all(isinstance(label, torch.Tensor) for label in labels):
            labels = torch.stack(labels)

        # Return as a tuple without trying to convert metadata to tensors
        return images, labels, paths, meta


def get_chexpert_dataset(
    data_dir: str,
    meta_dir: str,
    split: typing.Literal["train", "valid"] | None = None,
    frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = None,
    **kwargs,
):
    ds = ChexpertDataset(
        root=data_dir,
        split=split,
        meta_dir=meta_dir,
        frontal_lateral=frontal_lateral,
        transform=get_chexpert_transform(512),
        **kwargs,
    )
    ds_loader = DataLoader(
        ds,
        batch_size=12,
        shuffle=True,
        collate_fn=ChexpertDataset.collate,
    )
    return ds_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # images = get_chexpert_dataset(
    #     data_dir="/Volumes/T7/datasets/chexpert_plus/PNG/PNG/",
    #     split="train",
    #     meta_dir="/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv",
    #     frontal_lateral="Frontal",
    # )
    #
    # batch = next(iter(images))
    # model = get_seg_model()
    #
    # img, label, path, meta = batch
    #
    # print("metadata:", meta[0])
    #
    # base = img[0]

    img_path = "/Volumes/T7/projs/thesis/outputs/0ji4skar/patient02609_study1.tif"
    base = read_image(img_path)

    basefwd = 1 - base

    pred = model(basefwd)
    th = 0.6

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

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(base, cmap="gray")
    a = ax[1].imshow(d_only_lung_mask + d_bone_mask * 2 + d_rest_mask * 3)
    plt.colorbar(a)
    ax[2].hist(base.ravel(), bins=50, color="blue", alpha=0.5, label="Base")
    ax[2].hist(
        base[d_only_lung_mask.bool()].ravel(),
        bins=50,
        color="red",
        alpha=0.33,
        label="Only Lung",
    )
    ax[2].hist(
        base[d_bone_mask.bool()].ravel(),
        bins=50,
        color="green",
        alpha=0.33,
        label="All Bone",
    )
    ax[2].hist(
        base[d_rest_mask.bool()].ravel(),
        bins=50,
        color="yellow",
        alpha=0.33,
        label="Rest",
    )
    plt.legend()

    plt.show()
