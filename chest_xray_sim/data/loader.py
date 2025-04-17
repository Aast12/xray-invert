import os
import typing

import pandas as pd
import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class ChexpertMeta(typing.TypedDict):
    abs_img_path: typing.NotRequired[str]
    deid_patient_id: typing.NotRequired[str]
    frontal_lateral: typing.NotRequired[str]
    ap_pa: typing.NotRequired[str]


ItemT = tuple[torch.Tensor, int, str, ChexpertMeta]
CollateItemT = list[ItemT]
BatchItemT = tuple[torch.Tensor, torch.Tensor, list[str], list[ChexpertMeta]]


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

        img_path = self.imgs[index][0]

        item = self.metadata[self.metadata["abs_img_path"] == img_path].head(1)
        meta: ChexpertMeta = (
            ChexpertMeta(**item.iloc[0].to_dict()) if not item.empty else {}
        )

        # make a new tuple that includes original and the path
        out = original_tuple + (
            img_path,
            meta,
        )

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
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Resize(512),
                torchvision.transforms.CenterCrop(512),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        **kwargs,
    )
    ds_loader = DataLoader(
        ds,
        batch_size=12,
        shuffle=True,
        collate_fn=ChexpertDataset.collate,
    )
    return ds_loader


def get_seg_model():
    t = torchvision.transforms.Lambda(lambda x: (2 * x - 1.0) * 1024)
    model = xrv.baseline_models.chestx_det.PSPNet(
        cache_dir="/Volumes/T7/datasets/torchxrayvision"
    )

    def model_fn(img):
        if img.ndim == 3:
            img = img.squeeze()

        return model(t(img))

    model_fn.targets = model.targets

    return model_fn


if __name__ == "__main__":
    images = get_chexpert_dataset(
        data_dir="/Volumes/T7/datasets/chexpert_plus/PNG/PNG/",
        split="train",
        meta_dir="/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv",
        frontal_lateral="Frontal",
    )

    import matplotlib.pyplot as plt

    batch = next(iter(images))
    model = get_seg_model()

    img, label, path, meta = batch

    # base = [im for im, mt in zip(img, meta) if mt["frontal_lateral"] == "frontal"]

    print("metadata:", meta[0])

    base = img[0]

    with torch.no_grad():
        pred = model(base)
    pred = 1 / (1 + torch.exp(-pred))

    plt.figure(figsize=(22, 5))
    plt.subplot(1, len(model.targets) + 1, 1)
    plt.imshow(base[0], cmap="gray")
    for i in range(len(model.targets)):
        plt.subplot(1, len(model.targets) + 1, i + 2)
        plt.imshow(pred[0, i])
        plt.title(model.targets[i])
        plt.axis("off")
    plt.tight_layout()

    plt.show()
