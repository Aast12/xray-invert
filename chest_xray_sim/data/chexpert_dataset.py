import os
import typing

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

from chest_xray_sim.data.utils import filter_key, get_chexpert_transform


class ChexpertMeta(typing.TypedDict):
    """
    Metadata keys to include in dataset entries. Keys must match
    columns in the metadata CSV file.
    """

    abs_img_path: str
    deid_patient_id: typing.NotRequired[str]
    frontal_lateral: typing.NotRequired[str]
    ap_pa: typing.NotRequired[str]


ItemT = tuple[torch.Tensor, ChexpertMeta]
CollateItemT = list[ItemT]
BatchItemT = tuple[torch.Tensor, list[ChexpertMeta]]


class ChexpertDataset(torchvision.datasets.ImageFolder):
    metadata: pd.DataFrame
    root_dir: str

    def __init__(
        self,
        root: str,
        meta_dir: str,
        split: typing.Literal["train", "valid"] | None = None,
        frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = None,
        **kwargs,
    ):
        self.root_dir = root

        data_dir = os.path.join(root, split) if split else root
        self.metadata = pd.read_csv(meta_dir)

        self.metadata = filter_key(self.metadata, "frontal_lateral", frontal_lateral)
        is_valid_file: typing.Callable[[str], bool] | None = None
        if frontal_lateral:
            is_valid_file = lambda path: frontal_lateral.lower() in path.lower()  # noqa: E731

        super(ChexpertDataset, self).__init__(
            root=data_dir, is_valid_file=is_valid_file, allow_empty=True, **kwargs
        )

        self.metadata = filter_key(self.metadata, "split", split)

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

        img, _ = original_tuple
        img_path = self.imgs[index][0]

        item = self.metadata[self.metadata["abs_img_path"] == img_path].head(1)

        meta: ChexpertMeta = (
            ChexpertMeta(**item.iloc[0].to_dict()) if not item.empty else {}
        )  # pyright: ignore [reportAssignmentType]

        # make a new tuple that includes original and the path
        return img, meta

    @staticmethod
    def collate(batch: CollateItemT) -> BatchItemT:
        images, meta = zip(*batch)

        images = torch.stack(images)

        # Return as a tuple without trying to convert metadata to tensors
        return images, meta


def get_chexpert_dataset(
    data_dir: str,
    meta_dir: str,
    split: typing.Literal["train", "valid"] | None = None,
    frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = "Frontal",
    batch_size: int = 12,
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
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ChexpertDataset.collate,
    )
    return ds_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    images = get_chexpert_dataset(
        data_dir="/Volumes/T7/datasets/chexpert_plus/PNG/PNG/",
        split="train",
        meta_dir="/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv",
        frontal_lateral="Frontal",
    )
    batch = next(iter(images))

    img, meta = batch
    path = [meta[i]["abs_img_path"] for i in range(len(meta))]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        ax[i].imshow(img[i][0])
        ax[i].set_title(f"Path: {path[i]}")
        ax[i].axis("off")

    plt.show()
