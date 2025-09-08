import os
from pathlib import Path
import re
import typing

import pandas as pd
from torch.utils.data import DataLoader

from chest_xray_sim.data.meta import XRayImageMeta
from chest_xray_sim.data.utils import filter_key, get_chexpert_transform
from chest_xray_sim.data.dataset import FolderDataset, MetaDataset
from chest_xray_sim.types import ImageTorch


class ChexpertMeta(typing.TypedDict):
    """
    Metadata keys to include in dataset entries. Keys must match
    columns in the Chexpert metadata CSV file.
    """

    abs_img_path: str
    deid_patient_id: str
    frontal_lateral: typing.NotRequired[str]
    ap_pa: typing.NotRequired[str]


def to_xray_meta(chexpert_meta: ChexpertMeta) -> XRayImageMeta:
    meta: XRayImageMeta = {
        "img_id": chexpert_meta.get("deid_patient_id", ""),
        "local_img_path": chexpert_meta.get("abs_img_path", ""),
    }
    if "frontal_lateral" in chexpert_meta:
        meta["frontal_lateral"] = chexpert_meta["frontal_lateral"]
    if "ap_pa" in chexpert_meta:
        meta["ap_pa"] = chexpert_meta["ap_pa"]

    return meta


class ChexpertDataset(MetaDataset):
    frontal_lateral = None
    split = None
    root: str

    def __init__(
        self,
        root: str,
        meta_dir: str,
        split: typing.Literal["train", "valid"] | None = None,
        frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = None,
        **kwargs,
    ):
        self.root = root
        self.frontal_lateral = frontal_lateral
        self.split = split
        data_dir = os.path.join(root, split) if split else root

        super().__init__(data_dir, meta_dir, **kwargs)

    def is_valid_file(self, path: str) -> bool:
        if self.frontal_lateral is None and self.split is None:
            return True

        view_check = (
            self.frontal_lateral.lower() in path.lower()
            if self.frontal_lateral is not None
            else True
        )
        split_check = (
            self.split.lower() in path.lower() if self.split is not None else True
        )

        return view_check and split_check

    def filter_records(self, df: pd.DataFrame) -> pd.DataFrame:
        df = filter_key(df, "frontal_lateral", self.frontal_lateral)
        df = filter_key(self.metadata, "split", self.split)

        df["abs_img_path"] = df["path_to_image"].apply(
            lambda x: os.path.join(self.root, x.replace(".jpg", ".png"))
        )
        df["exists"] = df["abs_img_path"].apply(lambda x: os.path.exists(x))

        df = df[df["exists"]]
        df = df[list(ChexpertMeta.__annotations__.keys())]

        return df

    def get_meta_data(self, abs_path: str) -> XRayImageMeta:
        item = self.metadata[self.metadata["abs_img_path"] == abs_path].head(1)

        return (
            to_xray_meta(ChexpertMeta(**item.iloc[0].to_dict()))
            if not item.empty
            else {}
        )


class ChexpertMetaLoader:
    iterator = None
    lodaer: DataLoader

    def __init__(
        self,
        data_dir: str,
        meta_dir: str,
        split: typing.Literal["train", "valid"] | None = None,
        frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = "Frontal",
        batch_size: int = 12,
        **kwargs,
    ):
        self.ds = ChexpertDataset(
            root=data_dir,
            split=split,
            meta_dir=meta_dir,
            frontal_lateral=frontal_lateral,
            transform=get_chexpert_transform(512),
            **kwargs,
        )
        self.loader = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ChexpertDataset.collate,
        )

    def next(self) -> tuple[ImageTorch, list[XRayImageMeta]]:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


def get_chexpert_dataset(
    data_dir: str,
    meta_dir: str,
    split: typing.Literal["train", "valid"] | None = None,
    frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = "Frontal",
    batch_size: int = 12,
    image_size: int = 512,
    **kwargs,
):
    ds = ChexpertDataset(
        root=data_dir,
        split=split,
        meta_dir=meta_dir,
        frontal_lateral=frontal_lateral,
        transform=get_chexpert_transform(image_size),
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
