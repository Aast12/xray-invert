import cv2
import pandas as pd

from cv2.typing import MatLike
from typing import TypedDict
import typing
import os

import pandas as pd
import torch
import torchvision
import torchxrayvision as xrv
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

from torchvision.transforms import v2


class ChexpertDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, meta_dir, split=None, **kwargs):
        data_dir = os.path.join(root, split) if split else root
        super(ChexpertDataset, self).__init__(root=data_dir, **kwargs)
        self.metadata = pd.read_csv(meta_dir)
        # self.metadata = self.metadata[self.metadata["frontal_lateral"] == "Frontal"]

        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split]

        self.metadata["abs_img_path"] = self.metadata["path_to_image"].apply(
            lambda x: os.path.join(root, x.replace(".jpg", ".png"))
        )
        self.metadata["exists"] = self.metadata["abs_img_path"].apply(
            lambda x: os.path.exists(x)
        )

        self.metadata = self.metadata[self.metadata["exists"] == True]

    def __getitem__(self, index):
        original_tuple = super(ChexpertDataset, self).__getitem__(index)

        img_path = self.imgs[index][0]

        meta = self.metadata[self.metadata["abs_img_path"] == img_path].head(1)
        meta = meta.iloc[0].to_dict() if not meta.empty else {}

        # make a new tuple that includes original and the path
        return original_tuple + (
            img_path,
            meta,
        )

    @staticmethod
    def collate(batch):
        images, labels, paths, meta = zip(*batch)

        images = torch.stack(images)

        # For labels, ensure they're all tensors before stacking
        if all(isinstance(label, torch.Tensor) for label in labels):
            labels = torch.stack(labels)

        # Return as a tuple without trying to convert metadata to tensors
        return images, labels, paths, meta


def get_chexpert_dataset(data_dir, meta_dir, split=None, **kwargs):
    ds = ChexpertDataset(
        root=data_dir,
        split=split,
        meta_dir=meta_dir,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Resize(512),
                torchvision.transforms.CenterCrop(512),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    )
    ds_loader = DataLoader(
        ds,
        batch_size=12,
        shuffle=True,
        collate_fn=ChexpertDataset.collate,
    )
    return ds_loader


if __name__ == "__main__":
    images = get_chexpert_dataset(
        data_dir="/Volumes/T7/datasets/chexpert_plus/PNG/PNG/",
        split="train",
        meta_dir="/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv",
    )

    import matplotlib.pyplot as plt

    for i in range(1):
        batch = next(iter(images))
        img, label, path, meta = batch
        plt.imshow(
            img[0][0],
            cmap="gray",
        )

        plt.show()
