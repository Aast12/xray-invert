import os
import typing

import torch
from torch.utils.data import DataLoader

from chest_xray_sim.data.chexpert_dataset import ChexpertDataset, ChexpertMeta
from chest_xray_sim.data.segmentation import ChestSegmentation
from chest_xray_sim.data.utils import get_chexpert_transform


class SegmentationChexpertDataset(ChexpertDataset):
    """Dataset that yields both Chexpert images and their segmentation masks."""

    def __init__(
        self,
        root: str,
        meta_dir: str,
        mask_dir: str | None = None,
        compute_masks: bool = False,
        split: typing.Literal["train", "valid"] | None = None,
        frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = None,
        cache_dir: str | None = None,
        **kwargs: typing.Any,
    ):
        """

        Mask cache directory assumes and uses the same file structure as the image directory.

                Args:
                    root (str): Path to the dataset root directory.
                    meta_dir (str): Path to the metadata CSV file.
                    mask_dir (str | None): Directory with precomputed masks. If None, masks will be computed on-the-fly.
                    split (str | None): Dataset split ('train' or 'valid').
                    frontal_lateral (str | None): Filter for frontal or lateral images.
                    cache_dir (str | None): Directory for caching segmentation model data (default: ~/.torchxrayvision/).
        .
                    **kwargs: Additional arguments for ChexpertDataset.
        """

        super(SegmentationChexpertDataset, self).__init__(
            root=root,
            meta_dir=meta_dir,
            split=split,
            frontal_lateral=frontal_lateral,
            **kwargs,
        )

        self.mask_dir = mask_dir
        if mask_dir and not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        self.seg_model = ChestSegmentation(cache_dir=cache_dir)

    def _get_mask_path(self, path: str) -> str | None:
        if self.mask_dir is None:
            return None

        path_id = os.path.relpath(path, start=self.root_dir)
        path_id = os.path.splitext(path_id)[0]
        mask_path = os.path.join(self.mask_dir, f"{path_id}_masks.pt")

        return mask_path

    def _get_segmentation(self, path: str, image: torch.Tensor) -> torch.Tensor:
        mask_path = self._get_mask_path(path)

        if mask_path and os.path.exists(mask_path):
            return torch.load(mask_path)

        pred = self.seg_model(image)[0]
        assert pred.ndim == 3, f"Segmentation model returned {pred.ndim}D tensor."

        if mask_path:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            torch.save(pred, mask_path)

        return pred

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, ChexpertMeta]:
        # Get original data from parent class
        img, meta = super(SegmentationChexpertDataset, self).__getitem__(index)
        img_path = meta.get("abs_img_path")

        return img, self._get_segmentation(img_path, img), meta

    @staticmethod
    def collate(batch):
        images, masks, metas = zip(*batch)

        images = torch.stack(images)
        masks = torch.stack(masks)

        return images, masks, metas


def get_segmentation_dataset(
    data_dir: str,
    meta_dir: str,
    mask_dir: str,
    cache_dir: str,
    split: typing.Literal["train", "valid"] | None = None,
    frontal_lateral: typing.Literal["Frontal", "Lateral"] | None = "Frontal",
    batch_size: int = 12,
    **kwargs,
):
    print("using batch size", batch_size)
    ds = SegmentationChexpertDataset(
        root=data_dir,
        split=split,
        meta_dir=meta_dir,
        frontal_lateral=frontal_lateral,
        transform=get_chexpert_transform(512),
        mask_dir=mask_dir,
        cache_dir=cache_dir,
        **kwargs,
    )
    ds_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SegmentationChexpertDataset.collate,
    )
    return ds_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model_path = "/Volumes/T7/datasets/torchxrayvision"
    masks_path = "/Volumes/T7/datasets/chexpert_plus/masks/"

    ds = get_segmentation_dataset(
        data_dir="/Volumes/T7/datasets/chexpert_plus/PNG/PNG/",
        meta_dir="/Volumes/T7/datasets/chexpert_plus/df_chexpert_plus_240401.csv",
        mask_dir=masks_path,
        cache_dir=model_path,
        split="train",
        frontal_lateral="Frontal",
        batch_size=32,
    )

    batch = next(iter(ds))
    images, masks, meta = batch

    fig, ax = plt.subplots(3, 4, figsize=(15, 5))
    for i in range(4):
        im, sgm = images[i].squeeze(0), masks[i]
        print("sgm shape", sgm.shape)
        print("img shape", im.shape)

        bone = ChestSegmentation.get_group_mask(sgm, "bone", threshold=None)
        lung = ChestSegmentation.get_group_mask(sgm, "lung", threshold=None)

        ax[0, i].imshow(im, cmap="gray")
        ax[0, i].set_title(f"Label: {meta[i]['abs_img_path']}")
        ax[0, i].axis("off")

        a = ax[1, i].imshow(bone)
        plt.colorbar(a)
        a = ax[2, i].imshow(lung)
        plt.colorbar(a)

    plt.show()
