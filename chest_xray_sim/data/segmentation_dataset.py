import os
import typing

import torch

from chest_xray_sim.data.chexpert import ChexpertDataset
from chest_xray_sim.segmentation.model import (
    ChestSegmentation,
)


class SegmentationChexpertDataset(ChexpertDataset):
    """
    Dataset that yields both Chexpert images and their segmentation masks.
    """

    def __init__(
        self,
        root: str,
        meta_dir: str,
        mask_dir: str | None = None,
        cache_dir: str | None = None,
        **kwargs: typing.Any,
    ):
        """
        Initialize the SegmentationChexpertDataset.

        This dataset yields tuples of (image, segmentation, metadata).


        Args:
            root: Root directory of the dataset.
            meta_dir: Path to the chexpert metadata file.
            mask_dir: Path to the directory to cache the segmentation masks.
            cache_dir: Path to the ChestXDet model.
            **kwargs: Additional arguments to pass to ChexpertDataset
        """
        super().__init__(
            root=root,
            meta_dir=meta_dir,
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

    def __getitem__(self, index: int):
        img, meta = super(SegmentationChexpertDataset, self).__getitem__(index)
        img_path = meta["local_img_path"]

        return img, self._get_segmentation(img_path, img), meta

    @staticmethod
    def collate(batch):
        images, masks, metas = zip(*batch)

        images = torch.stack(images)
        masks = torch.stack(masks)

        return images, masks, metas


def iter_as_numpy(ds_loader):
    """Convert a DataLoader to a numpy array."""
    for images, masks, meta in ds_loader:
        yield images.cpu().numpy(), masks.cpu().numpy(), meta
