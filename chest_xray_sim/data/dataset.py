from jaxtyping import Float
import pandas as pd
import torch
import os
import torchvision

from chest_xray_sim.data.meta import XRayImageMeta, default_get_metadata_from_df, default_get_metadata_from_path
from chest_xray_sim.types import ImageTorch


ItemT = tuple[ImageTorch, XRayImageMeta]
BatchItemT = tuple[ImageTorch, list[XRayImageMeta]]


def is_file_type(path: str, extensions: tuple[str, ...] | None = None) -> bool:
    if extensions is None:
        return True
    return any(path.lower().endswith(ext) for ext in extensions)


class FolderDataset(torchvision.datasets.ImageFolder):
    """
    A dataset class that loads images from a folder structure and extracts metadata from filenames.

    Attributes:
        root_dir: root directory to scan for images
        extensions: valid image file extensions, set to None to allow all files
    """

    root_dir: str
    extensions: tuple[str, ...] | None

    def __init__(
        self,
        root: str,
        extensions: tuple[str, ...] | None = (".png", ".jpg", ".jpeg"),
        **kwargs,
    ):
        self.root_dir = root
        self.extensions = extensions

        is_valid_file = None if extensions is None else self.is_valid_file

        super().__init__(
            root=root, is_valid_file=is_valid_file, allow_empty=True, **kwargs
        )

    def is_valid_file(self, path: str) -> bool:
        """
        Check if a file is a valid image file based on its extension.

        Args:
            path: Path to the file

        Returns:
            True if the file has a valid image extension, False otherwise
        """
        return is_file_type(path, self.extensions)

    def get_meta_data(self, abs_path: str) -> XRayImageMeta:
        """Extract metadata from filename.

        This is a base implementation that extracts minimal metadata.
        Subclasses should override this method to implement custom extraction logic.

        Args:
            filepath: Full path to the image file

        Returns:
            Dictionary containing extracted metadata
        """
        return default_get_metadata_from_path(abs_path)

    def __getitem__(self, index: int) -> ItemT:
        original_tuple = super(FolderDataset, self).__getitem__(index)

        img, _ = original_tuple
        img_path = self.imgs[index][0]

        meta = self.get_meta_data(img_path)

        return img, meta

    @staticmethod
    def collate(batch: list[ItemT]) -> BatchItemT:
        images, meta = zip(*batch)

        images = torch.stack(images)

        return images, list(meta)


class MetaDataset(FolderDataset):
    """
    A dataset class that loads images from a folder structure and associates them with metadata from a CSV file.

    Attributes:
        metadata: DataFrame containing metadata for the images
        root_dir: root directory to scan for images
    """

    metadata: pd.DataFrame
    root_dir: str

    def __init__(
        self,
        root: str,
        meta_dir: str,
        **kwargs,
    ):
        self.root_dir = root
        self.metadata = pd.read_csv(meta_dir)
        self.metadata = self.filter_records(self.metadata)

        super().__init__(
            root=root, extensions=None, **kwargs
        )

    def filter_records(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_meta_data(self, abs_path: str) -> XRayImageMeta:
        return default_get_metadata_from_df(self.metadata, abs_path)
