import os
import re
import typing
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from chest_xray_sim.data.utils import get_chexpert_transform


class ChexpertMeta(typing.TypedDict):
    """
    Metadata keys to include in dataset entries.
    """
    abs_img_path: str
    deid_patient_id: str
    frontal_lateral: typing.NotRequired[str]
    ap_pa: typing.NotRequired[str]


ImageT = torch.Tensor
ItemT = tuple[ImageT, ChexpertMeta]
CollateItemT = list[ItemT]
BatchItemT = tuple[ImageT, list[ChexpertMeta]]


class FolderDataset(torchvision.datasets.ImageFolder):
    """Dataset that reads all images from a folder and extracts metadata from filenames.
    
    This class provides a base implementation that can be extended by subclasses
    to implement custom metadata extraction logic.
    """
    
    def __init__(
        self,
        root: str,
        extensions: tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
        transform=None,
        **kwargs,
    ):
        """Initialize the dataset.
        
        Args:
            root: Root directory containing images
            extensions: Valid image file extensions
            transform: Image transformation to apply
            **kwargs: Additional arguments passed to ImageFolder
        """
        self.root_dir = root
        self.extensions = extensions
        
        # Create a custom is_valid_file function
        def is_valid_file(path: str) -> bool:
            return any(path.lower().endswith(ext) for ext in extensions)
        
        super().__init__(
            root=root,
            transform=transform,
            is_valid_file=is_valid_file,
            **kwargs
        )
    
    def extract_metadata_from_filename(self, filepath: str) -> ChexpertMeta:
        """Extract CheXpert metadata from filename.
        
        This is a base implementation that extracts minimal metadata.
        Subclasses should override this method to implement custom extraction logic.
        
        Args:
            filepath: Full path to the image file
            
        Returns:
            Dictionary containing extracted metadata
        """
        filename = os.path.basename(filepath)
        
        # Default implementation - just extract basic info
        meta: ChexpertMeta = {
            'abs_img_path': filepath,
            'deid_patient_id': 'unknown',
        }
        
        # Try to extract frontal/lateral from filename
        if 'frontal' in filename.lower():
            meta['frontal_lateral'] = 'Frontal'
        elif 'lateral' in filename.lower():
            meta['frontal_lateral'] = 'Lateral'
            
        # Try to extract AP/PA from filename
        if '_ap_' in filename.lower() or '_ap.' in filename.lower():
            meta['ap_pa'] = 'AP'
        elif '_pa_' in filename.lower() or '_pa.' in filename.lower():
            meta['ap_pa'] = 'PA'
            
        return meta
    
    def __getitem__(self, index: int) -> ItemT:
        """Get an item from the dataset.
        
        Args:
            index: Index of the item
            
        Returns:
            Tuple of (image, metadata)
        """
        # Get the original image and label
        img, _ = super().__getitem__(index)
        
        # Get the image path
        img_path, _ = self.imgs[index]
        
        # Extract metadata from filename
        meta = self.extract_metadata_from_filename(img_path)
        
        return img, meta
    
    @staticmethod
    def collate(batch: CollateItemT) -> BatchItemT:
        """Collate function for DataLoader.
        
        Args:
            batch: List of (image, metadata) tuples
            
        Returns:
            Tuple of (stacked images, list of metadata)
        """
        images, meta = zip(*batch)
        images = torch.stack(images)
        return images, list(meta)


class ChexpertFolderDataset(FolderDataset):
    """Dataset for CheXpert images with custom filename parsing.
    
    This implementation assumes CheXpert filenames follow patterns like:
    - patient{id}_frontal.png
    - patient{id}_lateral.png
    - patient{id}_view{view}_frontal.png
    - CheXpert-v1.0/train/patient{id}/study{study}/view{view}_frontal.png
    """
    
    def extract_metadata_from_filename(self, filepath: str) -> ChexpertMeta:
        """Extract CheXpert metadata from filename using common patterns."""
        filename = os.path.basename(filepath)
        full_path = Path(filepath)
        
        meta: ChexpertMeta = {
            'abs_img_path': filepath,
            'deid_patient_id': 'unknown',
        }
        
        # Try to extract patient ID from various patterns
        # Pattern 1: patient{id}_
        patient_match = re.search(r'patient(\d+)', str(full_path))
        if patient_match:
            meta['deid_patient_id'] = f"patient{patient_match.group(1)}"
        
        # Pattern 2: Just a numeric ID at the start
        id_match = re.match(r'^(\d+)_', filename)
        if id_match and meta['deid_patient_id'] == 'unknown':
            meta['deid_patient_id'] = id_match.group(1)
        
        # Extract view information
        filename_lower = filename.lower()
        if 'frontal' in filename_lower:
            meta['frontal_lateral'] = 'Frontal'
        elif 'lateral' in filename_lower:
            meta['frontal_lateral'] = 'Lateral'
        
        # Extract AP/PA information
        if '_ap_' in filename_lower or '_ap.' in filename_lower or 'ap_' in filename_lower:
            meta['ap_pa'] = 'AP'
        elif '_pa_' in filename_lower or '_pa.' in filename_lower or 'pa_' in filename_lower:
            meta['ap_pa'] = 'PA'
        
        return meta


def get_folder_dataset(
    data_dir: str,
    batch_size: int = 12,
    transform_size: int = 512,
    dataset_class: type[FolderDataset] = FolderDataset,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for images in a folder.
    
    Args:
        data_dir: Directory containing images
        batch_size: Batch size for DataLoader
        transform_size: Size for image transformation
        dataset_class: Dataset class to use (default: FolderDataset)
        **kwargs: Additional arguments passed to dataset class
        
    Returns:
        DataLoader instance
    """
    ds = dataset_class(
        root=data_dir,
        transform=get_chexpert_transform(transform_size),
        **kwargs,
    )
    
    ds_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_class.collate,
    )
    
    return ds_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example usage with base FolderDataset
    loader = get_folder_dataset(
        data_dir="/path/to/images",
        batch_size=4,
    )
    
    # Example usage with ChexpertFolderDataset
    chexpert_loader = get_folder_dataset(
        data_dir="/path/to/chexpert/images",
        batch_size=4,
        dataset_class=ChexpertFolderDataset,
    )
    
    # Get a batch
    batch = next(iter(loader))
    imgs, metas = batch
    
    # Display first 3 images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(min(3, len(imgs))):
        axes[i].imshow(imgs[i][0], cmap='gray')
        axes[i].set_title(f"Patient: {metas[i]['deid_patient_id']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()