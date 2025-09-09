import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image

from chest_xray_sim.data.segmentation_dataset import (
    SegmentationChexpertDataset,
    iter_as_numpy,
)


class TestSegmentationChexpertDataset:
    """Test cases for SegmentationChexpertDataset class."""

    def test_constructor_with_mask_dir(self, mock_dataset_structure):
        """Test constructor with mask directory specified."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation"
        ) as mock_seg_model:
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
                cache_dir="/fake/cache",
                split="train",
                image_size=64,
            )

            assert os.path.exists(mask_dir)

            mock_seg_model.assert_called_once_with(cache_dir="/fake/cache")

            assert hasattr(dataset, "mask_dir")
            assert dataset.mask_dir == str(mask_dir)

    def test_constructor_without_mask_dir(self, mock_dataset_structure):
        """Test constructor without mask directory (masks not cached)."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation"
        ) as mock_seg_model:
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=None,
                cache_dir=None,
                split="train",
                image_size=64,
            )

            mock_seg_model.assert_called_once_with(cache_dir=None)

            assert dataset.mask_dir is None

    def test_get_mask_path_with_mask_dir(self, mock_dataset_structure):
        """Test _get_mask_path method with mask directory."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        with patch("chest_xray_sim.data.segmentation_dataset.ChestSegmentation"):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
            )

            img_path = str(temp_dir / "train" / "dummy_class" / "test_image.png")
            mask_path = dataset._get_mask_path(img_path)

            expected_path = os.path.join(
                mask_dir, "train", "dummy_class", "test_image_masks.pt"
            )
            assert mask_path == expected_path

    def test_get_mask_path_without_mask_dir(self, mock_dataset_structure):
        """Test _get_mask_path method without mask directory."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]

        with patch("chest_xray_sim.data.segmentation_dataset.ChestSegmentation"):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=None,
            )

            img_path = str(temp_dir / "train" / "dummy_class" / "test_image.png")
            mask_path = dataset._get_mask_path(img_path)

            assert mask_path is None

    def test_get_segmentation_with_cached_mask(self, mock_dataset_structure):
        """Test _get_segmentation method when mask is already cached."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        with patch("chest_xray_sim.data.segmentation_dataset.ChestSegmentation"):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
            )

            img_path = str(temp_dir / "train" / "dummy_class" / "test_image.png")
            mask_path = dataset._get_mask_path(img_path)

            assert mask_path is not None
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)

            mock_mask = torch.randn(5, 64, 64)
            torch.save(mock_mask, mask_path)

            test_image = torch.randn(1, 64, 64)
            result = dataset._get_segmentation(img_path, test_image)

            assert torch.equal(result, mock_mask)

    def test_get_segmentation_without_cached_mask(self, mock_dataset_structure):
        """Test _get_segmentation method when mask is not cached."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        mock_seg_model = MagicMock()
        mock_prediction = torch.randn(5, 64, 64)
        mock_seg_model.return_value = [mock_prediction]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation",
            return_value=mock_seg_model,
        ):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
            )

            img_path = str(temp_dir / "train" / "dummy_class" / "test_image.png")
            test_image = torch.randn(1, 64, 64)
            result = dataset._get_segmentation(img_path, test_image)

            mock_seg_model.assert_called_once_with(test_image)

            assert torch.equal(result, mock_prediction)

    def test_get_segmentation_saves_mask_to_cache(self, mock_dataset_structure):
        """Test that _get_segmentation saves generated mask to cache."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        # Create a mock segmentation model
        mock_seg_model = MagicMock()
        mock_prediction = torch.randn(5, 64, 64)
        mock_seg_model.return_value = [mock_prediction]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation",
            return_value=mock_seg_model,
        ):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
            )

            img_path = str(temp_dir / "train" / "dummy_class" / "test_image.png")
            test_image = torch.randn(1, 64, 64)
            _ = dataset._get_segmentation(img_path, test_image)

            mask_path = dataset._get_mask_path(img_path)
            assert mask_path is not None
            assert os.path.exists(mask_path)

            saved_mask = torch.load(mask_path)
            assert torch.equal(saved_mask, mock_prediction)

    def test_getitem_returns_correct_format(self, mock_dataset_structure):
        """Test that __getitem__ returns correct tuple format (image, mask, meta)."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        mock_seg_model = MagicMock()
        mock_prediction = torch.randn(5, 64, 64)
        mock_seg_model.return_value = [mock_prediction]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation",
            return_value=mock_seg_model,
        ):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
                split="train",
                image_size=64,
            )

            image, mask, meta = dataset[0]

            # Check types and shapes
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert isinstance(meta, dict)

            # Check image shape (should be grayscale)
            assert image.shape == (1, 64, 64)

            # Check mask shape (should be multi-channel)
            assert mask.shape == (5, 64, 64)

            # Check metadata contains expected keys
            assert "local_img_path" in meta
            assert "img_id" in meta

    def test_collate_method(self, mock_dataset_structure):
        """Test static collate method for batching."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        with patch("chest_xray_sim.data.segmentation_dataset.ChestSegmentation"):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
                split="train",
                image_size=64,
            )

            # Create mock batch data
            batch = [
                (torch.randn(1, 64, 64), torch.randn(5, 64, 64), {"img_id": "1"}),
                (torch.randn(1, 64, 64), torch.randn(5, 64, 64), {"img_id": "2"}),
            ]


            # todo: fix typing hints
            images, masks, metas = SegmentationChexpertDataset.collate(batch)  # pyright: ignore[reportArgumentType]

            assert images.shape == (2, 1, 64, 64)
            assert masks.shape == (2, 5, 64, 64)
            assert len(metas) == 2

            # Check that images and masks are stacked
            assert isinstance(images, torch.Tensor)
            assert isinstance(masks, torch.Tensor)
            assert isinstance(metas, tuple)

    def test_iter_as_numpy_function(self, mock_dataset_structure):
        """Test iter_as_numpy utility function."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        # Create a mock segmentation model
        mock_seg_model = MagicMock()
        mock_prediction = torch.randn(5, 64, 64)
        mock_seg_model.return_value = [mock_prediction]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation",
            return_value=mock_seg_model,
        ):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
                split="train",
                image_size=64,
            )

            # Create a mock DataLoader
            from torch.utils.data import DataLoader

            dataloader = DataLoader(
                dataset, batch_size=2, collate_fn=SegmentationChexpertDataset.collate
            )

            # Test iter_as_numpy
            for images_np, masks_np, meta in iter_as_numpy(dataloader):
                # Check that outputs are numpy arrays
                assert hasattr(images_np, "shape")
                assert hasattr(masks_np, "shape")
                assert isinstance(meta, tuple)
                break  # Just test first batch

    def test_segmentation_model_error_handling(self, mock_dataset_structure):
        """Test error handling when segmentation model returns wrong dimensions."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        # Create a mock segmentation model that returns wrong dimensions
        mock_seg_model = MagicMock()
        mock_prediction = torch.randn(5, 64)  # Wrong: 2D instead of 3D
        mock_seg_model.return_value = [mock_prediction]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation",
            return_value=mock_seg_model,
        ):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
            )

            # Test that assertion error is raised
            img_path = str(temp_dir / "train" / "dummy_class" / "test_image.png")
            test_image = torch.randn(1, 64, 64)

            with pytest.raises(AssertionError) as exc_info:
                dataset._get_segmentation(img_path, test_image)

            assert "Segmentation model returned 2D tensor" in str(exc_info.value)

    def test_integration_with_dataloader(self, mock_dataset_structure):
        """Test integration with PyTorch DataLoader."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]
        mask_dir = mock_dataset_structure["mask_dir"]

        # Create a mock segmentation model
        mock_seg_model = MagicMock()
        mock_prediction = torch.randn(5, 64, 64)
        mock_seg_model.return_value = [mock_prediction]

        with patch(
            "chest_xray_sim.data.segmentation_dataset.ChestSegmentation",
            return_value=mock_seg_model,
        ):
            dataset = SegmentationChexpertDataset(
                root=str(temp_dir),
                meta_dir=str(csv_path),
                mask_dir=str(mask_dir),
                split="train",
                image_size=64,
            )

            # Create DataLoader
            from torch.utils.data import DataLoader

            dataloader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=SegmentationChexpertDataset.collate,
                shuffle=False,
            )

            # Test iteration
            batch_count = 0
            for images, masks, metas in dataloader:
                batch_count += 1

                # Check batch shapes
                assert images.shape[0] <= 2  # batch_size
                assert images.shape[1:] == (1, 64, 64)  # grayscale, 64x64
                assert masks.shape[1:] == (5, 64, 64)  # 5 channels, 64x64
                assert len(metas) == images.shape[0]

                if batch_count >= 2:  # Test a few batches
                    break

            assert batch_count > 0


@pytest.fixture
def mock_dataset_structure(tmp_path_factory):
    """Create a mock file structure for testing SegmentationChexpertDataset.

    This fixture creates the same structure as the existing test_chexpert.py fixture
    but is specifically tailored for segmentation dataset testing.
    """
    base_dir = tmp_path_factory.mktemp("mock_seg_ds")

    # Create directory structure
    data_dir = base_dir / "data"
    meta_dir = base_dir / "metadata"
    mask_dir = base_dir / "masks"

    for split in ["train", "valid"]:
        # ImageFolder expects at least one class subdirectory
        (data_dir / split / "dummy_class").mkdir(parents=True, exist_ok=True)
        (mask_dir / split).mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Create sample images and metadata
    image_data = []
    for i, (split, view, orientation) in enumerate(
        [("train", "Frontal", "AP"), ("valid", "Frontal", "AP")]
    ):
        img_filename = (
            f"patient{i:03d}_{split}_{view.lower()}_{orientation.lower()}.png"
        )
        img_path = data_dir / split / "dummy_class" / img_filename

        # Create simple test image
        img = Image.new("L", (64, 64), color=128)
        img.save(img_path)

        image_data.append(
            {
                # Include class folder in the relative path, to match ImageFolder layout
                "path_to_image": f"{split}/dummy_class/{img_filename}",
                "deid_patient_id": f"patient{i:03d}",
                "frontal_lateral": view,
                "ap_pa": orientation,
                "split": split,
                "abs_img_path": str(img_path),
            }
        )

    # Create CSV metadata
    csv_path = meta_dir / "metadata.csv"
    pd.DataFrame(image_data).to_csv(csv_path, index=False)

    return {
        "data_dir": data_dir,
        "meta_dir": meta_dir,
        "mask_dir": mask_dir,
        "csv_path": csv_path,
    }


pytestmark = [
    pytest.mark.data_loading,
    pytest.mark.segmentation,
]
