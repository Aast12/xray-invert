import pandas as pd
import pytest
from unittest.mock import patch
import torch
from PIL import Image

from chest_xray_sim.data.chexpert import (
    ChexpertDataset,
    ChexpertMetaLoader,
    to_xray_meta,
)
from chest_xray_sim.data.utils import get_chexpert_transform, filter_key


class TestChexpertDataset:
    """Test cases documenting issues with ChexpertDataset class."""

    def test_constructor_requires_csv_file(self):
        """Test that ChexpertDataset constructor requires a real CSV file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ChexpertDataset(root="/fake/path", meta_dir="/fake/meta.csv", split="train")

        assert "No such file or directory" in str(exc_info.value)
        assert "/fake/meta.csv" in str(exc_info.value)

    def test_constructor_parameter_passing(self):
        """Test that ChexpertDataset passes parameters correctly to parent classes."""
        with patch(
            "chest_xray_sim.data.dataset.MetaDataset.__init__"
        ) as mock_meta_init:
            mock_meta_init.return_value = None

            with patch("pandas.read_csv") as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame()

                ChexpertDataset(
                    root="/fake/path",
                    meta_dir="/fake/meta.csv",
                    split="train",
                    frontal_lateral="Frontal",
                    image_size=128,
                )

                mock_meta_init.assert_called_once()

                call_args = mock_meta_init.call_args
                meta_root_dir, meta_meta_dir = call_args[0]

                assert meta_root_dir == "/fake/path/train"
                assert meta_meta_dir == "/fake/meta.csv"

    def test_is_valid_file_method(self):
        """Test is_valid_file method logic."""

        dataset = ChexpertDataset.__new__(ChexpertDataset)
        dataset.frontal_lateral = "Frontal"
        assert dataset.is_valid_file("test_frontal.jpg")
        assert not dataset.is_valid_file("test_lateral.jpg")
        assert not dataset.is_valid_file("test_unknown.jpg")

        dataset.frontal_lateral = None
        assert dataset.is_valid_file("test_frontal.jpg")
        assert dataset.is_valid_file("train_lateral.jpg")

        dataset.split = "train"
        assert not dataset.is_valid_file("test_frontal.jpg")
        assert dataset.is_valid_file("train_lateral.jpg")
        assert not dataset.is_valid_file("valid_lateral.jpg")

        dataset.split = None
        assert dataset.is_valid_file("test_frontal.jpg")
        assert dataset.is_valid_file("test_lateral.jpg")

        dataset.frontal_lateral = "Frontal"
        dataset.split = "train"
        assert dataset.is_valid_file("train_frontal.jpg")
        assert not dataset.is_valid_file("test_frontal.jpg")
        assert not dataset.is_valid_file("test_lateral.jpg")
        assert not dataset.is_valid_file("test_unknown.jpg")

    def test_successful_get_meta_data(self):
        """Test successful get_meta_data method."""
        dataset = ChexpertDataset.__new__(ChexpertDataset)
        dataset.metadata = pd.DataFrame(
            {
                "abs_img_path": ["/fake/path/test.jpg"],
                "deid_patient_id": ["12345"],
                "frontal_lateral": ["Frontal"],
                "ap_pa": ["AP"],
                "split": ["train"],
            }
        )
        meta = dataset.get_meta_data("/fake/path/test.jpg")
        assert meta == {
            "img_id": "12345",
            "local_img_path": "/fake/path/test.jpg",
            "frontal_lateral": "Frontal",
            "ap_pa": "AP",
        }

    def test_get_meta_data_not_found(self):
        """Test get_meta_data with non-existent path."""
        dataset = ChexpertDataset.__new__(ChexpertDataset)

        dataset.metadata = pd.DataFrame(columns=["abs_img_path"])

        with pytest.raises(ValueError) as exc_info:
            dataset.get_meta_data("/non/existent/path.png")

        assert "ChexpertMeta must contain deid_patient_id and abs_img_path" in str(
            exc_info.value
        )


class TestChexpertMetaLoader:
    """Test cases documenting issues with ChexpertMetaLoader class."""

    def test_constructor_creates_data_loader(self, mock_dataset_structure):
        """Test that ChexpertMetaLoader creates a DataLoader successfully."""

        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]

        loader = ChexpertMetaLoader(
            data_dir=str(temp_dir), meta_dir=str(csv_path), split="train", batch_size=1
        )

        assert hasattr(loader, "loader")
        assert hasattr(loader.loader, "batch_size")
        assert loader.loader.batch_size == 1

    def test_iteration(self, mock_dataset_structure):
        """Test ChexpertMetaLoader iteration."""

        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]

        loader = ChexpertMetaLoader(
            data_dir=str(temp_dir),
            meta_dir=str(csv_path),
            split=None,
            batch_size=2,
            image_size=64,
        )

        assert hasattr(loader, "loader")
        assert hasattr(loader.loader, "batch_size")

        item_count = 0
        for batch in loader:
            for img, meta in zip(batch[0], batch[1]):
                item_count += 1
                assert img.shape == (1, 64, 64)
                assert str(temp_dir) in meta["local_img_path"]

        assert item_count == 2


class TestGetChexpertDatasetIssues:
    """Test cases documenting issues with get_chexpert_dataset function."""

    def test_constructor_creates_data_loader(self, mock_dataset_structure):
        """Test that get_chexpert_dataset creates a DataLoader successfully."""
        temp_dir = mock_dataset_structure["data_dir"]
        csv_path = mock_dataset_structure["csv_path"]

        dataloader = ChexpertMetaLoader(
            data_dir=str(temp_dir),
            meta_dir=str(csv_path),
            split="train",
            frontal_lateral="Frontal",
            batch_size=2,
            image_size=64,
        )

        assert hasattr(dataloader, "loader")
        assert hasattr(dataloader.loader, "batch_size")
        assert dataloader.loader.batch_size == 2


class TestUtilsFunctions:
    """Test cases for utility functions."""

    def test_get_chexpert_transform(self):
        """Test get_chexpert_transform function."""
        transform = get_chexpert_transform(size=64)

        assert transform is not None
        assert hasattr(transform, "transforms")
        assert len(transform.transforms) > 0

        # Test with actual image
        sample_img = Image.new("RGB", (128, 128), color="red")
        transformed = transform(sample_img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 1  # Grayscale
        assert transformed.shape[1] == 64  # Resized
        assert transformed.shape[2] == 64  # Resized

    def test_filter_key_function(self):
        """Test filter_key function."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["x", "y", "x", "z", "x"]})

        result = filter_key(df, "B", "x")

        assert len(result) == 3
        assert all(result["B"] == "x")
        assert list(result["A"]) == [1, 3, 5]


class TestToXrayMeta:
    """Test cases for to_xray_meta function."""

    def test_to_xray_meta_conversion(self):
        """Test conversion from ChexpertMeta to XRayImageMeta."""
        from chest_xray_sim.data.chexpert import ChexpertMeta

        chexpert_meta: ChexpertMeta = {
            "abs_img_path": "/fake/path/test.jpg",
            "deid_patient_id": "patient123",
            "frontal_lateral": "Frontal",
            "ap_pa": "AP",
        }

        result = to_xray_meta(chexpert_meta)

        assert isinstance(result, dict)
        assert result.get("img_id") == "patient123"
        assert result.get("ap_pa") == "AP"
        assert result.get("frontal_lateral") == "Frontal"
        assert "local_img_path" in result

    def test_to_xray_meta_missing_optional_fields(self):
        """Test conversion when optional fields are missing."""
        from chest_xray_sim.data.chexpert import ChexpertMeta

        chexpert_meta: ChexpertMeta = {
            "abs_img_path": "/fake/path/test.jpg",
            "deid_patient_id": "patient123",
            "frontal_lateral": "Frontal",
            # Missing ap_pa
        }

        result = to_xray_meta(chexpert_meta)

        assert isinstance(result, dict)
        assert result.get("img_id") == "patient123"
        assert result.get("frontal_lateral") == "Frontal"
        # ap_pa is not included when missing
        assert "ap_pa" not in result


@pytest.fixture
def mock_dataset_structure(tmp_path_factory):
    """Create a mock file structure for testing all dataset variants.

    Uses tmp_path_factory to avoid premature cleanup while DataLoader reads files.
    """
    base_dir = tmp_path_factory.mktemp("mock_ds")

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

        # Create mask file matching dataset naming: mask_dir/<split>/<image_stem>_masks.pt
        mask_rel_dir = mask_dir / split
        mask_rel_dir.mkdir(parents=True, exist_ok=True)
        mask_stem = img_filename.replace(".png", "")
        mask_path = mask_rel_dir / f"{mask_stem}_masks.pt"

        # Build simple non-zero multi-channel masks (5 channels)
        masks = torch.zeros((5, 64, 64), dtype=torch.float32)
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        left_lung = ((xx - 20) ** 2 + (yy - 32) ** 2) < 14**2
        right_lung = ((xx - 44) ** 2 + (yy - 32) ** 2) < 14**2
        heart = ((xx - 32) ** 2 + (yy - 36) ** 2) < 8**2
        for r in range(8, 64, 12):
            masks[4, r : r + 2, :] = 1.0
        masks[1, left_lung] = 1.0
        masks[2, right_lung] = 1.0
        masks[3, heart] = 1.0
        torch.save(masks, mask_path)

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
    pytest.mark.chexpert,
]
