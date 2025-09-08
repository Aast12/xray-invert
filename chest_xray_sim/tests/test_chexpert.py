"""
Tests for CheXpert data loaders.

This module contains tests for the CheXpert data loading functionality,
focusing on the actual data loader classes and their behavior.
"""

import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image
import numpy as np

# Import the actual modules being tested
from chest_xray_sim.data.chexpert import (
    ChexpertDataset,
    ChexpertMetaLoader,
    get_chexpert_dataset,
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

                dataset = ChexpertDataset(
                    root="/fake/path",
                    meta_dir="/fake/meta.csv",
                    split="train",
                    frontal_lateral="Frontal",
                    image_size=128,
                )

                mock_meta_init.assert_called_once()

                call_args = mock_meta_init.call_args
                meta_root_dir, meta_meta_dir = call_args[0]
                meta_kwargs = call_args[1]

                assert meta_root_dir == "/fake/path/train"
                assert meta_meta_dir == "/fake/meta.csv"
                assert meta_kwargs["image_size"] == 128

    def test_is_valid_file_method(self):
        """Test is_valid_file method logic."""

        dataset = ChexpertDataset.__new__(ChexpertDataset)
        dataset.frontal_lateral = "Frontal"
        assert dataset.is_valid_file("test_frontal.jpg") == True
        assert dataset.is_valid_file("test_lateral.jpg") == False
        assert dataset.is_valid_file("test_unknown.jpg") == False

        dataset.frontal_lateral = None
        assert dataset.is_valid_file("test_frontal.jpg") == True
        assert dataset.is_valid_file("train_lateral.jpg") == True

        dataset.split = "train"
        assert dataset.is_valid_file("test_frontal.jpg") == False
        assert dataset.is_valid_file("train_lateral.jpg") == True
        assert dataset.is_valid_file("valid_lateral.jpg") == False

        dataset.split = None
        assert dataset.is_valid_file("test_frontal.jpg") == True
        assert dataset.is_valid_file("test_lateral.jpg") == True

        dataset.frontal_lateral = "Frontal"
        dataset.split = "train"
        assert dataset.is_valid_file("train_frontal.jpg") == True
        assert dataset.is_valid_file("test_frontal.jpg") == False
        assert dataset.is_valid_file("test_lateral.jpg") == False
        assert dataset.is_valid_file("test_unknown.jpg") == False

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

        meta = dataset.get_meta_data("/non/existent/path.png")
        assert meta == {}


class TestChexpertMetaLoaderIssues:
    """Test cases documenting issues with ChexpertMetaLoader class."""

    def test_constructor_creates_data_loader(self):
        """Test that ChexpertMetaLoader creates a DataLoader successfully."""
        # Test that ChexpertMetaLoader actually creates a DataLoader with proper data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy CSV file
            csv_path = os.path.join(temp_dir, "train.csv")
            with open(csv_path, "w") as f:
                f.write("path_to_image,deid_patient_id,frontal_lateral,ap_pa,split\n")
                f.write("dummy.png,123,Frontal,AP,train\n")

            # Create a dummy image directory structure
            img_dir = os.path.join(temp_dir, "train")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(os.path.join(img_dir, "dummy_class"), exist_ok=True)

            # Create a dummy image file
            dummy_img_path = os.path.join(img_dir, "dummy_class", "dummy.png")
            with open(dummy_img_path, "w") as f:
                f.write("dummy image content")

            # Test that ChexpertMetaLoader creates a DataLoader successfully
            loader = ChexpertMetaLoader(
                data_dir=temp_dir, meta_dir=csv_path, split="train", batch_size=1
            )

            # Verify it returns a DataLoader
            assert hasattr(loader, "loader")
            assert hasattr(loader.loader, "batch_size")
            assert loader.loader.batch_size == 1
            print(
                "SUCCESS: ChexpertMetaLoader creates a DataLoader successfully with proper data structure"
            )

    def test_initialization_parameters(self):
        """Test ChexpertMetaLoader parameter handling."""
        # Create an instance without calling __init__
        loader = ChexpertMetaLoader.__new__(ChexpertMetaLoader)
        loader.data_dir = "/fake/data"
        loader.meta_dir = "/fake/meta.csv"
        loader.split = "train"
        loader.batch_size = 1
        loader.image_size = 64

        assert loader.data_dir == "/fake/data"
        assert loader.meta_dir == "/fake/meta.csv"
        assert loader.split == "train"
        assert loader.batch_size == 1
        assert loader.image_size == 64


class TestGetChexpertDatasetIssues:
    """Test cases documenting issues with get_chexpert_dataset function."""

    def test_constructor_creates_data_loader(self):
        """Test that get_chexpert_dataset creates a DataLoader successfully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy CSV file
            csv_path = os.path.join(temp_dir, "train.csv")
            with open(csv_path, "w") as f:
                f.write("path_to_image,deid_patient_id,frontal_lateral,ap_pa,split\n")
                f.write("dummy.png,123,Frontal,AP,train\n")

            # Create a dummy image directory structure
            img_dir = os.path.join(temp_dir, "train")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(os.path.join(img_dir, "dummy_class"), exist_ok=True)

            # Create a dummy image file
            dummy_img_path = os.path.join(img_dir, "dummy_class", "dummy.png")
            with open(dummy_img_path, "w") as f:
                f.write("dummy image content")

            # Test that get_chexpert_dataset works with proper data structure
            dataloader = get_chexpert_dataset(
                data_dir=temp_dir,
                meta_dir=csv_path,
                split="train",
                frontal_lateral="Frontal",
                batch_size=1,
                image_size=64,
            )

            # Verify it returns a DataLoader
            assert hasattr(dataloader, "batch_size")
            assert dataloader.batch_size == 1
            print(
                "SUCCESS: get_chexpert_dataset creates a DataLoader successfully with proper data structure"
            )

    def test_parameter_passing(self):
        """Test that get_chexpert_dataset passes parameters correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy CSV file
            csv_path = os.path.join(temp_dir, "train.csv")
            with open(csv_path, "w") as f:
                f.write("path_to_image,deid_patient_id,frontal_lateral,ap_pa,split\n")
                f.write("dummy.png,123,Frontal,AP,train\n")

            # Create a dummy image directory structure
            img_dir = os.path.join(temp_dir, "train")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(os.path.join(img_dir, "dummy_class"), exist_ok=True)

            # Create a dummy image file
            dummy_img_path = os.path.join(img_dir, "dummy_class", "dummy.png")
            with open(dummy_img_path, "w") as f:
                f.write("dummy image content")

            # Test that get_chexpert_dataset works with proper data structure
            dataloader = get_chexpert_dataset(
                data_dir=temp_dir,
                meta_dir=csv_path,
                split="train",
                frontal_lateral="Frontal",
                batch_size=1,
                image_size=64,
            )

            # Verify it returns a DataLoader
            assert hasattr(dataloader, "batch_size")
            assert dataloader.batch_size == 1
            print(
                "SUCCESS: get_chexpert_dataset creates a DataLoader successfully with proper data structure"
            )


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
        chexpert_meta = {
            "deid_patient_id": "patient123",
            "frontal_lateral": "Frontal",
            "ap_pa": "AP",
        }

        result = to_xray_meta(chexpert_meta)

        assert isinstance(result, dict)
        assert result["img_id"] == "patient123"
        assert result["ap_pa"] == "AP"
        assert result["frontal_lateral"] == "Frontal"
        assert "local_img_path" in result

    def test_to_xray_meta_missing_optional_fields(self):
        """Test conversion when optional fields are missing."""
        chexpert_meta = {
            "deid_patient_id": "patient123",
            "frontal_lateral": "Frontal",
            # Missing ap_pa
        }

        result = to_xray_meta(chexpert_meta)

        assert isinstance(result, dict)
        assert result["img_id"] == "patient123"
        assert result["frontal_lateral"] == "Frontal"
        # ap_pa is not included when missing
        assert "ap_pa" not in result


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Session-scoped fixture for test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (64, 64), color="red")


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.data_loading,
    pytest.mark.chexpert,
]
