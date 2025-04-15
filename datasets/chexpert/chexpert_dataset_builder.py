"""Chexpert Plus dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os

from pathlib import Path

import csv
import typing


_DATA_DIR = "PNG/PNG"
_METADATA_PATH = "df_chexpert_plus_240401.csv"
_TRAIN_DIR = os.path.join(_DATA_DIR, "train")
_VALID_DIR = os.path.join(_DATA_DIR, "valid")


class ChexpertPlus(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for chexpert dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    You must register and agree to user agreement on the dataset page:
    https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1
    Afterwards, you have to put the dataset contents in the
    manual_dir. It should contain a metadata file (e.g. df_chexpert_plus_240401.csv), and PNG 
    folder with subdirectories: train/ and valid/.

    As of April 2025, DICOM files are not available to download and the metadata file mistakenly
    records image paths as jpg instead of png. 
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(chexpert): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "deid_patient_id": tfds.features.Text(),
                    "study": tfds.features.Text(),
                    "image": tfds.features.Image(),
                    "image_view": tfds.features.ClassLabel(
                        names=["Frontal", "Lateral"]
                    ),
                    "ap_pa": tfds.features.ClassLabel(names=["PA", "AP", ""]),
                }
            ),
            supervised_keys=None,
            homepage="https://aimi.stanford.edu/datasets/chexpert-plus",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        data_path = os.path.join(path, _DATA_DIR)
        train_path = os.path.join(path, _TRAIN_DIR)
        valid_path = os.path.join(path, _VALID_DIR)
        metadata_path = os.path.join(path, _METADATA_PATH)

        if not tf.io.gfile.exists(train_path) or not tf.io.gfile.exists(valid_path):
            raise AssertionError(
                "Chexpert Plus dataset must be downloaded and put into %s." % path
            )

        return {
            "train": self._generate_examples(
                "train", Path(data_path), Path(metadata_path)
            ),
            "valid": self._generate_examples(
                "valid", Path(data_path), Path(metadata_path)
            ),
        }

    def _generate_examples(
        self,
        split: typing.Literal["train", "valid"],
        data_path: Path,
        metadata_path: Path,
    ):
        with metadata_path.open() as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if row["split"].strip().lower() != split.strip().lower():
                    continue

                name = row["path_to_image"]
                # dataset csv is mistakenly using jpg suffix
                name = name.replace(".jpg", ".png")

                # assumes incomplete chunks
                if not os.path.exists(os.path.join(data_path, name)):
                    continue

                deid_patient_id = row["deid_patient_id"]
                image_view = row["frontal_lateral"]
                ap_pa = row["ap_pa"]

                study = name[name.find("study") :]
                study = study[: study.find("/")]
                yield (
                    name,
                    {
                        "deid_patient_id": deid_patient_id,
                        "image_view": image_view,
                        "study": study,
                        "ap_pa": ap_pa,
                        "image": os.path.join(data_path, name),
                    },
                )
