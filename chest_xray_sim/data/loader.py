import cv2
import pandas as pd

from cv2.typing import MatLike
from typing import TypedDict
import os


class ChexpertMeta(TypedDict):
    image_path: str
    patient_id: str
    image: MatLike
    study: str
    image_label: str


def load_chexpert(
    root_dir: str,
    df_meta_path: str | None = None,
    img_dir: str | None = None,
    chexpert_version: str = "240401",
    limit: int | None = None,
) -> list[ChexpertMeta]:
    df_meta_path = (
        df_meta_path
        if df_meta_path is not None
        else os.path.join(root_dir, f"df_chexpert_plus_{chexpert_version}.csv")
    )
    img_dir = img_dir if img_dir is not None else os.path.join(root_dir, "PNG")

    df_meta_path = os.path.abspath(df_meta_path)
    img_dir = os.path.abspath(img_dir)

    df_meta = pd.read_csv(df_meta_path)
    df_meta = df_meta[df_meta["frontal_lateral"] == "Frontal"]

    images: list[ChexpertMeta] = []

    for idx, row in df_meta.iterrows():
        img_path = f"{img_dir}/{row['path_to_image']}"
        patient_id = str(row["deid_patient_id"])

        if not os.path.exists(img_path):
            # chexpert csv reporting jpgs instead of pngs
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                continue

        if limit and len(images) >= limit:
            break

        img_label = img_path.split("/")[-1]
        study = img_path[img_path.find("study") :][:6]

        images.append(
            {
                "image": cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255.0,
                "image_path": img_path,
                "patient_id": patient_id,
                "study": study,
                "image_label": img_label,
            }
        )

    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    print("args:", args)

    root_path = args.root_path
    meta_path = args.meta_path
    img_dir = args.img_dir

    images = load_chexpert(root_path, meta_path, img_dir, limit=10)

    import matplotlib.pyplot as plt

    plt.imshow(images[len(images) // 2]["image"], cmap="gray")
    plt.show()
