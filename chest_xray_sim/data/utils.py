import typing

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2


def get_chexpert_transform(
    size: int = 512, dtype: torch.dtype = torch.float32
) -> v2.Compose:
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.PILToTensor(),
            v2.Resize(size),
            v2.CenterCrop(size),
            v2.ToDtype(dtype, scale=True),
        ]
    )


def read_image(path: str, size: int = 512) -> torch.Tensor:
    """
    Assumes image is in grayscale format, 8-bit unsigned integer.
    Other formats may result in unexpected behavior.
    """
    img = Image.open(path)
    img = img.convert("L")  # Convert to grayscale

    t = get_chexpert_transform(size)
    return t(img)


def remove_color_channels(x: torch.Tensor) -> torch.Tensor:
    # Assuming x is a 3D tensor (C, H, W)
    if x.ndim == 3:
        # Convert to grayscale by averaging the color channels
        x = x.mean(dim=0, keepdim=False)
    return x


def filter_key(df: pd.DataFrame, key: str, value: typing.Any) -> pd.DataFrame:
    if value is None:
        return df
    return df[df[key] == value]
