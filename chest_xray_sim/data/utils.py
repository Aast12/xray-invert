import torch
from PIL import Image
import torchvision
from torchvision.transforms import v2


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


def get_chexpert_transform(
    size: int = 512, dtype: torch.dtype = torch.float32
) -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            # v2.Grayscale(),
            v2.PILToTensor(),
            v2.Resize(size),
            v2.CenterCrop(size),
            v2.ToDtype(dtype, scale=True),
            v2.Lambda(remove_color_channels),
        ]
    )
