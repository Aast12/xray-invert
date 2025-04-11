import cv2
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from enum import Enum


class ImageDomain(Enum):
    NORMALIZED = 0
    BIT_8 = 1
    BIT_16 = 2
    BIT_32 = 3


DOMAIN_NP_DTYPES = {
    ImageDomain.NORMALIZED: np.float32,
    ImageDomain.BIT_8: np.uint8,
    ImageDomain.BIT_16: np.uint16,
    ImageDomain.BIT_32: np.uint32,
}

DOMAIN_JNP_DTYPES = {
    ImageDomain.NORMALIZED: jnp.float32,
    ImageDomain.BIT_8: jnp.uint8,
    ImageDomain.BIT_16: jnp.uint16,
    ImageDomain.BIT_32: jnp.uint32,
}


def read_image(image_path, dtype=np.float32):
    if dtype not in [np.float16, np.float32, np.float64]:
        raise ValueError("Output dtype must be float16, float32, or float64")

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Get the maximum value based on the image's dtype
    dtype_info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else np.finfo(image.dtype)
    max_value = dtype_info.max
    normalized_image = image.astype(dtype) / dtype(max_value)

    # Ensure all values are in [0, 1] range
    normalized_image = np.clip(normalized_image, 0., 1.)

    return normalized_image


def show_image_dist(img, hist_ax=None, img_ax=None, bins=256):
    if hist_ax is None and img_ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        hist_ax = ax[0]
        img_ax = ax[1]

    hist_ax.hist(img.ravel(), bins=bins)
    img_ax.imshow(img, cmap="gray")
