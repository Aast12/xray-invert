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

VAR_2 = 2


def read_image(path, mode=cv2.IMREAD_UNCHANGED, domain=ImageDomain.NORMALIZED):
    img = cv2.imread(path, mode)

    dtype = DOMAIN_NP_DTYPES[domain]
    img = np.array(img, dtype=np.float64)

    img = img / img.max()
    if domain == ImageDomain.NORMALIZED:
        return img

    bit_factor = 2 ** (8 * (DOMAIN_NP_DTYPES[domain]().itemsize)) - 1
    return (img * bit_factor).astype(dtype)


def show_image_dist(img, hist_ax=None, img_ax=None, bins=256):
    if hist_ax is None and img_ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        hist_ax = ax[0]
        img_ax = ax[1]

    hist_ax.hist(img.ravel(), bins=bins)
    img_ax.imshow(img, cmap="gray")
