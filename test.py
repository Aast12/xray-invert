import matplotlib.pyplot as plt
import cv2
import jax.numpy as jnp
import jax
import numpy as np

import src.main as m

import argparse
from pprint import pprint
import os

from src.processings import (
    LookupTableWeights,
    MultiscaleProcessingWeights,
    initialize_weights,
    range_mapping,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n_steps", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--raw", type=str, default="data/conventional_transmissionmap.tif")
parser.add_argument("--processed", type=str, default="data/conventional_processed.tif")


from jax import jacfwd, jacrev


def is_differentiable():
    test_image = jnp.ones((64, 64))
    test_weights = initialize_weights(test_image)
    # test_weights = (
    #     LookupTableWeights(
    #         breakpoints=jnp.array([0.0, 0.5, 1.0]),
    #         values=jnp.array([0.0, 0.5, 1.0]),
    #         partitions=jax.lax.stop_gradient(0.5),
    #     ),
    #     MultiscaleProcessingWeights(
    #         filter_sizes=jax.lax.stop_gradient(jnp.array([3, 9, 27])),
    #         unsharp_weights=jnp.ones(3) / 3,
    #     ),
    # )

    jacobian = jacrev(m.forward)(test_weights)
    print(jacobian)


if __name__ == "__main__":
    args = parser.parse_args()

    is_differentiable()

    conventional = cv2.imread(args.raw, cv2.IMREAD_UNCHANGED)
    conventional_processed = cv2.imread(args.processed, cv2.IMREAD_UNCHANGED)

    processed = (
        conventional_processed.astype(jnp.float32) / conventional_processed.max()
    )

    pprint("Parser arguments: ")
    pprint(args.__dict__)

    recovered, losses = m.optimize(processed, lr=args.lr, n_steps=args.n_steps)

    recovered_processed = m.forward(recovered)

    # print(
    #     "Recovered weights:",
    #     recovered.w_lookup_table,
    #     recovered.w_multiscale_processing,
    # )

    remapped = range_mapping(recovered.image, new_min=0, new_max=255)
    remapped_processed = range_mapping(recovered_processed, new_min=0, new_max=255)

    if not os.path.exists("./outputs/"):
        os.mkdir("./outputs/")

    print("recovered type:", type(recovered.image))
    print("recovered:", recovered.image)
    cv2.imwrite(
        "outputs/recovered.tif", np.array(recovered.image * 255, dtype=np.uint8)
    )
    cv2.imwrite("outputs/remapped.tif", np.array(remapped, dtype=np.uint8))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(conventional, cmap="gray")
    ax[0, 0].set_title("Conventional")
    ax[0, 1].imshow(conventional_processed, cmap="gray")
    ax[0, 1].set_title("Conventional Processed")
    ax[1, 0].imshow(recovered.image, cmap="gray")
    ax[1, 0].set_title("Recovered")
    ax[1, 1].imshow(recovered_processed, cmap="gray")
    ax[1, 1].set_title("Recovered Processed")

    plt.savefig("outputs/comparison.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(recovered.image, cmap="gray")
    ax[1].hist(recovered.image.ravel(), bins=256)

    plt.savefig("outputs/recovered_hist.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(remapped, cmap="gray")
    ax[1].hist(remapped.ravel(), bins=256)
    plt.title("Remapped")

    plt.savefig("outputs/remapped_hist.png")

    diff = conventional_processed - remapped_processed
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(diff, cmap="gray")
    ax[1].boxplot(diff.ravel())
    plt.title("Diff")

    plt.savefig("outputs/diff_hist.png")



