import matplotlib.pyplot as plt
import cv2
import jax.numpy as jnp

import src.main as m

import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--n_steps", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--raw", type=str, default="data/conventional_processed.tif")
parser.add_argument(
    "--processed", type=str, default="data/conventional_transmissionmap.tif"
)


if __name__ == "__main__":
    args = parser.parse_args()

    conventional = cv2.imread(args.raw, cv2.IMREAD_UNCHANGED)
    conventional_processed = cv2.imread(args.processed, cv2.IMREAD_UNCHANGED)

    processed = (
        conventional_processed.astype(jnp.float32) / conventional_processed.max()
    )

    pprint('Parser arguments: ')
    pprint(args.__dict__)

    recovered, losses = m.optimize(processed, lr=args.lr, n_steps=args.n_steps)


    print(
        "Recovered weights:",
        recovered.w_lookup_table,
        recovered.w_multiscale_processing,
    )

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(conventional_processed, cmap="gray")
    ax[1].imshow(recovered.image, cmap="gray")
    ax[2].imshow(conventional, cmap="gray")

    plt.show()
