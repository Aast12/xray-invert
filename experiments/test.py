import matplotlib.pyplot as plt
import numpy as np

from chest_xray_sim.data.utils import read_image

import sys

if __name__ == "__main__":
    p_tm = [
        "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
        "/Volumes/T7/projs/thesis/data/conventional_transmissionmap.tif",
    ]
    p_smooth_samples = [
        "/Volumes/T7/projs/thesis/outputs/v87ejb52/patient00007_view1_frontal.png",
        "/Volumes/T7/projs/thesis/outputs/v87ejb52/patient00015_view1_frontal.png",
        "/Volumes/T7/projs/thesis/outputs/v87ejb52/patient00049_view1_frontal.png",
    ]

    p_non_smooth_samples = [
        "/Volumes/T7/projs/thesis/outputs/9qrkq9f1/patient00007_view1_frontal.png",
        "/Volumes/T7/projs/thesis/outputs/9qrkq9f1/patient00015_view1_frontal.png",
        "/Volumes/T7/projs/thesis/outputs/9qrkq9f1/patient00049_view1_frontal.png",
    ]

    p_real_images = [
        "/Volumes/T7/datasets/chexpert_plus/PNG/PNG/train/patient00007/study1/view1_frontal.png",
        "/Volumes/T7/datasets/chexpert_plus/PNG/PNG/train/patient00015/study1/view1_frontal.png",
        "/Volumes/T7/datasets/chexpert_plus/PNG/PNG/train/patient00049/study1/view1_frontal.png",
    ]
    unproc_img = [read_image(p).squeeze(0) for p in p_tm]

    smooth_images = [read_image(p).squeeze(0) for p in p_smooth_samples]
    p_non_smooth_imags = [read_image(p).squeeze(0) for p in p_non_smooth_samples]
    real_images = [read_image(p).squeeze(0) for p in p_real_images]

    # make naive transmission maps
    naive_real_images = [1 - im for im in real_images]
    # adjust dynamic range to get up to 0.4
    # naive_real_images = [(im / 0.4).clip(0, 1) for im in naive_real_images]

    def smoothing(x):
        return (np.diff(x, axis=0) ** 2).sum() + (
            np.diff(smooth_images, axis=1) ** 2
        ).sum()

    naive_smoothings = [smoothing(im) for im in naive_real_images]
    smooth_smoothings = [smoothing(im) for im in smooth_images]
    tm_smoothings = [smoothing(im) for im in unproc_img]

    x_gradients = {
        "naive": np.array([np.diff(im, axis=0) for im in naive_real_images]),
        "smooth": np.array([np.diff(im, axis=0) for im in smooth_images]),
        "non_smooth": np.array([np.diff(im, axis=0) for im in p_non_smooth_imags]),
        "tm": np.array([np.diff(im, axis=0) for im in unproc_img]),
    }
    y_gradients = {
        "naive": np.array([np.diff(im, axis=1) for im in naive_real_images]),
        "smooth": np.array([np.diff(im, axis=1) for im in smooth_images]),
        "non_smooth": np.array([np.diff(im, axis=1) for im in p_non_smooth_imags]),
        "tm": np.array([np.diff(im, axis=1) for im in unproc_img]),
    }

    fig, ax = plt.subplots(3, 4, figsize=(15, 10), sharex='row', sharey='row')

    ydiff_non_smooth = y_gradients["non_smooth"]
    xdiff_non_smooth = x_gradients["non_smooth"]
    y_diff_smooth = y_gradients["smooth"]
    x_diff_smooth = x_gradients["smooth"]

    for i in range(len(p_non_smooth_imags)):
        # ax[i, 0].imshow(smooth_images[i], cmap="gray", vmin=0, vmax=1)
        # ax[i, 1].imshow(p_non_smooth_imags[i], cmap="gray", vmin=0, vmax=1)
        ax[i, 0].hist(np.abs(xdiff_non_smooth[i]).flatten(), bins=50 )
        ax[i, 1].hist(np.abs(ydiff_non_smooth[i]).flatten(), bins=50 )
        ax[i, 2].hist(np.abs(x_diff_smooth[i]).flatten(), bins=50)
        ax[i, 3].hist(np.abs(y_diff_smooth[i]).flatten(), bins=50)

    fig, ax = plt.subplots(3, 4, figsize=(15, 10), sharex='row', sharey='row')
    for i in range(len(p_non_smooth_imags)):
        ax[i, 0].imshow(xdiff_non_smooth[i], cmap="gray")
        ax[i, 1].imshow(ydiff_non_smooth[i], cmap="gray")
        ax[i, 2].imshow(x_diff_smooth[i], cmap="gray")
        ax[i, 3].imshow(y_diff_smooth[i], cmap="gray")
        ax[i, 0].set_title("X gradient non-smooth")
        ax[i, 1].set_title("Y gradient non-smooth")
        ax[i, 2].set_title("X gradient smooth")
        ax[i, 3].set_title("Y gradient smooth")


    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for i, im in enumerate(unproc_img):
        ax[i, 0].hist(np.abs(np.diff(im, axis=0)).flatten(), bins=50)
        ax[i, 1].hist(np.abs(np.diff(im, axis=1)).flatten(), bins=50)

    plt.show()

    

    sys.exit()

    fig, subplots = plt.subplots(3, 3, figsize=(15, 15))

    for i, (im, x_grad, y_grad) in enumerate(
        zip(
            [naive_real_images, smooth_images, unproc_img],
            x_gradients.values(),
            y_gradients.values(),
        )
    ):
        subplots[i, 0].imshow(x_grad[0], cmap="gray", vmin=-0.5, vmax=0.5)
        subplots[i, 1].imshow(y_grad[0], cmap="gray", vmin=-0.5, vmax=0.5)
        subplots[i, 2].imshow(im[0], cmap="gray")

        subplots[i, 0].set_title("X gradient")
        subplots[i, 1].set_title("Y gradient")
        subplots[i, 2].set_title("Gradient magnitude")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(naive_real_images[0], cmap="gray")
    ax[0].set_title("Naive smoothing")
    ax[1].imshow(smooth_images[0], cmap="gray")
    ax[1].set_title("Smooth smoothing")
    ax[2].imshow(unproc_img[0], cmap="gray")
    ax[2].set_title("TM smoothing")

    print("Naive smoothing", naive_smoothings)
    print("Smooth smoothing", smooth_smoothings)
    print("TM smoothing", tm_smoothings)

    plt.show()
