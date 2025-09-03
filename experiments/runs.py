import wandb
import joblib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
from chest_xray_sim.data.segmentation import ChestSegmentation, batch_get_exclusive_masks
from chest_xray_sim.data.utils import read_image
import chest_xray_sim.inverse.operators as ops

# from chest_xray_sim.inverse.eval


def operator(txm, weights, sigma, high_sigma):
    x = ops.window(
        txm,
        weights["window_center"],
        weights["window_width"],
        weights.get("gamma", 0.0),
        'sigmoid',
    )
    x = ops.negative_log(x)
    x = ops.unsharp_masking(x, sigma, weights["enhance_factor"])
    x = ops.clipping(x)
    x = ops.unsharp_masking(x, high_sigma, weights["high_enhance_factor"])
    x = ops.clipping(x)
    return x

run_id = 'c6r8oftq'

output_dir = '/Volumes/T7/projs/thesis/outputs/'

run_path = os.path.join(output_dir, run_id)


meta_path = os.path.join(run_path, 'meta.joblib')
weights_path = os.path.join(run_path, 'weights.joblib')

meta = joblib.load(meta_path)
weights = joblib.load(weights_path)

print(f"Loaded meta: {meta}")
print(f"Loaded weights: {weights}")

model = ChestSegmentation(cache_dir=os.environ.get("CACHE_DIR"))
for i, data in enumerate(meta):
    img_path = data['abs_img_path']
    filepath = os.path.join(run_path, data['filepath'])

    og = read_image(img_path)
    im = read_image(filepath)
    im = jnp.array(im).squeeze()

    mask = model(img_path)
    segmentations = jnp.array(mask.cpu().numpy())
    seg_labels, segmentations = batch_get_exclusive_masks(
    segmentations, 0.6 
    )

    mask = model(im)

    w = {k: v[i] for k, v in weights.items()} 
    print(w)

    # print(im)
    txm = operator(im, w, sigma=3.0, high_sigma=10.0)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(im, cmap='gray', vmin=0, vmax=1)
    ax[1].imshow(txm, cmap='gray', vmin=0, vmax=1)
    ax[2].hist(im.ravel(), bins=100, alpha=0.5, label='Image')

plt.show()
