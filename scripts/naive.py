from chest_xray_sim.data.utils import read_image
from chest_xray_sim.utils.results import naive_inversion
import matplotlib.pyplot as plt
import jax.numpy as jnp


reconstructed_path = (
    "/Volumes/T7/projs/thesis/outputs/1ry43vj1/patient03826_view1_frontal.png"
)
real_fwd_path = "/Volumes/T7/datasets/chexpert_plus/PNG/PNG/train/patient03826/study1/view1_frontal.png"
real_tm_path = "/Volumes/T7/projs/thesis/data/conventional_transmissionmap.tif"

reconstructed_tm = read_image(reconstructed_path).squeeze(0)
real_fwd = read_image(real_fwd_path).squeeze(0)
real_tm = read_image(real_tm_path).squeeze(0)

print("real range", real_fwd.min(), real_fwd.max())
print("real tm range", real_fwd.min(), real_fwd.max())
print("shape", real_fwd.shape, reconstructed_tm.shape, real_tm.shape)

naive_tm = naive_inversion(jnp.array(real_fwd))
naive_stretch = naive_tm / 2.0

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(naive_tm, cmap="gray", vmin=0, vmax=1)
ax[0, 1].hist(naive_tm.ravel(), bins=100, color="gray", alpha=0.7)
ax[1, 0].imshow(naive_stretch, cmap="gray", vmin=0, vmax=1)
ax[1, 1].hist(naive_stretch.ravel(), bins=100, color="gray", alpha=0.7)


naive_tm = 1.0 - jnp.array(real_fwd)
naive_stretch = naive_tm / 2.0

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(naive_tm, cmap="gray", vmin=0, vmax=1)
ax[0, 1].hist(naive_tm.ravel(), bins=100, color="gray", alpha=0.7)
ax[1, 0].imshow(naive_stretch, cmap="gray", vmin=0, vmax=1)
ax[1, 1].hist(naive_stretch.ravel(), bins=100, color="gray", alpha=0.7)

plt.show()
