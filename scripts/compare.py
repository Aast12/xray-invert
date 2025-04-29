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

fig, ax = plt.subplots(2, 3, figsize=(10, 5))

labels = ["Naive TM", "Reconstructed TM", "Real TM"]
for i, cmap in enumerate(["gray", "jet"]):
    ax[i, 0].imshow(naive_tm, cmap=cmap, )
    ax[i, 1].imshow(reconstructed_tm, cmap=cmap, vmin=0, vmax=1)
    ax[i, 2].imshow(real_tm, cmap=cmap, vmin=0, vmax=1)
    for j, lb in enumerate(labels):
        ax[i, j].set_title(lb)
        ax[i, j].axis("off")

plt.tight_layout()
plt.show()
# plt.imshow(rough_tm, cmap="nipy_spectral")
# plt.show()
