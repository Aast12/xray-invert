from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from chest_xray_sim.data.segmentation import (
    MaskGroupsT,
)

DEBUG = True
DTYPE = jnp.float32


@dataclass
class ExperimentInputs:
    images: Float[Array, "batch height width"]
    segmentations: Float[Array, "batch labels height width"]
    prior_labels: list[MaskGroupsT]
    priors: Float[Array, "labels 2"]
