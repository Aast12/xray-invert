from dataclasses import dataclass
import jax.numpy as jnp

from chest_xray_sim.segmentation.model import MaskGroupsT
from chest_xray_sim.types import SegmentationT, TransmissionMapT, ValueRangeT

DEBUG = True
DTYPE = jnp.float32


@dataclass
class ExperimentInputs:
    """Container for experiment input data."""

    images: TransmissionMapT
    segmentations: SegmentationT
    prior_labels: list[MaskGroupsT]
    priors: ValueRangeT


__all__ = ["ExperimentInputs"]
