# Re-export from core library for backward compatibility
from chest_xray_sim.data.priors import (
    extract_region_value_ranges,
    get_priors,
    map_range,
)

import jax.numpy as jnp

DEBUG = True
DTYPE = jnp.float32

# Default paths for backward compatibility
REAL_TM_PATHS = [
    "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-oprocess.tif",
]
FWD_REAL_TM_PATHS = [
    "/Volumes/T7/projs/thesis/data/Processed vs unprocessed real GE scanner/Z01-process.tif",
]

__all__ = ["extract_region_value_ranges", "get_priors", "map_range", "REAL_TM_PATHS", "FWD_REAL_TM_PATHS"]
