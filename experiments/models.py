# Re-export from core library for backward compatibility
from chest_xray_sim.types import ExperimentInputs

import jax.numpy as jnp

DEBUG = True
DTYPE = jnp.float32

__all__ = ["ExperimentInputs"]
