"""Projection functions for parameter constraints."""

from typing import Callable

import optax
from jaxtyping import PyTree

import chest_xray_sim.utils.projections as proj
from chest_xray_sim.types import TransmissionMapT, WeightsT
from .config import PipelineConfig


def build_projection_spec(config: PipelineConfig) -> dict[str, Callable]:
    """Build a projection specification from config."""
    spec = {
        "enhance_factor": proj.box(config.low_min_enhancement, config.max_enhancement),
        "high_enhance_factor": proj.box(
            config.high_min_enhancement, config.max_enhancement
        ),
        "window_center": proj.box(0.1, 0.5),
        "window_width": proj.box(0.1, 1.0),
    }
    if config.windowing_type == "sigmoid":
        spec["gamma"] = proj.box(1, 20)
    return spec


def build_projection_fn(
    config: PipelineConfig,
) -> Callable[[TransmissionMapT, WeightsT], tuple[TransmissionMapT, WeightsT]]:
    """Build a composed projection function from config."""
    spec = build_projection_spec(config)

    def projection_fn(
        txm_state: TransmissionMapT,
        weights_state: WeightsT,
    ) -> tuple[TransmissionMapT, WeightsT]:
        new_txm = optax.projections.projection_hypercube(txm_state)
        new_weights = optax.projections.projection_non_negative(weights_state)
        new_weights = proj.projection_spec(new_weights, spec)
        return new_txm, new_weights

    return projection_fn


def default_weights(config: PipelineConfig) -> dict[str, float]:
    """Initial weight values based on config."""
    params: dict[str, float] = {
        "enhance_factor": 0.5,
        "window_center": 0.2,
        "window_width": 0.2,
        "high_enhance_factor": 0.2,
    }
    if config.windowing_type == "sigmoid":
        params["gamma"] = 5.0
    return params
