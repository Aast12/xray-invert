from typing import Any, Callable
import optax


def hypercube():
    return optax.projections.projection_hypercube


def non_negative():
    return optax.projections.projection_non_negative


def box(lower_bound, upper_bound):
    return lambda state: optax.projections.projection_box(
        state, lower_bound, upper_bound
    )


def projection_spec(state: dict[str, Any], spec: dict[str, Callable]):
    for key, projection in spec.items():
        if key not in state:
            raise ValueError(f"Key {key} not found in state.")
        if not callable(projection):
            raise ValueError(f"Projection for {key} is not callable.")
        state[key] = projection(state[key])

    return state
