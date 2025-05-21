from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

DTYPE = jnp.float32


def generate_range_normal(
    key: jax.Array,
    shape: tuple[int, ...] = (),
    val_range: tuple[float, float] = (0.0, 1.0),
    axis: int | None = 0,
    **kwargs,
) -> jax.Array:
    minval, maxval = val_range
    arr = jax.random.normal(key, shape=shape, dtype=DTYPE)
    ratio = (arr - arr.min(axis=axis)) / (
        arr.max(axis=axis) - arr.min(axis=axis)
    )
    return ratio * (maxval - minval) + minval


def _range_init(
    key: jax.Array,
    shape: tuple[int, ...],
    val_range: tuple[float, float] | None,
    mode: Literal["normal", "uniform"] = "uniform",
    axis: int | None = 0,
    **kwargs,
) -> jax.Array:
    arr = None
    assert val_range is not None, (
        "val_range must be provided for normal distribution"
    )

    if mode == "normal":
        return generate_range_normal(key, shape, val_range, axis, *kwargs)
    elif mode == "uniform":
        minval, maxval = val_range
        arr = jax.random.uniform(
            key, minval=minval, maxval=maxval, shape=shape, dtype=DTYPE
        )
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")

    return arr


def parse_init_params(
    hyperparams: Any, images: Float[Array, "batch rows cols"]
):
    init_mode, init_config = hyperparams.tm_init_params

    init_params: dict[str, Any] = dict(mode=init_mode)
    if init_mode in ["uniform", "normal"]:
        init_params["val_range"] = init_config
    elif init_mode in ["target", "negative"]:
        init_params["target"] = images

    return init_mode, init_params


def initialize(
    seed: int,
    shape: tuple[int, ...] = (),
    mode: Literal["normal", "uniform", "target", "zeros"] = "uniform",
    val_range: tuple[float, float] | None = None,
    target: jax.Array | None = None,
    axis: int | None = 0,
    **kwargs,
) -> jax.Array:
    key = jax.random.key(seed)
    if mode == "normal" or mode == "uniform":
        return _range_init(key, shape, val_range, mode, axis, *kwargs)
    elif mode == "target":
        assert target is not None, "target must be provided for copy mode"
        return target.copy().astype(DTYPE)
    elif mode == "negative":
        assert target is not None, "target must be provided for copy mode"
        return 1 - target.copy().astype(DTYPE)
    elif mode == "zeros":
        return jnp.zeros(shape, dtype=DTYPE) + 1e-6
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")


# deprecated
def get_random(
    key: int,
    shape: tuple[int, ...],
    distribution: Literal["normal", "uniform"] = "uniform",
    val_range=(0.0, 1.0),
    axis: int | None = 0,
):
    arr = None
    minval, maxval = val_range
    if distribution == "normal":
        arr = jax.random.normal(key, shape=shape)
        ratio = (arr - arr.min(axis=axis)) / (
            arr.max(axis=axis) - arr.min(axis=axis)
        )
        arr = ratio * (maxval - minval) + minval
    else:
        arr = jax.random.uniform(key, minval=minval, maxval=maxval, shape=shape)

    return arr
