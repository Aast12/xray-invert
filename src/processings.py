from pprint import pprint
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Num, Int, Float
from jax.tree_util import register_dataclass
from dataclasses import dataclass, field
from typing import Callable


ImageType = Num[Array, "height width"]


def mean_filter(image, size):
    kernel = jnp.ones((size, size)) / (size * size)
    return convolve2d(image, kernel, mode="same")


@jax.tree_util.register_pytree_node_class
@dataclass
class MultiscaleProcessingWeights:
    # filter sizes are static for now, not trivial to optimize discrete values
    filter_sizes: Int[Array, "levels"] = field(
        metadata=dict(static=True), compare=False
    )
    unsharp_weights: Float[Array, "levels"]
    filter_fn: Callable[[ImageType, Int], ImageType] = field(
        default=mean_filter, metadata=dict(static=True), compare=False
    )

    def tree_flatten(self):
        children = (self.unsharp_weights,)
        aux_data = ("unsharp_weights",), {
            "filter_sizes": self.filter_sizes,
            "filter_fn": self.filter_fn,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        children_keys, aux = aux_data
        children_data = dict(zip(children_keys, children))

        all_args = {**aux, **children_data}

        return cls(**all_args)


# @jit
def multiscale_processing(
    image: Num[Array, "height width"],
    unsharp_weights: Float[Array, "levels"],
    filter_sizes: Int[Array, "levels"] = field(metadata=dict(static=True)),
    filter_fn=mean_filter,
) -> Num[Array, "height width"]:

    assert filter_sizes.shape[0] == unsharp_weights.shape[0]

    blurred_pyramid = [image]

    for size in filter_sizes:
        blurred = filter_fn(blurred_pyramid[-1], int(size))
        blurred_pyramid.append(blurred)

    result = image.copy()
    for i in range(len(unsharp_weights)):
        detail = blurred_pyramid[i] - blurred_pyramid[i + 1]
        result = result + unsharp_weights[i] * detail

    return result


@register_dataclass
@dataclass
class LookupTableWeights:
    breakpoints: Float[Array, "breakpoints"]
    values: Float[Array, "breakpoints"]
    partitions: Float


def create_lookup_table(
    x: ImageType,
    breakpoints: Float[Array, "breakpoints"],
    values: Float[Array, "breakpoints"],
):
    """Create a piecewise linear lookup table breakpoints: list of x coordinates for the breakpoints values: list of y coordinates for the breakpoints"""
    lut = jnp.zeros_like(x, dtype=float)
    breakpoints = breakpoints
    values = values

    lut = lut.at[x <= breakpoints[0]].set(values[0])

    lut = lut.at[x >= breakpoints[-1]].set(values[-1])

    for i in range(len(breakpoints) - 1):
        mask = (x > breakpoints[i]) & (x <= breakpoints[i + 1])
        x1, x2 = breakpoints[i], breakpoints[i + 1]
        y1, y2 = values[i], values[i + 1]
        slope = (y2 - y1) / (x2 - x1)
        lut = lut.at[mask].set(y1 + slope * (x[mask] - x1))

    return lut


def apply_lut(
    image: ImageType,
    partitions: Int = 1000,
    breakpoints: Float[Array, "breakpoints"] = None,
    values: Float[Array, "breakpoints"] = None,
    lut: Float[Array, "partitions"] = None,
):
    assert len(breakpoints) == len(
        values
    ), "breakpoints and values must have the same length"

    x = jnp.linspace(0.0, 1.0, partitions)
    lut = create_lookup_table(x, breakpoints, values)

    return jnp.interp(image, x, lut)
