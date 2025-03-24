from pprint import pprint
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Num, Int, Float
from jax.tree_util import register_dataclass
from dataclasses import dataclass, field
from typing import Callable, Optional

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
        print("size:", size)
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
    separation_eps=1e-6,
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
        slope = (y2 - y1) / jnp.maximum(x2 - x1, separation_eps)
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


def range_mapping(image, new_min=0, new_max=1):
    old_min = image.min()
    old_max = image.max()
    return ((image - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


@register_dataclass
@dataclass
class DynamicRangeWeights:
    """
    Parameters for sigmoid-based dynamic range compression
    Ref: Prokop & Neitzel (2011)
    """

    window_center: Float[Array, ""]  # Single float value
    window_width: Float[Array, ""]  # Single float value
    gamma: Float[Array, ""]  # Power law correction


@register_dataclass
@dataclass
class MultiscaleWeights:
    """
    Parameters for multiscale decomposition and enhancement
    Ref: Stahl et al. (2000)
    """

    filter_sizes: Int[Array, "levels"] = field(
        metadata=dict(static=True), compare=False
    )
    enhancement_weights: Float[Array, "levels"]
    edge_weights: Float[Array, "levels"]


@register_dataclass
@dataclass
class NoiseReductionWeights:
    """
    Parameters for noise reduction
    Ref: Li et al. (2005)
    """

    conductance: Float[Array, ""]
    num_iterations: Int[Array, ""] = field(metadata=dict(static=True), compare=False)


@register_dataclass
@dataclass
class DisplayLUTWeights:
    """
    Two options for display transformation:
    1. Piecewise linear with differentiable interpolation
    2. Parameterized sigmoid curve
    Ref: DICOM PS3.14
    """

    # For sigmoid approach
    window_center: Float[Array, ""]
    window_width: Float[Array, ""]
    gamma: Float[Array, ""]

    # # For piecewise approach (optional)
    # breakpoints: Optional[Float[Array, "points"]] = None
    # values: Optional[Float[Array, "points"]] = None


def apply_display_lut_sigmoid(
    image: Float[Array, "height width"], weights: DisplayLUTWeights
) -> Float[Array, "height width"]:
    """Sigmoid-based display transformation"""
    normalized = (image - weights.window_center) / weights.window_width
    return jax.nn.sigmoid(normalized) ** weights.gamma


@register_dataclass
@dataclass
class PipelineWeights:
    image: ImageType
    dynamic_range: DynamicRangeWeights
    multiscale: MultiscaleWeights
    noise: NoiseReductionWeights
    # display: DisplayLUTWeights


def dynamic_range_compression(
    image: Float[Array, "height width"], weights: DynamicRangeWeights
) -> Float[Array, "height width"]:
    """Sigmoid-based dynamic range compression"""
    normalized = (image - weights.window_center) / weights.window_width
    return jax.nn.sigmoid(normalized) ** weights.gamma


def multiscale_enhance(
    image: Float[Array, "height width"],
    weights: MultiscaleWeights,
    filter_fn: Callable = None,
) -> Float[Array, "height width"]:
    """Multi-scale enhancement with edge preservation"""
    if filter_fn is None:

        def gaussian_filter(img, size):
            kernel = jnp.exp(-jnp.linspace(-2, 2, size) ** 2)[:, None] * jnp.exp(
                -jnp.linspace(-2, 2, size)[None, :] ** 2
            )
            kernel = kernel / jnp.sum(kernel)
            return jax.scipy.signal.convolve2d(img, kernel, mode="same")

        filter_fn = gaussian_filter

    scales = []
    previous = image
    laplacian_kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    for size, weight, edge_weight in zip(
        weights.filter_sizes, weights.enhancement_weights, weights.edge_weights
    ):
        filtered = filter_fn(previous, size)
        detail = previous - filtered

        scales.append(weight * detail)

        # edge = jax.scipy.signal.convolve2d(filtered, laplacian_kernel, mode="same")

        # # edge = jnp.abs(jax.scipy.signal.laplace(filtered))
        # scales.append(weight * detail + edge_weight * edge)
        # previous = filtered

    return image + jnp.sum(jnp.stack(scales, axis=0), axis=0)


def anisotropic_diffusion(
    image: Float[Array, "height width"], weights: NoiseReductionWeights
) -> Float[Array, "height width"]:
    """Perona-Malik anisotropic diffusion"""

    def single_step(img):
        gradN = jnp.roll(img, -1, axis=0) - img
        gradS = jnp.roll(img, 1, axis=0) - img
        gradE = jnp.roll(img, -1, axis=1) - img
        gradW = jnp.roll(img, 1, axis=1) - img

        cN = jnp.exp(-((gradN / weights.conductance) ** 2))
        cS = jnp.exp(-((gradS / weights.conductance) ** 2))
        cE = jnp.exp(-((gradE / weights.conductance) ** 2))
        cW = jnp.exp(-((gradW / weights.conductance) ** 2))

        return img + 0.25 * (cN * gradN + cS * gradS + cE * gradE + cW * gradW)

    result = image
    for _ in range(weights.num_iterations):
        result = single_step(result)
    return result


def apply_display_lut(
    image: Float[Array, "height width"], weights: DisplayLUTWeights
) -> Float[Array, "height width"]:
    """Apply display LUT transformation"""

    return apply_display_lut_sigmoid(image, weights)
    # return jnp.interp(image, weights.breakpoints, weights.values)


def forward(
    image: Float[Array, "height width"], weights: PipelineWeights
) -> Float[Array, "height width"]:
    """Complete pipeline forward pass"""
    # Start with -log transform
    x = -jnp.log(jnp.maximum(image, 1e-6))

    # Apply each stage
    x = dynamic_range_compression(x, weights.dynamic_range)
    x = anisotropic_diffusion(x, weights.noise)
    x = multiscale_enhance(x, weights.multiscale)
    x = apply_display_lut(x, weights.display)

    return jnp.clip(x, 0.0, 1.0)


def initialize_weights(image, pixel_spacing_mm=0.2) -> PipelineWeights:
    """Initialize pipeline weights with research-based defaults"""
    return PipelineWeights(
        image=image,
        dynamic_range=DynamicRangeWeights(
            window_center=jnp.array(2.0),
            window_width=jnp.array(4.0),
            gamma=jnp.array(1.0),
        ),
        multiscale=MultiscaleWeights(
            filter_sizes=jax.lax.stop_gradient(
                jnp.array(
                    [
                        int(0.5 / pixel_spacing_mm),
                        int(2.0 / pixel_spacing_mm),
                        int(8.0 / pixel_spacing_mm),
                    ]
                )
            ),
            enhancement_weights=jnp.array([0.5, 0.3, 0.2]),
            edge_weights=jnp.array([0.3, 0.5, 0.2]),
        ),
        noise=NoiseReductionWeights(
            conductance=jnp.array(2.0),
            num_iterations=jax.lax.stop_gradient(jnp.array(5)),
        ),
    )
