from typing import Callable, Union

from jaxtyping import Array, Float, PyTree
from torch import Tensor

ArrayT = Union[Array, Tensor]

WeightsT = PyTree
BatchT = Float[Array, "batch rows cols"]
# SegmentationT = Float[Array, "batch labels rows cols"]

# TODO: duplicate definition segmentation.py
# TODO: some usages might expect different dimension sizes for mask_groups,
# some parts use the entire segmentation labels, others a reduced amount
TransmissionMapT = Float[Array, "batch height width"]
ForwardT = Float[Array, "batch height width"]
# annotate segmentation types with full labels (non-merged masks )
FullSegmentationT = Float[Array, "batch segmentation_labelsheight width"]
ValueRangeT = Float[Array, "mask_groups 2"]
SegmentationT = Float[Array, "batch mask_groups height width"]

# SegmentationT = PyTree  # dict of mask ids - batch channels rows cols
LossFnT = Callable[[BatchT, WeightsT, BatchT, BatchT], Float[Array, ""]]
SegLossFnT = Callable[
    [BatchT, WeightsT, BatchT, BatchT, SegmentationT], Float[Array, ""]
]
ForwardFnT = Callable[[BatchT, WeightsT], BatchT]
ProjectFnT = Callable[[PyTree, WeightsT], tuple[PyTree, WeightsT]]
SegProjectFnT = Callable[
    [PyTree, WeightsT, SegmentationT], tuple[PyTree, WeightsT]
]
