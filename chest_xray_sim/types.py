from dataclasses import dataclass
from typing import Callable, Union, TypeVar, Generic

from jaxtyping import Array, Float, PyTree, Integer
from torch import Tensor


ArrayT = Union[Array, Tensor]


WeightsT = PyTree

ImageJax = Float[ArrayT, "*batch height width"]
ImageTorch = Float[Tensor, "*batch height width"]
ImageT = Union[ImageJax, ImageTorch]

SingleMaskJax = Integer[ArrayT, "*batch height width"]
SingleMaskTorch = Integer[Tensor, "*batch height width"]
SingleMaskT = Union[SingleMaskJax, SingleMaskTorch]

MasksJax = Integer[ArrayT, "*batch labels height width"]
MasksTorch = Integer[Tensor, "*batch labels height width"]
MasksT = Union[MasksJax, MasksTorch]

SegmentationTorch = Integer[Tensor, "*batch labels height width"]
SegmentationJax = Integer[ArrayT, "*batch labels height width"]
SegmentationT = Union[SegmentationTorch, SegmentationJax]

# todo: deprecated
BatchT = Float[ArrayT, "*batch rows cols"]
fBatchT = Float[ArrayT, "*batch rows cols"]
iBatchT = Integer[ArrayT, "*batch rows cols"]


# TODO: duplicate definition segmentation.py
# TODO: some usages might expect different dimension sizes for mask_groups,
# some parts use the entire segmentation labels, others a reduced amount
TransmissionMapT = Float[Array, "batch height width"]
ForwardT = Float[Array, "batch height width"]
# annotate segmentation types with full labels (non-merged masks )
FullSegmentationT = Float[Array, "*batch segmentation_labels height width"]
ValueRangeT = Float[Array, "mask_groups 2"]

# SegmentationT = PyTree  # dict of mask ids - batch channels rows cols
LossFnT = Callable[[BatchT, WeightsT, BatchT, BatchT], Float[Array, ""]]
SegLossFnT = Callable[
    [BatchT, WeightsT, BatchT, BatchT, SegmentationT], Float[Array, ""]
]
ForwardFnT = Callable[[BatchT, WeightsT], BatchT]
ProjectFnT = Callable[[PyTree, WeightsT], tuple[PyTree, WeightsT]]
SegProjectFnT = Callable[[PyTree, WeightsT, SegmentationT], tuple[PyTree, WeightsT]]


# @dataclass
# class ExperimentInputs:
#     """Container for experiment input data."""

#     images: TransmissionMapT
#     segmentations: SegmentationT
#     prior_labels: list[MaskGroupsT]
#     priors: ValueRangeT
