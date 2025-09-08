import typing
import os
import pandas as pd


from jaxtyping import Float
from ..types import ArrayT


class XRayImageMeta(typing.TypedDict):
    """
    Meta information for a single X-Ray image.

    Attributes:
        img_id: unique identifier to fetch from source
        local_img_path: local path used to load the image file
        frontal_lateral: if the x-ray image is frontal or lateral
        ap_pa: if the x-ray image is anterior-posterior (AP) or posterior-anterior (PA)
    """

    img_id: typing.NotRequired[str]
    local_img_path: typing.NotRequired[str]
    frontal_lateral: typing.NotRequired[str]
    ap_pa: typing.NotRequired[str]


ImageT = Float[ArrayT, "*channels height width"]
ItemT = tuple[ImageT, XRayImageMeta]
CollateItemT = list[ItemT]
BatchItemT = tuple[ImageT, list[XRayImageMeta]]


def default_get_metadata_from_path(abs_path: str) -> XRayImageMeta:
    filename = os.path.basename(abs_path)

    meta: XRayImageMeta = {
        "img_id": abs_path,
        "local_img_path": abs_path,
    }

    # Try to extract frontal/lateral from filename
    if "frontal" in filename.lower():
        meta["frontal_lateral"] = "Frontal"
    elif "lateral" in filename.lower():
        meta["frontal_lateral"] = "Lateral"

    # Try to extract AP/PA from filename
    if "_ap_" in filename.lower() or "_ap." in filename.lower():
        meta["ap_pa"] = "AP"
    elif "_pa_" in filename.lower() or "_pa." in filename.lower():
        meta["ap_pa"] = "PA"

    return meta


def default_get_metadata_from_df(
    metadata: pd.DataFrame, abs_path: str
) -> XRayImageMeta:
    item = metadata[metadata["abs_img_path"] == abs_path].head(1)
    item = item.iloc[0]
    item = item[list(XRayImageMeta.__annotations__.keys())]

    return XRayImageMeta(**item.to_dict()) if not item.empty else {}

