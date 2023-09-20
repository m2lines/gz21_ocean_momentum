import xarray as xr

from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import List

import yaml

@dataclass
class BoundingBox():
    """A rectangle defined by two latitudes and two longitudes.

    Initialization order is `lat_min, lat_max, long_min, long_max`.
    """
    lat_min:  float
    lat_max:  float
    long_min: float
    long_max: float

#@staticmethod
def validate_nonempty(bbox: BoundingBox) -> bool:
    """Validate that a bounding box represents a non-empty region."""
    return bbox.lat_max > bbox.lat_min and bbox.long_max > bbox.long_min

def bound_dataset(
        dim_lat: str, dim_long: str,
        data: xr.Dataset, bbox: BoundingBox
        ):
    """Bound an xarray `Dataset` to the given `BoundingBox` using the given
    dimension names as spatial axes to bound along.

    The spatial dimensions should be `float`s. Argument order is latitude (y)
    followed by longitude (x).
    """
    return data.sel({
            dim_lat:  slice(bbox.lat_min,  bbox.lat_max),
            dim_long: slice(bbox.long_min, bbox.long_max)})

def load_bounding_boxes_yaml(path: str) -> list[BoundingBox]:
    """Load a YAML file of bounding boxes.

    The YAML value must be a list where each element contains `float` fields
    `lat-min`, `lat-max`, `long-min` and `long-max`.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    bboxes = []
    for el in data:
        bboxes.append(BoundingBox(
            el["lat-min"],  el["lat-max"],
            el["long-min"], el["long-max"]))
    return bboxes
