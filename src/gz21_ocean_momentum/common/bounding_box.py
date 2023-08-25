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

def load_bounding_boxes_yaml(path: str) -> List[BoundingBox]:
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
