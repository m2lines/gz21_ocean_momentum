from dataclasses import dataclass
from typing import Optional
from typing import Tuple

@dataclass
class BoundingBox():
    """A rectangle defined by two latitudes and two longitudes.

    Initialization order is `lat_min, lat_max, long_min, long_max`.
    """
    lat_min:  float
    lat_max:  float
    long_min: float
    long_max: float
