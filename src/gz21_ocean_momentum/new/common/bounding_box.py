from dataclasses import dataclass
from typing import Optional
from typing import Tuple

@dataclass
class BoundingBox():
    lat_min:  float
    lat_max:  float
    long_min: float
    long_max: float
