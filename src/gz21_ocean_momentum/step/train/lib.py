import torch

import gz21_ocean_momentum.models.transforms

from gz21_ocean_momentum.common.bounding_box import BoundingBox

def select_subdomains(ds: torch.utils.data.Dataset, sds: list[BoundingBox]) -> list[torch.utils.data.Dataset]:
    """TODO requires xu_ocean, yu_ocean"""
    return [ ds.sel(xu_ocean=slice(sd.long_min, sd.long_max),
                    yu_ocean=slice(sd.lat_min,  sd.lat_max)) for sd in sds ]
