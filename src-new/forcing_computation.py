from typing import assert_never
from typing import Literal
from typing import Tuple
import enum
from enum import Enum

import xarray as xr

@dataclass
class CO2Change(enum.Enum):
    Control0 = enum.auto()
    "TODO control"

    AnnualIncrease1 = enum.auto()
    "TODO annual increase"

@dataclass
class BoundingBox():
    lat_min:  float
    lat_max:  float
    long_min: float
    long_max: float

def download_cmip(
        grid,
        surface_fields,
        bounding_box: Optional[BoundingBox],
        ntimes,
        cyclize: bool,
        factor: int,
        *selected_vars: str,
        ) -> Tuple[xr.Dataset, xr.Dataset]:

    # transform non-primary coords into vars
    grid = grid.reset_coords()[["dxu", "dyu", "wet"]]

    if bounding_box is not None:
        surface_fields = surface_fields .sel(
            xu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max,  None),
            yu_ocean=slice(bounding_box.long_min, bounding_box.long_max, None))
        grid = grid.sel(
            xu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max,  None),
            yu_ocean=slice(bounding_box.long_min, bounding_box.long_max, None))

    if ntimes is not None:
        surface_fields = surface_fields.isel(time=slice(0, ntimes))

    if len(selected_vars) is not 0:
        surface_fields = surface_fields[list(selected_vars)]

    if cyclize:
        # make the dataset cyclic along longitude
        logger.info("Cyclic data... Making the dataset cyclic along longitude...")
        surface_fields = cyclize_dataset(surface_fields, "xu_ocean", factor)
        grid = cyclize_dataset(grid, "xu_ocean", factor)

        # Rechunk along the cyclized dimension
        surface_fields = surface_fields.chunk({"xu_ocean": -1})
        grid = grid.chunk({"xu_ocean": -1})

    return surface_fields, grid

def prepare_cmip(resolution_degrading_factor, make_cyclic):
    return 0

def compute_forcing(resolution_degrading_factor):
    """
    Returns an xarray.
    """
    return 0
