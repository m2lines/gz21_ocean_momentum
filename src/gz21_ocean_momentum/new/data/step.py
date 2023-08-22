import gz21_ocean_momentum.new.data.coarsen as coarsen
import gz21_ocean_momentum.new.data.utils   as utils
from   gz21_ocean_momentum.new.common.bounding_box import BoundingBox

import xarray as xr
import intake

from typing import Optional
from typing import Tuple

def preprocess_and_compute_forcings(
        grid: xr.Dataset,
        surface_fields: xr.Dataset,
        bounding_box: Optional[BoundingBox],
        ntimes: Optional[int],
        cyclize: bool,
        resolution_degrading_factor: int,
        *selected_vars: str,
        ) -> xr.Dataset:
    """
    Perform various preprocessing on a dataset.
    """

    # transform non-primary coords into vars
    grid = grid.reset_coords()[["dxu", "dyu", "wet"]]

    if bounding_box is not None:
        surface_fields = surface_fields.sel(
            xu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max,  None),
            yu_ocean=slice(bounding_box.long_min, bounding_box.long_max, None))
        grid = grid.sel(
            xu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max,  None),
            yu_ocean=slice(bounding_box.long_min, bounding_box.long_max, None))

    if ntimes is not None:
        surface_fields = surface_fields.isel(time=slice(0, ntimes))

    if len(selected_vars) != 0:
        surface_fields = surface_fields[list(selected_vars)]

    if cyclize:
        # TODO logger
        #logger.info("Cyclic data... Making the dataset cyclic along longitude...")
        surface_fields = utils.cyclize(
                surface_fields, "xu_ocean", resolution_degrading_factor)
        grid = utils.cyclize(
                grid,           "xu_ocean", resolution_degrading_factor)

        # rechunk along the cyclized dimension
        surface_fields = surface_fields.chunk({"xu_ocean": -1})
        grid = grid.chunk({"xu_ocean": -1})

    # TODO should this be earlier? later? never? ???
    # TODO logger
    #logger.debug("Getting grid data locally")
    # grid data is saved locally, no need for dask
    grid = grid.compute()

    # calculate eddy-forcing dataset for that particular patch
    return coarsen.eddy_forcing(surface_fields, grid, resolution_degrading_factor)

def download_cm2_6(
        catalog_url: str,
        co2_increase: bool,
        ) -> Tuple[xr.Dataset, xr.Dataset]:
    """Run data step on CM2.6 dataset."""
    catalog = intake.open_catalog(catalog_url)
    grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    grid = grid.to_dask()
    if co2_increase:
        surface_fields = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    else:
        surface_fields = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
    surface_fields = surface_fields.to_dask()
    return surface_fields, grid
