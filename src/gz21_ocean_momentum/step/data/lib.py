import gz21_ocean_momentum.step.data.coarsen as coarsen

import xarray as xr
import intake

from typing import Optional
from typing import Tuple

def preprocess_and_compute_forcings(
        grid: xr.Dataset,
        surface_fields: xr.Dataset,
        cyclize: bool,
        resolution_degrading_factor: int,
        *selected_vars: str,
        ) -> xr.Dataset:
    """
    Perform various preprocessing on a dataset.
    """

    # transform non-primary coords into vars
    grid = grid.reset_coords()[["dxu", "dyu", "wet"]]

    if len(selected_vars) != 0:
        surface_fields = surface_fields[list(selected_vars)]

    if cyclize:
        # TODO logger
        #logger.info("Cyclic data... Making the dataset cyclic along longitude...")
        surface_fields = _cyclize(
                surface_fields, "xu_ocean", resolution_degrading_factor)
        grid = _cyclize(
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

def retrieve_cm2_6(
        catalog_uri: str,
        co2_increase: bool,
        ) -> Tuple[xr.Dataset, xr.Dataset]:
    """Retrieve the CM2.6 dataset via the given intake catalog URI.

    Will download if given an `http://` URI. Will use local files such as
    `/home/user/catalog.yaml` directly.
    """
    catalog = intake.open_catalog(catalog_uri)
    grid = catalog.GFDL_CM2_6.GFDL_CM2_6_grid
    grid = grid.to_dask()
    if co2_increase:
        surface_fields = catalog.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    else:
        surface_fields = catalog.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
    surface_fields = surface_fields.to_dask()
    return surface_fields, grid

def _cyclize(ds: xr.Dataset, coord_name: str, nb_points: int):
    """
    Generate a cyclic dataset from non-cyclic input.

    Return a cyclic dataset, with nb_points added on each end, along
    the coordinate specified by coord_name.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to process.
    coord_name : str
        Name of the coordinate along which the data is made cyclic.
    nb_points : int
        Number of points added on each end.

    Returns
    -------
    New extended dataset.
    """
    # TODO make this flexible
    cycle_length = 360.0
    left = ds.roll({coord_name: nb_points}, roll_coords=True)
    right = ds.roll({coord_name: nb_points}, roll_coords=True)
    right = right.isel({coord_name: slice(0, 2 * nb_points)})
    left[coord_name] = xr.concat(
        (left[coord_name][:nb_points] - cycle_length, left[coord_name][nb_points:]),
        coord_name,
    )
    right[coord_name] = xr.concat(
        (right[coord_name][:nb_points], right[coord_name][nb_points:] + cycle_length),
        coord_name,
    )
    new_ds = xr.concat((left, right), coord_name)
    return new_ds
