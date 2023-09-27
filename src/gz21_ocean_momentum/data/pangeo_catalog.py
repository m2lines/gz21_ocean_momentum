#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routines for fetching data from pangeo and converting to xarray Dataset."""

import intake
import xarray as xr
import numpy as np

from intake.config import conf

CATALOG_URL = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml"


def get_patch(
    catalog_url: str = CATALOG_URL,
    ntimes: int = None,
    bounds: list = None,
    CO2_level=0,
    *selected_vars,
):
    """
    Return a tuple with a patch of uv velocities along with the grid details.

    Parameters
    ----------
    catalog_url : str
        url where the catalog lives.
    ntimes : int, optional
        Number of days to use. The default is None which corresponds to all.
    bounds : list, optional
        Bounds of the path, (lat_min, lat_max, long_min, long_max). Note that
        the order matters!
    CO2_level : int, optional
        CO2 level, 0 (control) or 1 (1 percent increase C02 per year).
        The default is 0.
    *selected_vars : str
        Variables selected from the surface velocities dataset.

    Returns
    -------
    uv_data : xarray dataset
        xarray dataset containing the requested u,v velocities.
    grid_data : xarray dataset
        xarray dataset with the grid details.
    """
    catalog = intake.open_catalog(catalog_url)
    if CO2_level == 0:
        source = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
        # TODO variable cache_folder currently unused. Remove?
        # cache_folder = CACHE_FOLDER
    elif CO2_level == 1:
        source = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
    else:
        raise ValueError("Unrecognized CO2 level. Should be O or 1.")
    s_grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_grid
    # Convert to dask
    uv_data = source.to_dask()
    grid_data = s_grid.to_dask()
    # Following line is necessary to transform non-primary coords into vars
    grid_data = grid_data.reset_coords()[["dxu", "dyu", "wet"]]
    if bounds is not None:
        uv_data = uv_data.sel(xu_ocean=slice(*bounds[2:]), yu_ocean=slice(*bounds[:2]))
        grid_data = grid_data.sel(
            xu_ocean=slice(*bounds[2:]), yu_ocean=slice(*bounds[:2])
        )
    if ntimes is not None:
        uv_data = uv_data.isel(time=slice(0, ntimes))

    if len(selected_vars) == 0:
        return uv_data, grid_data
    return uv_data[list(selected_vars)], grid_data


def get_grid():
    """
    Obtain the CM2.6 grid information, necessary for instance to compute derivatives

    Returns
    -------
    grid_data : xarray Dataset
        Grid information as an xarray dataset
    """
    catalog = intake.open_catalog(CATALOG_URL)
    s_grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_grid
    grid_data = s_grid.to_dask()
    # Following line is necessary to transform non-primary coords into vars
    grid_data = grid_data.reset_coords()[["dxu", "dyu", "wet"]]
    return grid_data


def get_whole_data(url, c02_level):
    """
    Obtain all surface data and grid data for a given CO2 level.

    Parameters
    ----------
    url : str
        url where to download the CM2.6 data from. Should correspond to an
        intake catalog, made available by PANGEO.
    c02_level : 0 or 1
        0 for pre-industrial level, 1 for 1 percent increase per year

    Returns
    -------
    data : xarray dataset
        xarray dataset containing the requested u,v velocities.
    grid : xarray dataset
        xarray dataset with the grid details.
    """
    data, grid = get_patch(url, None, None, c02_level, "usurf", "vsurf")
    return data, grid


if __name__ == "__main__":
    import os

    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = "~/.config/gcloud/application_default_credentials.json"
    CATALOG_URL = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
        /master/intake-catalogs/master.yaml"
    retrieved_data = get_whole_data(CATALOG_URL, 0)
