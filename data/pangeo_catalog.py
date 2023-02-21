#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routines for fetching data from pangeo and converting to xarray Dataset."""

import intake
import xarray as xr
import numpy as np

from intake.config import conf

# TODO variable currently unused? Remove? Or is it setting environment variable?
conf["persist_path"] = "/scratch/ag7531/"
# TODO variable currently unused. Remove?
CACHE_FOLDER = "/scratch/ag7531/cm26_cache"
# TODO: Not sure this should be hard coded here. Isn't it also hard coded in cmip26.py?
CATALOG_URL = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml"


def get_patch(
    catalog_url: str = CATALOG_URL,
    ntimes: int = None,
    bounds: list = None,
    CO2_level=0,
    *selected_vars
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
        cache_folder = CACHE_FOLDER
    elif CO2_level == 1:
        source = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
        # TODO variable cache_folder currently unused. Remove?
        cache_folder = CACHE_FOLDER + "1percent"
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
    Description?.  # AB

    Returns
    -------
    grid_data : type?  # AB
        description?  # AB
    """
    catalog = intake.open_catalog(CATALOG_URL)
    s_grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_grid
    grid_data = s_grid.to_dask()
    # Following line is necessary to transform non-primary coords into vars
    grid_data = grid_data.reset_coords()[["dxu", "dyu", "wet"]]
    return grid_data


def get_whole_data(url, c02_level):
    """
    Description?.  # AB

    Parameters
    ----------
    url : type?  # AB
        description?  # AB
    c02_level : type?  # AB
        description?  # AB

    Returns
    -------
    data : xarray dataset
        xarray dataset containing the requested u,v velocities.
    grid : xarray dataset
        xarray dataset with the grid details.
    """
    data, grid = get_patch(url, None, None, c02_level, "usurf", "vsurf")
    return data, grid


def get_cm2_5_grid():
    """
    Description?.  # AB

    Returns
    -------
    grid : xarray Dataset
        description?  # AB
    dx_u : type?  # AB
        description?  # AB
    dy_u : type?  # AB
        description?  # AB
    """
    grid = xr.open_dataset("/home/arthur/ocean.static.nc")
    dy_u = np.diff(grid["yu_ocean"]) / 360 * 2 * np.pi * 6400 * 1e3
    # dx_u = np.diff(grid['xu_ocean']) * np.cos(grid['yu_ocean'] / 360 * 2 * np.pi)
    dx_u = None
    return grid, dx_u, dy_u


if __name__ == "__main__":
    import os

    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = "/home/arthur/\
access_key.json"
    CATALOG_URL = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
        /master/intake-catalogs/master.yaml"
    retrieved_data = get_whole_data(CATALOG_URL, 0)
