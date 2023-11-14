#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data step API: CM2.6 downloading, forcing generation and coarsening."""

import xarray as xr
import intake
from scipy.ndimage import gaussian_filter
import numpy as np

from typing import Optional
from typing import Tuple

import logging

logger = logging.getLogger(__name__)

def retrieve_cm2_6(
        catalog_uri: str,
        co2_increase: bool,
        ) -> Tuple[xr.Dataset, xr.Dataset]:
    """Retrieve the CM2.6 dataset via the given intake catalog URI.

    Returns a tuple of `(uv dataset, grid dataset)`.

    Will download if given an `http://` URI. Will use local files such as
    `/home/user/catalog.yaml` directly.
    """

    catalog = intake.open_catalog(catalog_uri)
    grid = catalog.GFDL_CM2_6.GFDL_CM2_6_grid
    grid = grid.to_dask()

    # transform non-primary coords into vars
    grid = grid.reset_coords()[["dxu", "dyu", "wet"]]

    if co2_increase:
        logger.info("using 1% annual CO2 increase dataset")
        surface_fields = catalog.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
    else:
        logger.info("using control dataset -> no annual CO2 increase")
        surface_fields = catalog.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    surface_fields = surface_fields.to_dask()

    return surface_fields, grid

def cyclize(dim_name: str, ds: xr.Dataset, nb_points: int) -> xr.Dataset:
    """
    Generate a cyclic dataset from non-cyclic input.

    Return a cyclic dataset, with `nb_points` added on each end, along
    the dimension specified by `dim_name`.

    Parameters
    ----------
    dim_name: str
        Name of the dimension along which the data is made cyclic.
    ds : xr.Dataset
        Dataset to process.
    nb_points : int
        Number of points added on each end.

    Returns
    -------
    New extended dataset.
    """
    # 2023-09-20 raehik: old note from original import: "make this flexible"
    cycle_length = 360.0
    left = ds.roll({dim_name: nb_points}, roll_coords=True)
    right = left.isel({dim_name: slice(0, 2 * nb_points)})
    left[dim_name] = xr.concat(
        (left[dim_name][:nb_points] - cycle_length, left[dim_name][nb_points:]),
        dim_name,
    )
    right[dim_name] = xr.concat(
        (right[dim_name][:nb_points], right[dim_name][nb_points:] + cycle_length),
        dim_name,
    )
    return xr.concat((left, right), dim_name)


def compute_forcings_and_coarsen_cm2_6(
    u_v_dataset: xr.Dataset,
    grid_data: xr.Dataset,
    scale: int,
    nan_or_zero: str = "zero",
) -> xr.Dataset:
    """
    Coarsen and compute subgrid forcings for the given ocean surface velocities.
    Takes in high-resolution data, outputs low-resolution with associated
    subgrid forcings.

    Designed for CM2.6 simulation data.

    Rough outline:

      * apply a Gaussian filter
      * compute subgrid forcing from filtered data, save in filtered dataset
      * coarsen this amended filtered dataset

    See Guillaumin (2021) 2.2 for further details.

    Parameters
    ----------
    u_v_dataset : xarray Dataset
        High-resolution velocity field in "usurf" and "vsurf".
    grid_data : xarray Dataset
        High-resolution grid details.
    scale : float
        gaussian filtering & coarsening factor
    nan_or_zero: str, optional
        String set to either 'nan' or 'zero'. Determines whether we keep the
        nan values in the initial surface velocities array or whether we
        replace them by zeros before applying the procedure.
        In the second case, remaining zeros after applying the procedure will
        be replaced by NaNs for consistency.
        The default is 'zero'.

    Returns
    -------
    forcing : xarray Dataset
        Dataset containing the low-resolution velocity field in "usurf" and
        "vsurf", and forcing in data variables "S_x" and "S_y".
    """
    # Replace nan values with zeros.
    if nan_or_zero == "zero":
        u_v_dataset = u_v_dataset.fillna(0.0)

    # Interpolate temperature
    # interp_coords = dict(xt_ocean=u_v_dataset.coords['xu_ocean'],
    #                      yt_ocean=u_v_dataset.coords['yu_ocean'])
    # u_v_dataset['temp'] = u_v_dataset['surface_temperature'].interp(
    #     interp_coords)

    # High res advection terms
    adv = _advections(u_v_dataset, grid_data)
    # Filtered advections
    filtered_adv = _spatial_filter_dataset(adv, grid_data, scale/2)

    # Filtered u,v field and temperature
    u_v_filtered = _spatial_filter_dataset(u_v_dataset, grid_data, scale/2)
    # Advection term from filtered velocity field
    adv_filtered = _advections(u_v_filtered, grid_data)

    # Forcing
    ds_forcing = adv_filtered - filtered_adv
    ds_forcing = ds_forcing.rename({"adv_x": "S_x", "adv_y": "S_y"})
    # Merge filtered u,v, temperature and forcing terms
    ds_merged = ds_forcing.merge(u_v_filtered)
    logger.debug("uncoarsened forcings follow below:")
    logger.debug(ds_merged)

    # Coarsen
    ds_merged_coarse = ds_merged.coarsen(
        {"xu_ocean": int(scale), "yu_ocean": int(scale)}, boundary="trim"
    ).mean()

    if nan_or_zero == "zero":
        # Replace zeros with nans for consistency
        ds_merged_coarse = ds_merged_coarse.where(ds_merged_coarse["usurf"] != 0)

    # Specify input vs output type for each variable of the dataset. Might
    # be used later on for training or testing.
    ds_merged_coarse["S_x"].attrs["type"] = "output"
    ds_merged_coarse["S_y"].attrs["type"] = "output"
    ds_merged_coarse["usurf"].attrs["type"] = "input"
    ds_merged_coarse["vsurf"].attrs["type"] = "input"

    return ds_merged_coarse


def _advections(u_v_field: xr.Dataset, grid_data: xr.Dataset) -> xr.Dataset:
    """
    Compute advection terms corresponding to the passed velocity field.

    Parameters
    ----------
    u_v_field : xarray Dataset
        Velocity field, must contains variables "usurf" and "vsurf", coordinates
        "xu_ocean" and "yu_ocean"
    grid_data : xarray Dataset
        grid data, must contain variables "dxu" and "dyu"

    Returns
    -------
    result : xarray Dataset
        Advection components, under variable names "adv_x" and "adv_y"
    """
    dxu = grid_data["dxu"]
    dyu = grid_data["dyu"]
    gradient_x = u_v_field.diff(dim="xu_ocean") / dxu
    gradient_y = u_v_field.diff(dim="yu_ocean") / dyu
    # Interpolate back the gradients
    interp_coords = {
        "xu_ocean": u_v_field.coords["xu_ocean"],
        "yu_ocean": u_v_field.coords["yu_ocean"],
    }
    gradient_x = gradient_x.interp(interp_coords)
    gradient_y = gradient_y.interp(interp_coords)
    u, v = u_v_field["usurf"], u_v_field["vsurf"]
    adv_x = u * gradient_x["usurf"] + v * gradient_y["usurf"]
    adv_y = u * gradient_x["vsurf"] + v * gradient_y["vsurf"]
    result = xr.Dataset({"adv_x": adv_x, "adv_y": adv_y})
    # check if we can simply prevent the previous operation from adding chunks
    # result = result.chunk(dict(xu_ocean=-1, yu_ocean=-1))
    return result

def _spatial_filter_dataset(
        dataset: xr.Dataset, grid_data: xr.Dataset, sigma: float
        ) -> xr.Dataset:
    """
    Apply spatial filtering to the dataset across the spatial dimensions.

    Parameters
    ----------
    dataset : xarray Dataset
        Dataset to filter. First dimension must be time, followed by spatial dimensions
    grid_data: xarray Dataset
        grid data,  must include variables "dxu" and "dyu"
    sigma : float
        Scale of the filtering, same unit as those of the grid (often, meters)

    Returns
    -------
    filt_dataset : xarray Dataset
        Filtered dataset
    """
    area_u = grid_data["dxu"] * grid_data["dyu"] / 1e8

    # Normalisation term, so that if the quantity we filter is constant
    # over the domain, the filtered quantity is constant with the same value
    norm = xr.apply_ufunc(
        lambda x: gaussian_filter(x, sigma, mode="constant"),
        area_u,
        dask="parallelized",
        output_dtypes=[
            float,
        ],
    )
    filtered = xr.apply_ufunc(
        lambda x: _spatial_filter(x, sigma),
        dataset * area_u,
        dask="parallelized",
        output_dtypes=[
            float,
        ],
    )
    return filtered / norm

def _spatial_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian filter to spatial data.

    Apply scipy Gaussian filter to along all dimensions except first one, which
    corresponds to time.

    Parameters
    ----------
    data : ndarray
        Data to filter.
    sigma : float
        Unitless scale of the filter.

    Returns
    -------
    result : ndarray
        Filtered data
    """
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        data_t = data[t, ...]
        result_t = gaussian_filter(data_t, sigma, mode="constant")
        result[t, ...] = result_t
    return result
