#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routines for coarsening a dataset."""

import logging
import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np

def eddy_forcing(
    u_v_dataset: xr.Dataset,
    grid_data: xr.Dataset,
    scale: int,
    nan_or_zero: str = "zero",
) -> xr.Dataset:
    """
    Compute the sub-grid forcing terms using mean coarse-graining.

    Parameters
    ----------
    u_v_dataset : xarray Dataset
        High-resolution velocity field.
        Is changed in function.
    grid_data : xarray Dataset
        High-resolution grid details.
    scale : float
        factor (TODO)
    nan_or_zero: str, optional
        String set to either 'nan' or 'zero'. Determines whether we keep the
        nan values in the initial surface velocities array or whether we
        replace them by zeros before applying the procedure.
        In the second case, remaining zeros after applying the procedure will
        be replaced by nans for consistency.
        The default is 'zero'.

    Returns
    -------
    forcing : xarray Dataset
        Dataset containing the low-resolution velocity field and forcing.
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
    adv = advections(u_v_dataset, grid_data)
    # Filtered advections
    filtered_adv = spatial_filter_dataset(adv, grid_data, scale/2)
    # Filtered u,v field and temperature
    u_v_filtered = spatial_filter_dataset(u_v_dataset, grid_data, scale/2)
    # Advection term from filtered velocity field
    adv_filtered = advections(u_v_filtered, grid_data)
    # Forcing
    forcing = adv_filtered - filtered_adv
    forcing = forcing.rename({"adv_x": "S_x", "adv_y": "S_y"})
    # Merge filtered u,v, temperature and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # TODO logging
    #logging.debug(forcing)

    # Coarsen
    forcing_coarse = forcing.coarsen(
        {"xu_ocean": int(scale), "yu_ocean": int(scale)}, boundary="trim"
    )
    forcing_coarse = forcing_coarse.mean()

    if nan_or_zero == "zero":
        # Replace zeros with nans for consistency
        forcing_coarse = forcing_coarse.where(forcing_coarse["usurf"] != 0)

    # Specify input vs output type for each variable of the dataset. Might
    # be used later on for training or testing.
    forcing_coarse["S_x"].attrs["type"] = "output"
    forcing_coarse["S_y"].attrs["type"] = "output"
    forcing_coarse["usurf"].attrs["type"] = "input"
    forcing_coarse["vsurf"].attrs["type"] = "input"

    return forcing_coarse

def advections(u_v_field: xr.Dataset, grid_data: xr.Dataset):
    """
    Compute advection terms corresponding to the passed velocity field.

    Parameters
    ----------
    u_v_field : xarray Dataset
        Velocity field, must contains variables "usurf" and "vsurf"
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
    # TODO got "ValueError: zero-size array to reduction operation fmin which has
    # no identity" when given 0 bounding box
    gradient_x = gradient_x.interp(interp_coords)
    gradient_y = gradient_y.interp(interp_coords)
    u, v = u_v_field["usurf"], u_v_field["vsurf"]
    adv_x = u * gradient_x["usurf"] + v * gradient_y["usurf"]
    adv_y = u * gradient_x["vsurf"] + v * gradient_y["vsurf"]
    result = xr.Dataset({"adv_x": adv_x, "adv_y": adv_y})
    # TODO check if we can simply prevent the previous operation from adding
    # chunks
    # result = result.chunk(dict(xu_ocean=-1, yu_ocean=-1))
    return result

def spatial_filter_dataset(
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
        lambda x: spatial_filter(x, sigma),
        dataset * area_u,
        dask="parallelized",
        output_dtypes=[
            float,
        ],
    )
    return filtered / norm

def spatial_filter(data: np.ndarray, sigma: float):
    """
    Apply a gaussian filter to spatial data.

    Apply scipy gaussian filter to along all dimensions except first one, which
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
