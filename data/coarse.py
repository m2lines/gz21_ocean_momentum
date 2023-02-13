#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020

@author: arthur
"""

import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np
import logging


def advections(u_v_field: xr.Dataset, grid_data: xr.Dataset):
    """
    Return the advection terms corresponding to the passed velocity field.
    Note that the velocities sit on U-grids

    Parameters
    ----------
    u_v_field : xarray dataset
        Velocity field, must contains variables usurf and vsurf.
    grid_data : xarray dataset
        Dataset with grid details, must contain variables dxu and dyu.

    Returns
    -------
    advections : xarray dataset
        Advection components, under variable names adv_x and adv_y.

    """
    dxu = grid_data['dxu']
    dyu = grid_data['dyu']
    gradient_x = u_v_field.diff(dim='xu_ocean') / dxu
    gradient_y = u_v_field.diff(dim='yu_ocean') / dyu
    # Interpolate back the gradients
    interp_coords = dict(xu_ocean=u_v_field.coords['xu_ocean'],
                         yu_ocean=u_v_field.coords['yu_ocean'])
    gradient_x = gradient_x.interp(interp_coords)
    gradient_y = gradient_y.interp(interp_coords)
    u, v = u_v_field['usurf'], u_v_field['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = u * gradient_x['vsurf'] + v * gradient_y['vsurf']
    result = xr.Dataset({'adv_x': adv_x, 'adv_y': adv_y})
    # TODO check if we can simply prevent the previous operation from adding
    # chunks
    #result = result.chunk(dict(xu_ocean=-1, yu_ocean=-1))
    return result


def spatial_filter(data: np.ndarray, sigma: float):
    """
    Apply a gaussian filter along all dimensions except first one, which
    corresponds to time.

    Parameters
    ----------
    data : numpy array
        Data to filter.
    sigma : float
        Unitless scale of the filter.

    Returns
    -------
    result : numpy array
        Filtered data.

    """
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        data_t = data[t, ...]
        result_t = gaussian_filter(data_t, sigma, mode='constant')
        result[t, ...] = result_t
    return result


def spatial_filter_dataset(dataset: xr.Dataset, grid_info: xr.Dataset,
                           sigma: float):
    """
    Apply spatial filtering to the dataset across the spatial dimensions.

    Parameters
    ----------
    dataset : xarray dataset
        Dataset to which filtering is applied. Time must be the first
        dimension, whereas spatial dimensions must come after.
    grid_info : xarray dataset
        Dataset containing details on the grid, in particular must have
        variables dxu and dyu.
    sigma : float
        Scale of the filtering, same unit as those of the grid (often, meters)

    Returns
    -------
    filt_dataset : xarray dataset
        Filtered dataset.

    """
    area_u = grid_info['dxu'] * grid_info['dyu'] / 1e8
    dataset = dataset * area_u
    # Normalisation term, so that if the quantity we filter is constant
    # over the domain, the filtered quantity is constant with the same value
    norm = xr.apply_ufunc(lambda x: gaussian_filter(x, sigma, mode='constant'),
                          area_u, dask='parallelized', output_dtypes=[float, ])
    filtered = xr.apply_ufunc(lambda x: gaussian_filter(x, sigma, mode='constant'),
                              dataset,
                              input_core_dims=[['yu_ocean', 'xu_ocean']],
                              output_core_dims=[['yu_ocean', 'xu_ocean']],
                              dask='parallelized',
                              vectorize=True,
                              output_dtypes=[float, ])
    return filtered / norm


def eddy_forcing(u_v_dataset : xr.Dataset, grid_data: xr.Dataset,
                 scale: int, method: str = 'mean',
                 nan_or_zero: str = 'zero', scale_mode: str = 'factor',
                 debug_mode=False) -> xr.Dataset:
    """
    Compute the sub-grid forcing terms.

    Parameters
    ----------
    u_v_dataset : xarray dataset
        High-resolution velocity field.
    grid_data : xarray dataset
        High-resolution grid details.
    scale : float
        Scale, in meters, or factor, if scale_mode is set to 'factor'
    method : str, optional
        Coarse-graining method. The default is 'mean'.
    nan_or_zero: str, optional
        String set to either 'nan' or 'zero'. Determines whether we keep the
        nan values in the initial surface velocities array or whether we
        replace them by zeros before applying the procedure.
        In the second case, remaining zeros after applying the procedure will
        be replaced by nans for consistency.
        The default is 'zero'.
    scale_mode: str, optional
        DEPRECIATED, should always be left as 'factor'
    Returns
    -------
    forcing : xarray dataset
        Dataset containing the low-resolution velocity field and forcing.

    """
    # Replace nan values with zeros.
    if nan_or_zero == 'zero':
        u_v_dataset = u_v_dataset.fillna(0.0)
    if scale_mode == 'factor':
        print('Using factor mode')
        scale_x = scale
        scale_y = scale
    # Interpolate temperature
    # interp_coords = dict(xt_ocean=u_v_dataset.coords['xu_ocean'],
    #                      yt_ocean=u_v_dataset.coords['yu_ocean'])
    # u_v_dataset['temp'] = u_v_dataset['surface_temperature'].interp(
    #     interp_coords)

    scale_filter = (scale_x / 2, scale_y / 2)
    # High res advection terms
    adv = advections(u_v_dataset, grid_data)
    # Filtered advections
    filtered_adv = spatial_filter_dataset(adv, grid_data, scale_filter)
    # Filtered u,v field and temperature
    u_v_filtered = spatial_filter_dataset(u_v_dataset, grid_data, scale_filter)
    # Advection term from filtered velocity field
    adv_filtered = advections(u_v_filtered, grid_data)
    # Forcing
    forcing = adv_filtered - filtered_adv
    forcing = forcing.rename({'adv_x': 'S_x', 'adv_y': 'S_y'})
    # Merge filtered u,v, temperature and forcing terms
    forcing = forcing.merge(u_v_filtered)
    logging.debug(forcing)
    # Coarsen
    print('scale factor: ', scale)
    forcing_coarse = forcing.coarsen({'xu_ocean': int(scale_x),
                                      'yu_ocean': int(scale_y)},
                                     boundary='trim')
    if method == 'mean':
        forcing_coarse = forcing_coarse.mean()
    else:
        raise ValueError('Passed coarse-graining method not implemented.')
    if nan_or_zero == 'zero':
        # Replace zeros with nans for consistency
        forcing_coarse = forcing_coarse.where(forcing_coarse['usurf'] != 0)
    if not debug_mode:
        return forcing_coarse
    u_v_dataset = u_v_dataset.merge(adv)
    filtered_adv = filtered_adv.rename({'adv_x': 'f_adv_x',
                                        'adv_y': 'f_adv_y'})
    adv_filtered = adv_filtered.rename({'adv_x': 'adv_f_x',
                                        'adv_y': 'adv_f_y'})
    u_v_filtered = u_v_filtered.rename({'usurf': 'f_usurf',
                                        'vsurf': 'f_vsurf'})
    u_v_dataset = xr.merge((u_v_dataset, u_v_filtered, adv, filtered_adv,
                            adv_filtered, forcing[['S_x', 'S_y']]))
    return u_v_dataset, forcing_coarse
