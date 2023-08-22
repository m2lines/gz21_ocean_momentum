#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for handling data."""
import mlflow
import xarray as xr

def cyclize(ds: xr.Dataset, coord_name: str, nb_points: int):
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
