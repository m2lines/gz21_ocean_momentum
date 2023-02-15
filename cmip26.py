#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute subgrid forcing on requested dataset.

Script to compute the subgrid forcing for a given region, using
data from cmip2.6 on one of the pangeo data catalogs.
Command line parameters include region specification.
Reads data from the CM2.6 and apply coarse graining.
Stores the resulting dataset into an MLFLOW
experiment within a specific run.

"""
from os.path import join

import argparse
import configparser
import tempfile

import json
from dataclasses import dataclass
import xarray as xr
from dask.diagnostics import ProgressBar
import mlflow

from gz_ocean_momentum.data.utils import cyclize_dataset
from gz_ocean_momentum.data.coarse import eddy_forcing
from gz_ocean_momentum.data.pangeo_catalog import get_patch


@dataclass
class RunParams:
    """
    dataclass to hold variables defining a particular dataset configuration.

    Attributes
    ----------
    ntimes : int
        number of days starting from first day
    bounds : list of float
        minimum and maximum latitudes and longitudes in degrees to run for
        [min_lat, max_lat, min_lon, max_lon]
    factor : int
        factor of degrading, should be integer > 1, default = 0.
    chunk_size : int
        chunk size along the time dimension, default = 50
    CO2 : int
        CO2 level, O (control) or 1 (1 percent CO2 increase)
    global_data : bool
        True if global data. In this case the data is made cyclic along longitude.
    """

    ntimes = 10000
    bounds = [-85.0, 85.0, -280.0, 80.0]
    # TODO says default value of factor is 0, but also that it has to be > 1??
    factor = 0
    chunk_size = 50
    CO2 = 0
    global_data = False

    @classmethod
    def load_json_params(cls, jsonpath):
        """
        Load parameters for dataset and processing from a json file into dataclass.

        Parameters
        ----------
        jsonpath : str
            path to json file

        Returns
        -------
        json_run_params : RunParams
            dataclass populated with data read from json file

        """
        json_run_params = cls()
        with open(jsonpath, "r", encoding="utf-8") as read_file:
            paramsdict = json.load(read_file)
        json_run_params.ntimes = paramsdict["ntimes"]
        json_run_params.bounds = [
            paramsdict["lat_min"],
            paramsdict["lat_max"],
            paramsdict["lon_min"],
            paramsdict["lon_max"],
        ]
        json_run_params.factor = paramsdict["factor"]
        json_run_params.chunk_size = paramsdict["chunk_size"]
        json_run_params.CO2 = paramsdict["CO2"]
        json_run_params.global_data = paramsdict["global_data"]

        return json_run_params


# read config file
# This is unneccessary when MLFlow removed
config = configparser.ConfigParser()
config.read("config.ini")

# Script parameters
CATALOG_URL = (
    "https://raw.githubusercontent.com/"
    "pangeo-data/pangeo-datastore/master/intake-catalogs/master.yaml"
)


# Parse parameters file from command line, or use default values defined above
DESCRIPTION = (
    "Read data from the CM2.6 and "
    "apply coarse graining. Stores the resulting dataset into an MLFLOW "
    "experiment within a specific run."
)
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    "--paramsfile", type=str, help="path to json file with run parameters."
)
input_args = parser.parse_args()
if input_args.paramsfile:
    try:
        params = RunParams.load_json_params(input_args.paramsfile)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Input json file of run parameters "
            f"'{input_args.paramsfile}' not found.\nCheck file exists."
        ) from e
else:
    params = RunParams()

if params.ntimes < 1:
    raise ValueError(f"'ntimes' should be > 0, currently set to {params.ntimes}.")
# TODO: need to check what acceptable values for factor are. See above.
if params.factor <= 1:
    raise ValueError(f"'factor' should be > 1, currently set to {params.factor}.")
if not params.CO2 in [0, 1]:
    raise ValueError(f"'CO2' should be 0 or 1, currently set to {params.CO2}.")

# Retrieve the patch of data specified in the input args from pangeo
patch_data, grid_data = get_patch(
    CATALOG_URL, params.ntimes, params.bounds, params.CO2, "usurf", "vsurf"
)

# If global data, make the dataset cyclic along longitude
if params.global_data == 1:
    # TODO this breaks currently. Not urgent though.
    patch_data = cyclize_dataset(patch_data, "xu_ocean", params.factor)
    grid_data = cyclize_dataset(grid_data, "xu_ocean", params.factor)
    # Rechunk along the cyclized dimension
    patch_data = patch_data.chunk({"xu_ocean": -1})
    grid_data = grid_data.chunk({"xu_ocean": -1})

# grid data is saved locally, no need for dask
grid_data = grid_data.compute()

# Calculate eddy-forcing dataset for that particular patch
scale_m = params.factor


def func(block):
    """
    Description?.

    Parameters
    ----------
    block : type?
        description?

    Returns
    -------
    eddy_forcing : type?
        description?

    """
    return eddy_forcing(block, grid_data, scale=scale_m)


template = patch_data.coarsen(
    {"xu_ocean": int(scale_m), "yu_ocean": int(scale_m)}, boundary="trim"
).mean()
template2 = template.copy()
template2 = template2.rename({"usurf": "S_x", "vsurf": "S_y"})
template = xr.merge((template, template2))
forcing = xr.map_blocks(func, patch_data, template=template)
# forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean',
#                        scale_mode='factor')

# Progress bar
ProgressBar().register()

# Specify input vs output type for each variable of the dataset. Might
# be used later on for training or testing.
forcing["S_x"].attrs["type"] = "output"
forcing["S_y"].attrs["type"] = "output"
forcing["usurf"].attrs["type"] = "input"
forcing["vsurf"].attrs["type"] = "input"

# Crop according to bounds passed as CLI options
bounds = params.bounds
forcing = forcing.sel(
    xu_ocean=slice(bounds[2], bounds[3]), yu_ocean=slice(bounds[0], bounds[1])
)

print(forcing)

# export data
data_location = tempfile.mkdtemp(dir=config["MLFLOW"]["TEMP_DATA_LOCATION"])
forcing.to_zarr(join(data_location, "forcing"), mode="w")

# Log processed dataset as an artifact the created zarr
mlflow.log_artifact(join(data_location, "forcing"))
