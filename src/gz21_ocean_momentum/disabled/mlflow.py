#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mlflow
import xarray as xr
import yaml

def load_data_from_run(run_id):
    """
    Load data from a previous run from mlflow files.

    TODO silly, should be removed

    Parameters
    ----------
    run_id : str
        unique mlflow identifier for run to load

    Returns
    -------
    xr_dataset : xr.dataset
        xarray dataset populated with data for requested run
    """
    mlflow_client = mlflow.tracking.MlflowClient()
    data_file = mlflow_client.download_artifacts(run_id, "forcing")
    xr_dataset = xr.open_zarr(data_file)
    return xr_dataset


def load_data_from_runs(run_ids):
    """
    Load data from previous runs from mlflow files.

    Parameters
    ----------
    run_id : list of str
        list of unique mlflow identifiers for runs to load

    Returns
    -------
    xr_datasets : list of xr.dataset
        list of xarray datasets populated with data for requested runs
    """
    xr_datasets = []
    for run_id in run_ids:
        xr_datasets.append(load_data_from_run(run_id))
    return xr_datasets


def load_training_datasets(ds: xr.Dataset, config_fname: str):
    """
    Load training data from a previous run from mlflow files.

    Parameters
    ----------
    run_id : str
        unique mlflow identifier for run to load

    Returns
    -------
    results : list of ???
        description?
    """
    results = []
    with open(config_fname, encoding="utf-8") as config_file:
        try:
            # AB TODO check that safe_load() is OK rather than load()
            # TODO 2023-05-12 raehik: `full_load()` used in another changeset.
            # safe_load gives errors.
            #subdomains = yaml.safe_load(config_file)
            subdomains = yaml.full_load(config_file)
        except FileNotFoundError as e:
            raise type(e)("Configuration file of subdomains not found")
        for subdomain in subdomains:
            coords = subdomain[1]
            lats = slice(coords["lat_min"], coords["lat_max"])
            lons = slice(coords["lon_min"], coords["lon_max"])
            results.append(ds.sel(xu_ocean=lons, yu_ocean=lats))
    return results
