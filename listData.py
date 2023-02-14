#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:39:23 2020

@author: arthur
Script that lists the data runs with the relevant information
"""

import mlflow
from mlflow.tracking import MlflowClient
import xarray as xr
import matplotlib.pyplot as plt
from src.gz_ocean_momentum.analysis import select_experiment


def show_data_sample(forcing, index: int):
    """Plots the data for the given index"""
    sample = forcing.isel(time=index)
    for var in sample.values():
        plt.figure()
        if var.name in ('usurf', 'vsurf'):
            var.plot(cmap='coolwarm')
            plt.figure()
            var.plot.hist(bins=100)
        elif var.name in ('S_x', 'S_y'):
            var.plot(cmap='coolwarm', vmin=-5, vmax=5)
            plt.figure()
            var.plot.hist(bins=100, range=(-5, 5))
    plt.show()


data_experiment_name = select_experiment()
data_experiment = mlflow.get_experiment_by_name(data_experiment_name)
data_experiment_id = data_experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[data_experiment_id, ])
runs_short = runs[['run_id', 'start_time', 'params.scale', 'params.ntimes', 
            'params.lat_min', 'params.lat_max', 
            'params.long_min', 'params.long_max']]

print('Select a run by entering its integer id (0, 1, ...) as listed\
      from below, in order to obtain more details.')
print(runs_short)
while True:
    selection = input('Select run: ')
    try:
        selection = int(selection)
    except ValueError as e:
        break
    print(runs.iloc[selection, :])
    run_id = runs.iloc[selection, :]['run_id']
    client = MlflowClient()
    forcing_data = client.download_artifacts(run_id, 'forcing')
    forcing = xr.open_zarr(forcing_data)
    forcing = forcing / forcing.std()
    show_data_sample(forcing, 0)

