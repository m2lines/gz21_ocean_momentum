# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:04:37 2020

@author: Arthur
Script to analyze the outputs from the 'multiscale' experiment, corresponding
to the script testing/multiscale.py.
"""

from analysis.loadmlflow import LoadMLFlow
from analysis.utils import select_run, view_predictions, DisplayMode
import mlflow

# We'll run this locally
mlflow.set_tracking_uri("file:///d:\\Data sets\\NYU\\mlruns")

# Setting the experiment
mlflow.set_experiment("multiscale")

# Select a run and load the predictions and targets for that id.
cols = ["params.scale_coarse", "params.scale_fine", "metrics.test mse"]
merge = [
    ("data", "params.data_run_id", "run_id"),
    ("Default", "params.model_run_id", "run_id"),
]
run_id, experiment_id = select_run(merge=merge, cols=cols)
loader = LoadMLFlow(run_id, experiment_id, "d:\\Data sets\\NYU\\mlruns")
predictions = loader.predictions[:1000, ...]
targets = loader.true_targets[:1000, ...]

view_predictions(
    predictions.take(0, axis=1), targets.take(0, axis=1), display_mode=DisplayMode.rmse
)
