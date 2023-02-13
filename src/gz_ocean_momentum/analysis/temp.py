# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:08:24 2020

@author: Arthur
Analysis script.

TODO
-Allow to load a trained model to do some tests on it. For instance I'd like
to check what a zero input gives, see if it can explain the behaviour on 
the east border, and see if we can correct that by enforcing zero bias on all
layers.
-also show the input field for analysis
-might want to study the correlation between the input field and the error,
see if there is anything remaining.
- do something similar for the multiscale
"""
# These two lines are required to plot over ssh
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import numpy as np
from .utils import select_run, view_predictions, DisplayMode
from .utils import play_movie
import mlflow
from mlflow.tracking import MlflowClient
import argparse

# Parse parameter
parser =  argparse.ArgumentParser()
parser.add_argument('--time', type=int, default=0)
params = parser.parse_args()
time = params.time



# If the runs dataframe already exists we use it. Note: this means you must
# restart the interpreter if the list of runs has changed.
run = select_run(sort_by='metrics.test mse', experiment_ids=['2',])

# Display some info about the train and validation sets for this run
train_split = run['params.train_split']
test_split = run['params.test_split']
print(f'Train split: {train_split}')
print(f'Test split: {test_split}')

# Download predictions and targets arrays
client = MlflowClient()
run_id = run['run_id']
predictions_file = client.download_artifacts(run_id, 
                                             'model_output/predictions.npy')
targets_file = client.download_artifacts(run_id, 'model_output/truth.npy')
predictions = np.load(predictions_file)
targets = np.load(targets_file)

# Plot the sample at the given time
plt.figure()
plt.subplot(121)
plt.imshow(targets[time, 0, ...], cmap='coolwarm', vmin=-2, vmax=2)
plt.colorbar()
plt.subplot(122)
plt.imshow(predictions[time, 0, ...], cmap='coolwarm', vmin=-2, vmax=2)
plt.colorbar()
plt.show()

# Plot the time series of the MSE
mse_time_series = np.mean(np.power(predictions - targets, 2), 
                          axis = (1, 2, 3))
plt.figure()
plt.plot(mse_time_series)
plt.show()

min_pred, max_pred = np.min(predictions), np.max(predictions)
print(f'Min prediction is {min_pred}')
print(f'Max prediction is {max_pred}')

# TODO Also show input data
predictions = predictions[:, 0, ...]
targets = targets[:, 0, ...]
view_predictions(predictions, targets, DisplayMode.rmse)
view_predictions(predictions, targets, DisplayMode.correlation)

#play_movie(truth, title='target')
animation1 = play_movie(predictions, title='prediction')
