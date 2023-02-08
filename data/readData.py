# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:13:22 2019

@author: Arthur
"""
# TODO Add more params to log, including some specifying the model used for the
# generation as well as its parameters (for instance, reynolds number, dt, etc)


import numpy as np
import matplotlib.pyplot as plt
from data._utils import load_tom_data, build_training_data, block_loop
import os.path
import mlflow

# Set a specific experiment name for the data
mlflow.set_tracking_uri('file:///d:\\Data sets\\NYU\\mlruns')
mlflow.set_experiment('data')
mlflow.start_run()

# Dynamic model parameters
# grid steps (m)
dx = 7.5e3
dy = dx
# Scale of the coarse-graining
scale_fine = (dx, dy)
scale_coarse = (35e3, 35e3)

factor = scale_coarse[0] / scale_fine[0]

# Logging parameters
mlflow.log_param('scale_coarse', scale_coarse)
mlflow.log_param('scale_fine', scale_fine)

# Dataset location
READ_DATA = True
filepath = 'D:\\Data sets\\NYU\\QG Data for Eugene-20191121T200122Z-002'
filepath_new_data = 'D:\\Data sets\\NYU\\processed_data'
filename = 'out513b-600-75-8_psi1a.mat'

# Read the data
data = load_tom_data(filepath, filename)

# For some reason the sample index is last in the stored data.
data = np.swapaxes(data, 0, 2)
n_times = data.shape[0]
print('nb of samples: {}'.format(n_times))
length_x, length_y = data.shape[1:]
print(data.shape[1:])

# Processing of the data
n_times_per_loop = 10
shape_result = (n_times, 3, int(length_x // factor), int(length_y // factor))
func = lambda x: build_training_data(x, dx, dy, scale_coarse[0])
processed_data = block_loop(n_times, n_times_per_loop, data,
                            func, shape_result)
psi_coarse = processed_data[:, 0, :, :]
sx_coarse = processed_data[:, 1, :, :]
sy_coarse = processed_data[:, 2, :, :]

print(psi_coarse.shape)
print(sx_coarse.shape)
print(sy_coarse.shape)

# Saving to disk
np.save(os.path.join(filepath_new_data, 'psi_coarse'), psi_coarse)
np.save(os.path.join(filepath_new_data, 'sx_coarse'), sx_coarse)
np.save(os.path.join(filepath_new_data, 'sy_coarse'), sy_coarse)

# log those as artifacts
mlflow.log_artifact(os.path.join(filepath_new_data, 'psi_coarse.npy'))
mlflow.log_artifact(os.path.join(filepath_new_data, 'sx_coarse.npy'))
mlflow.log_artifact(os.path.join(filepath_new_data, 'sy_coarse.npy'))

mlflow.end_run()
