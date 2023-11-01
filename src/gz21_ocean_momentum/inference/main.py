# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:25:09 2020

@author: Arthur
Here we test a trained model on an unseen region. The user is prompted to
select a trained model within a list and a new region to test that model.
Fine-tuning is an option through the n_epochs parameter of the script. 
If n_epochs takes is default value 0, no fine-tuning is performed.

We allow for different modes of training:
    - training of all parameters
    - training of last layer only
    - training of batch norm layers only

"""
from copy import deepcopy
from sys import modules
import os.path
import tempfile
import logging

import torch
import numpy as np
import mlflow
from torch.utils.data import DataLoader
import xarray as xr
from gz21_ocean_momentum.utils import select_run, select_experiment, TaskInfo
from gz21_ocean_momentum.train.utils import learning_rates_from_string
from gz21_ocean_momentum.data.datasets import (
    RawDataFromXrDataset,
    DatasetTransformer,
    Subset_,
    DatasetWithTransform,
    ComposeTransforms,
    MultipleTimeIndices,
    DatasetPartitioner,
)
from gz21_ocean_momentum.train.base import Trainer
from gz21_ocean_momentum.train import losses
from gz21_ocean_momentum.inference.utils import (
    create_large_test_dataset,
    BatchSampler,
    pickle_artifact,
)
from gz21_ocean_momentum.inference.metrics import MSEMetric, MaxMetric
from gz21_ocean_momentum.models.utils import load_model_cls
from gz21_ocean_momentum.models.transforms import SoftPlusTransform

import argparse


from dask.diagnostics import ProgressBar

from gz21_ocean_momentum.data.xrtransforms import SeasonalStdizer
from gz21_ocean_momentum.models import submodels


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=0)
parser.add_argument("--lr_ratio", type=float, default=1)
parser.add_argument("--train_mode", type=str, default="all")
parser.add_argument("--n_test_times", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--to_experiment", type=str, default="test")
parser.add_argument("--n_splits", type=int, default=1)

script_params = parser.parse_args()
n_epochs = script_params.n_epochs
lr_ratio = script_params.lr_ratio
to_experiment = script_params.to_experiment
n_test_times = script_params.n_test_times
batch_size = script_params.batch_size
n_splits = script_params.n_splits

# Location used to write generated data before it is logged through MLFlow
data_location = tempfile.mkdtemp()
model_output_dir = "model_output"

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Logging to experiment {to_experiment}...")
mlflow.set_experiment(to_experiment)
mlflow.start_run()

# Prompt user to retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
print("First, select a trained model (experiment then run)...")
models_experiment_id, _ = select_experiment()
cols = [
    "metrics.test loss",
    "start_time",
    "params.time_indices",
    "params.model_cls_name",
    "params.source.run-id",
    "params.submodel",
]
model_run = select_run(
    sort_by="start_time",
    cols=cols,
    experiment_ids=[
        models_experiment_id,
    ],
)

# Load some extra parameters of the model.
# TODO allow general time_indices
time_indices = [
    0,
]
train_split = float(model_run["params.train_split"])
test_split = float(model_run["params.test_split"])
batch_size = batch_size if batch_size else int(model_run["params.batchsize"])
model_module_name = model_run["params.model_module_name"]
model_cls_name = model_run["params.model_cls_name"]
loss_cls_name = model_run["params.loss_cls_name"]
learning_rates = learning_rates_from_string(model_run["params.learning_rate"])
submodel_name = model_run["params.submodel"]

learning_rate = learning_rates[0] * lr_ratio

# Load the model's file
client = mlflow.tracking.MlflowClient()
model_file = client.download_artifacts(model_run.run_id, "models/trained_model.pth")

# TODO temporary fix for backward compatibility
if not isinstance(submodel_name, str):
    submodel_name = "transform3"
submodel = getattr(submodels, submodel_name)

# metrics saved independently of the training criterion
metrics = {"mse": MSEMetric(), "Inf Norm": MaxMetric()}

# Prompt user to select the test dataset
print("Second, select a dataset (experiment and run)...")
data_experiment_id, _ = select_experiment()
cols = ["params.lat_min", "params.lat_max", "params.factor", "params.CO2"]
data_run = select_run(
    cols=cols,
    experiment_ids=[
        data_experiment_id,
    ],
)

# Recover the data (velocities and forcing)
client = mlflow.tracking.MlflowClient()
data_file = client.download_artifacts(data_run.run_id, "forcing")

mlflow.log_param("model_run_id", model_run.run_id)
mlflow.log_param("data_run_id", data_run.run_id)
mlflow.log_param("n_epochs", n_epochs)

# Read the dataset file
print("loading dataset...")
xr_dataset = xr.open_zarr(data_file)

with ProgressBar(), TaskInfo("Applying transforms to dataset"):
    xr_dataset = submodel.fit_transform(xr_dataset)

# To PyTorch Dataset
dataset = RawDataFromXrDataset(xr_dataset)
dataset.index = "time"
dataset.add_input("usurf")
dataset.add_input("vsurf")
dataset.add_output("S_x")
dataset.add_output("S_y")

if n_epochs > 0:
    train_index = int(train_split * len(dataset))
    test_index = int(test_split * len(dataset))
else:
    # TODO check this. Right now we have done this to align with chunks.
    train_index = batch_size
    test_index = batch_size

n_test_times = n_test_times if n_test_times else (len(dataset) - test_index)
train_dataset = Subset_(dataset, np.arange(train_index))

print("Adding transforms...")
features_transform_ = ComposeTransforms()
targets_transform_ = ComposeTransforms()
transform = DatasetTransformer(features_transform_, targets_transform_)
transform.fit(train_dataset)
dataset = DatasetWithTransform(dataset, transform)
# TODO implement back. Right now causing issues
# dataset = MultipleTimeIndices(dataset)
dataset.time_indices = [
    0,
]
train_dataset = Subset_(dataset, np.arange(train_index))
test_dataset = Subset_(dataset, np.arange(test_index, test_index + n_test_times))

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
partitioner = DatasetPartitioner(n_splits)
partition = partitioner.get_partition(test_dataset)
loaders = (
    DataLoader(d, batch_size=None, sampler=BatchSampler(d, batch_size))
    for d in partition
)
sampler = BatchSampler(test_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=None, sampler=sampler)

# Set inverse transform for metrics
for metric in metrics.values():
    metric.inv_transform = lambda x: test_dataset.inverse_transform_target(x)

# Set up training criterion and select parameters to train
try:
    n_targets = dataset.n_targets
    criterion = getattr(losses, loss_cls_name)(n_targets)
except AttributeError as e:
    raise type(e)("Could not find the loss class used for training, ", loss_cls_name)

# On first testdataset load the model. Or if we train to reset the model
logging.info("Creating the neural network model")
model_cls = load_model_cls(model_module_name, model_cls_name)
net = model_cls(dataset.n_features, criterion.n_required_channels)
transformation = pickle_artifact(model_run.run_id, "models/transformation")
net.final_transformation = transformation

# Load parameters of pre-trained model
logging.info("Loading the neural net parameters")
net.load_state_dict(torch.load(model_file))
print(net)

# Adding transforms required by the model
net.cpu()
dataset.add_transforms_from_model(net)

print("Size of training data: {}".format(len(train_dataset)))
print("Size of validation data : {}".format(len(test_dataset)))
print("Input height: {}".format(train_dataset.height))
print("Input width: {}".format(train_dataset.width))
print(train_dataset[0][0].shape)
print(train_dataset[0][1].shape)
print("Features transform: ", transform.transforms["features"].transforms)
print("Targets transform: ", transform.transforms["targets"].transforms)

# Net to GPU
with TaskInfo("Put neural network on device"):
    net.to(device)

print("width: {}, height: {}".format(dataset.width, dataset.height))


# Training itself
if n_epochs > 0:
    with TaskInfo("Training"):
        trainer = Trainer(net, device)
        trainer.criterion = criterion
        # Register metrics
        for metric_name, metric in metrics.items():
            trainer.register_metric(metric_name, metric)
        parameters = net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for i_epoch in range(n_epochs):
            train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
            test_loss, metrics_results = trainer.test(test_dataloader)
            print("Epoch {}".format(i_epoch))
            print("Train loss for this epoch is {}".format(train_loss))
            print("Test loss for this epoch is {}".format(test_loss))

    with TaskInfo("Validation"):
        train_loss, train_metrics_results = trainer.test(train_dataloader)
        print(f"Final train loss is {train_loss}")

# Test
with ProgressBar(), TaskInfo("Create output dataset"):
    out = create_large_test_dataset(net, criterion, partition, loaders, device)
    file_path = os.path.join(data_location, f"test_output_0")
    ProgressBar().register()
    print("Start of actual computations...")
    out = out.chunk(dict(time=32))
    out.to_zarr(file_path)
    mlflow.log_artifact(file_path)
    print(f"Size of output data is {out.nbytes/1e9} GB")

mlflow.end_run()
print("Done")
