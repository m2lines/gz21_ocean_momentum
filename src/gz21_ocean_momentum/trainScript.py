#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script that performs training of a model on data."""
import os
import os.path
import copy
import argparse
import importlib
import pickle
import tempfile
from dask.diagnostics import ProgressBar
import numpy as np
import mlflow
import xarray as xr

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

# These imports are used to create the training datasets
from data.datasets import (
    DatasetWithTransform,
    DatasetTransformer,
    RawDataFromXrDataset,
    ConcatDataset_,
    Subset_,
    ComposeTransforms,
)

# Some utils functions
from train.utils import (
    DEVICE_TYPE,
    learning_rates_from_string,
)
from train.base import Trainer
from inference.utils import create_test_dataset
from inference.metrics import MSEMetric, MaxMetric
import train.losses
from models import transforms, submodels


from utils import TaskInfo

import gz21_ocean_momentum.step.train.lib as lib
from   gz21_ocean_momentum.common.bounding_box import load_bounding_boxes_yaml

from typing import Any

torch.autograd.set_detect_anomaly(True)


def _check_dir(dir_path):
    """
    Create directory if it does not already exist.

    Parameters
    ----------
    dir_path : str
        string of directory to check/make
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def negative_int(value: str):
    """
    Convert string input to negative integer.

    Parameters
    ----------
    value : str
        string to convert

    Returns
    -------
    : int
        negative integer of input string
    """
    return -int(value)


def check_str_is_None(string_in: str):
    """
    Return None if string is "none".

    Parameters
    ----------
    string_in : str
        string to check

    Returns
    -------
    string_in or None : str or None
        returns None if string_in is none, else returns string_in
    """
    return None if string_in.lower() == "none" else string_in


# --------------------
# READ IN DATA FOR RUN
# --------------------
description = (
    "Trains a model on a chosen dataset from the store."
    "Allows to set training parameters via the CLI."
    "Use one of either --run-id or --forcing-data-path."
)
parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "--run-id",
    type=str,
    help="MLflow run ID of data step containing forcing data to use",
)

# access input forcing data via absolute filepath
parser.add_argument(
    "--forcing-data-path", type=str, help="Filepath of the forcing data"
)

parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument(
    "--learning_rate", type=learning_rates_from_string, default={"0\1e-3"}
)
parser.add_argument("--train_split", type=float, default=0.8, help="Between 0 and 1")
parser.add_argument(
    "--test_split",
    type=float,
    default=0.8,
    help="Between 0 and 1, greater than train_split.",
)
parser.add_argument("--time_indices", type=negative_int, nargs="*")
parser.add_argument("--printevery", type=int, default=20)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.05,
    help="Depreciated. Controls the weight decay on the linear " "layer",
)
parser.add_argument(
    "--model_module_name",
    type=str,
    default="models.fully_conv_net",
    help="Name of the module containing the nn model",
)
parser.add_argument(
    "--model_cls_name",
    type=str,
    default="FullyCNN",
    help="Name of the class defining the nn model",
)
parser.add_argument(
    "--loss_cls_name",
    type=str,
    default="HeteroskedasticGaussianLossV2",
    help="Name of the loss function used for training.",
)
parser.add_argument(
    "--transformation_cls_name",
    type=str,
    default="SquareTransform",
    help="Name of the transformation applied to outputs "
    "required to be positive. Should be defined in "
    "models.transforms.",
)
parser.add_argument("--submodel", type=str, default="transform1")
parser.add_argument(
    "--features_transform_cls_name", type=str, default="None", help="Depreciated"
)
parser.add_argument(
    "--targets_transform_cls_name", type=str, default="None", help="Depreciated"
)
parser.add_argument(
    "--subdomains-file", type=str, required=True, help="YAML file describing subdomains to use (bounding boxes. TODO format"
)
params = parser.parse_args()


def argparse_get_mlflow_artifact_path_or_direct_or_fail(
    mlflow_artifact_name: str, params: dict[str, Any]
) -> str:
    """Obtain a filepath either from an MLflow run ID and artifact name, or a
    direct path if provided.

    params must have keys run_id and forcing_data_path.

    Only one of run_id and path should be non-None.

    Note that the filepath is not checked for validity (but for run_id, MLflow
    probably will assert that it exists).

    Effectful: errors result in immediate program exit.
    """
    if params.run_id is not None and params.run_id != "None":
        if params.forcing_data_path is not None and params.forcing_data_path != "None":
            # got run ID and direct path: bad
            raise TypeError(
                "overlapping options provided (--forcing-data-path and --exp-id)"
            )

        # got only run ID: obtain path via MLflow
        mlflow.log_param("source.run-id", params.run_id)
        mlflow_client = mlflow.tracking.MlflowClient()
        return mlflow_client.download_artifacts(params.run_id, mlflow_artifact_name)

    if params.forcing_data_path is not None and params.forcing_data_path != "None":
        # got only direct path: use
        return params.forcing_data_path

    # if we get here, neither options were provided
    raise TypeError("require one of --run-id or --forcing-data-path")


forcings_path = argparse_get_mlflow_artifact_path_or_direct_or_fail("forcing", params)

# --------------------------
# SET UP TRAINING PARAMETERS
# --------------------------
# Note that we use two indices for the train/test split. This is because we
# want to avoid the time correlation to play in our favour during test.
batch_size = params.batchsize
learning_rates = params.learning_rate
weight_decay = params.weight_decay
n_epochs = params.n_epochs
train_split = params.train_split
test_split = params.test_split
model_module_name = params.model_module_name
model_cls_name = params.model_cls_name
loss_cls_name = params.loss_cls_name
transformation_cls_name = params.transformation_cls_name
# Transforms applied to the features and targets
temp = params.features_transform_cls_name
features_transform_cls_name = check_str_is_None(temp)
temp = params.targets_transform_cls_name
targets_transform_cls_name = check_str_is_None(temp)
# Submodel (for instance monthly means)
submodel = params.submodel


# --------------------------
# SET UP INPUT PARAMETERS
# --------------------------
# Parameters specific to the input data
# past specifies the indices from the past that are used for prediction
indices = params.time_indices

# Other parameters
print_loss_every = params.printevery
MODEL_NAME = "trained_model.pth"

# Directories where temporary data will be saved
data_location = tempfile.mkdtemp()
print("Created temporary dir at  ", data_location)

FIGURES_DIRECTORY = "figures"
MODELS_DIRECTORY = "models"
MODEL_OUTPUT_DIR = "model_output"

for directory in [FIGURES_DIRECTORY, MODELS_DIRECTORY, MODEL_OUTPUT_DIR]:
    _check_dir(os.path.join(data_location, directory))

# Device selection. If available we use the GPU.
# TODO Allow CLI argument to select the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = DEVICE_TYPE.GPU if torch.cuda.is_available() else DEVICE_TYPE.CPU
print("Selected device type: ", device_type.value)


# ------------------
# LOAD TRAINING DATA
# ------------------
global_ds = xr.open_zarr(forcings_path)
subdomains = load_bounding_boxes_yaml(params.subdomains_file)
xr_datasets = lib.select_subdomains(global_ds, subdomains)
# Split into train and test datasets
datasets, train_datasets, test_datasets = [], [], []


for xr_dataset in xr_datasets:
    # TODO this is a temporary fix to implement seasonal patterns
    submodel_transform = copy.deepcopy(getattr(submodels, submodel))
    print(submodel_transform)
    xr_dataset = submodel_transform.fit_transform(xr_dataset)
    with ProgressBar(), TaskInfo("Computing dataset"):
        # Below line only for speeding up debugging
        # xr_dataset = xr_dataset.isel(time=slice(0, 1000))
        xr_dataset = xr_dataset.compute()
    print(xr_dataset)
    dataset = RawDataFromXrDataset(xr_dataset)
    dataset.index = "time"
    dataset.add_input("usurf")
    dataset.add_input("vsurf")
    dataset.add_output("S_x")
    dataset.add_output("S_y")
    # TODO temporary addition, should be made more general
    if submodel == "transform2":
        dataset.add_output("S_x_d")
        dataset.add_output("S_y_d")
    if submodel == "transform4":
        dataset.add_input("s_x_formula")
        dataset.add_input("s_y_formula")
    train_index = int(train_split * len(dataset))
    test_index = int(test_split * len(dataset))
    features_transform = ComposeTransforms()
    targets_transform = ComposeTransforms()
    transform = DatasetTransformer(features_transform, targets_transform)
    dataset = DatasetWithTransform(dataset, transform)
    # dataset = MultipleTimeIndices(dataset)
    # dataset.time_indices = [0, ]
    train_dataset = Subset_(dataset, np.arange(train_index))
    test_dataset = Subset_(dataset, np.arange(test_index, len(dataset)))
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    datasets.append(dataset)

# Concatenate datasets. This adds shape transforms to ensure that all regions
# produce fields of the same shape, hence should be called after saving
# the transformation so that when we're going to test on another region
# this does not occur.
train_dataset = ConcatDataset_(train_datasets)
test_dataset = ConcatDataset_(test_datasets)

# Dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)

print(f"Size of training data: {len(train_dataset)}")
print(f"Size of validation data : {len(test_dataset)}")


# -------------------
# LOAD NEURAL NETWORK
# -------------------
# Load the loss class required in the script parameters
n_target_channels = datasets[0].n_targets
criterion = getattr(train.losses, loss_cls_name)(n_target_channels)

# Recover the model's class, based on the corresponding CLI parameters
try:
    models_module = importlib.import_module(model_module_name)
    model_cls = getattr(models_module, model_cls_name)
except ModuleNotFoundError as e:
    raise type(e)("Could not find the specified module for : " + str(e))
except AttributeError as e:
    raise type(e)("Could not find the specified model class: " + str(e))
net = model_cls(datasets[0].n_features, criterion.n_required_channels)
try:
    transformation_cls = getattr(transforms, transformation_cls_name)
    transformation = transformation_cls()
    transformation.indices = criterion.precision_indices
    net.final_transformation = transformation
except AttributeError as e:
    raise type(e)("Could not find the specified transformation class: " + str(e))

print("--------------------")
print(net)
print("--------------------")
print("***")


# Log the text representation of the net into a txt artifact
with open(
    os.path.join(data_location, MODELS_DIRECTORY, "nn_architecture.txt"),
    "w",
    encoding="utf-8",
) as f:
    print("Writing neural net architecture into txt file.")
    f.write(str(net))

# Add transforms required by the model.
for dataset in datasets:
    dataset.add_transforms_from_model(net)


# -------------------
# TRAINING OF NETWORK
# -------------------
# Adam optimizer
# To GPU
net.to(device)

# Optimizer and learning rate scheduler
params = list(net.parameters())
optimizer = optim.Adam(params, lr=learning_rates[0], weight_decay=weight_decay)
lr_scheduler = MultiStepLR(optimizer, list(learning_rates.keys())[1:], gamma=0.1)

trainer = Trainer(net, device)
trainer.criterion = criterion
trainer.print_loss_every = print_loss_every

# metrics saved independently of the training criterion.
metrics = {"R2": MSEMetric(), "Inf Norm": MaxMetric()}
for metric_name, metric in metrics.items():
    metric.inv_transform = lambda x: test_dataset.inverse_transform_target(x)
    trainer.register_metric(metric_name, metric)

for i_epoch in range(n_epochs):
    print(f"Epoch number {i_epoch}.")
    # TODO remove clipping?
    train_loss = trainer.train_for_one_epoch(
        train_dataloader, optimizer, lr_scheduler, clip=1.0
    )
    test = trainer.test(test_dataloader)
    if test == "EARLY_STOPPING":
        print(test)
        break
    test_loss, metrics_results = test
    # Log the training loss
    print("Train loss for this epoch is ", train_loss)
    print("Test loss for this epoch is ", test_loss)

    for metric_name, metric_value in metrics_results.items():
        print(f"Test {metric_name} for this epoch is {metric_value}")
    mlflow.log_metric("train loss", train_loss, i_epoch)
    mlflow.log_metric("test loss", test_loss, i_epoch)
    mlflow.log_metrics(metrics_results)
# Update the logged number of actual training epochs
mlflow.log_param("n_epochs_actual", i_epoch + 1)


# ------------------------------
# SAVE THE TRAINED MODEL TO DISK
# ------------------------------
net.cpu()
full_path = os.path.join(data_location, MODELS_DIRECTORY, MODEL_NAME)
torch.save(net.state_dict(), full_path)
net.to(device=device)

# Save other parts of the model
# TODO this should not be necessary
print("Saving other parts of the model")
full_path = os.path.join(data_location, MODELS_DIRECTORY, "transformation")
with open(full_path, "wb") as f:
    pickle.dump(transformation, f)

with TaskInfo("Saving trained model"):
    mlflow.log_artifact(os.path.join(data_location, MODELS_DIRECTORY))


# ----------
# DEBUT TEST
# ----------
for i_dataset, dataset, test_dataset, xr_dataset in zip(
    range(len(datasets)), datasets, test_datasets, xr_datasets
):
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    output_dataset = create_test_dataset(
        net,
        criterion.n_required_channels,
        xr_dataset,
        test_dataset,
        test_dataloader,
        test_index,
        device,
    )

    # Save model output on the test dataset
    output_dataset.to_zarr(
        os.path.join(data_location, MODEL_OUTPUT_DIR, f"test_output{i_dataset}")
    )


# -----------------------
# LOG ARTIFACTS IN MLFLOW
# -----------------------
print("Logging artifacts...")
mlflow.log_artifact(os.path.join(data_location, FIGURES_DIRECTORY))
mlflow.log_artifact(os.path.join(data_location, MODEL_OUTPUT_DIR))
print("Done...")
