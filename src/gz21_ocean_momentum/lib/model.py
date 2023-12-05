# Common functions relating to neural net model, training data.

import xarray as xr
import numpy as np
import torch.utils.data as torch

from gz21_ocean_momentum.common.assorted import at_idx_pct

from gz21_ocean_momentum.data.datasets import (
    DatasetWithTransform,
    DatasetTransformer,
    RawDataFromXrDataset,
    ConcatDataset_,
    Subset_,
    ComposeTransforms,
)

def cm26_xarray_to_torch(ds_xr: xr.Dataset) -> torch.Dataset:
    """
    Obtain a PyTorch `Dataset` view over an xarray dataset, specifically for
    CM2.6 ocean velocity data annotated with forcings in `S_x` and `S_y`.
    """
    ds_torch = RawDataFromXrDataset(ds_xr)
    ds_torch.index = "time"
    ds_torch.add_input("usurf")
    ds_torch.add_input("vsurf")
    ds_torch.add_output("S_x")
    ds_torch.add_output("S_y")
    return ds_torch

def gz21_train_data_subdomain_xr_to_torch(ds_xr: xr.Dataset) -> torch.Dataset:
    """
    Convert GZ21 training data (coarsened CM2.6 data with diagnosed forcings)
    into a PyTorch dataset.

    Intended to take in a single spatial subdomain of the "main" dataset.
    Apply submodel transforms first.
    Perform dataset splits after.
    """
    ds_torch = cm26_xarray_to_torch(ds_xr)

    # prep empty transform, filled in later by custom torch DataLoaders
    features_transform = ComposeTransforms()
    targets_transform = ComposeTransforms()
    transform = DatasetTransformer(features_transform, targets_transform)
    ds_torch_with_transform = DatasetWithTransform(ds_torch, transform)

    return ds_torch_with_transform

def prep_train_test_dataloaders(
        dss: list,
        pct_train_end:  float,
        pct_test_start: float,
        batch_size: int):
    """
    Split a list of PyTorch datasets into two dataloaders: one for training,
    one for testing.

    Parameters
    ----------
    pct_train_end: float
        Training data will be from 0->x of the dataset. 0<=x<=1

    pct_test_start: float
        Test data will be from x->end of the dataset. pct_train_end<=x<=1

    Returns
    -------
    Two PyTorch DataLoaders: train, test.
    """
    # split dataset according to requested lengths
    train_range = lambda x: np.arange(0, at_idx_pct(pct_train_end, x))
    test_range  = lambda x: np.arange(at_idx_pct(pct_test_start, x), len(x))

    train_datasets = [ Subset_(x, train_range(x)) for x in dss ]
    test_datasets  = [ Subset_(x, test_range(x))  for x in dss ]

    # Concatenate datasets. This adds shape transforms to ensure that all
    # regions produce fields of the same shape, hence should be called after
    # saving the transformation so that when we're going to test on another
    # region this does not occur.
    train_dataset = ConcatDataset_(train_datasets)
    test_dataset = ConcatDataset_(test_datasets)

    # Dataloaders
    train_dataloader = torch.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    test_dataloader = torch.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_dataloader, test_dataloader
