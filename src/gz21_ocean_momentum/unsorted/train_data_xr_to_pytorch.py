import xarray as xr
import numpy as np

from gz21_ocean_momentum.data.datasets import (
    DatasetWithTransform,
    DatasetTransformer,
    RawDataFromXrDataset,
    ConcatDataset_,
    Subset_,
    ComposeTransforms,
)

def at_idx_pct(pct: float, a) -> int:
    """
    Obtain the index into the given list-like to the given percent.

    e.g. `at_idx_pct(0.5, [0,1,2]) == 1`

    Must be able to `len(a)`.

    Invariant: `0<=pct<=1`.

    Returns a valid index into `a`.
    """
    return int(pct * len(a))

def gz21_train_data_subdomain_xr_to_torch(ds_xr: xr.Dataset):
    """
    Convert GZ21 training data (coarsened CM2.6 data with diagnosed forcings)
    into a PyTorch dataset.

    Intended to take in a single spatial subdomain of the "main" dataset.
    Apply submodel transforms first.
    Perform dataset splits after.
    """
    # TODO disabled
    #with ProgressBar(), TaskInfo("Computing dataset"):
        # Below line only for speeding up debugging
        # xr_dataset = xr_dataset.isel(time=slice(0, 1000))
        #xr_dataset = xr_dataset.compute()
    ds_torch = RawDataFromXrDataset(ds_xr)
    ds_torch.index = "time"
    ds_torch.add_input("usurf")
    ds_torch.add_input("vsurf")
    ds_torch.add_output("S_x")
    ds_torch.add_output("S_y")
    features_transform = ComposeTransforms()
    targets_transform = ComposeTransforms()
    transform = DatasetTransformer(features_transform, targets_transform)
    ds_torch_with_transform = DatasetWithTransform(ds_torch, transform)
    # dataset = MultipleTimeIndices(dataset)
    # dataset.time_indices = [0, ]
    return ds_torch_with_transform

def prep_train_test_dataloaders(
        dss: list,
        pct_train_end:  float,
        pct_test_start: float,
        batch_size: int):

    # split dataset according to requested lengths
    train_datasets = [ Subset_(x, np.arange(0, at_idx_pct(pct_train_end, x)))       for x in dss ]
    test_datasets  = [ Subset_(x, np.arange(at_idx_pct(pct_test_start, x), len(x))) for x in dss ]

    # Concatenate datasets. This adds shape transforms to ensure that all
    # regions produce fields of the same shape, hence should be called after
    # saving the transformation so that when we're going to test on another
    # region this does not occur.
    train_dataset = ConcatDataset_(train_datasets)
    test_dataset = ConcatDataset_(test_datasets)

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_dataloader, test_dataloader
