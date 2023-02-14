#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:13:35 2020

@author: arthur
"""
import numpy as np
import xarray as xr
import mlflow
import os.path
from scipy.stats import norm


class TestDataset:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        return self.ds[i]

    def __setitem__(self, name, value):
        self.ds[name] = value

    def errors(self, normalized=False):
        sx_error = self['S_xpred'] - self['S_x']
        sy_error = self['S_ypred'] - self['S_y']
        if normalized:
            sx_error *= self['S_xscale']
            sy_error *= self['S_yscale']
        return xr.Dataset({'S_x': sx_error, 'S_y': sy_error})

    def rmse(self, dim: str, normalized=False):
        errors = self.errors(normalized)
        return np.sqrt((errors['S_x']**2 + errors['S_y']**2).mean(dim=dim))

    def __getattr__(self, attr_name):
        if hasattr(self.ds, attr_name):
            return getattr(self.ds, attr_name)
        else:
            raise AttributeError()

    def __setattr__(self, name, value):
        if name != 'ds' and hasattr(self.ds, name):
            setattr(self.ds, name, value)
        else:
            self.__dict__[name] = value


def get_test_datasets(run_id: str):
    """Return a list of the test datasets for the provided run id"""
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    test_outputs = list()
    for a in artifacts:
        basename = os.path.basename(a.path)
        print('.', basename, '.')
        if basename.startswith('test_output_'):
            print('loading')
            ds = xr.open_zarr(client.download_artifacts(run_id, basename))
            test_outputs.append(TestDataset(ds))
    return test_outputs


class DataQuantiles:
    def __init__(self):
        pass

    def __get__(self, obj, type=None):
        if not obj._data_quantiles_computed:
            obj._update_data_quantiles()
        return obj._data_quantiles

    def __set__(self, obj, value):
        raise NotImplementedError('Cannot set the data quantiles manually.')


class QuantileCompare:
    """A class to compare the quantiles of the data with that of a given
    distribution"""

    data_quantiles = DataQuantiles()
    default_dim = 'time'

    def __init__(self, distribution=norm, quantiles=[]):
        self.quantiles = quantiles
        self.distribution = distribution
        self._data_quantiles_computed = False
        self.dim = self.default_dim

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        self._distribution = value
        self._update_quantiles()

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, value):
        self._quantiles = {k: None for k in value}
        self._update_quantiles()
        self._data_quantiles_computed = False

    def _update_quantiles(self):
        if hasattr(self, 'distribution'):
            for k in self.quantiles.keys():
                self.quantiles[k] = self.distribution.ppf(k)

    def _update_data_quantiles(self):
        if not hasattr(self, 'data'):
            raise AttributeError('The data has not been set.')
        self._data_quantiles = {k: self.data.quantile(k, dim=self.dim)
                                for k in self.quantiles.keys()}

    def qq_diff(self):
        return {k: self.data_quantiles[k] - v
                for k,v in self.quantiles.items()}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: xr.DataArray):
        self._data = value
