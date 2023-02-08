#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:16:33 2020

@author: arthur
Unit tests for the coarse-graining operation from coarse.py
"""

import unittest
from xarray import DataArray
from xarray import Dataset
import numpy as np
from numpy import ma
from .coarse import spatial_filter_dataset, spatial_filter, eddy_forcing
import matplotlib.pyplot as plt

class TestEddyForcing(unittest.TestCase):
    def test_spatial_filter(self):
        a = np.random.randn(10, 4, 4)
        filtered_a = spatial_filter(a, 5)
        self.assertEqual(a.ndim, filtered_a.ndim)

    def test_spatial_filter_of_constant(self):
        """Check that when filtering over a constant field, the filtered
        field and the passed filed are equal to that constant."""
        a = np.ones((100, 5, 5))
        xs = np.arange(5)
        ys = np.arange(5) * 2
        times = np.arange(100)
        xs_, ys_ = np.meshgrid(xs, ys)
        dxs = np.ones_like(xs_)
        dys = np.ones_like(ys_) * 2
        dxs = DataArray(dxs, dims=('xu_ocean', 'yu_ocean'),
                        coords={'xu_ocean': xs, 'yu_ocean': ys})
        dys = DataArray(dys, dims=('xu_ocean', 'yu_ocean'),
                        coords={'xu_ocean': xs, 'yu_ocean': ys})
        grid_info = Dataset(dict(dxu=dxs, dyu=dys))
        data = Dataset(dict(a=DataArray(a,
                       dims=('time', 'xu_ocean', 'yu_ocean'),
                       coords={'time': times, 'xu_ocean': xs, 'yu_ocean': ys})))
        filtered_data = spatial_filter_dataset(data, grid_info, (5, 5))
        self.assertTrue(np.all(data['a'].values == filtered_data['a'].values
                               + 1))

    # def test_spatial_filter_dataset(self):
    #     a1 = DataArray(data = np.zeros((10, 4, 4)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(10) * 3,
    #                              'x' : np.arange(4) * 7,
    #                              'y' : np.arange(4) * 11})
    #     ds = Dataset({'var0' : a1})
    #     filtered = spatial_filter_dataset(ds, 2)
    #     self.assertEqual(filtered.dims, ds.dims)
    #
    # def test_spatial_filter_dataset2(self):
    #     a1 = DataArray(data = np.random.randn(1000, 40, 40),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     ds = Dataset({'var0' : a1})
    #     ds = ds.chunk({'time' : 100})
    #     filtered = spatial_filter_dataset(ds, 100).compute()
    #     filtered2 = spatial_filter(ds['var0'].compute().data, 100)
    #     # ds['var0'].isel(time=0).plot(cmap='coolwarm')
    #     # plt.figure()
    #     # filtered['var0'].isel(time=0).plot(cmap='coolwarm')
    #     # plt.figure()
    #     # plt.imshow(filtered2[0, :, :], cmap='coolwarm', origin='lower')
    #     # plt.colorbar()
    #     test = (filtered.to_array().values == filtered2).all()
    #     self.assertTrue(test.item())
    #
    # def test_spatial_filter_dataset3(self):
    #     """Checks that the average of the filtered dataset is equal to that
    #     of the original data within some threshold"""
    #     a1 = DataArray(data = np.random.randn(1000, 40, 40),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     ds = Dataset({'var0' : a1})
    #     filtered = spatial_filter_dataset(ds, sigma=3)
    #     mean_origin = ds.mean(dim='x').mean(dim='y')
    #     mean_filtered = filtered.mean(dim='x').mean(dim='y')
    #     self.assertTrue((mean_filtered - mean_origin
    #                      <= 0.1 * mean_origin ).all())
    #
    # def test_eddy_forcing(self):
    #     a1 = DataArray(data = np.random.randn(1000, 100, 100),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(100) * 10,
    #                              'y' : np.arange(100) * 11})
    #     a2 = DataArray(data = np.random.randn(1000, 100, 100),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(100) * 10,
    #                              'y' : np.arange(100) * 11})
    #     ds = Dataset({'usurf' : a1, 'vsurf' : a2})
    #     forcing = eddy_forcing(ds, 40)
    #     self.assertTrue(forcing.dims != ds.dims)
    #
    # def test_eddy_forcing2(self):
    #     a1 = DataArray(data = ma.zeros((100, 10, 10)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(100) * 3,
    #                              'x' : np.arange(10) * 10,
    #                              'y' : np.arange(10) * 11})
    #     a1.data.mask = np.zeros((100, 10, 10), dtype = np.bool)
    #     a1.data.mask[0] = True
    #     a2 = DataArray(data = np.zeros((100, 10, 10)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(100) * 3,
    #                              'x' : np.arange(10) * 10,
    #                              'y' : np.arange(10) * 11})
    #     ds = Dataset({'usurf' : a1, 'vsurf' : a2})
    #     forcing = eddy_forcing(ds, 40)

    # def test_advections(self):
    #     a1 = DataArray(data = np.zeros((1000, 40, 40)), 
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     a2= DataArray(data = np.zeros((1000, 40, 40)), 
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     ds = Dataset({'usurf' : a1, 'vsurf' : a2})
    #     adv = advections(ds)
    #     self.assertTrue((adv == 0).all().to_array().all().item())
