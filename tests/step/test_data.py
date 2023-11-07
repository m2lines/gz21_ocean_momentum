#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the data step."""

import pytest
import xarray as xr
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import gz21_ocean_momentum.step.data.lib as lib

class TestEddyForcing:
    "Class to test eddy forcing routines."

    def test_spatial_filter(self):
        """
        Check that number of dimensions stay the same through spatial_filter().
        """
        a = np.random.randn(10, 4, 4)
        filtered_a = lib._spatial_filter(a, 5)
        assert a.ndim == filtered_a.ndim

    def test_spatial_filter_of_constant(self):
        """
        Check that a constant field is unchanged by spatial_filter() within tolerance.
        """
        xs = np.arange(5)
        ys = np.arange(5) * 2
        times = np.arange(100)
        xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')
        dxs = np.ones_like(xs_)
        dys = np.ones_like(ys_) * 2

        dxs = xr.DataArray(
            dxs, dims=("xu_ocean", "yu_ocean"), coords={"xu_ocean": xs, "yu_ocean": ys}
        )
        dys = xr.DataArray(
            dys, dims=("xu_ocean", "yu_ocean"), coords={"xu_ocean": xs, "yu_ocean": ys}
        )
        grid_info = xr.Dataset({"dxu": dxs, "dyu": dys})
        data = xr.Dataset(
            {
                "a": xr.DataArray(
                    np.ones((100, 5, 5)),
                    dims=("time", "xu_ocean", "yu_ocean"),
                    coords={"time": times, "xu_ocean": xs, "yu_ocean": ys},
                )
            }
        )

        filtered_data = lib._spatial_filter_dataset(data, grid_info, (5, 5))

        assert data["a"].values == pytest.approx(filtered_data["a"].values)

    def test_eddy_forcing_chunks(self):
        """
        Check that the computations are done along the spatial dimensions.
        """
        xs = np.arange(100)
        ys = np.arange(100) * 2
        times = np.arange(2)
        xs_, ys_ = np.meshgrid(xs, ys)
        dxs = np.ones_like(xs_)
        dys = np.ones_like(ys_) * 2

        # TODO renamed vars. confusing
        dxs = xr.DataArray(
                dxs, dims=("xu_ocean", "yu_ocean"), coords={"xu_ocean": xs, "yu_ocean": ys}
        )
        dys = xr.DataArray(
                dys, dims=("xu_ocean", "yu_ocean"), coords={"xu_ocean": xs, "yu_ocean": ys}
        )
        grid_info = xr.Dataset({"dxu": dxs, "dyu": dys})
        data = xr.Dataset(
                {
                    "usurf": xr.DataArray(
                            np.stack((np.zeros((100, 100)),
                                      np.ones((100, 100))), axis=0),
                            dims=("time", "xu_ocean", "yu_ocean"),
                            coords={"time": times, "xu_ocean": xs, "yu_ocean": ys},
                    ),
                    "vsurf": xr.DataArray(
                            np.stack((np.zeros((100, 100)),
                                      np.ones((100, 100))), axis=0),
                            dims=("time", "xu_ocean", "yu_ocean"),
                            coords={"time": times, "xu_ocean": xs, "yu_ocean": ys},
                    ),
                }
        )
        forcing = lib.compute_forcings_and_coarsen_cm2_6(data, grid_info, 4)
        usurf_0, usurf_1 = forcing.usurf.isel(time=0), forcing.usurf.isel(time=1)
        # remove nan values at the boundaries from the test
        usurf_0 = usurf_0.data[~np.isnan(usurf_0)]
        usurf_1 = usurf_1.data[~np.isnan(usurf_1)]
        print(usurf_0)
        assert np.all(usurf_0 == 0)
        assert not np.all(usurf_1 == 0)

    def test_eddy_forcing_chunking(self):
        """
        Assert that applying eddy_forcing using different chunking gives the
        same result.

        The original implementation used `xarray.map_blocks`. Somewhere along
        the way, the behaviour of this function changed, ballooning memory
        usage. We removed this call and swapped some things up-- this test
        asserts that our change doesn't impact behaviour.
        """

        scale_m = 1

        xs = np.random.rand(100)
        ys = np.random.rand(100) * 2
        times = np.arange(2)
        xs_, ys_ = np.meshgrid(xs, ys)
        dxs = np.ones_like(xs_)
        dys = np.ones_like(ys_) * 2

        # TODO overlapping vars. confusing
        dxs = xr.DataArray(
                dxs, dims=("xu_ocean", "yu_ocean"), coords={"xu_ocean": xs, "yu_ocean": ys}
        )
        dys = xr.DataArray(
                dys, dims=("xu_ocean", "yu_ocean"), coords={"xu_ocean": xs, "yu_ocean": ys}
        )
        grid_info = xr.Dataset({"dxu": dxs, "dyu": dys})

        data = xr.Dataset(
                {
                    "usurf": xr.DataArray(
                            np.stack((np.zeros((100, 100)),
                                      np.ones((100, 100))), axis=0),
                            dims=("time", "xu_ocean", "yu_ocean"),
                            coords={"time": times, "xu_ocean": xs, "yu_ocean": ys},
                    ),
                    "vsurf": xr.DataArray(
                            np.stack((np.zeros((100, 100)),
                                      np.ones((100, 100))), axis=0),
                            dims=("time", "xu_ocean", "yu_ocean"),
                            coords={"time": times, "xu_ocean": xs, "yu_ocean": ys},
                    ),
                }
        )

        # new forcing: apply
        forcing_new = lib.compute_forcings_and_coarsen_cm2_6(data, grid_info, scale=scale_m)

        # new forcing: post-chunk
        for var in forcing_new:
          forcing_new[var].encoding = {}
        forcing_new = forcing_new.chunk(dict(time=1))

        # old forcing: helper
        def f(block):
            """
            Description: Apply coarsening operations + computations of subgrid
                momentum forcing to a chunk of data along the time axis.
            Parameters
            ----------
            block :
                description: an xarray dataset containing variables usurf and vsurf
                for the two components of surface velocities.
            Returns
            -------
            eddy_forcing : xr.Dataset
                Dataset containing 4 variables:
                - usurf and vsurf, the two components of the coarse-grained & filtered
                surface velocities
                - S_x and S_y, the two components of the diagnosed subgrid momentum
                forcing
            """
            return lib.compute_forcings_and_coarsen_cm2_6(block, grid_info, scale=scale_m)

        # old forcing: apply
        template = data.coarsen(
            {"xu_ocean": int(scale_m), "yu_ocean": int(scale_m)},
            boundary="trim").mean()
        template2 = template.copy()
        template2 = template2.rename({"usurf": "S_x", "vsurf": "S_y"})
        template = xr.merge((template, template2))
        forcing_old = xr.map_blocks(f, data, template=template)

        assert forcing_new == forcing_old

    # def test_spatial_filter_dataset(self):
    #     a1 = xr.DataArray(data = np.zeros((10, 4, 4)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(10) * 3,
    #                              'x' : np.arange(4) * 7,
    #                              'y' : np.arange(4) * 11})
    #     ds = xr.Dataset({'var0' : a1})
    #     filtered = spatial_filter_dataset(ds, 2)
    #     self.assertEqual(filtered.dims, ds.dims)
    #
    # def test_spatial_filter_dataset2(self):
    #     a1 = xr.DataArray(data = np.random.randn(1000, 40, 40),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     ds = xr.Dataset({'var0' : a1})
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
    #     """Checks that the average of the filtered xr.Dataset is equal to that
    #     of the original data within some threshold"""
    #     a1 = xr.DataArray(data = np.random.randn(1000, 40, 40),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     ds = xr.Dataset({'var0' : a1})
    #     filtered = spatial_filter_dataset(ds, sigma=3)
    #     mean_origin = ds.mean(dim='x').mean(dim='y')
    #     mean_filtered = filtered.mean(dim='x').mean(dim='y')
    #     self.assertTrue((mean_filtered - mean_origin
    #                      <= 0.1 * mean_origin ).all())
    #
    # def test_eddy_forcing(self):
    #     a1 = xr.DataArray(data = np.random.randn(1000, 100, 100),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(100) * 10,
    #                              'y' : np.arange(100) * 11})
    #     a2 = xr.DataArray(data = np.random.randn(1000, 100, 100),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(100) * 10,
    #                              'y' : np.arange(100) * 11})
    #     ds = xr.Dataset({'usurf' : a1, 'vsurf' : a2})
    #     forcing = eddy_forcing(ds, 40)
    #     self.assertTrue(forcing.dims != ds.dims)
    #
    # def test_eddy_forcing2(self):
    #     a1 = xr.DataArray(data = ma.zeros((100, 10, 10)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(100) * 3,
    #                              'x' : np.arange(10) * 10,
    #                              'y' : np.arange(10) * 11})
    #     a1.data.mask = np.zeros((100, 10, 10), dtype = np.bool)
    #     a1.data.mask[0] = True
    #     a2 = xr.DataArray(data = np.zeros((100, 10, 10)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(100) * 3,
    #                              'x' : np.arange(10) * 10,
    #                              'y' : np.arange(10) * 11})
    #     ds = xr.Dataset({'usurf' : a1, 'vsurf' : a2})
    #     forcing = eddy_forcing(ds, 40)

    # def test_advections(self):
    #     a1 = xr.DataArray(data = np.zeros((1000, 40, 40)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     a2= xr.DataArray(data = np.zeros((1000, 40, 40)),
    #                    dims = ['time', 'x', 'y'],
    #                    coords = {'time' : np.arange(1000) * 3,
    #                              'x' : np.arange(40) * 7,
    #                              'y' : np.arange(40) * 11})
    #     ds = xr.Dataset({'usurf' : a1, 'vsurf' : a2})
    #     adv = advections(ds)
    #     self.assertTrue((adv == 0).all().to_array().all().item())
