# TODO: Currently unsure if this file is used
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:21:16 2020

@author: arthur
"""

import xarray as xr
from abc import ABC, abstractmethod
import pickle
from dask import delayed
import dask.array as da
import numpy as np
from typing import List
from regression.functions import bz


class Transform(ABC):
    """Abstract transform class for xarray datasets"""

    requires_fit = False
    repr_params = []
    fit_only_once = True

    def __init__(self, inverse: bool = True):
        self.fitted = False
        if not inverse:
            self.inv_transform = lambda x: x

    @abstractmethod
    def transform(self, x: xr.Dataset):
        pass

    def apply(self, x: xr.Dataset):
        return self.transform(x)

    def __call__(self, x: xr.Dataset):
        return self.apply(x)

    def fit_transform(self, x: xr.Dataset):
        if hasattr(self, "fit"):
            self.fit(x)
        return self.transform(x)

    def inv_transform(self, x: xr.DataArray):
        raise NotImplementedError("Inverse transform not implemented.")

    def dump(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __init_subclass__(cls, *args):
        """We use this method to make sure that for all subclasses, the
        transform method conserves the attributes of the datasets"""
        raw_transform = cls.transform
        raw_inv_transform = cls.inv_transform

        def new_transform(self, x):
            if self.requires_fit and not self.fitted:
                raise RuntimeError("The transform needs to be fitted first.")
            new_ds = raw_transform(self, x)
            new_ds.attrs.update(x.attrs)
            return new_ds

        def new_inv_transform(self, x):
            if not self.fitted and self.requires_fit:
                raise RuntimeError("The transform needs to be fitted first.")
            return raw_inv_transform(self, x)

        cls.transform = new_transform
        cls.inv_transform = new_inv_transform

        # The following ensures that fitted is set to True if fit is called
        if hasattr(cls, "fit"):
            cls.requires_fit = True
            raw_fit = cls.fit

            def new_fit(self, x):
                if self.fitted and self.fit_only_once:
                    raise RuntimeError(
                        "The transform has already been\
                                       fitted."
                    )
                raw_fit(self, x)
                self.fitted = True

            cls.fit = new_fit

    def __repr__(self):
        params = ", ".join(
            [param + "=" + str(getattr(self, param)) for param in self.repr_params]
        )
        return "".join((str(type(self)), "(", params, ")"))


class ChainedTransform(Transform):
    def __init__(self, transforms, *args, **kargs):
        super().__init__(*args, **kargs)
        self.transforms = transforms
        if not any([t.requires_fit for t in self.transforms]):
            self.requires_fit = False

    def fit(self, x: xr.Dataset):
        for transform in self.transforms:
            x = transform.fit_transform(x)

    def transform(self, x: xr.Dataset):
        for transform in self.transforms:
            x = transform.apply(x)
        return x

    def inv_transform(self, x: xr.Dataset):
        for transform in reversed(self.transforms):
            x = transform.inv_transform(x)
        return x

    def __repr__(self, level=0):
        tabs = "\t" * (level + 1)
        s = "ChainedTransform(\n" + tabs
        reprs = [
            t.__repr__(level + 1) if isinstance(t, ChainedTransform) else t.__repr__()
            for t in self.transforms
        ]
        s2 = (",\n" + tabs).join(reprs)
        s3 = "\n" + tabs[:-1] + ")"
        return "".join((s, s2, s3))


class TargetedTransform(Transform):
    def __init__(self, transform: Transform, targets: List[str], *args, **kargs):
        super().__init__(*args, **kargs)
        self.base_transform = transform
        self.targets = targets
        self.requires_fit = self.base_transform.requires_fit

    def fit(self, x: xr.Dataset):
        if hasattr(self.transform, "fit"):
            temp_ds = x[self.targets]
            self.transform.fit(temp_ds)

    def transform(self, x: xr.Dataset):
        temp_ds = x[self.targets]
        temp_ds = self.base_transform(temp_ds)
        new_ds = x.copy()
        new_ds.update(temp_ds)
        return new_ds

    def inv_transform(self, x: xr.Dataset):
        temp_ds = x[self.targets]
        temp_ds = self.base_transform.inv_transform(temp_ds)
        new_ds = x.copy()
        new_ds.update(temp_ds)
        return new_ds

    def __repr__(self):
        targets = ", ".join(self.targets)
        return "".join((self.base_transform.__repr__(), " on ", targets))


class ScalingTransform(Transform):
    repr_params = [
        "factor",
    ]

    def __init__(self, factor: dict = None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.factor = factor

    def transform(self, x: xr.Dataset):
        return self.factor * x

    def inv_transform(self, x: xr.Dataset):
        return 1 / self.factor * x


class SeasonalStdizer(Transform):
    repr_params = ["apply_std", "by"]

    def __init__(
        self,
        by: str = "time.month",
        dim: str = "time",
        std: bool = True,
        *args,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        self.by = by
        self.dim = dim
        self._means = None
        self._stds = None
        self._grouped = None
        self.apply_std = std

    @property
    def grouped(self):
        return self._grouped

    @grouped.setter
    def grouped(self, value):
        self._grouped = value

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, value):
        self._means = value

    @property
    def stds(self):
        return self._stds

    @stds.setter
    def stds(self, value):
        self._stds = value

    def fit(self, x):
        self.grouped = x.groupby(self.by)
        self.means = self.grouped.mean(dim=self.dim).compute()
        self.stds = self.grouped.std(dim=self.dim).compute()

    def get_transformed(self, data):
        times = data.time
        months = times.dt.month
        r = data - self.means.sel(month=months)
        if self.apply_std:
            stds = self.stds.sel(month=months)
            r = r / stds
            stds = stds.rename({raw_name: raw_name + "_d" for raw_name in stds.keys()})
            r.update(stds)
        del r["month"]
        return r

    @delayed
    def get_inv_transformed(self, data, var_name):
        times = data.time
        months = times.dt.month
        result = data
        if self.apply_std:
            result = result * self.stds[var_name].sel(month=months)
        result = result + self.means[var_name].sel(month=months)
        del result["month"]
        return result.values

    def transform(self, data):
        template = data.copy()
        if self.apply_std:
            template.update(
                {raw_name + "_d": value for raw_name, value in template.items()}
            )
        return data.map_blocks(self.get_transformed, template=template)

    def inv_transform(self, data):
        sub_datasets = []
        nb_samples = len(data.time)
        for start in range(0, nb_samples, 8):
            sub_data = data.isel(time=slice(start, min(start + 8, nb_samples)))
            sub_coords = sub_data.coords
            new_xr_arrays = {}
            for k, val in sub_data.items():
                new_shape = val.shape
                dims = val.dims
                transformed = self.get_inv_transformed(val, k)
                dask_array = da.from_delayed(
                    transformed, shape=new_shape, dtype=np.float64
                )
                new_xr_array = xr.DataArray(
                    data=dask_array, coords=sub_coords, dims=dims
                )
                new_xr_arrays[k] = new_xr_array
            new_ds = xr.Dataset(new_xr_arrays)
            sub_datasets.append(new_ds)
        return xr.concat(sub_datasets, dim="time")


class CropToNewShape(Transform):
    def __init__(self, new_shape: dict = None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.new_shape = new_shape

    @staticmethod
    def get_slice(length: int, length_to: int):
        d_left = max(0, (length - length_to) // 2)
        d_right = d_left + max(0, (length - length_to)) % 2
        return slice(d_left, length - d_right)

    def transform(self, x):
        dims = x.dims
        idx = {
            dim_name: self.get_slice(dims[dim_name], dim_size)
            for dim_name, dim_size in self.new_shape.items()
        }
        return x.isel(idx)

    def __repr__(self):
        return f"CropToNewShape({self.new_shape})"


class CropToMinSize(CropToNewShape):
    def __init__(self, datasets, dim_names: list, *args, **kargs):
        super().__init__(*args, **kargs)
        new_shape = {
            dim_name: min([dataset.dims[dim_name] for dataset in datasets])
            for dim_name in dim_names
        }
        super().__init__(new_shape)

    def __repr__(self):
        return super().__repr__() + "(CropToMinSize)"


class CropToMultipleOf(CropToNewShape):
    def __init__(self, multiples: dict, *args, **kargs):
        super().__init__(*args, **kargs)
        self.multiples = multiples

    @staticmethod
    def get_multiple(p: int, m: int):
        return p // m * m

    def transform(self, x):
        dims = x.dims
        new_sizes = {
            dim_name: self.get_multiple(dims[dim_name], m)
            for dim_name, m in self.multiples.items()
        }
        idx = {
            dim_name: self.get_slice(dims[dim_name], new_sizes[dim_name])
            for dim_name, multiple in self.multiples.items()
        }
        return x.isel(idx)

    def __repr__(self):
        return f"CropToMultipleOf({self.multiples})"


class FormulaTransform(Transform):
    """Adds an extra dimension to the input corresponding to a closed-form
    equation of the surface velocity field"""

    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def transform(self, x):
        s_x_formula, s_y_formula = self.equation(x)
        out = x.update(dict(s_x_formula=s_x_formula, s_y_formula=s_y_formula))
        out = out.fillna(0.0)
        return out

    def __repr__(self):
        return "FormulaTranform()"


class BZFormulaTransform(FormulaTransform):
    def __init__(self):
        super().__init__(bz)
