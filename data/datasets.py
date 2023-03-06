# -*- coding: utf-8 -*-
"""
Routines for handling datasets.

TODO:
- balance the weights when mixing data sets
"""
import warnings
import bisect
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import xarray as xr


def call_only_once(func):
    """
    Decorator to ensure a function is only called once for a given set of parameters.

    Parameters
    ----------
    func : type?  # AB
        description?  # AB

    Returns
    -------
    new_func : type?  # AB
        description?  # AB
    """
    func.called = []

    def new_func(*args, **kargs):
        if not (args, kargs) in func.called:
            func.called.append((args, kargs))
            return func(*args, **kargs)
        raise Exception(
            "This method should be called at most once for a given set of parameters."
        )

    return new_func


class FeaturesTargetsDataset(Dataset):
    """
    Class describing a Pytorch Dataset based on a set of features
    and targets both passed as numpy arrays.

    Attributes
    ----------
    features : ndarray
        Numpy array of features with the first dimension indexing
        various samples
    targets : ndarray
        Numpy array of target values with the first dimension indexing
        various samples
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets
        assert len(self.features) == len(self.targets)

    def __getitem__(self, index: int):
        """
        getter for target and feature of a particular index.

        Parameters
        ----------
        index : int
            index of the feature.

        Returns
        -------
        (self.features[index], self.targets[index]) : tuple of ?  # AB
            description?  # AB
        """
        return (self.features[index], self.targets[index])

    def __len__(self):
        """
        get length of features attribute.

        Returns
        -------
        len(self.features) : int
            length of features attribute from FeaturesTargetsDataset class
        """
        return len(self.features)


def prod(l):
    """
    Returns the product of the elements of an iterable.

    Returns product of all elements, or 1 if iterable is empty


    Parameters
    ----------
    l : iterable
        description?  # AB

    Returns
    -------
    l[0] * prod(l[1:]) : type?  # AB
        product of all elements, or 1 if len(l)=0
    """
    if len(l) == 0:
        return 1
    return l[0] * prod(l[1:])


class DatasetTransformer:
    """
    Class to describe a transform that can be applied on a dataset.

    Attributes
    ----------
    features_transform : ArrayTransform
        Transform that will be applied to the features
    targets_transform : ArrayTransform
        Transform that will be applied to the targets
    """
    def __init__(self, features_transform, targets_transform=None):
        self.transforms = {}
        self.transforms["features"] = features_transform
        if targets_transform is None:
            targets_transform = deepcopy(features_transform)
        self.transforms["targets"] = targets_transform

    def add_features_transform(self, transform: ArrayTransform):
        """
        Add a transform to the list of transforms that will be
        applied on the features of a dataset

        Parameters
        ----------
        transform : ArrayTransform
            Transform to be added to the list of transforms
        """
        feature_t = self.transforms["features"]
        if not isinstance(feature_t, ComposeTransforms):
            self.transforms["features"] = ComposeTransforms(
                [
                    feature_t,
                ]
            )
        self.transforms["features"].add_transform(transform)

    def add_targets_transform(self, transform):
        """
        Add a transform to the list of transforms that will be
        applied on the targets of a dataset

        Parameters
        ----------
        transform : ArrayTransform
            Transform to be added to the list of target transforms
        """
        target_t = self.transforms["targets"]
        if not isinstance(target_t, ComposeTransforms):
            self.transforms["targets"] = ComposeTransforms(
                [
                    target_t,
                ]
            )
        self.transforms["targets"].add_transform(transform)

    def fit(self, x: torch.utils.data.Dataset):
        """
        Call the fit method of all array transforms in the list
        of features and target transforms on the passed Dataset.
        # TODO Arthur check whether we actually still use this

        Parameters
        ----------
        x : torch.utils.data.Dataset
            Pytorch Dataset to use for fitting.

        Returns
        -------
        self : DatasetTransformer
            The DatasetTransformer after applying the fitting.
        """
        # TODO Arthur check this
        features, targets = x[:]
        self.transforms["features"].fit(features)
        self.transforms["targets"].fit(targets)
        return self

    def transform(self, x):
        """
        Applies features and targets transforms to inputs and returns.

        Parameters
        ----------
        x : tuple of (numpy array, numpy array)
            Arrays of features and targets on which to apply the transforms

        Returns
        -------
        new_features, new_targets : tuple of (numpy array, numpy array)
            Transformed features and transformed targets
        """
        features, targets = x
        new_features = self.transforms["features"].transform(features)
        new_targets = self.transforms["targets"].transform(targets)
        return new_features, new_targets

    def get_features_coords(self, coords):
        """
        Get the coordinates of the transformed features. These might change for instance
        if we apply a crop transform to the features.
        #TODO Arthur Can we find a nicer way to achieve this? Say using missing values?

        Parameters
        ----------
        coords : dict of ?  # AB
            description?  # AB

        Returns
        -------
        result : dict of ?  # AB
            description?  # AB
        """
        result = {}
        for k, v in coords.items():
            result[k] = self.transforms["features"].transform_coordinate(v, k)
        return result

    def get_targets_coords(self, coords):
        """
        Apply targets transform to input coords.

        Parameters
        ----------
        coords : dict of ?  # AB
            description?  # AB

        Returns
        -------
        result : dict of ?  # AB
            description?  # AB
        """
        result = {}
        for k, v in coords.items():
            result[k] = self.transforms["targets"].transform_coordinate(v, k)
        return result

    def __call__(self, x):
        """
        description?  # AB

        Parameters
        ----------
        x : type  # AB
            description?  # AB

        Returns
        -------
        self.transform(x) : type?  # AB
            description?  # AB
        """
        return self.transform(x)

    def inverse_transform_target(self, target):
        """
        Return the inverse transform of the passed transformed target
        """
        return self.transforms["targets"].inverse_transform(target)

    def inverse_transform(self, x: Dataset):
        """
        getter for target and feature of a particular index.

        Parameters
        ----------
        x : type?  # AB torch.utils.Dataset in hints, but tuple in function?
            description?  # AB

        Returns
        -------
        FeaturesTargetsDataset(new_features, new_targets) : type?  # AB
            description?  # AB
        """
        features, targets = x
        new_features = self.transforms["features"].inverse_transform(features)
        new_targets = self.transforms["targets"].inverse_transform(targets)
        return FeaturesTargetsDataset(new_features, new_targets)


class ArrayTransform(ABC):
    def __call__(self, x):
        return self.transform(x)

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def transform_coordinate(self, coord, dim):
        pass


class ComposeTransforms(ArrayTransform):
    def __init__(self, *transforms):
        self.transforms = list(transforms)

    def fit(self, x):
        for transform in self.transforms:
            transform.fit(x)
            y = []
            for i in range(x.shape[0]):
                y.append(transform(x[i, ...]))
            x = np.array(y)

    def transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse_transform(self, x):
        for transform in self.transforms[::-1]:
            if hasattr(transform, "inverse_transform"):
                x = transform.inverse_transform(x)
        return x

    def transform_coordinate(self, coord, dim):
        x = coord
        for transform in self.transforms:
            # TODO arthur line below is ugly
            if isinstance(transform, (CropToNewShape, CyclicRepeat)):
                x = transform.transform_coordinate(coord, dim)
        return x

    def add_transform(self, transform):
        self.transforms.append(transform)


class CropToNewShape(ArrayTransform):
    """Crops to a new shape. Keeps the array centered, modulo 1 in which case
    the top left is priviledged. If the passed data is smaller than the
    requested shape, the data is unchanged."""

    def __init__(self, height=None, width=None):
        self.height = height
        self.width = width

    def fit(self, x):
        pass

    def get_slice(self, length: int, length_to: int):
        d_left = max(0, (length - length_to) // 2)
        d_right = d_left + max(0, (length - length_to)) % 2
        return slice(d_left, length - d_right)

    def transform(self, x):
        if self.height is None:
            self.fit(x)
        height, width = x.shape[1:]
        return x[
            :, self.get_slice(height, self.height), self.get_slice(width, self.width)
        ]

    def transform_coordinate(self, coords, dim):
        length = len(coords)
        if dim == "height":
            return coords[self.get_slice(length, self.height)]
        if dim == "width":
            return coords[self.get_slice(length, self.width)]

    def __repr__(self):
        return f"CropToNewShape({self.height}, {self.width})"


class CyclicRepeat(ArrayTransform):
    """Repeats the dataset in a cyclic way. The provided data should
    cover a complete cycle, with no repetition."""

    def __init__(self, axis: int, dim_name: str, cycle_length: float, nb_points: int):
        """
        Constructor.

        Parameters
        ----------
        axis : int
            Index of the dimension along which the data is repeated.

        dim_name: str
            Name of the dimension corresponding to the axis.
            # TODO This is redundant

        cycle_length: float
            Length of one cycle. For instance for longitude in degrees this
            would be 360.

        nb_points : int
            Number of points repeated on each end. If nb_points is 10, the
            transformed array will have 20 more points along the specified
            axis.

        Returns
        -------
        None.

        """
        self.axis = axis
        self.dim_name = dim_name
        self.length = cycle_length
        self.nb_points = nb_points

    def fit(self, x):
        pass

    def transform(self, x):
        nb = self.nb_points
        left = np.take(x, np.arange(-nb, 0), self.axis)
        right = np.take(x, np.arange(nb), self.axis)
        return np.concatenate((left, x, right), axis=self.axis)

    def transform_coordinate(self, coords, dim):
        print(f"{dim}, {self.dim_name}")
        if dim == self.dim_name:
            left = coords[-self.nb_points :] - self.length
            right = coords[: self.nb_points] + self.length
            return np.concatenate((left, coords, right))
        return coords

    def __repr__(self):
        return (
            f"CyclicRepeat({self.axis}, {self.dim_name},"
            f" {self.length}, {self.nb_points})"
        )


class CropToMultipleof(CropToNewShape):
    """Transformation that crops arrays to ensure that they have width and
    height multiple of the passed parameter. This is used for instance
    for Unet"""

    def __init__(self, multiple_of: int = 2):
        super().__init__()
        self.multiple_of = multiple_of

    def fit(self, x):
        shape = x.shape
        self.height = shape[2] // self.multiple_of * self.multiple_of
        self.width = shape[3] // self.multiple_of * self.multiple_of

    def __repr__(self):
        return f"CropToMultipleOf({self.multiple_of})"


class SignedSqrt(ArrayTransform):
    def fit(self, x):
        pass

    def transform(self, x):
        x = np.sign(x) * np.sqrt(np.abs(x))
        return x

    def inverse_transform(self, x):
        return x**2 * torch.sign(x)


class PerChannelNormalizer(ArrayTransform):
    def __init__(self, use_mean=False, fit_only_once=False):
        self.fit_only_once = fit_only_once
        self._std = None
        self._mean = None
        self._use_mean = use_mean

    def fit(self, x: np.ndarray):
        if (not self.fit_only_once) or (self._mean is None):
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            std = np.std(x, axis=(0, 2, 3), keepdims=True)
            self._mean = mean.reshape(mean.shape[1:])
            self._std = std.reshape(std.shape[1:])

    def transform(self, x: np.ndarray):
        assert self._mean is not None
        if self._use_mean:
            x = x - self._mean
        return x / self._std

    def inverse_transform(self, X):
        if self._use_mean:
            return X * self._std + self._mean
        return X * self._std


class FixedNormalizer(ArrayTransform):
    def fit(self, x):
        pass

    def transform(self, x):
        return x / self.std

    def inverse_transform(self, x):
        return x * self.std


class FixedVelocityNormalizer(FixedNormalizer):
    # std = 0.1
    std = 1


class FixedForcingNormalizer(FixedNormalizer):
    # std = 1e-7
    std = 1


class ArctanPerChannelNormalizer(PerChannelNormalizer):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def transform(self, x):
        assert self._mean is not None
        if self._use_mean:
            x = x - self._mean
        return np.arctan(x)


class PerLocationNormalizer(ArrayTransform):
    def __init__(self):
        self._std = None
        self._mean = None

    def fit(self, x: np.ndarray):
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        self._mean = mean.reshape(mean.shape[1:])
        self._std = std.reshape(std.shape[1:])

    def transform(self, x: np.ndarray):
        assert self._mean is not None
        return (x - self._mean) / self._std


class PerInputNormalizer(ArrayTransform):
    def fit(self, x):
        pass

    def transform(self, x: np.ndarray):
        mean = np.mean(x, axis=(1, 2), keepdims=True)
        std = np.std(x, axis=(1, 2), keepdims=True)
        return (x - mean) / std


class RawDataFromXrDataset(Dataset):
    """This class allows to define a Pytorch Dataset based on an xarray
    dataset easily, specifying features and targets."""

    def __init__(self, dataset: xr.Dataset):
        self.xr_dataset = dataset
        self.input_arrays = []
        self.output_arrays = []
        self.index = None

    @property
    def output_coords(self):
        # return dict([(k, v.data) for k, v in self.xr_dataset.coords.items()])
        return {k: v.data for k, v in self.xr_dataset.coords.items()}

    @property
    def input_coords(self):
        # return dict([(k, v.data) for k, v in self.xr_dataset.coords.items()])
        return {k: v.data for k, v in self.xr_dataset.coords.items()}

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: str):
        self._index = index

    @property
    def output_arrays(self):
        return self._output_arrays

    @output_arrays.setter
    def output_arrays(self, str_list: list):
        for array_name in str_list:
            self._check_varname(array_name)
        self._output_arrays = str_list

    @property
    def input_arrays(self):
        return self._input_arrays

    @input_arrays.setter
    def input_arrays(self, str_list):
        for array_name in str_list:
            self._check_varname(array_name)
        self._input_arrays = str_list

    @property
    def features(self):
        return self.xr_dataset[self.input_arrays]

    @property
    def targets(self):
        return self.xr_dataset[self.output_arrays]

    @property
    def n_targets(self):
        return len(self.targets)

    @property
    def n_features(self):
        return len(self.features)

    def add_output(self, varname):
        self._check_varname(varname)
        self._output_arrays.append(varname)

    def add_input(self, varname: str):
        self._check_varname(varname)
        self._input_arrays.append(varname)

    @property
    def width(self):
        dims = self.xr_dataset.dims
        candidates = list(filter(lambda x: x.startswith("x"), dims))
        if len(candidates) == 1:
            x_dim_name = candidates[0]
        elif "x" in candidates:
            x_dim_name = "x"
        else:
            raise Exception(
                "Could not determine width axis according \
                            to convention"
            )
        return len(self.xr_dataset[x_dim_name])

    @property
    def height(self):
        dims = self.xr_dataset.dims
        candidates = list(filter(lambda x: x.startswith("y"), dims))
        if len(candidates) == 1:
            y_dim_name = candidates[0]
        elif "y" in candidates:
            y_dim_name = "y"
        else:
            raise Exception(
                "Could not determine width axis according \
                            to convention"
            )
        return len(self.xr_dataset[y_dim_name])

    def __getitem__(self, index):
        try:
            features = self.features.isel({self._index: index})
            features = features.to_array().data
            targets = self.targets.isel({self._index: index})
            targets = targets.to_array().data
            # to_array method stacks variables along first dim, hence next line
            if not isinstance(index, (int, np.int64, np.int_)):
                features = features.swapaxes(0, 1)
                targets = targets.swapaxes(0, 1)
        except ValueError as e:
            raise type(e)(
                "Make sure you have defined the index, inputs,\
                          and outputs: "
                + str(e)
            )
        if hasattr(features, "compute"):
            features = features.compute()
            targets = targets.compute()
        return features, targets

    def __len__(self):
        """
        Return the number of samples of the datasets. Requires that the
        index property has been set.

        Raises
        ------
        KeyError
            Raised if the index has not been defined or is not one of the
            dimensions of the xarray dataset.

        Returns
        -------
        int
            Number of samples of the dataset.

        """
        try:
            return len(self.xr_dataset[self._index])
        except KeyError as e:
            raise type(e)("Make sure you have defined the index: " + str(e))

    def _check_varname(self, var_name: str):
        if var_name not in self.xr_dataset:
            raise KeyError("Variable not in the xarray dataset.")
        if var_name in self._input_arrays or var_name in self._output_arrays:
            raise ValueError("Variable already added as input or output.")

    def __getattr__(self, attr_name):
        if hasattr(self.xr_dataset, attr_name):
            return getattr(self.xr_dataset, attr_name)
        raise AttributeError()


class DatasetWithTransform:
    def __init__(self, dataset, transform: DatasetTransformer):
        self.dataset = dataset
        self.transform = transform

    @property
    def output_coords(self):
        coords = {
            "height": self.dataset.output_coords["yu_ocean"],
            "width": self.dataset.output_coords["xu_ocean"],
        }
        new_coords = self.transform.get_targets_coords(coords)
        return {
            "yu_ocean": new_coords["height"],
            "xu_ocean": new_coords["width"],
            "time": self.coords["time"],
        }

    @property
    def input_coords(self):
        coords = {
            "height": self.dataset.input_coords["yu_ocean"],
            "width": self.dataset.input_coords["xu_ocean"],
        }
        new_coords = self.transform.get_features_coords(coords)
        return {
            "yu_ocean": new_coords["height"],
            "xu_ocean": new_coords["width"],
            "time": self.coords["time"],
        }

    @property
    def height(self):
        """Since the transform can modify the height..."""
        x = self[0][0]
        return x.shape[1]

    @property
    def width(self):
        x = self[0][0]
        return x.shape[2]

    @property
    def output_height(self):
        return self[0][1].shape[1]

    @property
    def output_width(self):
        return self[0][1].shape[2]

    def __getitem__(self, index: int):
        raw_features, raw_targets = self.dataset[index]
        new_features, new_targets = [], []
        if hasattr(index, "__iter__"):
            n_samples = raw_features.shape[0]
            # The following is because the transform applies to a single sample
            for i in range(n_samples):
                temp = self.transform((raw_features[i, ...], raw_targets[i, ...]))
                new_features.append(temp[0])
                new_targets.append(temp[1])
            return np.stack(new_features), np.stack(new_targets)
        return self.transform(self.dataset[index])

    def __getattr__(self, attr):
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        raise AttributeError()

    def __len__(self):
        return len(self.dataset)

    def add_transforms_from_model(self, model):
        features_transforms = self.add_features_transform_from_model(model)
        targets_transforms = self.add_targets_transform_from_model(model)
        return {
            "features transform": features_transforms,
            "targets_transform": targets_transforms,
        }

    def add_features_transform_from_model(self, model):
        """Automatically adds features transform required by the model.
        For instance Unet will require features to be reshaped to multiples
        of 2, 4 or higher depending on the number of scales of the Unet."""
        # If the underlying xarray dataset has attribute cycle we automatically
        # add a cycle repeat transform. This should depend on the model as
        # well, right now this has been done manually by fixing the number of
        # added points on each end to 10.
        # TODO make this adaptable
        if self.attrs.get("cycle") is not None:
            cycle_length = self.attrs["cycle"]
            cycle_repeat = CyclicRepeat(2, "width", cycle_length, 10)
            self.add_features_transform(cycle_repeat)
        if hasattr(model, "get_features_transform"):
            transform = model.get_features_transform()
            self.add_features_transform(transform)
            return transform
        return None

    def add_targets_transform_from_model(self, model):
        """Automatically reshapes the targets of the dataset to match
        the shape of the output of the model."""
        output_height = model.output_height(self.height, self.width)
        output_width = model.output_width(self.height, self.width)
        transform = CropToNewShape(output_height, output_width)
        self.add_targets_transform(transform)
        return transform

    def inverse_transform(self, x):
        return self.transform.inverse_transform(x)

    def inverse_transform_target(self, x):
        return self.transform.inverse_transform_target(x)

    def add_features_transform(self, transform):
        self.transform.add_features_transform(transform)

    def add_targets_transform(self, transform):
        self.transform.add_targets_transform(transform)


class Subset_(Subset):
    """Extends the Pytorch Subset class to allow for attributes of the
    dataset to be propagated to the subset dataset"""

    def __init__(self, dataset, indices):
        super(Subset_, self).__init__(dataset, indices)

    @property
    def output_coords(self):
        new_coords = self.dataset.output_coords
        new_coords["time"] = new_coords["time"][self.indices].data
        return new_coords

    @property
    def input_coords(self):
        new_coords = self.dataset.input_coords
        new_coords["time"] = new_coords["time"][self.indices]
        return new_coords

    def __getattr__(self, attr):
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        raise AttributeError()


class DatasetPartitioner:
    """Helper class to create partitions of a Dataset with each subset
    having a memory size small enough to fit in memory"""

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def get_partition(self, dataset):
        """
        Return a partition of the passed dataset, with a number of splits
        specified by the objet's corresponding attribute.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to be partitioned

        Returns
        -------
        l_subsets : list[Subset_]
            List of self.n_splits subsets, the union of which is equal to
            dataset.

        """
        length = len(dataset)
        split_length = int(length / self.n_splits)
        indexes = np.arange(0, length, split_length)
        l_subsets = []
        for i in indexes:
            start = i
            end = min(i + split_length, length)
            l_subsets.append(Subset_(dataset, np.arange(start, end)))
        return l_subsets


class ConcatDataset_(ConcatDataset):
    """Extends the Pytorch Concat Dataset in two ways:
    - enforces (by default) the concatenated dataset to have the same
    shapes
    - passes on attributes (from the first dataset, assuming they are
                            equal accross concatenated datasets)
    """

    def __init__(self, datasets, enforce_same_dims=True):
        super(ConcatDataset_, self).__init__(datasets)
        self.enforce_same_dims = enforce_same_dims
        if enforce_same_dims:
            heights = [dataset.height for dataset in self.datasets]
            widths = [dataset.width for dataset in self.datasets]
        self.height = min(heights)
        self.width = min(widths)
        for dataset in self.datasets:
            crop_transform = CropToNewShape(self.height, self.width)
            dataset.add_features_transform(crop_transform)
            dataset.add_targets_transform(crop_transform)

    def __getattr__(self, attr):
        if hasattr(self.datasets[0], attr):
            return getattr(self.datasets[0], attr)
        raise AttributeError()

    # def __setattr__(self, attr_name, value):
    #     if 'coord' in attr_name:
    #         for ds in self.datasets:
    #             setattr(ds, attr_name, value)
    #     else:
    #         self.__dict__[attr_name] = value


class LensDescriptor:
    def __get__(self, obj, type=None):
        lens = np.array([len(dataset) for dataset in obj.datasets])
        obj.__dict__[self.name] = lens
        return lens

    def __set_name__(self, owner, name):
        self.name = name


class RatiosDescriptor:
    def __get__(self, obj, type=None):
        if not obj.balanced:
            ratios = (obj.lens * obj.precision) // np.min(obj.lens)
        else:
            ratios = np.ones((len(obj.lens),))
        obj.__dict__[self.name] = ratios
        return ratios

    def __set_name__(self, owner, name):
        self.name = name


class MixedDatasets(Dataset):
    """Similar to the ConcatDataset from pytorch, with the difference that
    the datasets are not concatenated one after another, but instead mixed.
    For instance if we mix two datasets d1 and d2 that have the same size,
    and d = MixedDatasets((d1, d2)), then d[0] returns the first element of
    d1, d[1] returns the first element of d2, d[2] returns the second element
    of d1, and so on. In the case where the two datasets do not have the same
    size, see the __getitem__ documentation for more information of selection
    behaviour."""

    lens = LensDescriptor()
    ratios = RatiosDescriptor()

    def __init__(self, datasets, transforms=None, balanced=True):
        self.datasets = datasets
        self.precision = 1
        self.transforms = transforms
        self.balanced = balanced

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
        self._datasets = datasets
        # Delete instance attribute lens if it exists so that the descriptor
        # is called on next access to re-compute
        self.__dict__.pop("lens", None)
        self.__dict__.pop("ratios", None)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value: int):
        self._precision = value
        self.__dict__.pop("ratios", None)

    @property
    def balanced(self):
        return self._balanced

    @balanced.setter
    def balanced(self, balanced):
        self._balanced = balanced
        self.__dict__.pop("ratios", None)

    def __len__(self):
        return min(self.lens // self.ratios) * np.sum(self.ratios)

    def __getitem__(self, index):
        cum_sum = np.cumsum(self.ratios)
        remainer = index % cum_sum[-1]
        dataset_idx = bisect.bisect_right(cum_sum, remainer)
        sub_idx = index // cum_sum[-1]
        if self.transforms is not None:
            transform = self.transforms[dataset_idx]
        else:
            transform = lambda x: x
        return transform(self.datasets[dataset_idx][sub_idx * cum_sum[-1]])


class MixedDataFromXrDataset(MixedDatasets):
    def __init__(self, datasets, index: str, transforms):
        self.datasets = list(map(RawDataFromXrDataset, datasets))
        self.index = index
        super().__init__(self.datasets, transforms)

    @staticmethod
    def all_equal(l):
        v = l[0]
        for value in l:
            if value != v:
                return False
        return True

    @property
    def features(self):
        for dataset in self.datasets:
            yield dataset.features

    @property
    def targets(self):
        for dataset in self.datasets:
            yield dataset.targets

    @property
    def n_features(self):
        n_features = [d.n_features for d in self.datasets]
        if not self.all_equal(n_features):
            raise ValueError(
                "All datasets do not have the same number of features"
            )
        else:
            return n_features[0]

    @property
    def n_targets(self):
        n_targets = [d.n_targets for d in self.datasets]
        if not self.all_equal(n_targets):
            raise ValueError(
                "All datasets do not have the same number of targets"
            )
        return n_targets[0]

    @property
    def height(self):
        heights = [dataset.height for dataset in self.datasets]
        if not self.all_equal(heights):
            warnings.warn("Concatenated datasets do not have the same height")
        return heights[0]

    @property
    def width(self):
        widths = [dataset.width for dataset in self.datasets]
        if not self.all_equal(widths):
            warnings.warn("Concatenated datasets do not have the same height")
        return widths[0]

    def add_input(self, var_name: str) -> None:
        for dataset in self.datasets:
            dataset.add_input(var_name)

    def add_output(self, var_name: str) -> None:
        for dataset in self.datasets:
            dataset.add_output(var_name)

    @property
    def index(self):
        for dataset in self.datasets:
            yield dataset.index

    @index.setter
    def index(self, index: str):
        for dataset in self.datasets:
            dataset.index = index


class MultipleTimeIndices(Dataset):
    """Class to create a dataset based on an existing dataset where we
    concatenate multiple time indices along the channel dimension to create a
    new feature"""

    def __init__(self, dataset: Dataset, time_indices: list() = None):
        self.dataset = dataset
        self._time_indices = None
        self._shift = 0
        if time_indices is not None:
            self.time_indices = time_indices
        else:
            self.time_indices = [
                0,
            ]

    @property
    def time_indices(self):
        if self._time_indices:
            return np.array(self._time_indices)

    @time_indices.setter
    def time_indices(self, indices: list):
        for i in indices:
            if i > 0:
                raise ValueError("The indices should be 0 or negative")
        self._time_indices = indices
        self._shift = max([abs(v) for v in indices])

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value: int):
        raise Exception(
            "The shift cannot be set manually. Instead set \
                        the time indices."
        )

    @property
    def n_features(self):
        return self.dataset.n_features * len(self.time_indices)

    def __getitem__(self, index):
        """Returns the sample indexed by the passed index."""
        # TODO check this does not slows things down. Hopefully should not,
        # as it should just be a memory view.
        indices = index + self.shift + self.time_indices
        features = [self.dataset[i][0] for i in indices]
        feature = np.concatenate(features)
        target = self.dataset[index + self.shift][1]
        return (feature, target)

    def __len__(self):
        """Returns the number of samples available in the dataset. Note that
        this might be less than the actual size of the first dimension
        if self.indices contains values other than 0, i.e. if we are
        using some data from the past to make predictions"""
        return len(self.dataset) - self.shift

    def __getattr__(self, attr_name):
        if hasattr(self.dataset, attr_name):
            return getattr(self.dataset, attr_name)
        raise AttributeError()


if __name__ == "__main__":
    from xarray import DataArray
    from xarray import Dataset as xrDataset
    from numpy.random import randint

    da = DataArray(
        data=randint(0, 10, (20, 32, 48)), dims=("time", "yu_ocean", "xu_ocean")
    )
    da2 = DataArray(
        data=randint(0, 3, (20, 32, 48)), dims=("time", "yu_ocean", "xu_ocean")
    )
    da3 = DataArray(
        data=randint(0, 100, (20, 32, 48)) * 10, dims=("time", "yu_ocean", "xu_ocean")
    )
    da4 = DataArray(
        data=randint(0, 2, (20, 32, 48)) * 20, dims=("time", "yu_ocean", "xu_ocean")
    )
    ds = xrDataset(
        {"in0": da, "in1": da2, "out0": da3, "out1": da4},
        coords={
            "time": np.arange(20),
            "xu_ocean": np.arange(48) * 5,
            "yu_ocean": np.arange(32) * 2,
        },
    )
    ds = ds.chunk({"time": 2})
    ds = ds.where(ds["in0"] > 3)
    dataset1 = RawDataFromXrDataset(ds)
    dataset1.index = "time"
    dataset1.add_input("in0")
    dataset1.add_input("in1")
    dataset1.add_output("out0")
    dataset1.add_output("out1")

    # loader = DataLoader(dataset1, batch_size=7, drop_last=True)

    # ds2 = ds.isel(yu_ocean=slice(0, 28), xu_ocean=slice(0, 37))
    # dataset2 = RawDataFromXrDataset(ds2)
    # dataset2.index = 'time'
    # dataset2.add_input('in0')
    # dataset2.add_input('in1')
    # dataset2.add_output('out0')
    # dataset2.add_output('out1')
    # t = DatasetTransformer(ComposeTransforms(CropToMultipleof(5),
    #                                          FixedVelocityNormalizer(),
    #                                          SignedSqrt()))
    # t.add_features_transform(CropToMultipleof(3))
    # t2 = deepcopy(t)
    # train_dataset1 = Subset_(dataset1, np.arange(5))
    # train_dataset2 = Subset_(dataset2, np.arange(5))
    # t.fit(train_dataset1)
    # t2.fit(train_dataset2)
    # new_dataset = DatasetWithTransform(dataset1, t)
    # new_dataset2 = DatasetWithTransform(dataset2, t2)
    # datasets = (new_dataset, new_dataset2)
    # c = ConcatDataset_(datasets)
