#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines transformations applied to model outputs.

Define some transformations applied to model outputs.
Allows us to separate these operations from the models themselves.
In particular, when we use a heteroskedastic loss, we compare two
transformations that ensure that the precision is positive.

TODO
----
- Not immediately clear that ABC (abstract base class) is useful here?
- Seem to define lots of child classes to work to a fial point, can this be reduced?
"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn.functional import softplus


class Transform(nn.Module, ABC):
    """
    Base Class for all transforms, child of a Torch NN Module, does nothing.

    Methods
    -------
    transform
    forward
    """

    # AB This does nothing, should it be defined in later classes directly?
    @abstractmethod
    def transform(self, input):
        """Does nothing."""
        pass

    def forward(self, input_):
        # AB Do we need this method if it only applies the transform method?
        """
        Apply self.transform to input.

        Parameters
        ----------
        input_ : type?  # AB
            description  # AB
        """
        return self.transform(input_)

    # AB This does nothing, should it be defined in later classes directly?
    @abstractmethod
    def __repr__(self):
        """Does nothing."""
        pass


class PrecisionTransform(Transform):
    """
    Class description?  # AB

    Attributes
    ----------
    min_value : float
      description?  # AB

    Methods
    -------
    min_value : type?  # AB
        description?  # AB
    indices : type?  # AB
        description?  # AB
    transform
    transform_precision
    """

    def __init__(self, min_value=0.1):
        super().__init__()
        self.min_value = nn.Parameter(torch.tensor(min_value))

    @property
    def min_value(self):
        """Applies softplus activation function to min_value."""
        return softplus(self._min_value)

    @min_value.setter
    def min_value(self, value):
        """Convert float input min_value into a Torch tensor."""
        self._min_value = nn.Parameter(torch.tensor(value))

    @property
    def indices(self):
        """Return the indices transformed."""
        return self._indices

    @indices.setter
    def indices(self, values):
        self._indices = values

    def transform(self, input_):
        # AB is this docstring correct? It looks to me like it
        """
        Split into sections of size 2 along channel dimension.

        What is this method for?, longer description of purpose?  # AB
        Note: the split argument is the size of the sections, not the number of them,
        although this does not matter for 4 channels.

        Parameters
        ----------
        input_ : type?  # AB
            description?  # AB

        Returns
        -------
        result : type?  # AB
            description?  # AB
        """
        result = torch.clone(input_)
        result[:, self.indices, :, :] = (
            self.transform_precision(input_[:, self.indices, :, :]) + self.min_value
        )
        return result

    @staticmethod
    @abstractmethod
    def transform_precision(precision):
        """Does nothing."""
        pass


class MixedPrecisionTransform(PrecisionTransform):
    """
    Class description?  # AB

    Methods
    -------
    mean_indices
    transform
    """

    def __init__(self):
        super().__init__()

    @property
    def mean_indices(self):
        """
        description?  # AB

        Returns
        -------
        : list
            description  # AB
        """
        indices_set = set(self.indices)
        range_set = set(list(range(4)))
        return list(range_set.difference(indices_set))

    def transform(self, input_):
        """
        Description?  # AB

        Parameters
        ----------
        input_ : type?  # AB
            description?  # AB

        Returns
        -------
        result : type?  # AB
            description?  # AB
        """
        result = super().transform(input_)
        result = torch.clone(result)
        result[:, self.indices, :, :] = (
            1
            / ((torch.abs(result[:, self.mean_indices, :, :]) + 0.01))
            * result[:, self.indices, :, :]
        )
        return result


class SoftPlusTransform(PrecisionTransform):
    """
    Class description?  # AB

    Attributes
    ----------
    min_value : float
      description?  # AB

    Methods
    -------
    mean_indices
    transform_precision
    """

    # AB This repeats the __init__ for PrecisionTransform so is unneccessary?
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return "".join(("SoftPlusTransform(", str(self.min_value), ")"))


class MixedSoftPlusTransform(MixedPrecisionTransform):
    """
    Class description?  # AB

    Attributes
    ----------
    min_value : float
      description?  # AB

    Methods
    -------
    transform_precision
    """

    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return "".join(("MixedSoftPlusTransform(", str(self.min_value), ")"))


class SquareTransform(PrecisionTransform):
    """
    Class description?  # AB

    Attributes
    ----------
    min_value : float
      description?  # AB

    Methods
    -------
    transform_precision
    """

    # AB This repeats the __init__ for PrecisionTransform so is unneccessary?
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return precision**2

    def __repr__(self):
        return "".join(("SquareTransform(", str(self.min_value), ")"))


class MeanTransform(Transform):
    """
    Class description?  # AB

    Methods
    -------
    transform
    """

    def transform(self, input):
        result = torch.clone(input)
        result[:, 0:2, :, :] = torch.tan(result[:, 0:2, :, :])
        return result

    def __repr__(self):
        return self.__class__.__name__


class ComposeTransform(Transform):
    """
    Class description?  # AB

    Methods
    -------
    transform
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, input):
        for transform in self.transforms:
            input = transform(input)
        return input

    def __repr__(self):
        return " o ".join([transform.__repr__() for transform in self.transforms])


class MeanPrecisionTransform(ComposeTransform):
    """Class description?"""  # AB

    def __init__(self):
        transform_1 = SoftPlusTransform()
        transform_1.indices = [2, 3]
        transform_2 = MeanTransform()
        super().__init__((transform_1, transform_2))
