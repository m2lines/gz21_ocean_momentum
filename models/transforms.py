#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file we define some transformations applied to the output of our 
models. This allows us to keep separate these from the models themselves.
In particular, when we use a heteroskedastic loss, we compare two
transformations that ensure that the precision is positive.
"""

from abc import ABC, abstractmethod
import torch
from torch.nn import Module, Parameter
from torch.nn.functional import softplus


class Transform(Module, ABC):
    """Abstract Base Class for all transforms"""

    @abstractmethod
    def transform(self, input):
        pass

    def forward(self, input_):
        return self.transform(input_)

    @abstractmethod
    def __repr__(self):
        pass


class PrecisionTransform(Transform):
    def __init__(self, min_value=0.1):
        super().__init__()
        self.min_value = Parameter(torch.tensor(min_value))

    @property
    def min_value(self):
        return softplus(self._min_value)

    @min_value.setter
    def min_value(self, value):
        self._min_value = Parameter(torch.tensor(value))

    @property
    def indices(self):
        """Return the indices transformed"""
        return self._indices

    @indices.setter
    def indices(self, values):
        self._indices = values

    def transform(self, input_):
        # Split in sections of size 2 along channel dimension
        # Careful: the split argument is the size of the sections, not the
        # number of them (although does not matter for 4 channels)
        result = torch.clone(input_)
        result[:, self.indices, :, :] = (
            self.transform_precision(input_[:, self.indices, :, :]) + self.min_value
        )
        return result

    @staticmethod
    @abstractmethod
    def transform_precision(precision):
        pass


class MixedPrecisionTransform(PrecisionTransform):
    def __init__(self):
        super().__init__()

    @property
    def mean_indices(self):
        s = set(self.indices)
        s2 = set(list(range(4)))
        return list(s2.difference(s))

    def transform(self, input_):
        result = super().transform(input_)
        result = torch.clone(result)
        result[:, self.indices, :, :] = (
            1
            / ((torch.abs(result[:, self.mean_indices, :, :]) + 0.01))
            * result[:, self.indices, :, :]
        )
        return result


class SoftPlusTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return "".join(("SoftPlusTransform(", str(self.min_value), ")"))


class MixedSoftPlusTransform(MixedPrecisionTransform):
    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return "".join(("MixedSoftPlusTransform(", str(self.min_value), ")"))


class SquareTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return precision**2

    def __repr__(self):
        return "".join(("SquareTransform(", str(self.min_value), ")"))


class MeanTransform(Transform):
    def transform(self, input):
        result = torch.clone(input)
        result[:, 0:2, :, :] = torch.tan(result[:, 0:2, :, :])
        return result

    def __repr__(self):
        return self.__class__.__name__


class ComposeTransform(Transform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, input):
        for t in self.transforms:
            input = t(input)
        return input

    def __repr__(self):
        return " o ".join([t.__repr__() for t in self.transforms])


class MeanPrecisionTransform(ComposeTransform):
    def __init__(self):
        t1 = SoftPlusTransform()
        t1.indices = [2, 3]
        t2 = MeanTransform()
        super().__init__((t1, t2))
