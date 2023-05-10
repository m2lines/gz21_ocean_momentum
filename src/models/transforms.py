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
Arthur replies: Yeah, in the end I only used the Precision Transform, removing the rest
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
    # Arthur replies: this is to force implementation of this method, but overall, this might be all too complicated
    # for something quite simple. Might want to make things simpler later on.
    @abstractmethod
    def transform(self, input):
        """Does nothing."""
        pass

    def forward(self, input_):
        """
        Apply self.transform to input.

        Parameters
        ----------
        input_ : tensor
            tensor output by the neural network, should have shape (N, C, H, W)
        """
        return self.transform(input_)


class PrecisionTransform(Transform):
    """
    General class for a transform that acts on the precision
    channels of the outputs. The min value is a Parameter
    of the neural network that can be trained during the SGD.

    Attributes
    ----------
    min_value : float
      minimum positive value for the output precision

    Methods
    -------
    transform
    transform_precision
    """

    def __init__(self, min_value=0.1):
        super().__init__()
        # TODO Arthur should this not just be = min_value, we already convert to parameter in the setter
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
        """
        Applies the transform_precision method to precision channels,
        specified by the indices property

        Parameters
        ----------
        input_ : tensor
            tensor output by the neural network

        Returns
        -------
        result : tensor
            tensor with precision channels transformed to be positive
        """
        result = torch.clone(input_)
        result[:, self.indices, :, :] = (
            self.transform_precision(input_[:, self.indices, :, :]) + self.min_value
        )
        return result

    @staticmethod
    @abstractmethod
    def transform_precision(precision):
        """Does nothing, to be implemented in subclasses"""
        pass


class MixedPrecisionTransform(PrecisionTransform):
    """
    Scales the precision channel inverse proportionally to the mean channels.
    # TODO do not think this is used anymore in the final paper, but need to double check.
    # TODO check whether there is a potential issue with the lack of ordering in indices.

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
        Indices of the output channels corresponding to the means (usually 0 and 1)

        Returns
        -------
        : list
            List of the indices in the output corresponding to the means.
        """
        indices_set = set(self.indices)
        range_set = set(list(range(4)))
        return list(range_set.difference(indices_set))

    def transform(self, input_):
        """
        Applies the transform to precision channels in two steps:
        1. Apply a simple precision transform (for instance a softplus)
        2. Scales inverse-proportionally to the mean channel

        Parameters
        ----------
        input_ : tensor
            tensor output by the neural network

        Returns
        -------
        result : tensor
            tensor with the precision channels made positive
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
    Class used to apply a simple softplus on the precision channels

    Attributes
    ----------
    min_value : float
      min value for the precision

    Methods
    -------
    transform_precision
    """

    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return "".join(("SoftPlusTransform(", str(self.min_value), ")"))


class MixedSoftPlusTransform(MixedPrecisionTransform):
    """
    Class used to apply a softplus on each precision channels, followed
    by an inverse scaling from the class MixedPrecisionTransform.

    Attributes
    ----------
    min_value : float
      Minimum positive value for the precision

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
    Class used to apply a simple square value to the precision channels

    Attributes
    ----------
    min_value : float
      minimum value of the final precision

    Methods
    -------
    transform_precision
    """

    @staticmethod
    def transform_precision(precision):
        return precision**2

    def __repr__(self):
        return "".join(("SquareTransform(", str(self.min_value), ")"))
