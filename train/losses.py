#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:08:26 2020

@author: arthur
In this module we define custom loss functions. In particular we define
a loss function based on the Gaussian likelihood with two parameters, 
mean and precision.
"""
import torch
from torch.nn.modules.loss import _Loss
from enum import Enum
from abc import ABC
import numpy as np
from torch.autograd import Function


class VarianceMode(Enum):
    variance = 0
    precision = 1



# DEPRECIATED
class HeteroskedasticGaussianLoss(_Loss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, 2, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        term1 = - 1 / 2 * torch.log(precision)
        term2 = 1 / 2 * (target - mean)**2 * precision
        return (term1 + term2).mean()



class HeteroskedasticGaussianLossV2(_Loss):
    """Class for Gaussian likelihood"""

    def __init__(self, n_target_channels: int = 1, bias: float = 0.,
                 mode=VarianceMode.precision):
        super().__init__()
        self.n_target_channels = n_target_channels
        self.bias = bias
        self.mode = mode

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    @property
    def channel_names(self):
        return ['S_x', 'S_y', 'S_xscale', 'S_yscale']

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        if self.mode is VarianceMode.precision:
            term1 = - torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias))**2 * precision**2
        elif self.mode is VarianceMode.variance:
            term1 = torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias))**2 / precision**2
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias


class HeteroskedasticGaussianLossV3(_Loss):
    """Loss to be used with transform2 from models/submodels.py"""

    def __init__(self, *args, **kargs):
        super().__init__()
        self.base_loss = HeteroskedasticGaussianLossV2(*args, **kargs)

    def __getattr__(self, name: str):
        try:
            # This is necessary as the class Module defines its own __getattr__
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_loss, name)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.base_loss.forward(input, target)

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        raw_loss = self._base_loss(input, target[:, :self.n_target_channels, ...])
        return raw_loss + torch.log(target[:, self.n_target_channels: self.n_target_channels + 1, ...])
