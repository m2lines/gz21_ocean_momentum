#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:42:20 2020

@author: arthur
In here we define some metrics that are used on the test data to compare
the efficiency of models. These metrics classes allow to define an inverse
transform that ensures that the metric is calculated independently of the
normalization applied.
"""

import numpy as np
import torch
from torch.nn.functional import mse_loss
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, metric_func, name: str = None):
        if name is None:
            self.name = self.__class__.name
        else:
            self.name = name
        self.func = metric_func
        self.i_batch = 0
        self.value = 0

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func):
        self._func = func

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def inv_transform(self):
        if hasattr(self, "_inv_transform"):
            return self._inv_transform
        else:
            return lambda x: x

    @inv_transform.setter
    def inv_transform(self, inv_transform):
        self._inv_transform = inv_transform

    def __call__(self, y_hat, y):
        y_hat = self.inv_transform(y_hat)
        y = self.inv_transform(y)
        return self.func(y_hat, y)

    @abstractmethod
    def update(self, y_hat, y):
        pass

    @abstractmethod
    def reset(self):
        pass


class MSEMetric(Metric):
    def __init__(self):
        def func(x, y):
            squared_error = (x - y) ** 2
            return squared_error[~torch.isnan(y)].mean()

        super(MSEMetric, self).__init__(func)
        self._mse_zero_estimator = 0
        self._mse = 0

    def update(self, y_hat, y):
        mse = self(y_hat, y).item()
        mse_zero = self(torch.zeros_like(y), y).item()
        self._mse_zero_estimator = self.update_mean(
            self._mse_zero_estimator, mse_zero, self.i_batch
        )
        self._mse = self.update_mean(self._mse, mse, self.i_batch)
        self.value = self._mse / self._mse_zero_estimator
        self.i_batch += 1

    @staticmethod
    def update_mean(mean_value, value, i):
        mean_value = i / (i + 1) * mean_value
        mean_value += 1 / (i + 1) * value
        return mean_value

    def reset(self):
        self.value = 0
        self._mse_zero_estimator = 0
        self._mse = 0
        self.i_batch = 0


class MaxMetric(Metric):
    def __init__(self):
        def func(x, y):
            diff = torch.abs(x - y)
            diff = diff[~torch.isnan(y)]
            return torch.max(diff)

        super(MaxMetric, self).__init__(func)

    def update(self, y_hat, y):
        value = self(y_hat, y).item()
        self.value = max(value, self.value)

    def reset(self):
        self.value = 0
        self.i_batch = 0
