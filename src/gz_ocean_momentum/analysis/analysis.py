# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:04:01 2020

@author: Arthur

TODOS
-write a decorator that allows to save the figure plotted by a function
by just passing a save_fig=True to its parameters.
BUGS
-decorator allow_hold_on only works when the function is supposed to generate
only one figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

data_location = '/data/ag7531/'
figures_directory = 'figures'


def allow_hold_on(f):
    """Decorator that allows to specify a hold_on parameter that makes the
    plotting use the current figure instead of creating a new one."""
    def wrapper_f(*args, **kargs):
        if 'hold_on' in kargs and kargs['hold_on']:
            plt.gcf()
            del kargs['hold_on']
        else:
            plt.figure()
        f(*args, **kargs)
    return wrapper_f


class TimeSeriesForPoint:
    """Analysis class that allows to study the time series of the true
    value of the target at a specific point over time verses its predictions.
    """
    def __init__(self, predictions: np.ndarray, truth: np.ndarray):
        self._predictions = predictions
        self._truth = truth
        self._time_series = dict()
        self._point = None
        self._fig = None
        self._name = ''

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, point):
        self._point = point
        self._name = 'point{}-{}'.format(*point)
        self._build_time_series()

    def _build_time_series(self):
        x, y = self.point
        self._time_series['predictions'] = self._predictions[:, y, x]
        self._time_series['true values'] = self._truth[:, y, x]

    @property
    def predictions(self):
        return self._time_series['predictions']

    @property
    def true_values(self):
        return self._time_series['true values']

    @allow_hold_on
    def plot_pred_vs_true(self):
        """Plots the predictions and the true target accross time for the
        instance's point."""
        plt.figure()
        plt.plot(self.predictions)
        plt.plot(self.true_values)
        plt.legend(('Prediction', 'True values'))
        plt.title('Predictions for point {}, {}'.format(*self.point))
        plt.figure()
        plt.plot(self.predictions - self.true_values)
        plt.title('Prediction errors for point {}, {}'.format(*self.point))
        plt.show()

    def save_fig(self):
        if not self._fig:
            self.plot_pred_vs_true()
        plt.savefig(join(data_location, figures_directory, self.name))