# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:12:49 2020

@author: Arthur
Contains a class for loading an MLFlow run.
TODOS
-write a generic loadmlflow class from which more specific classes can inherit
from
"""
import torch
from os.path import join
import numpy as np
import warnings
import json
from mlflow.tracking import MlflowClient

class LoadMLFlow:
    """Class to load an MLFlow run. In particular this allows to load the
    pytorch model if it was logged as an artifact, as well as the train and
    test split indices, and the predictions that were made on the test
    set."""
    def __init__(self,  run_id: str, experiment_id: int = 0,
                 mlruns_path: str = 'mlruns'):
        self.mlruns_path = mlruns_path
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.path = join(mlruns_path, str(experiment_id), run_id)
        self.paths = dict()
        self.paths['params'] = join(self.path, 'params')
        self.paths['artifacts'] = join(self.path, 'artifacts')
        # Neural net attributes
        self._net_class = None
        self._net_filename = ''
        self._net = None
        # Dataset attributes
        self._train_split = None
        self._test_split = None
        self._train_dataset = None
        self._test_dataset = None
        # Prediction attirbutes
        self._predictions = None
        self._true_targets = None
        # mlflow client
        client = MlflowClient()

    @property
    def net_class(self):
        return self._net_class

    @net_class.setter
    def net_class(self, net_class: type):
        """Specifies the class used for the neural network"""
        self._net_class = net_class

    @property
    def net_filename(self):
        return join(self.paths['artifacts'], self._net_filename)

    @net_filename.setter
    def net_filename(self, net_filename: str):
        self._net_filename = net_filename

    @property
    def net(self):
        if not self._net:
            self._load_net()
        return self._net

    def _load_net(self, *net_params):
        net = self._net_class(*net_params)
        net.load_state_dict(torch.load(self.net_filename))
        self._net = net

    def load_param(self, param_name: str):
        """Loads the parameter"""
        if not hasattr(self, '_' + param_name) or getattr(self, 
                      '_' + param_name) is None:
            with open(join(self.paths['params'], param_name)) as f:
                setattr(self, '_' + param_name, f.readline())
        return getattr(self, '_' + param_name)

    @property
    def time_indices(self):
        return json.loads(self.load_param('time_indices'))

    @property
    def batch_size(self):
        return int(self.load_param('batch_size'))

    @property
    def train_split(self):
        # TODO generalize this by writing a single method for all params.
        if self._train_split is None:
            with open(join(self.paths['params'], 'train_split')) as f:
                self._train_split = float(f.readline())
        return self._train_split

    @train_split.setter
    def train_split(self, train_split: int):
        raise Exception('This should not be set by the user.')

    @property
    def test_split(self):
        # TODO generalize this by writing a single method for all params.
        if self._test_split is None:
            with open(join(self.paths['params'], 'test_split')) as f:
                self._test_split = float(f.readline())
        return self._test_split

    @test_split.setter
    def test_split(self, test_split: int):
        raise Exception('This should not be set by the user.')

    @property
    def predictions(self):
        """Returns the predictions made on the test dataset. These are loaded
        directly from the artifacts."""
        if self._predictions is None:
            try:
                self._predictions = np.load(join(self.paths['artifacts'],
                                                 'predictions.npy'))
            except FileNotFoundError:
                print('Predictions file not found for this run.')
        return self._predictions

    @property
    def true_targets(self):
        """Returns the true targets of the test dataset. These are loaded
        directly from the artifacts."""
        if self._true_targets is None:
            try:
                self._true_targets = np.load(join(self.paths['artifacts'],
                                                  'truth.npy'))
            except FileNotFoundError:
                try:
                    self._true_targets = np.load(join(self.paths['artifacts'],
                                                  'targets.npy'))
                except FileNotFoundError:
                    warnings.warn('True targets not found for this run')
        return self._true_targets

    @property
    def psi(self):
        if self._psi is None:
            path_psi = join(r'D:\Data sets\NYU\processed_data', 
                            'psi_coarse.npy')
            self._psi = np.load(path_psi)
        return self._psi
