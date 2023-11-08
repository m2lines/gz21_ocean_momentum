#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines base classes with methods to be used in pytorch models of models1.py.

TODO
----
- Should these really be defined here, or should they perhaps be defined directly in
  FullyCNN of models1.py?  # AB
- FinalTransformationMixin is not used anywhere as far as I can tell, should we remove?
- output_width and output_height could probably be combined for one output_size method.
- As far as I can tell the methods of DetectOutputSizeMixin are only used in
  inference/utils.py and data/datasets.py so may be better off there than as part of the
  model?
"""
import torch


class DetectOutputSizeMixin:
    """Class to detect the shape of a neural net."""

    # TODO: protect this with `@no_grad` decorator to conserve memory/time etc.
    def output_width(self, input_height, input_width):
        """
        Generate a tensor and run forward model to get output width.

        Parameters
        ----------
        input_height, output_height : int
            The dimensions of the model input tensor

        Returns
        -------
        dummy_out.size(3) : int
            width of the output tensor
        """
        # TODO: following 2 lines can be combined for speedup as
        #       e.g. `torch.zeros(10, 10, device=self.device)`
        dummy_in = torch.zeros((1, self._n_in_channels, input_height, input_width))
        dummy_in = dummy_in.to(device=self.device)
        # AB - Self here is assuming access to a neural net forward method?
        #      If so I think this should really be contained in FullyCNN.
        #      We can discuss and I am happy to perform the refactor.
        dummy_out = self(dummy_in)
        return dummy_out.size(3)

    # TODO: protect this with `@no_grad` decorator to conserve memory/time etc.
    def output_height(self, input_height, input_width):
        """
        Generate a tensor and run forward model to get output height.

        Parameters
        ----------
        input_height, output_height : int
            The dimensions of the model input tensor

        Returns
        -------
        dummy_out.size(2) : int
            height of the output tensor
        """
        # TODO: following 2 lines can be combined for speedup as
        #       e.g. `torch.zeros(10, 10, device=self.device)`
        dummy_in = torch.zeros((1, self._n_in_channels, input_height, input_width))
        dummy_in = dummy_in.to(device=self.device)
        dummy_out = self(dummy_in)
        return dummy_out.size(2)

    @property
    def device(self):
        """
        Return the device model uses.

        Returns
        -------
        Device
            Device where the neural network lives
        """
        # TODO: This can probably just be `return self.parameters[0].device`
        params = list(self.parameters())[0]
        return params.device


class FinalTransformationMixin:
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        return self.final_transformation(x)
