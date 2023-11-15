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
  testing/utils.py and data/datasets.py so may be better off there than as part of the
  model?
"""
import torch


class DetectOutputSizeMixin:
    """Class to detect the shape of a neural net."""

    # use inference mode to reduce memory and time cost
    @torch.no_grad()
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
        dummy_in = torch.zeros((1, self.n_in_channels, input_height, input_width), device=self.device)
        # AB - Self here is assuming access to a neural net forward method?
        #      If so I think this should really be contained in FullyCNN.
        #      We can discuss and I am happy to perform the refactor.
        dummy_out = self(dummy_in)
        return dummy_out.size(3)

    # use inference mode to reduce memory and time cost
    @torch.no_grad()
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
        dummy_in = torch.zeros((1, self.n_in_channels, input_height, input_width), device=self.device)
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
