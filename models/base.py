#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch


class DetectOutputSizeMixin:
    """
    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    """

    # TODO Should there be an __init__ method here, or is this a child class?
    # See class FullyCNN in models1.py - combine & tidy

    def output_width(self, input_height, input_width):
        """
        Returns the width of the neural net?

        Parameters
        ----------
        input_height : TYPE?
            Description?
        input_width : TYPE?
            Description

        Returns
        -------
        y.size(3) : TYPE?
            Description

        """
        # TODO Complete Docstring
        # TODO consider renaming x and y to meaningful names?

        # TODO Where is n_in_channels_defined?
        x = torch.zeros((1, self.n_in_channels, input_height, input_width))
        x = x.to(device=self.device)
        # TODO Not sure what this line is doing, self should not be callable...
        # Calls the foward method of the network - make more explicit.
        # Gets size of output of NN for a given height
        y = self(x)

        # temporary fix for student loss
        if isinstance(y, tuple):
            y = y[0]

        return y.size(3)

    def output_height(self, input_height, input_width):
        """
        Returns the height of the neural net?

        Parameters
        ----------
        input_height : TYPE?
            Description?
        input_width : TYPE?
            Description

        Returns
        -------
        y.size(3) : TYPE?
            Description

        """
        # TODO Complete Docstring
        # TODO consider renaming x and y to meaningful names?

        x = torch.zeros((1, self.n_in_channels, input_height, input_width))
        x = x.to(device=self.device)
        y = self(x)

        # temporary fix for student loss
        if isinstance(y, tuple):
            y = y[0]
        return y.size(2)

    @property
    def device(self):
        p = list(self.parameters())[0]
        return p.device


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
