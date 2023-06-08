# 2023-05-12 raehik: try to replace with fully_conv_net. should be identical.

# -*- coding: utf-8 -*-
"""
Module of PyTorch ML models.

TODOs:
------
-Try some standard image classification network whose last layer you'll change
- change the color map of plots
- study different values of time indices
-  Log the data run that is used to create the dataset. Log any
   transformation applied to the data. Later we might want to allow from
   stream datasets.

BUGS
----
-when we run less than 100 epochs the figures from previous runs are
logged.
"""
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn.functional import pad

import numpy as np


# THIS IS THE MODEL USED IN THE FINAL PAPER
class FullyCNN(nn.Sequential):
    """
    Fully Convolutional Neural Net used for modelling ocean momentum.

    Attributes
    ----------
    n_in_channels : int
        number of input chanels to model, 2 for the two surface velocity
        components
    n_out_channels : int
        number of output channels from model, 4 with 2 per component
        of the subgrid forcing
    padding : str
        padding argument passed on to Conv2d layers
    batch_norm : bool
        whether to normalise batches

    Methods
    -------
    final_transformation
    forward
    """

    _n_in_channels: int

    def get_n_in_channels(self):
        """Return the input channels to model.

        Used by internal methods.
        """
        return self._n_in_channels

    def __init__(
        self,
        n_in_channels: int = 2,
        n_out_channels: int = 4,
        padding=None,
        batch_norm=False,
    ):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == "same":
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError("Unknow value for padding parameter.")

        self._n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 128, 5, padding=padding_5)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(128, 64, 5, padding=padding_5)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(64, 32, 3, padding=padding_3)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(32, n_out_channels, 3, padding=padding_3)

        nn.Sequential.__init__(
            self, *block1, *block2, *block3, *block4, *block5, *block6, *block7, conv8
        )

    @property
    def final_transformation(self):
        """
        Return the final transformation of the model.

        Returns
        -------
        self._final_transformation : Module
            Final layer applied to precision channels to ensure
            positiveness.
        """
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        """
        Setter method for the final_transformation attribute.

        Parameters
        ----------
        transformation : Module
            Final layer applied that transforms precision channels to ensure
            positiveness. It leaves other channels unchanged.
        """
        self._final_transformation = transformation

    def forward(self, x):
        """
        Forward method for the model.

        Parameters
        ----------
        x : tensor
            Input coarse velocities, with shape (N, C, H, W)

        Returns
        -------
        self.final_transformation(x) : tensor
            input with final_transformation operation performed
        """
        x = super().forward(x)
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        """
        Building unit for a convolution subblock followed
        by a ReLU.

        Parameters
        ----------
        conv : Conv2d
            Conv2d Module instance

        Returns
        -------
        subblock : Module
            A Module composed of a Conv2d followed by a ReLU,
            potentially with Batchnorm if activated via
            the batchnorm attribute.
        """
        subblock = [conv, nn.ReLU()]
        if self.batch_norm:
            subblock.append(nn.BatchNorm2d(conv.out_channels))
        return subblock

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
        dummy_in = torch.zeros((1, self.get_n_in_channels(), input_height, input_width))
        dummy_in = dummy_in.to(device=self.get_device())
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
        dummy_in = torch.zeros((1, self.get_n_in_channels(), input_height, input_width))
        dummy_in = dummy_in.to(device=self.get_device())
        dummy_out = self(dummy_in)
        return dummy_out.size(2)

    def get_device(self):
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
