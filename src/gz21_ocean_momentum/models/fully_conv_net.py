# -*- coding: utf-8 -*-
"""
Fully convolution network (FCN) for modelling ocean momentum.

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

from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Sequential, Conv2d
import torch

from .blocks_2d import ConvBlock

class FullyCNN(torch.nn.Sequential):
    """
    Fully convolution network (FCN) for modelling ocean momentum.

    Attributes
    ----------
    in_chans : int
        Number of input channels to model
    out_chans : int
        Number of output channels from model
    padding : str, optional
        The user-supplied padding argument: ``None`` or ``"same"``.
    batch_norm : bool
        Boolean switch determing whether ``BatchNorm2d`` layers are placed
        after the ``ReLU`` activations.

    """

    _n_in_channels: int

    def get_n_in_channels(self):
        """Return the input channels to model.

        Used by internal methods.
        """
        return self._n_in_channels

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 4,
        padding: Optional[str] = None,
        batch_norm=False,
    ):

        if padding is None:
            padding_3 = 0
            padding_5 = 0
        elif padding == "same":
            padding_3 = 1
            padding_5 = 2
        else:
            raise ValueError(f"Unsupported padding value (expected None or \"some\"): {padding}")

        super().__init__(
            ConvBlock(in_chans, 128, 5, padding_5, batch_norm),
            ConvBlock(128, 64, 5, padding_5, batch_norm),
            ConvBlock(64, 32, 3, padding_3, batch_norm),
            ConvBlock(32, 32, 3, padding_3, batch_norm),
            ConvBlock(32, 32, 3, padding_3, batch_norm),
            ConvBlock(32, 32, 3, padding_3, batch_norm),
            ConvBlock(32, 32, 3, padding_3, batch_norm),
            Conv2d(32, out_chans, 3, padding=padding_3),
        )

        # store in_chans as attribute
        self._n_in_channels = in_chans

    @property
    def final_transformation(self):
        """
        Return the final transformation of the model.

        Returns
        -------
        self._final_transformation : type?  # AB
            description?  # AB

        """
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        """
        Setter method for the final_transformation attribute.

        Parameters
        ----------
        transformation : type?  # AB
            description?  # AB
        """
        self._final_transformation = transformation

    def forward(self, batch: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """
        Forward method for the model.

        Parameters
        ----------
        x : type?  # AB
            description?  # AB

        Returns
        -------
        self.final_transformation(x) : type?  # AB
            input with final_transformation operation performed
        """
        return self.final_transformation(super().forward(batch))


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
        # TODO 2023-05-30 raehik: where does `self.parameters()` come from??
        params = list(self.parameters())[0]
        return params.device
