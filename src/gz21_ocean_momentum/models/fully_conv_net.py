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
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Sequential, Conv2d


from .blocks_2d import ConvBlock
from .base import DetectOutputSizeMixin


class FullyCNN(DetectOutputSizeMixin, Sequential):
    """
    Fully Convolutional Neural Net used for modelling ocean momentum.

    Attributes
    ----------
    in_chans : int
        Number of input chanels to model
    out_chans : int
        Number of output channels from model
    padding : str, optional
        The user-supplied padding argument: ``None`` or ``"same"``.
    batch_norm : bool
        Boolean switch determing whether ``BatchNorm2d`` layers are placed
        after the ``ReLU`` activations.

    """

    _n_in_channels: int

    def get_n_in_channels():
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
        """Build ``FullyCNN``."""
        padding_3, padding_5 = self._process_padding(padding)

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


    @staticmethod
    def _process_padding(padding: Optional[str] = None) -> Tuple[int, int]:
        """Process the padding argument.

        Parameters
        ----------
        padding : str or None
            Use-supplied padding argument.

        Returns
        -------
        padding_3 : int
            Stuff.
        padding_5 : int
            Stuff.

        Raises
        ------
        ValueError
            If ``padding`` is not ``None`` or ``same``.

        """
        if padding is None:
            return 0, 0
        if padding == "same":
            return 1, 2

        raise ValueError(f"Unknown value '{padding}' for padding parameter.")

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
