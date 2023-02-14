# THIS FILE CAN BE REMOVED
# WAS USED IN THE RESEARCH PROCESS BUT WAS SUPERCEDED BY MODELS1

# During training you can pass a choice of model
# This could be passed as an alternative 



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from src.gz_ocean_momentum.data.datasets import CropToMultipleof
from .base import DetectOutputSizeMixin, FinalTransformationMixin


class Unet_(nn.Module, DetectOutputSizeMixin):
    """
    Parameters
    ----------
    n_in_channels : int
        Description?
    n_out_channels : int
        Description?
    n_scales : int
        Description?
    depth : TYPE?
        Description?
    kernel_sizes : TYPE?
        Description?
    batch_norm : bool
        Description?
    padding : bool
        Description?

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        n_in_channels: int = 2,
        n_out_channels: int = 4,
        n_scales: int = 2,
        depth=64,
        kernel_sizes=None,
        batch_norm=False,
        padding=False,
    ):
        nn.Module.__init__(self)
        DetectOutputSizeMixin.__init__(self)
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_scales = n_scales
        self.depth = depth
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        self.kernel_sizes = self._repeat(kernel_sizes, n_scales)
        self.batch_norm = batch_norm
        self.padding = padding
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.final_convs = None
        self.conv_layers = []
        self.linear_layer = None
        self.linear_layer = None
        self._build_convs()

    @staticmethod
    def _repeat(l, n_times):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        for i in range(n_times - len(l)):
            l.append(l[-1])
        return l

    @staticmethod
    def _crop_spatial(t: torch.Tensor, size: torch.Size):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        size_t = t.size()
        dh = (size_t[2] - size[2]) // 2
        dw = (size_t[3] - size[3]) // 2
        return t[:, :, dh : size_t[2] - dh, dw : size_t[3] - dw]

    def _padding(self, k_size: int):
        """
        Returns the padding, depending on the padding parameter
 
        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        if self.padding:
            return k_size // 2
        return 0

    def forward(self, x: torch.Tensor):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        blocks = []
        for i in range(self.n_scales):
            # Convolutions layers for that scale
            x = self.down_convs[i](x)
            if i < self.n_scales - 1:
                blocks.append(x)
                # Downscaling
                x = self.down(x)
        blocks.reverse()
        for i in range(self.n_scales - 1):
            x = self.up(x, i)
            # Concatenate to the finer scale, after cropping
            y = self._crop_spatial(blocks[i], x.size())
            x = torch.cat((x, y), 1)
            # Convolutions for that scale
            x = self.up_convs[i](x)
        final = self.final_convs(x)
        return final

    def down(self, x):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        return F.max_pool2d(x, 2)

    def up(self, x, i):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        return self.up_samplers[i](x)

    def _make_subblock(self, conv):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc

    def _build_convs(self):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        for i in range(self.n_scales):
            if i == 0:
                n_in_channels = self.n_in_channels
                n_out_channels = self.depth
            else:
                n_in_channels = n_out_channels
                n_out_channels = 2 * n_out_channels
            k_size = self.kernel_sizes[i]
            padding = self._padding(k_size)
            conv1 = torch.nn.Conv2d(
                n_in_channels, n_out_channels, k_size, padding=padding
            )
            conv2 = torch.nn.Conv2d(
                n_out_channels, n_out_channels, k_size, padding=padding
            )
            block1 = self._make_subblock(conv1)
            block2 = self._make_subblock(conv2)
            submodule = nn.Sequential(*block1, *block2)
            self.down_convs.append(submodule)
            self.conv_layers.append(conv1)
            self.conv_layers.append(conv2)
        for i in range(self.n_scales - 1):
            # Add the upsampler
            up_sampler = nn.Upsample(mode="bilinear", scale_factor=2)
            conv = torch.nn.Conv2d(n_out_channels, n_out_channels // 2, 1)
            self.up_samplers.append(nn.Sequential(up_sampler, conv))
            # The up convs
            n_in_channels = n_out_channels
            n_out_channels = n_out_channels // 2
            k_size = self.kernel_sizes[-i]
            padding = self._padding(k_size)
            conv1 = torch.nn.Conv2d(
                n_in_channels, n_out_channels, k_size, padding=padding
            )
            conv2 = torch.nn.Conv2d(
                n_out_channels, n_out_channels, k_size, padding=padding
            )
            block1 = self._make_subblock(conv1)
            block2 = self._make_subblock(conv2)
            submodule = nn.Sequential(*block1, *block2)
            self.up_convs.append(submodule)
            self.conv_layers.append(conv1)
            self.conv_layers.append(conv2)
        # Final convs
        conv1 = torch.nn.Conv2d(
            n_out_channels, n_out_channels // 2, 3, padding=self._padding(3)
        )

        conv3 = torch.nn.Conv2d(
            n_out_channels // 2, self.n_out_channels, 3, padding=self._padding(3)
        )
        block1 = self._make_subblock(conv1)
        self.final_convs = nn.Sequential(*block1, conv3)

    def get_features_transform(self):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        return CropToMultipleof(2**self.n_scales)


class Unet(FinalTransformationMixin, Unet_):
    """
    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, *args, **kargs):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        Unet_.__init__(self, *args, **kargs)


class Unet3scales(Unet):
    """
    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, *args, **kargs):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        super().__init__(n_scales=3)


class Unet32(Unet):
    """
    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, *args, **kargs):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        super().__init__(depth=32)


class Unet32_3scales(Unet32):
    """
    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, *args, **kargs):
        """
        Function Purpose?

        Parameters
        ----------
        input : TYPE?
            Description?

        Returns
        -------

        """
        super().__init__(n_scales=3)
