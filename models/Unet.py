# THIS FILE CAN BE REMOVED
# WAS USED IN THE RESEARCH PROCESS BUT WAS SUPERCEDED BY MODELS1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:22:59 2020

@author: arthur
Implementation of the U-net structure
"""


import torch
from torch.nn import Module, ModuleList, Upsample, Sequential
from torch.nn import functional as F
from torch.nn.functional import pad
import torch.nn as nn
from .base import DetectOutputSizeMixin, FinalTransformationMixin
from data.datasets import CropToMultipleof


class Unet_(Module, DetectOutputSizeMixin):
    def __init__(
        self,
        n_in_channels: int = 2,
        n_out_channels: int = 4,
        n_scales: int = 2,
        depth=64,
        kernel_sizes=[3, 3],
        batch_norm=False,
        padding=False,
    ):
        Module.__init__(self)
        DetectOutputSizeMixin.__init__(self)
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_scales = n_scales
        self.depth = depth
        self.kernel_sizes = self._repeat(kernel_sizes, n_scales)
        self.batch_norm = batch_norm
        self.padding = padding
        self.down_convs = ModuleList()
        self.up_convs = ModuleList()
        self.up_samplers = ModuleList()
        self.final_convs = None
        self.conv_layers = []
        self.linear_layer = None
        self.linear_layer = None
        self._build_convs()

    @staticmethod
    def _repeat(l, n_times):
        for i in range(n_times - len(l)):
            l.append(l[-1])
        return l

    @staticmethod
    def _crop_spatial(t: torch.Tensor, size: torch.Size):
        size_t = t.size()
        dh = (size_t[2] - size[2]) // 2
        dw = (size_t[3] - size[3]) // 2
        return t[:, :, dh : size_t[2] - dh, dw : size_t[3] - dw]

    def _padding(self, k_size: int):
        """Returns the padding, depending on the padding parameter"""
        if self.padding:
            return k_size // 2
        else:
            return 0

    def forward(self, x: torch.Tensor):
        blocks = list()
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
        return F.max_pool2d(x, 2)

    def up(self, x, i):
        return self.up_samplers[i](x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc

    def _build_convs(self):
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
            submodule = Sequential(*block1, *block2)
            self.down_convs.append(submodule)
            self.conv_layers.append(conv1)
            self.conv_layers.append(conv2)
        for i in range(self.n_scales - 1):
            # Add the upsampler
            up_sampler = Upsample(mode="bilinear", scale_factor=2)
            conv = torch.nn.Conv2d(n_out_channels, n_out_channels // 2, 1)
            self.up_samplers.append(Sequential(up_sampler, conv))
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
            submodule = Sequential(*block1, *block2)
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
        self.final_convs = Sequential(*block1, conv3)

    def get_features_transform(self):
        return CropToMultipleof(2**self.n_scales)


class Unet(FinalTransformationMixin, Unet_):
    def __init__(self, *args, **kargs):
        Unet_.__init__(self, *args, **kargs)


class Unet3scales(Unet):
    def __init__(self, *args, **kargs):
        super().__init__(n_scales=3)


class Unet32(Unet):
    def __init__(self, *args, **kargs):
        super().__init__(depth=32)


class Unet32_3scales(Unet32):
    def __init__(self, *args, **kargs):
        super().__init__(n_scales=3)
