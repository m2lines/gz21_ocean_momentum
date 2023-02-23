#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:18:53 2020

@author: arthur
"""
import torch


class DetectOutputSizeMixin:
    def output_width(self, input_height, input_width):
        x = torch.zeros((1, self.n_in_channels, input_height, input_width))
        x = x.to(device=self.device)
        y = self(x)
        # temporary fix for student loss
        if isinstance(y, tuple):
            y = y[0]
        return y.size(3)

    def output_height(self, input_height, input_width):
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
