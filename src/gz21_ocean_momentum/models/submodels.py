#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:34:58 2020

@author: arthur
"""

from gz21_ocean_momentum.data.xrtransforms import (
    ScalingTransform,
    SeasonalStdizer,
    ChainedTransform,
    TargetedTransform,
)

# v CM2.6 specific v

# velocities (usurf, vsurf) are metres/s
velocity_vars = ["usurf", "vsurf"]

# forcing unitless -- common scale is ?
forcing_vars = ["S_x", "S_y"]

velocity_scaler = TargetedTransform(ScalingTransform(10.0), velocity_vars)
forcing_scaler = TargetedTransform(ScalingTransform(1e7, inverse=False), forcing_vars)

monthly_stdizer_means = SeasonalStdizer(std=False)
monthly_stdizer_means_stds = SeasonalStdizer(std=True)

transform1 = ChainedTransform((velocity_scaler, forcing_scaler, monthly_stdizer_means))

transform2 = ChainedTransform(
    (velocity_scaler, forcing_scaler, monthly_stdizer_means_stds)
)

transform3 = ChainedTransform((velocity_scaler, forcing_scaler))
