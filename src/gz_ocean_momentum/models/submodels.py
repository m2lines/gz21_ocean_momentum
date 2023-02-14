#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of data operations to work on u and v data. called from
"""
from src.gz_ocean_momentum.data.xrtransforms import (
    ScalingTransform,
    SeasonalStdizer,
    ChainedTransform,
    TargetedTransform,
    BZFormulaTransform,
)

velocity_vars = ["usurf", "vsurf"]
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

transform4 = ChainedTransform((transform3, BZFormulaTransform()))
