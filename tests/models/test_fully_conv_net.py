# -*- coding: utf-8 -*-

import numpy as np
import torch

from gz21_ocean_momentum.models.fully_conv_net import *


def test_construct_valid():
    """Construct a valid FullyCNN instance.

    Simple check migrated from `models.models1`.
    """
    net = FullyCNN()
    net._final_transformation = lambda x: x
    input_ = torch.randint(0, 10, (17, 2, 35, 30)).to(dtype=torch.float)
    input_[0, 0, 0, 0] = np.nan
    output = net(input_)

    # no assertion; above constructor will raise exception on erroneous input
