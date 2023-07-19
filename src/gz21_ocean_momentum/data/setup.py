# TODO Don't think there should be a setup.py here... Remove/refactor?
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:11:09 2019

@author: Arthur
"""
from distutils.core import setup

import numpy
import setuptools
from Cython.Build import cythonize

setup(
    name="utils",
    ext_modules=cythonize("_utils.pyx"),
    include_dirs=[numpy.get_include()],
)
