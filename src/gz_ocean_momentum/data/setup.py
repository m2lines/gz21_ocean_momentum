# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:11:09 2019

@author: Arthur
"""
import setuptools
from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(name='utils', ext_modules=cythonize('_utils.pyx'),
      include_dirs=[numpy.get_include()])