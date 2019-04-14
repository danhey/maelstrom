#!/usr/bin/env python

import os
from setuptools import setup, Extension

setup(
    name="maelstrom",
    license="MIT",
    packages=["maelstrom"],
    url='https://github.com/danielhey/maelstrom',
    install_requires=['numpy>=1.10', 'astropy>=1.0', 'corner', 'pymc3',
                      'theano', 'exoplanet'],
    zip_safe=True,
)
