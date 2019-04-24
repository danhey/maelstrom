#!/usr/bin/env python

import os
from setuptools import setup, Extension

# Load the __version__ variable without importing the package already
exec(open('maelstrom/version.py').read())

# Get dependencies
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='lightkurve',
      version=__version__,
      description="A package to model the orbits of pulsating stars in binaries",
      long_description=open('README.rst').read(),
      license='MIT',
      package_dir={
            'maelstrom': 'maelstrom',},
      packages=['maelstrom'],
      install_requires=install_requires,
      url='https://github.com/danielhey/maelstrom',
      include_package_data=True,
)