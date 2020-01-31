#!/usr/bin/env python

import os
import sys
from setuptools import setup, Extension

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/echelle*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open("maelstrom/version.py").read())

# Get dependencies
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="maelstrom",
    version=__version__,
    description="A package to model the orbits of pulsating stars in binaries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    package_dir={"maelstrom": "maelstrom",},
    packages=["maelstrom"],
    install_requires=install_requires,
    url="https://github.com/danhey/maelstrom",
    include_package_data=True,
)
