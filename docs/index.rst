Maelstrom
=========

.. image:: https://github.com/danhey/maelstrom/workflows/maelstrom-tests/badge.svg
.. image:: https://github.com/danhey/maelstrom/workflows/Docs/badge.svg
   :target: https://danhey.github.io/maelstrom/
.. image:: https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat
   :target: https://docs.pymc.io
.. image:: https://img.shields.io/badge/powered_by-exoplanet-EB5368.svg?style=flat
    :target: https://github.com/dfm/exoplanet
.. image:: https://codecov.io/gh/danhey/maelstrom/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/danhey/maelstrom
 
*maelstrom* is a set of custom PyMC3 Models and solvers for modelling binary orbits through the `phase modulation technique <https://arxiv.org/abs/1607.07879/>`_.
Unlike previous codes, *maelstrom* fits each individual datapoint in the time series by forward modelling the time delay onto the light curve. This approach fully captures variations in a light curve caused by 
an orbital companion.

To install the current version::

    git clone https://github.com/danhey/maelstrom.git
    cd maelstrom
    pip install -e .

To get started::
   
   from maelstrom import Maelstrom
   ms = Maelstrom(time, flux)
   ms.optimize()


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   user/citation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/Getting started.ipynb
   notebooks/Estimating frequencies.ipynb
   notebooks/Custom priors.ipynb
   notebooks/Recovering weak signals.ipynb
   notebooks/FAQ.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Case studies from paper

   case_studies/9651065.ipynb
   case_studies/6780873.ipynb
   case_studies/10080943.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API
   
   api/maelstrom
   api/orbit
   api/periodogram
   api/utils
   api/estimator
   api/eddy

License & attribution
---------------------

Copyright 2019 Daniel Hey, Daniel Foreman-Mackey, and Simon Murphy.

The source code is made available under the terms of the MIT license.


Changelog
---------

.. include:: ../CHANGES.rst
