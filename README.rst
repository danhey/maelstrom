Maelstrom
=========

.. image:: https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat
   :target: https://docs.pymc.io
.. image:: https://img.shields.io/badge/powered_by-exoplanet-EB5368.svg?style=flat
    :target: https://github.com/dfm/exoplanet

*maelstrom* is a set of custom PyMC3 Models and solvers for
modelling binary orbits through the `phase modulation technique <https://arxiv.org/abs/1607.07879/>`_.
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
   
Read the docs here:
   https://danhey.github.io/maelstrom
