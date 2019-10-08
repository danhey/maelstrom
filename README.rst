maelstrom
=========

.. image:: https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat
   :target: https://docs.pymc.io
.. image:: https://img.shields.io/badge/powered_by-exoplanet-EB5368.svg?style=flat
    :target: https://github.com/dfm/exoplanet

*maelstrom* is a collection of custom PyMC3 Models built on top of the *exoplanet* package for
modelling binary orbits through the phase modulation technique.
This approach fully captures variations in a light curve caused by 
an orbital companion.

To install the current version::

    git clone https://github.com/danielhey/maelstrom.git
    cd maelstrom
    pip install -e .