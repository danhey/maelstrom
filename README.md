<h1 align="center">Maelstrom</h1>

<p align="center">
   <a href="https://github.com/danhey/maelstrom/actions?query=workflow%3Amaelstrom-tests"><img
   src="https://github.com/danhey/maelstrom/workflows/maelstrom-tests/badge.svg"/></a>
   <a href="https://danhey.github.io/maelstrom/"><img
   src="https://github.com/danhey/maelstrom/workflows/Docs/badge.svg"/></a>
   <a href="https://codecov.io/gh/danhey/maelstrom"><img
   src="https://codecov.io/gh/danhey/maelstrom/branch/master/graph/badge.svg"/></a>
   <br/>
   <a href="https://docs.pymc.io"><img
   src="https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat"/></a>
   <a href="https://github.com/dfm/exoplanet"><img
   src="https://img.shields.io/badge/powered_by-exoplanet-EB5368.svg?style=flat"/></a>
</p>

  
*maelstrom* is a set of custom PyMC3 Models and solvers for
modelling binary orbits through the [phase modulation technique](https://arxiv.org/abs/1607.07879/).
Unlike previous codes, *maelstrom* fits each individual datapoint in the time series by forward modelling the time delay onto the light curve. This approach fully captures variations in a light curve caused by 
an orbital companion.

To install the current version

```python
git clone https://github.com/danhey/maelstrom.git
cd maelstrom
pip install -e .
```

To get started

```python
from maelstrom import Maelstrom
ms = Maelstrom(time, flux)
ms.optimize()
```
   
[Read the docs here](https://danhey.github.io/maelstrom)
