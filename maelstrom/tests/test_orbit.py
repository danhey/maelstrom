from __future__ import division, print_function

import sys
import pytest

from ..orbit import Orbit

import pymc3 as pm
import exoplanet as xo
import numpy as np


def test_orbit_basics():
    with pm.Model() as model:
        orbit = Orbit(
            period=pm.Flat("period", testval=100),
            lighttime=pm.Flat("lighttime", testval=100),
            eccen=pm.Flat("eccen", testval=0.0),
            omega=pm.Flat("omega", testval=0.0),
            phi=pm.Flat("phi", testval=0.0),
            freq=pm.Flat("freq", testval=np.array(1.0)),
        )

        times = np.linspace(0, 100, 100)
        fluxes = np.random.randn(100)

        # Check we can run everything fine ...
        xo.eval_in_model(orbit.get_time_delay(times))
        xo.eval_in_model(orbit.get_radial_velocity(times))
        xo.eval_in_model(orbit.get_lightcurve_model(times, fluxes))


def test_lighttime_shapes():
    with pm.Model() as model:
        orbit = Orbit(
            period=pm.Flat("period", testval=100),
            lighttime=pm.Flat("lighttime", testval=np.array([100, -100]), shape=2),
            eccen=pm.Flat("eccen", testval=0.0),
            omega=pm.Flat("omega", testval=0.0),
            phi=pm.Flat("phi", testval=0.0),
            freq=pm.Flat("freq", testval=np.array(1.0)),
        )

        times = np.linspace(0, 100, 100)

        # Check tensors are shaped correctly
        res = xo.eval_in_model(orbit.get_time_delay(times))
        assert res.shape == (len(times), 2)
        res = xo.eval_in_model(orbit.get_radial_velocity(times))
        assert res.shape == (len(times), 2)
