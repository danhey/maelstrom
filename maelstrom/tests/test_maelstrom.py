from __future__ import division, print_function

import sys
import pytest

from ..maelstrom import Maelstrom, PB1Model
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo


def test_maelstrom_basics():
    time, flux = np.linspace(0, 100, 10000), np.random.randn(10000)

    # Check we can instantiate under different circumstances
    ms = Maelstrom(time, flux, max_peaks=3)
    ms = Maelstrom(time, flux, freq=np.array([10, 20, 30]))
    ms.get_period_estimate()

    # Check plotting
    ms.first_look()
    ms.plot_time_delay_periodogram()
    ms.plot_time_delay_periodogram_period()
    ms.plot_time_delay()
    ms.plot_periodogram()


def test_maelstrom_model():
    time, flux = np.linspace(0, 100, 10000), np.random.randn(10000)
    ms = Maelstrom(time, flux, max_peaks=1)
    ms.setup_orbit_model(period=1)
    opt = ms.optimize()
    # Also make sure we can profile it
    ms.profile()


def test_maelstrom_shape():
    time, flux = np.linspace(0, 100, 10000), np.random.randn(10000)
    total_freqs = 3
    ms = Maelstrom(time, flux, max_peaks=total_freqs)
    ms.setup_orbit_model(period=1)
    with ms:
        assert xo.eval_in_model(ms.lighttime.shape) == total_freqs


def test_maelstrom_pinning():
    np.random.seed(2)
    time, flux = np.linspace(0, 100, 10000), np.random.randn(10000)
    ms = Maelstrom(time, flux, max_peaks=3)
    ms.setup_orbit_model(period=1)

    # Make sure pinning works
    res = ms.pin_orbit_model()
    assert res.name == "PB1"

    # Make sure we can optimize the model
    opt = res.optimize()


def test_maelstrom_inputs():

    time, flux = np.linspace(0, 100, 10000), np.random.randn(500)

    # Let's raise some errors..
    with pytest.raises(ValueError):
        ms = Maelstrom(time, flux, max_peaks=3)

    time, flux = np.linspace(0, 100, 10000), np.random.randn(10000)
    flux[42] = np.nan
    with pytest.raises(ValueError):
        ms = Maelstrom(time, flux)


# def test_maelstrom_uncertainty():
#     time, flux = np.linspace(0, 100, 100), np.random.randn(100)
#     ms = Maelstrom(time, flux, freq=np.array([5]))
#     ms.setup_orbit_model(period=1)
#     opt = ms.optimize()
#     ms.uncertainty(opt)


def test_maelstrom_gp():
    time, flux = np.linspace(0, 100, 100), np.random.randn(100)
    ms = PB1Model(time, flux, freq=np.array([5]))
    ms.init_orbit(period=1, asini=1, with_gp=True)
