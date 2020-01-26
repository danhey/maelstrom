from __future__ import division, print_function

import sys
import pytest

from ..utils import (
    mass_function,
    unique_colors,
    amplitude_spectrum,
    dft_phase,
    smooth,
    phase_error,
)
import astropy.units as u
import numpy as np


def test_mass_function():
    value = mass_function(1 * u.day, 1 * u.s)
    # Assert units
    assert value.unit == u.M_sun
    # Assert value
    assert np.isclose(0.001, value.value, atol=1e-3)

    # Check range of input units
    time_units = [u.day, u.year, u.second]
    for time_unit in time_units:
        value = mass_function(1 * time_unit, 1 * time_unit)
        assert value.unit == u.M_sun


def test_unique_colors():
    colors = unique_colors(5)
    assert len(colors) == 5
