from __future__ import division, print_function

import sys
import pytest

from ..estimator import estimate_frequencies
import matplotlib.pyplot as plt
import pytest
import numpy as np

def test_estimate_frequencies_basics():
    # Generate some data
    frequencies = np.array([1., 10., 23., 100.])
    amplitudes = np.array([1,0.5,0.7,0.2])
    x = np.linspace(0,100,100000)
    y = np.sum(amplitudes[:,None]*np.sin(2*np.pi*frequencies[:,None]*x), axis=0)
    # Ensure we recover original frequencies
    vals = np.sort(estimate_frequencies(x, y, max_peaks=4))
    assert np.all(np.isclose(frequencies, vals))
    
def test_estimate_frequencies_no_optimization():
    frequencies = np.array([1., 10., 23., 100.])
    amplitudes = np.array([1,0.5,0.7,0.2])
    x = np.linspace(0,100,100000)
    y = np.sum(amplitudes[:,None]*np.sin(2*np.pi*frequencies[:,None]*x), axis=0)
    # Ensure we recover original frequencies
    vals = np.sort(estimate_frequencies(x,y, max_peaks=4, optimize_freq=False))
    assert np.all(np.isclose(frequencies, vals))