from __future__ import division, print_function

import sys
import pytest

from ..maelstrom import Periodogram
import numpy as np
import matplotlib.pyplot as plt


def test_periodogram_basics():
    time, flux = np.linspace(0, 100, 100), np.random.randn(100)
    pg = Periodogram(time, flux, np.array([1]))
    res = pg.fit(periods=np.linspace(1, 10, 10))
    pg.diagnose()
