# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .estimator import estimate_frequencies
from .utils import unique_colors, amplitude_spectrum, dft_phase, phase_error

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
import tqdm
import pymc3 as pm
from pymc3.model import Model
import exoplanet as xo
from exoplanet.orbits import get_true_anomaly
from astropy.stats import LombScargle

__all__ = ["Periodogram"]


class Periodogram:
    """A class to brute force the likelihood in search of small signals.
    For each orbital period, it will iterate over a simplified Maelstrom
    model, optimise the values, and then record the loglikelihood. 

    This is useful for planets, and short period binaries.
    
    Args:
        time (array): Time values
        mag (array): Magnitude (or flux) values. Must be relative
        freq (array): Array of frequencies
    """

    def __init__(self, time, mag, freq):

        self.time = time
        self.mag = mag
        self.freq = freq
        # Initialise the model
        self.model = pm.Model()

        self.results = []

        with self.model as model:
            period = pm.Flat("period", testval=10.0)
            nu = pm.Flat("nu", testval=15)

            phi = xo.distributions.Angle("phi")
            logasini = pm.Uniform(
                "logasini", lower=np.log(1), upper=np.log(1000), testval=np.log(10)
            )
            drift = pm.Normal("drift", mu=0, sd=1.0)

            M = 2.0 * np.pi * self.time / period - phi

            factor = 2.0 * np.pi * nu
            A = factor * (1 + drift) * self.time
            B = -factor * (tt.exp(logasini) / 86400) * tt.sin(M)

            sinarg = tt.sin(A + B)
            cosarg = tt.cos(A + B)

            DT = tt.stack((sinarg, cosarg, tt.ones_like(sinarg)))
            w = tt.slinalg.solve(tt.dot(DT, DT.T), tt.dot(DT, self.mag))
            pm.Deterministic("w", w)
            pm.Deterministic("phase", tt.arctan2(w[1], w[0]))
            lc_model = tt.dot(DT.T, w)

            pm.Normal("obs", mu=lc_model, observed=self.mag)

            self.fit_params = [v for v in model.vars if v.name not in ["period", "nu"]]

    def _run_fit(self, p, nu):
        with self.model as model:
            start = dict(model.test_point)
            start["period"] = p
            start["nu"] = nu
            point, info = xo.optimize(
                start, vars=self.fit_params, return_info=True, verbose=False
            )
        return -info.fun, point

    def fit(self, periods):
        """Run the periodogram model for a given array of periods.
        
        Args:
            periods (array): Orbital periods over which to iterate.
        
        Returns:
            array: Results
        """
        results = []
        for f in self.freq:
            results.append([self._run_fit(p, f) for p in tqdm.tqdm(periods)])
        self.results = results
        self.periods = periods
        return self.results

    def diagnose(self):
        """Generate diagnostic plots of the fit values. After running 
        `Periodogram.fit()`, `diagnose` will plot the resulting period vs 
        likelihood and period vs optimised asini values.
        
        Returns:
            array: array of matplotlib axes.
        """
        results = self.results
        fig, axes = plt.subplots(1, 2, figsize=[11, 4])

        ax = axes[0]
        ys = np.array([[r[0] for r in row] for row in results])
        sm = np.sum(ys, axis=0)
        period_ind = np.argmax(sm)
        ax.plot(self.periods, sm)
        ax.axvline(self.periods[period_ind], c="red", linestyle="dashed")
        ax.set_xlabel("Period [day]")
        ax.set_ylabel("Model likelihood")
        ax.set_xlim(self.periods[0], self.periods[-1])

        ax = axes[1]
        ys = np.array([[np.exp(r[1]["logasini"]) for r in row] for row in results])
        ax.plot(self.periods, ys.T)
        ax.set_xlabel("Period [day]")
        ax.set_ylabel("asini (s)")
        ax.set_xlim(self.periods[0], self.periods[-1])

        return axes
