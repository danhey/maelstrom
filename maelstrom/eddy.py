# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .estimator import estimate_frequencies
from .utils import (
    unique_colors,
    amplitude_spectrum,
    dft_phase,
    phase_error,
    mass_function,
)
from .orbit import Orbit

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
import pymc3 as pm
from pymc3.model import Model

import exoplanet as xo
from exoplanet.orbits import get_true_anomaly
from astropy.stats import LombScargle
import tqdm

__all__ = ["Eddy"]


class Eddy(Model):
    """A PyMC3 model for modelling light travel time variations in a binary
        system using the subdividing method.
        
        Args:
            time (array): Time observations
            tau (array): Flux observations
            period_guess (float, optional): Initial guess of the orbital period,
            in units of days. Defaults to None.
            asini_guess (float, optional): Initial guess of the projected
            semi-major axis, in units of seconds. Defaults to None.
    """

    def __init__(
        self, time: np.ndarray, tau: np.ndarray, period_guess=None, asini_guess=None
    ):
        self.time = time
        self.tau = tau

        if period_guess is None:
            period_guess = 100.0
        if asini_guess is None:
            asini_guess = 100.0

        period = pm.Normal("period", mu=period_guess, sd=100)
        varpi = xo.distributions.Angle("varpi")
        eccen = pm.Uniform("eccen", lower=1e-3, upper=0.999)
        lighttime = pm.Uniform(
            "lighttime", lower=-2000, upper=2000, testval=asini_guess
        )
        phi = xo.distributions.Angle("phi")

        M = 2.0 * np.pi * self.time / period - phi
        # True anom
        f = get_true_anomaly(M, eccen + tt.zeros_like(M))
        factor = 1.0 - tt.square(eccen)
        factor /= 1.0 + eccen * tt.cos(f)
        psi = -factor * tt.sin(f + varpi)

        tau = lighttime * psi
        taumodel = pm.Deterministic("tau", tau - tt.mean(tau))
        pm.Normal("obs", mu=taumodel, sd=tt.exp(logs), observed=self.tau)

    def optimize(self, vars=None):
        """Optimises the model.
        
        Args:
            vars (list, optional): List of parameters in the model to optimize.
            If none, will optimize all parameters at once. Defaults to None.
        
        Returns:
            dict: Optimisation results
        """
        with self as model:
            if vars is None:
                map_params = xo.optimize(start=model.test_point, vars=[self.logs])
                map_params = xo.optimize(
                    start=map_params, vars=[self.eccen, self.omega]
                )

                map_params = xo.optimize(start=map_params, vars=[self.phi])
                map_params = xo.optimize(start=map_params)
            else:
                map_params = xo.optimize(vars=vars)
        return map_params

    def sample(self, tune=3000, draws=3000, start=None, target_accept=0.9, **kwargs):
        """Samples the midel using the exoplanet PyMC3 sampler. By default, this 
        will sample from 2 chains over 2 cores simultaneously.
        
        Args:
            tune (int, optional): Number of tuning steps. Defaults to 3000.
            draws (int, optional): Number of samples. Defaults to 3000.
            start (dict, optional): Starting location of the sampler. If none is 
            supplied, the sampler will first optimize the model. Defaults to None.
            target_accept (float, optional): Target acceptance ratio of the NUTS 
            sampler. Defaults to 0.9.
        
        Returns:
            trace: A PyMC3 trace of the posterior.
        """

        with self:
            trace = pm.sample(
                tune=tune,
                draws=draws,
                step=xo.get_dense_nuts_step(target_accept=target_accept),
                start=start,
                **kwargs
            )
        return trace
