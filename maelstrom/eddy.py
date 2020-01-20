# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .estimator import estimate_frequencies
from .utils import unique_colors, amplitude_spectrum, dft_phase, phase_error, mass_function

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano
import pymc3 as pm
from pymc3.model import Model
from maelstrom.orbit import Orbit

import exoplanet as xo
from exoplanet.orbits import get_true_anomaly
from astropy.stats import LombScargle
import tqdm

__all__ = ["Eddy"]

class Eddy(Model):

    def __init__(self, time: np.ndarray, tau: np.ndarray, period_guess=None,
                        asini_guess=None):
        """A simplified model for modelling light travel time variations
        in a binary system. This class directly models the time delay
        information and should be used on longer period binaries.
        
        Parameters
        ----------
        time : np.ndarray
            Time observations
        tau : np.ndarray
            Extracted time delay for each `time` point. Should be weighted
            by the amplitude of the modes
        period_guess : float, optional
            Initial guess of the orbital period, in units of days, by default None
        asini_guess : float, optional
            Initial guess of the projected semi-major axis in units of seconds, by default None
        """
        self.time = time
        self.tau = tau

        if period_guess is None:
            period_guess = 100.
        if asini_guess is None:
            asini_guess = 100.

        period = pm.Normal("period", mu=period_guess, sd=100)
        varpi = xo.distributions.Angle("varpi")
        eccen = pm.Uniform("eccen", lower=1e-3, upper=0.999)
        lighttime = pm.Uniform('lighttime', lower=-2000, upper=2000, testval=asini_guess)
        phi = xo.distributions.Angle('phi')
        
        M = 2. * np.pi * self.time / period - phi
        # True anom
        f = get_true_anomaly(M, eccen + tt.zeros_like(M))
        factor = 1.0 - tt.square(eccen)
        factor /= 1.0 + eccen * tt.cos(f)
        psi = -factor * tt.sin(f + varpi)
        
        tau = lighttime * psi
        taumodel = pm.Deterministic('tau', tau - tt.mean(tau))
        pm.Normal("obs", mu=taumodel, sd=tt.exp(logs), observed=self.tau)

    def optimize(self, vars=None):
        """Optimises the model.
        
        Parameters
        ----------
        vars : array of model parameters, optional
            parameters of the model to be optimized, by default None
        
        Returns
        -------
        dict
            optimisation results
        """
        
        with self as model:
            if vars is None:
                map_params = xo.optimize(start=model.test_point, vars=[self.logs])
                map_params = xo.optimize(start=map_params, vars=[self.eccen, self.omega])
                    
                map_params = xo.optimize(start=map_params, vars=[self.phi])
                map_params = xo.optimize(start=map_params)
            else:
                map_params = xo.optimize(vars=vars)
        return map_params

    def sample(self, tune=3000, draws=3000, start=None, target_accept=0.9, **kwargs):
        """
        Samples the model using the exoplanet PyMC3 sampler. By default,
        this will sample from 2 chains over 2 cores simultaneously.
        
        Parameters
        ----------
        tune : float, optional
            Number of tuning steps for the sampler (default 3000)
        draws : float, optional
            Number of samples from which to populate the trace (default 3000)
        start : dict, optional
            Starting location of the sampler. If none is supplied, the sampler
            will first optimize the model.
        target_accept : float, optional
            The target acceptance ratio of the NUTS sampler (default 0.9).
        **kwargs : 
            Keyword arguments to pass to sample.tune and sample.sample

        Returns
        -------
        trace : `pm trace object?`

        """
        with self:
            trace = pm.sample(tune=tune, draws=draws,
                            step=xo.get_dense_nuts_step(target_accept=target_accept), start=start, **kwargs)
        return trace