# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import theano.tensor as tt
import theano
import pymc3 as pm
import exoplanet as xo
from exoplanet.orbits import get_true_anomaly

__all__ = ["Orbit"]


class Orbit:
    """
    This class defines an orbit model which solves the time delay equations
        for given input parameters and times. The orbit model can return
        either a synthetic light curve composed of `freq` sinusoids which are
        phase modulated with the orbital parameters, or instead can return a
        synthetic time delay curve. `Orbit` will also solve radial velocity
        curves given the same parameters.
    
    Args:
        period (pymc3.model.FreeRV, optional): Orbital period tensor. Defaults to None.
        lighttime (pymc3.model.FreeRV, optional): Projected semi-major axis tensor. Defaults to None.
        freq (array or pymc3.model.FreeRV, optional): Frequencies used in the model. Defaults to None.
        eccen (pymc3.model.FreeRV, optional): Eccentricity tensor. Defaults to None.
        omega (pymc3.model.FreeRV, optional): Periapsis tensor. Defaults to None.
        phi (pymc3.model.FreeRV, optional): Phase of periapsis tensor. Defaults to None.
    """

    def __init__(
        self, period=None, lighttime=None, freq=None, eccen=None, omega=None, phi=None
    ):
        self.period = period
        self.lighttime = lighttime
        self.omega = omega
        self.eccen = eccen
        self.phi = phi

        self.freq = freq

    def get_time_delay(self, time):
        """Calculates the time delay for the given time values.
        
        Args:
            time (array): Time values at which to calculate tau.
        
        Returns:
            array: Time delay values for each `time`
        """
        # Mean anom
        M = 2.0 * np.pi * time / self.period - self.phi

        # Negative psi to agree with Hey+2019. Implies closest star has negative
        # time delay
        if self.eccen is None:
            psi = -tt.sin(M)
        else:
            f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
            psi = (
                -1
                * (1 - tt.square(self.eccen))
                * tt.sin(f + self.omega)
                / (1 + self.eccen * tt.cos(f))
            )

        tau = (self.lighttime / 86400) * psi[:, None]
        return tau

    def get_lightcurve_model(self, time, flux):
        """Calculates a synthetic light curve given the orbital parameters of 
        the `Orbit` object and supplied times and fluxes. The `orbit.freq` are
        phase modulated with binary motion.
        
        Args:
            time (array): Time-stamps
            flux (array): Flux values for each `time`
        
        Returns:
            array: Synthetic light curve
        """
        tau = self.get_time_delay(time)

        arg = 2.0 * np.pi * self.freq * (time[:, None] - tau)
        D = tt.concatenate((tt.cos(arg), tt.sin(arg)), axis=-1)
        w = tt.slinalg.solve(tt.dot(D.T, D), tt.dot(D.T, flux))
        lc_model = tt.dot(D, w)
        self.full_lc = lc_model

        return self.full_lc

    def get_radial_velocity(self, time):
        """Calculates the radial velocity for the given time values
        
        Args:
            time (array): time values
        
        Returns:
            array: RV values
        """
        M = 2.0 * np.pi * time[:, None] / self.period - self.phi
        f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
        rv = -1 * (
            (self.lighttime / 86400)
            * (
                2.0
                * np.pi
                * (1 / self.period)
                * (1 / tt.sqrt(1.0 - tt.square(self.eccen)))
                * (tt.cos(f + self.omega) + self.eccen * tt.cos(self.omega))
            )
        )
        rv *= 299792.458  # c in km/s

        return tt.squeeze(rv)
