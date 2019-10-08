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
    def __init__(self, 
                period=None, 
                lighttime=None,
                eccen=None,
                omega=None, 
                phi=None, 
                freq=None,
                with_rv=False,
                gammav=None):
        """This class defines an orbit model which will solve equation 10 of 
        Hey+2019 for given input values, defined within Theano.
        
        Parameters
        ----------
        period : pymc3.model.FreeRV, optional
            The orbital period prior, in days, by default None
        lighttime : pymc3.model.FreeRV, optional
            The convolved semi-major axis (asini) prior, given in seconds, by default None
        eccen : pymc3.model.FreeRV, optional
            The eccentricity prior, must be constrained between 0 and 1, by default None
        omega : pymc3.model.FreeRV, optional
            The argument of periapsis prior, given in angular coordinates, by default None
        phi : pymc3.model.FreeRV, optional
            The phase of periapsis, by default None
        freq : pymc3.model.FreeRV, optional
            Frequency prior, by default None
        with_rv : bool, optional
            Whether to include radial velocity calculations. If True, a prior
            on the systemic velocity (gammav) must be provided., by default False
        gammav : pymc3.model.FreeRV, optional
            The systemic velocity prior. Must be included if with_rv is True.
            by default None
        
        """
        self.period = period
        self.lighttime = lighttime
        self.omega = omega
        self.eccen = eccen
        self.phi = phi

        self.freq = freq

        if with_rv:
            if gammav is None:
                raise ValueError("The systemic velocity prior must be provided")
            else:
                self.gammav = gammav

    def get_time_delay(self, time):
        """Calculates the time delay under the given values
        
        Parameters
        ----------
        time : array-like
            Values of time at which to calculate the time delay
        
        Returns
        -------
        array-like
            Values of the time-delay
        """
        # Mean anom
        M = 2.0 * np.pi * time / self.period - self.phi
        
        # Negative psi to agree with Hey+2019. Implies closest star has negative
        # time delay
        if self.eccen is None:
            psi = -tt.sin(M)
        else:
            f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
            psi = -(1 - tt.square(self.eccen)) * tt.sin(f+self.omega) / (1 + self.eccen*tt.cos(f))

        tau = (self.lighttime / 86400) * psi[:, None]
        return tau

    def get_lightcurve_model(self, time, flux):
        """Calculates the light curve from the input time and flux data, with
        phase offsets from the time delays.
        
        Parameters
        ----------
        time : array-like
            Values of time at which to calculate the light curve
        flux : array-like
            Values of flux corresponding to `time`.
        
        Returns
        -------
        array-like
            Light curve
        """
        tau = self.get_time_delay(time)

        arg = 2. * np.pi * self.freq * (time[:, None] - tau)
        D = tt.concatenate((tt.cos(arg), tt.sin(arg)), axis=-1)
        w = tt.slinalg.solve(tt.dot(D.T, D), tt.dot(D.T, flux))
        lc_model = tt.dot(D, w)
        self.full_lc = lc_model
 
        return self.full_lc

    def get_radial_velocity(self, time):
        """Calculates the radial velocity within the framework of the Orbit for
        given input times.
        
        Parameters
        ----------
        time : array-like
            Time-values for each RV measurement
        
        Returns
        -------
        [type]
            [description]
        """
        M = 2.0 * np.pi * time / self.period - self.phi
        f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
        rv = ((self.lighttime / 86400) * (2.0 * np.pi * (1 / self.period) \
            * (1/tt.sqrt(1.0 - tt.square(self.eccen))) \
                * (tt.cos(f + self.omega) + self.eccen*tt.cos(self.omega))))
        rv *= 299792.458  # c in km/s
        rv += self.gammav # km/s

        return rv