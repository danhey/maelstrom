from __future__ import division, print_function

from exoplanet.utils import eval_in_model
from exoplanet.orbits import get_true_anomaly
import pymc3 as pm
import theano.tensor as tt
import numpy as np

__all__ = ["SyntheticBinary"]


class SyntheticBinary(object):
    """This class generates a synthetic light curve and injects a time delay
    signal corresponding to the input binary orbit parameters. Note that the 
    input parameters do *NOT* need to be PyMC3 tensors, unlike the `Orbit` module.
    
    Args:
        time (array): time-stamps 
        freq (array): oscillation modes of the light curve
        amplitude (array): amplitudes corresponding to each `freq`
        period (float): Orbital period in days
        eccen (float): Orbital eccentricity, must be between 0 and 1
        asini (float): The projected semi-major axis of the orbit in seconds.
        varpi (float): Angle between the ascending node and periapsis
        tref (float): Reference time of periapsis
        tau (bool, optional): Whether to inject binary motion or not. Defaults to True.
    """

    def __init__(
        self, time, freq, amplitude, period, eccen, asini, varpi, tref, tau=True
    ):
        self.time = time
        self.freq = freq
        self.amplitude = amplitude
        self.period, self.eccen, self.asini, self.varpi, self.tref = (
            period,
            eccen,
            asini,
            varpi,
            tref,
        )

        self.make_lightcurve(tau=tau)
        self.flux = self.flux_true

    def make_lightcurve(self, tau=True):
        """Generates the light curve with added binarity.
        
        Args:
            tau (bool, optional): Whether to include binarity. Defaults to True.
        """
        self.flux_true = np.zeros(len(self.time))
        # self.tau = 0
        if tau:
            with pm.Model() as model:
                # M = tt.zeros_like(tt.constant(self.time) - self.tref) + 2.0 * np.pi * (tt.constant(self.time)) / self.period
                M = 2.0 * np.pi * (tt.constant(self.time) - self.tref) / self.period
                f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
                tau_tens = (
                    -(1 - tt.square(self.eccen))
                    * tt.sin(f + self.varpi)
                    / (1 + self.eccen * tt.cos(f))
                ) * (self.asini / 86400.0)
                self.tau = eval_in_model(tau_tens)
        else:
            self.tau = 0

        for j, A_j in zip(self.freq, self.amplitude):
            omega = 2 * np.pi * j
            # self.flux_true += A_j * np.cos(omega * (self.time - self.tau)) + A_j * np.sin(omega * (self.time - self.tau))
            self.flux_true += A_j * np.sin(omega * (self.time - self.tau))

    def add_noise(self, snr=5):
        """Sets the SNR of the oscillation signal.
        
        Args:
            snr (float, optional): SNR. Defaults to 5.
        """
        # Add noise
        s_rms = self.amplitude.max() / (np.sqrt(np.pi / len(self.time)) * snr)
        self.flux = self.flux_true + np.random.normal(
            loc=0.0, scale=s_rms, size=len(self.time)
        )
