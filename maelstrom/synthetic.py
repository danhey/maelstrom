from __future__ import division, print_function

from exoplanet.utils import eval_in_model
from exoplanet.orbits import get_true_anomaly
import pymc3 as pm
import theano.tensor as tt
import numpy as np

__all__ = ["SyntheticBinary"]

class SyntheticBinary(object):
    """
    This class makes a synthetic light curve and injects a time delay signal corresponding to the input binary orbit parameters    
    """
    def __init__(self, time, freq, amplitude, 
                 period, eccen, asini, varpi, tref, tau=True):
        self.time = time
        self.freq = freq
        self.amplitude = amplitude
        self.period, self.eccen, self.asini, self.varpi, self.tref = period, eccen, asini, varpi, tref
        
        self.make_lightcurve(tau=tau)
        self.flux = self.flux_true
        
    def make_lightcurve(self, tau=True):
        self.flux_true = np.zeros(len(self.time))
        #self.tau = 0
        if tau:
            with pm.Model() as model:
                #M = tt.zeros_like(tt.constant(self.time) - self.tref) + 2.0 * np.pi * (tt.constant(self.time)) / self.period
                M = 2.0 * np.pi * (tt.constant(self.time) - self.tref) / self.period
                f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
                tau_tens = (- (1 - tt.square(self.eccen)) * tt.sin(f+self.varpi) / (1 + self.eccen*tt.cos(f))) * (self.asini / 86400.)
                self.tau = eval_in_model(tau_tens)
        else:
            self.tau = 0
            
        for j, A_j in zip(self.freq, self.amplitude):
            omega = 2 * np.pi * j
            #self.flux_true += A_j * np.cos(omega * (self.time - self.tau)) + A_j * np.sin(omega * (self.time - self.tau))
            self.flux_true += A_j * np.sin(omega * (self.time - self.tau))
    
    def add_noise(self, snr=5):
        # Add noise
        s_rms = self.amplitude.max() / (np.sqrt(np.pi / len(self.time)) * snr)
        #print(s_rms, self.amplitude)
        #print(np.random.normal(loc=0.0, scale=s_rms, size=len(self.time)))
        self.flux = self.flux_true + np.random.normal(loc=0.0, scale=s_rms, size=len(self.time))
        #self.flux_err = noise * np.std(self.flux_true)
        #self.flux = self.flux_true + self.flux_err * np.random.randn(len(self.flux_true))