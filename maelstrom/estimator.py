# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from astropy.stats import LombScargle
from scipy import optimize
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt

__all__ = ["estimate_frequencies"]

def estimate_frequencies(x, y, fmin=None, fmax=None, max_peaks=3,
                         oversample=4.0, optimize_freq=True):
    """
    Attempts to pick out the best frequencies
    for use with phase modulation. 

    Parameters
    ----------
    x : float, optional
        Number of tuning steps for the sampler (default 3000)
    y : float, optional
        Number of samples from which to populate the trace (default 3000)
    fmin : bool, optional
        If set to True, the sampler will optimize the model before
        attempting to sample. If False (default), the sampler will
        initialise at the testpoints of your priors.
    fmax : float, optional
        The target acceptance ratio of the NUTS sampler (default 0.9).
    max_peaks : int
        Maximum number of frequencies to return (default 3)
    oversample : float
        Oversample factor for the spectrum (default 4.)
    optimize_freq : bool
        Whether to optimize the frequencies according to the
        Maelstrom model using Scipy.optimize (default True)

    Returns
    -------
    peaks : `numpy.ndarray`
        Array of frequencies of length `max_peaks`

    """
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = ny

    freq = np.arange(fmin, fmax, df / oversample)
    power = LombScargle(x, y).power(freq)

    # Find peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for j in range(max_peaks):
        i = peak_inds[0]
        freq0 = freq[i]
        alias = 2.0*ny - freq0

        m = np.abs(freq[peak_inds] - alias) > 25*df
        m &= np.abs(freq[peak_inds] - freq0) > 25*df

        peak_inds = peak_inds[m]
        peaks.append(freq0)
    peaks = np.array(peaks)

    if optimize_freq:
        def chi2(nu):
            arg = 2*np.pi*nu[None, :]*x[:, None]
            D = np.concatenate([np.cos(arg), np.sin(arg),
                        np.ones((len(x), 1))],
                        axis=1)

            # Solve for the amplitudes and phases of the oscillations
            DTD = np.matmul(D.T, D)
            DTy = np.matmul(D.T, y[:, None])
            w = np.linalg.solve(DTD, DTy)
            model = np.squeeze(np.matmul(D, w))

            chi2_val = np.sum(np.square(y - model))
            return chi2_val

        res = optimize.minimize(chi2, [peaks], method="L-BFGS-B")
        return res.x
    else:
        return peaks


def __estimate_frequencies_pm(x, y, max_peaks=9, oversample=4.0, fmin=None, fmax=None):
    """Deprecated function, uses pymc3 to estimate frequencies 
    but is much slower than the scipy equivalent.
    """
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = ny

    freq = np.arange(fmin, fmax, df / oversample)
    power = LombScargle(x, y).power(freq)

    # Find peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for j in range(max_peaks):
        i = peak_inds[0]
        freq0 = freq[i]
        alias = 2.0*ny - freq0

        m = np.abs(freq[peak_inds] - alias) > 25*df
        m &= np.abs(freq[peak_inds] - freq0) > 25*df

        peak_inds = peak_inds[m]
        peaks.append(freq0)
    peaks = np.array(peaks)

    with pm.Model():
        nu = pm.Bound(pm.Normal, lower=fmin, upper=fmax)("nu", mu=peaks, sd=10, testval=peaks, shape=peaks.shape)
        arg = 2*np.pi*nu[None, :]*x[:, None]
        D = tt.concatenate((tt.cos(arg), tt.sin(arg), tt.ones((len(x), 1))), axis=1)
        DTD = tt.dot(D.T, D)
        DTy = tt.dot(D.T, y)
        w = tt.slinalg.solve(DTD, DTy)
        mod = tt.dot(D, w)
        pm.Normal("obs", mu=mod, observed=y)
        map_soln = xo.optimize()
    
    return map_soln["nu"]