# -*- coding: utf-8 -*-
from __future__ import division, print_function

import seaborn as sns
import numpy as np
from astropy.stats import LombScargle
from astropy.convolution import convolve, Box1DKernel
from astropy import constants as const
import astropy.units as u

__all__ = ["unique_colors", "amplitude_spectrum", "dft_phase", "mass_function"]


def mass_function(period, asini):
    si = (4*np.pi**2 * (1*const.c**3)) / (1*const.G) * 1 / (period.to(u.s)**2) * (asini**3)
    return si.to(u.M_sun)

def unique_colors(n, cmap="hls"):
    """ 
    Calculates n unique colours for plotting in a given colorspace
    
    Parameters
    ----------
        n : `int`
            Number of colours required. 
        cmap : `str`
            Colormap (either "hls" or "colorblind")   
    
    Returns:
    ----------
        phase : `list`
            A list of phases for the given frequencies
    """
    colors = np.array(sns.color_palette(cmap, 
                    n_colors=n))
    return colors

def amplitude_spectrum(t, y, fmin=None, fmax=None, nyq_mult=1., oversample_factor=5.):
    """ 
    Calculates the amplitude spectrum of a given signal
    
    Parameters
    ----------
        t : `array`
            Time values 
        y : `array`
            Flux or magnitude measurements
        fmin : float (default None)
            Minimum frequency to calculate spectrum (default None)
        fmax : float
            Maximum frequency to calculate spectrum
        nyq_mult : float  
    
    Returns:
    ----------
        phase : `list`
            A list of phases for the given frequencies
    """
    tmax = t.max()
    tmin = t.min()
    df = 1.0 / (tmax - tmin)
    
    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = (0.5 / np.median(np.diff(t)))*nyq_mult

    freq = np.arange(fmin, fmax, df / oversample_factor)
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    fct = np.sqrt(4./len(t))
    amp = np.sqrt(sc) * fct
    
    return freq, amp

def dft_phase(x, y, freq):
    """ 
    Discrete fourier transform to calculate the ASTC phase
    given x, y, and an array of frequencies
    
    Parameters
    ----------
        x : `array`
            Array in which to calculate 
        x : `array`
    
    Returns:
    ----------
        phase : `list`
            A list of phases for the given frequencies
    """

    freq = np.asarray(freq)
    x = np.array(x)
    y = np.array(y)
    phase = []
    for f in freq:
        expo = 2.0 * np.pi * f * x
        ft_real = np.sum(y * np.cos(expo))
        ft_imag = np.sum(y * np.sin(expo))
        phase.append(np.arctan2(ft_imag,ft_real))
    return phase

def smooth(freq, power, method='boxkernel', filter_width=2.):
    fs = np.mean(np.diff(freq))
    box_kernel = Box1DKernel(np.ceil((filter_width/fs)))
    smooth_power = convolve(power, box_kernel)
    return smooth_power

def phase_error(x, y, freq):
    """Calculates the phase uncertainty on the given frequency
    
    Args:
        x (np.array): Time values
        y (np.array): flux values
        freq (Array-like or list): Frequency values
    
    Returns:
        np.array: Array of uncertainties for each phase calculated
    """
    error = []
    freq = np.asarray(freq)
    x = np.array(x)
    y = np.array(y)
    freqs, amps = amplitude_spectrum(x, y)
    smoothed = smooth(freqs, amps)
    sig_a = np.sqrt(2 / np.sqrt(np.pi)) * np.std(smoothed)
    for f in freq:
        model = LombScargle(x, y)
        sc = model.power(f, method="fast", normalization="psd")
        fct = np.sqrt(4./len(x))
        amp = np.sqrt(sc) * fct
        err = sig_a / amp
        error.append(err)
    return error