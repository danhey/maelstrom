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
import seaborn as sns
import exoplanet as xo
from exoplanet.orbits import get_true_anomaly
from astropy.stats import LombScargle
import tqdm

__all__ = ["Maelstrom", "PB1Model", "BaseOrbitModel"]


class BaseOrbitModel(Model):
    
    orange = [0.84627451, 0.28069204, 0.00410611]

    def __init__(self, time, flux, freq=None, is_flux=True, name='', 
                model=None, **kwargs):
        """A base orbit model from which all other orbit models inherit.
        
        Parameters
        ----------
        time : Array-like
            Time values
        flux : Array-like
            Flux values
        freq : Array-like, optional
            Frequencies of the model, by default None
        is_flux : bool, optional
            [description], by default True
        name : str, optional
            [description], by default ''
        model : [type], optional
            [description], by default None
        
        Raises
        ------
        ValueError
            [description]
        ValueError
            [description]
        """
        
        super(BaseOrbitModel, self).__init__(name, model)
        
        # Input validation
        if not (len(time) == len(flux)):
            raise ValueError("Input arrays have different lengths."
                             " len(time)={}, len(flux)={}"
                             .format(len(time), len(flux)))

        if len(flux[flux==np.nan])>0:
            raise ValueError("Flux array must not have nan values")

        # Find frequencies if none are supplied
        if freq is None:
            freq = estimate_frequencies(time, flux, **kwargs)

        # Subtract and record time mid-point.
        self.time_mid = (time[0] + time[-1]) / 2.
        self._time = theano.shared(time - self.time_mid)

        # Relative flux in ppt
        self._flux = theano.shared((flux - np.mean(flux)) * 1e3)
        self.freq = np.array(freq)

    @property
    def time(self):
        return self._time.get_value()

    @property
    def flux(self):
        return self._flux.get_value()

    @flux.setter
    def flux(self, value):
        pass

    def uncertainty(self, map_soln):
        # This is broken for celerite models
        with self:
            return np.sqrt(np.diag(np.linalg.solve(pm.find_hessian(map_soln))))

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

        if start is None:
            start = self.optimize()

        sampler = xo.PyMC3Sampler(start=50, window=50, finish=300)
        with self:
            sampler.tune(tune=tune, start=start, **kwargs,
                                  step_kwargs=dict(target_accept=target_accept))
            trace = sampler.sample(draws=draws, **kwargs)
        return trace
        
    def evaluate(self, var, opt=None):
        """
        Convenience function which wraps exoplanet.utils.eval_in_model()
        """
        with self:
            return xo.utils.eval_in_model(var, opt)

    def get_period_estimate(self, **kwargs):
        """
        Estimates the period from the segmented time delay

        Returns
        ----------
        period : float
            Estimate of the period from extracted time delays.
        """
        # This should really use a weighted average periodogram
        t0s, time_delay = self.get_time_delay(**kwargs)
        ls_model = LombScargle(t0s, time_delay.T[0])
        ls_frequencies = np.linspace(1e-3, 0.5/np.median(np.diff(t0s)), 10000)
        power = ls_model.power(ls_frequencies, method="fast",
                               normalization="psd")
        period = 1/ls_frequencies[np.argmax(power)]
        return period

    def _estimate_segment(self):
        """Don't even ask.
        """
        return 510.753 * np.median(np.diff(self.time)) - 0.215054

    def plot_time_delay(self, ax=None, show_weighted=True, **kwargs):
        """Plots the time delay. **kwargs go into `get_time_delay`.
        """
        t0s, time_delay = self.get_time_delay(**kwargs)
        if ax is None:
            fig, ax = plt.subplots()

        colors = np.array(sns.color_palette('Blues', 
                    n_colors=len(self.freq)))[::-1]
        for td, color in zip(time_delay.T, colors):
            ax.plot(t0s, td, c=color)

        if show_weighted:
            ax.plot(t0s, np.average(time_delay, axis=1, 
                    weights=self.get_weights()), c=self.orange)
        ax.set_xlabel('Time [day]')
        ax.set_ylabel('Time delay [s]')
        ax.set_xlim(t0s[0], t0s[-1])
        return ax

    def plot_periodogram(self, ax=None):
        """Plots the periodogram of the light curve with 
        the model frequencies overlain.
        
        Parameters
        ----------
        ax : matplotlib axis, optional
            do you want an axis or not, by default None
        
        Returns
        -------
        ax
            yes, this is also an axis.
        """
        if ax is None:
            fig, ax = plt.subplots()
        colors = np.array(sns.color_palette('Blues', 
                    n_colors=len(self.freq)))[::-1]

        nyq = 0.5 / np.median(np.diff(self.time))
        if np.any(self.freq > nyq):
            fmax = np.max(self.freq) + 10.
        else:
            fmax = None
        freq, amp = amplitude_spectrum(self.time, self.flux, fmax=fmax)
        ax.plot(freq, amp, linewidth=0.7, c='black')
        weights = self.get_weights(norm=False)

        for f, weight, color in zip(self.freq, weights, colors):
            ax.scatter(f, weight, color=color, marker='v')

        ax.set_xlim(freq[0], freq[-1])
        ax.set_ylim(0, None)
        ax.set_xlabel(r'Frequency [day$^{-1}$]')
        ax.set_ylabel('Amplitude [ppt]')
        return ax

    def plot_time_delay_periodogram_period(self, min_period=None, max_period=None,
                                    ax=None, annotate=True, return_res=False, **kwargs):
        """ Plots the time delay periodogram
        """
        t0s, time_delay = self.get_time_delay(**kwargs)
        nyq = 0.5 / np.median(np.diff(t0s))
        if min_period is None:
            min_period = 1 / nyq
        if max_period is None:
            max_period = self.time[-1] - self.time[0]

        if ax is None:
            fig, ax = plt.subplots()
        full = np.average(time_delay, axis=1, weights=self.get_weights())
        m = np.isfinite(full)

        colors = np.array(sns.color_palette('Blues', 
                    n_colors=len(self.freq)))[::-1]

        for td, color in zip(time_delay.T, colors):
            res = xo.estimators.lomb_scargle_estimator(t0s[m], td[m], min_period=min_period, max_period=max_period)
            f, p = res["periodogram"]
            ax.plot(1 / f, p / np.max(p), c=color)

        res = xo.estimators.lomb_scargle_estimator(t0s[m], full[m], min_period=min_period, max_period=max_period)
        f, p = res["periodogram"]
        ax.plot(1 / f, p / np.max(p), c=self.orange)
        ax.set_xlabel("Period [day]")
        ax.set_ylabel("Power")
        ax.set_yticks([]);
        """
        if annotate:
            period_guess = res["peaks"][0]["period"]
            arg = 2*np.pi*t0s[:-1][m]/period_guess
            D = np.concatenate((np.sin(arg)[:, None],
                                np.cos(arg)[:, None],
                                np.ones((len(phases[m]), 1))), axis=-1)
            w = np.linalg.solve(np.dot(D.T, D), np.dot(D.T, full[m]))
            a_guess = np.sqrt(np.sum(w[:2]**2)) * 86400
            a = mass_function(period_guess*u.day, a_guess*u.s)

            ax.annotate('Period: ' + str(np.round(period_guess, 2)) + ' d \n' +
                        '$asini$: ' + str(np.round(a_guess, 2)) + ' s \n' +
                        '$f(M)$: ' + '{:.2e}'.format(a.value) + ' $M_\odot$', (0.75,0.75), xycoords='axes fraction')"""

        ax.set_xlim((1 / f)[-1], (1 / f)[0])
        ax.set_ylim(0, None)

        if return_res:
            return res
        return

    def plot_time_delay_periodogram(self, ax=None, **kwargs):
        """ Plots the time delay periodogram
        """
        t0s, time_delay = self.get_time_delay(**kwargs)
        
        if ax is None:
            fig, ax = plt.subplots()
        full = np.average(time_delay, axis=1, weights=self.get_weights())
        m = np.isfinite(full)

        colors = np.array(sns.color_palette('Blues', 
                    n_colors=len(self.freq)))[::-1]

        for td, color in zip(time_delay.T, colors):
            f, p = amplitude_spectrum(t0s[m], td[m])
            ax.plot(f, p / np.max(p), c=color)

        f, p = amplitude_spectrum(t0s[m], full[m])
        ax.plot(f, p / np.max(p), c=self.orange)
        ax.set_xlabel(r"Frequency [day$^{-1}$]")
        ax.set_ylabel("Power")
        ax.set_yticks([]);
        ax.set_xlim(f[0], f[-1])
        ax.set_ylim(0, None)
        return

    def get_phase(self, nu, t, y):
        """Some black magic to calculate the phase for a given
        segment of data for set frequencies
        
        Parameters
        ----------
        nu : array-like
            Frequencies for which the phase will be calculated
        t : array-like
            Time-stamps
        y : array-like
            Flux values corresponding to `t`
        
        Returns
        -------
        phases
            Given phases for each frequency.
        """
        arg = 2*np.pi*nu[None, :]*t[:, None]
        D = np.concatenate((np.sin(arg), np.cos(arg),
                            np.ones((len(t), 1))), axis=1)
        DT = D.T
        DTD = np.dot(DT, D)
        w = np.linalg.solve(DTD, np.dot(D.T, y))
        return np.arctan(w[:len(nu)] / w[len(nu):2*len(nu)])

    def first_look(self, segment_size=None, save_path=None, **kwargs):
        """ 
        Shows the light curve, it's amplitude spectrum, and 
        any time delay signal for the current self.freq in the model.
        This is useful if you want to check whether a star
        may be a PM binary. However, sometimes only the strongest
        peak in the star will show a TD signal, and can be drowned
        out by the others.
        
        Parameters
        ----------
        segment_size : `float`
            Segment size in which to separate the light curve, in units of
            the light curve time. For example, the default segment size of 10 
            will separate a 1000 d long light curve in 100 segments of 10 d
            each. If none is specified, Maelstrom will do its best
        save_path : `string`
            If `save_path` is not `None`, will save a copy of the first_look
            plots into the given path.
        
        Returns
        -------
        """
        if segment_size is None:
            segment_size = self._estimate_segment()

        # TODO: THIS ABSOLUTELY NEEDS TO BE FIXED TO 
        # NOT CALCULATE TD TWICE!!! DO IT DANIEL YOU NERD

        fig, axes = plt.subplots(2, 2, figsize=[12,7])
        axes = axes.flatten()

        # Light curve
        ax = axes[0]
        ax.plot(self.time, self.flux, '.k')
        ax.set_xlim(self.time[0], self.time[-1])
        
        # Plot the light curve periodogram
        ax = axes[1]
        self.plot_periodogram(ax=ax)

        ax = axes[2]
        self.plot_time_delay(ax=ax, segment_size=segment_size)

        ax = axes[3]
        self.plot_time_delay_periodogram(ax=ax, segment_size=segment_size)

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close("all")

        return axes

    def _assign_test_value(self, opt):
        """
        Updates the test value of a model
        with optimization results.
        """
        with self as model:
            for x in opt:
                model[x].tag.test_value = opt[x]

    def period_search(self, periods=None):
        """Optimizes a model over a grid of periods
        
        Parameters
        ----------
        periods : Array-like, optional
            Grid of periods over which to optimize, by default None
        """
        from .periodogram import Periodogram
        pg = Periodogram(self.time, self.flux, self.freq)
        return pg

    def get_time_delay(self, segment_size=None):
        """ 
        Calculates the time delay signal, splitting the light curve into 
        chunks of width segment_size. A smaller segment size will increase
        the scatter of the time delay signal, especially for low frequencies.
        
        Parameters
        ----------
        segment_size : `float`
            Segment size in which to separate the light curve, in units of
            the light curve time. For example, the default segment size of 10 
            will separate a 1000 d long light curve in 100 segments of 10 d
            each.
        
        Returns
        -------
        time_midpoint : `numpy.ndarray`
            Midpoints of time for each segment in the light curve
        time_delay: `numpy.ndarray`
            Values of the extracted time delay in each segment.
        """
        if segment_size is None:
            segment_size = self._estimate_segment()

        uHz_conv = 1e-6 * 24 * 60 * 60
        time_0 = self.time[0]
        time_slice, mag_slice, phase = [], [], []
        time_delays, time_midpoints = [], []

        # Iterate over lightcurve
        for t, y in zip(self.time, self.flux):
            time_slice.append(t)
            mag_slice.append(y)
            
            # In each segment
            if t - time_0 > segment_size:
                # Append the time midpoint
                time_midpoints.append(np.mean(time_slice))
                
                # And the phases for each frequency
                phase.append(dft_phase(time_slice, mag_slice, self.freq))
                time_0 = t
                time_slice, mag_slice = [], []
                
        phase = np.array(phase).T
        # Phase wrapping patch
        for ph, f in zip(phase, self.freq):
            #mean_phase = np.mean(ph)
            #ph[np.where(ph - mean_phase > np.pi/2)] -= np.pi
            #ph[np.where(ph - mean_phase < -np.pi/2)] += np.pi
            #ph -= np.mean(ph)
            ph = np.unwrap(ph)
            ph -= np.mean(ph)

            td = ph / (2*np.pi*(f / uHz_conv * 1e-6))
            time_delays.append(td)
        time_delays = np.array(time_delays).T
        return np.array(time_midpoints), time_delays

    def get_weights(self, norm=True):
        """Calculates the amplitudes of each frequency, returning
        an array of amplitudes. This is useful for 
        calculating the weighted average time delay.
        
        Returns
        -------
        weights
            (potentially) normalised amplitudes of each frequency
        """
        weights = np.zeros(len(self.freq))
        for i, f in enumerate(self.freq):
            model = LombScargle(self.time, self.flux)
            sc = model.power(f, method="fast", normalization="psd")
            fct = np.sqrt(4./len(self.time))
            weights[i] = np.sqrt(sc) * fct
        if norm:
            weights /= np.max(weights)
        return weights


    def profile(self):
        """Profiles the current model, returning the runtime of each
        node in your Theano graph
        
        Returns
        -------
        [type]
            [description]
        """
        with self as model:
            func = xo.utils.get_theano_function_for_var(model.logpt, profile=True)
            #     func = xo.utils.get_theano_function_for_var(theano.grad(model.logpt, model.vars), profile=True)
            #args = xo.utils.get_args_for_theano_function(model.test_point)
            #print(func(*args))
            
            #%timeit func(*args)
        return func.profile.summary()

class Maelstrom(BaseOrbitModel):
    def __init__(self, time, flux, freq=None, name='', model=None, **kwargs):
        """
        The real deal. This class provides an orbit model
        for an arbitrarily sized binary system, where each
        frequency `freq` is assigned a separate lighttime (asini).
        The time delays are forward-modelled directly onto the
        light curve `time` and `flux` data.

        Parameters
        ----------
        time : array-like
            Time values of light curve
        flux : array-like
            Flux values for for every time point.
        freq : array-like
            Frequencies on which to model the time delay.
            If none are supplied, Maelstrom will attempt to find
            the most optimal frequencies.
        **kwargs : 
            Arguments for the `maelstrom.utils.estimate_frequencies function`

        """

        super(Maelstrom, self).__init__(time, flux, freq=freq, name=name, model=model, **kwargs)
            
    def setup_orbit_model(self, period=None, eccen=None):
        """
        Generates an unpinned orbit model for the system. Each
        frequency in the model will be assigned a lighttime (asini).
        Optimizing this model will reveal which frequencies, if any, are associated
        with which stars. In general, this model should not be sampled.

        Parameters
        ----------
        period : float
            Initial period of the orbital system. If none is supplied (default),
            Maelstrom will attempt to find a suitable period by directly
            examining the time delays.
        eccen : float
            Initial eccentricity of the system, defaults to 0.5 if None supplied.
        """
        
        # Get estimate of period if none supplied
        if period is None:
            period = self.get_period_estimate()

        with self:
            # Orbital parameters
            logperiod = pm.Normal("logperiod", mu=np.log(period),
                                       sd=100)
            period = pm.Deterministic("period", tt.exp(logperiod))
            t0 = pm.Normal("t0", mu=0, sd=100.0)
            varpi = xo.distributions.Angle("varpi")
            eccen = pm.Uniform("eccen", lower=1e-5, upper=1.0 - 1e-5,
                                    testval=eccen)
            logs = pm.Normal('logs', mu=np.log(np.std(self.flux)), sd=100)
            lighttime = pm.Normal('lighttime', mu=0.0, sd=100.0,
                                       shape=len(self.freq))
            
            # Better parameterization for the reference time
            sinw = tt.sin(varpi)
            cosw = tt.cos(varpi)
            opsw = 1 + sinw
            E0 = 2 * tt.arctan2(tt.sqrt(1-eccen)*cosw, 
                                tt.sqrt(1+eccen)*opsw)
            M0 = E0 - eccen * tt.sin(E0)
            tref = pm.Deterministic("tref", t0 - M0 * period /
                                    (2*np.pi))
            
            # Mean anom
            M = 2.0 * np.pi * (self.time - tref) / period

            # True anom
            f = get_true_anomaly(M, eccen + tt.zeros_like(M))
            psi = (- (1 - tt.square(eccen)) * tt.sin(f+varpi) /
                   (1 + eccen*tt.cos(f)))
            
            # tau in d
            self.tau = - (lighttime / 86400.)[None, :] * psi[:, None]
            
            # Sample in the weights parameters
            factor = 2. * np.pi * self.freq[None, :]
            arg = factor * self.time[:, None] - factor * self.tau
            mean_flux = pm.Normal("mean_flux", mu=0.0, sd=100.0)
            W_hat_cos = pm.Normal("W_hat_cos", mu=0.0, sd=100.0,
                                  shape=len(self.freq))
            W_hat_sin = pm.Normal("W_hat_sin", mu=0.0, sd=100.0,
                                  shape=len(self.freq))
            model_tensor = tt.dot(tt.cos(arg), W_hat_cos[:, None])
            model_tensor += tt.dot(tt.sin(arg), W_hat_sin[:, None])
            self.lc_model = tt.squeeze(model_tensor) + mean_flux
                
            # Condition on the observations
            pm.Normal("obs", mu=self.lc_model, sd=tt.exp(logs), observed=self.flux)

    def pin_orbit_model(self, opt=None):
        
        """ 
        Pins the orbit model to attribute the frequencies
        to the correct stars. In doing so, the lighttimes are collapsed 
        into a single value for each star.
        
        Parameters
        ----------
        opt : `dict`
            Results of the `optimize` function for the model. If none is supplied,
            Maelstrom will optimize first.
        
        Returns
        -------
        `PB2Model` or `PB1Model`, both instances of a `PyMC3.model.Model`
        """

        if opt is None:
            opt = self.optimize()
            
        lt = opt['lighttime']

        lt_ivar = np.arange(len(self.freq)).astype(np.int32)
        chi = lt * np.sqrt(lt_ivar)
        mask_lower = chi < -1.0        
        mask_upper = chi > 1.0

        if np.any(mask_lower) and np.any(mask_upper):
            m1 = lt >= 0
            m2 = ~m1
            lt = np.array([
                np.sum(lt_ivar[m1]*lt[m1]) / np.sum(lt_ivar[m1]),
                np.sum(lt_ivar[m2]*lt[m2]) / np.sum(lt_ivar[m2]),
            ])
            inds = 1 - m1.astype(np.int32)
        else:
            inds = np.zeros(len(lt), dtype=np.int32)
            lt = np.array([np.sum(lt_ivar*lt) / np.sum(lt_ivar)])
        pinned_lt = lt

        # Get frequencies for each star
        nu_arr_negative = self.freq[np.where(inds==1)]
        nu_arr_positive = self.freq[np.where(inds==0)]
        
        if len(pinned_lt)>1:
            # PB2 system:
            #return PB2Model(self.time, self.flux, nu_arr_positive, nu_arr_negative)
            raise ValueError('PB2 systems have not been implemented yet.')
        else:
            # PB1 system, all frequencies belong to one star
            new_model = PB1Model(self.time, self.flux, freq=self.freq)
            new_model.init_params(period=opt['period'])
            new_model.init_orbit()
            # Ugly hack to pass optimized params into new model
            for x in opt:
                if x == 'lighttime':
                    new_model[new_model.name + '_' + x+'_a'].tag.test_value = pinned_lt[0]
                else:
                    new_model[new_model.name + '_' + x].tag.test_value = opt[x]
            return new_model

    def optimize(self, vars=None, verbose=False, **kwargs):
        with self:
            if vars is None:
                map_soln = xo.optimize(start=model.test_point, vars=[self.mean_flux, self.W_hat_cos, self.W_hat_sin], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[self.logs, self.mean_flux, model.W_hat_cos, model.W_hat_sin], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.lighttime, model.t0], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.logs, model.mean_flux, model.W_hat_cos, model.W_hat_sin], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.logperiod, model.t0], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.eccen, model.varpi], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.logperiod, model.t0], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.lighttime], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.eccen, model.varpi], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.logs, model.mean_flux, model.W_hat_cos, model.W_hat_sin], verbose=False)
                map_soln = xo.optimize(start=map_soln, verbose=verbose)
            else:
                map_soln = xo.optimize(vars=vars, verbose=verbose, **kwargs)
        self._assign_test_value(map_soln)
        return map_soln

    def to_eddy(self):
        """
        Passes the Maelstrom information to the Eddy class, where
        basic modelling can be performed
        """
        pass


class PB1Model(BaseOrbitModel):
    
    def __init__(self, time, flux, freq=None,
                 name='PB1', model=None):
            
        """
        A PyMC3 custom model object for a binary system in which 
        only one star is pulsating. This model inherits from the
        BaseOrbitModel class.
        """
        super(PB1Model, self).__init__(time, flux, freq=freq, name=name, model=model)

    def init_orbit(self, period=None, with_eccen=True, eccen=None, asini=None, with_gp=True):
    
        self.with_eccen = with_eccen
        self.with_gp = with_gp
        
        if period is None:
            period = self.get_period_estimate()
        
        if eccen is None:
            eccen = 0.5

        if asini is None:
            asini = 100

        with self:
            
            # Orbital period
            logP = pm.Bound(pm.Normal,
                        lower=np.log(10),
                        upper=np.log(1000))("logP", mu=np.log(period), sd=1.0,
                                        testval=np.log(period))
            self.period = pm.Deterministic("period", pm.math.exp(logP))
            
            # The time of conjunction
            self.phi = xo.distributions.Angle("phi")
            self.logs_lc = pm.Normal('logs_lc', mu=np.log(np.std(self.flux)), sd=10, testval=0.)
            logasini = pm.Bound(pm.Normal,
                                lower=np.log(10),
                                upper=np.log(1000))('logasini', mu=np.log(184), sd=1,
                                                    testval=np.log(184))
            self.asini = pm.Deterministic("asini", tt.exp(logasini))
            
            # The baseline flux
            mean = pm.Normal("mean", mu=0.0, sd=10.0, testval=np.mean(self.flux))
            
            # Sampling in the weights parameter is faster than solving the matrix.
            lognu = pm.Normal("lognu", mu=np.log(self.freq), sd=0.1, shape=len(self.freq))
            nu = pm.Deterministic("nu", tt.exp(lognu))

            # Mean anom
            M = 2.0 * np.pi * self.time / self.period - self.phi

            if with_eccen:
                # Periastron sampled from uniform angle
                self.omega = xo.distributions.Angle("omega")
                # Eccentricity
                self.eccen = pm.Uniform("eccen", lower=0, upper=1-1e-3, testval=eccen)

                kepler_op = xo.theano_ops.kepler.KeplerOp()
                sinf, cosf = kepler_op(M, self.eccen + np.zeros(len(self.time)))
                
                factor = 1.0 - tt.square(self.eccen)
                factor /= 1.0 + self.eccen * cosf
                psi = factor * (sinf*tt.cos(self.omega)+cosf*tt.sin(self.omega))
            else:
                psi = -tt.sin(M)
            
            
            factor = 2. * np.pi * nu
            
            arg = ((factor)[None, :] * self.time[:, None]
                - (factor * self.asini / 86400)[None, :] * psi[:, None])

            phase = xo.distributions.Angle("phase", shape=len(self.freq))
            log_min_amp = np.log(0.1 * np.std(self.flux))  # np.log(np.median(np.abs(np.diff(mag))))
            log_max_amp = np.log(np.std(self.flux))
            log_mean_amp = 0.5*(log_min_amp + log_max_amp)
            logamp = pm.Bound(pm.Normal,
                            lower=log_min_amp,
                            upper=log_max_amp)("logamp", mu=log_mean_amp, sd=10.0, shape=len(self.freq),
                                                testval=log_mean_amp)
            lc_model = tt.sum(tt.exp(logamp)[None, :] * tt.sin(arg - phase[None, :]), axis=1) + mean
            if with_gp:
                logw0 = pm.Bound(pm.Normal,
                                lower=np.log(2*np.pi/100.0),
                                upper=np.log(2*np.pi/0.05))("logw0", mu=np.log(2*np.pi/10), sd=10,
                                                            testval=np.log(2*np.pi/10))
                logpower = pm.Normal("logpower", mu=np.log(np.var(self.flux)), sd=10)
                logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)
                kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
                self.gp = xo.gp.GP(kernel, self.time, tt.exp(2*self.logs_lc) + tt.zeros(len(self.time)), J=2)

                pm.Potential("obs", self.gp.log_likelihood(self.flux - lc_model))
            
            else:
                pm.Normal("obs", mu=lc_model, sd=tt.exp(self.logs_lc), observed=self.flux)
                
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
                all_but = [v for v in model.vars if v.name not in ["logP_interval__", 
                                                                    "logasini_interval__"]]

                map_params = xo.optimize(start=None, vars=[self.mean])
                map_params = xo.optimize(start=map_params, vars=[self.logs_lc])
                
                if self.with_gp:
                    map_params = xo.optimize(start=map_params, vars=[self.logpower, self.logw0])
                    map_params = xo.optimize(start=map_params, vars=[self.phase, self.logamp])
                    
                if self.with_eccen:
                    map_params = xo.optimize(start=map_params, vars=[self.eccen, self.omega])
                    
                map_params = xo.optimize(start=map_params, vars=[self.phi])
                map_params = xo.optimize(start=map_params, vars=[self.lognu])
                map_params = xo.optimize(start=map_params, vars=all_but)

                map_params = xo.optimize(start=map_params, vars=[self.logasini])
                map_params = xo.optimize(start=map_params, vars=all_but)

                map_params = xo.optimize(start=map_params, vars=[self.logP])
                self.map_params = xo.optimize(start=map_params, vars=all_but)
            else:
                self.map_params = xo.optimize(start=None, vars=vars)
            
        self._assign_test_value(self.map_soln)
        return self.map_params

    def add_radial_velocity(self, time, rv, err=None, lighttime='a'):
        """[summary]
        
        Parameters
        ----------
        time : [type]
            [description]
        rv : [type]
            [description]
        err : [type], optional
            [description], by default None
        lighttime : str, optional
            [description], by default 'a'
        
        Raises
        ------
        ValueError
            [description]
        """
        
        # Subtract mid time
        time -= self.time_mid
        
        # Input validation for lighttime type
        if lighttime not in ('a', 'b'):
            raise ValueError("The lighttime must be assigned to either star a or b")
            
        with self:
            # Account for uncertainties in RV data
            if err is None:
                logs_rv = pm.Normal('logs_RV_'+lighttime, mu=0., sd=10)
            else:
                logs_rv = pm.Normal('logs_RV_'+lighttime, mu=np.log(np.median(err)), sd=10)
            
            # Solve Kepler's equation for the RVs
            rv_mean_anom = (2.0 * np.pi * (time - self.tref) / self.period)
            rv_true_anom = get_true_anomaly(rv_mean_anom, self.eccen +
                                            tt.zeros_like(rv_mean_anom))

            if lighttime=='a':
                self.rv_vrad_a = ((self.lighttime_a / 86400) * (-2.0 * np.pi * (1 / self.period) * (1/tt.sqrt(1.0 - tt.square(self.eccen))) * (tt.cos(rv_true_anom + self.varpi) + self.eccen*tt.cos(self.varpi))))
                self.rv_vrad_a *= 299792.458  # c in km/s
                self.rv_vrad_a += self.gammav

                pm.Normal("obs_radial_velocity_"+lighttime, mu=self.rv_vrad_a, sd=tt.exp(logs_rv), observed=rv)
            elif lighttime=='b':
                # There's no second pulsating star in the PB1 model
                # In this case, we make a new lighttime solely used
                # by the radial velocity data.
                lighttime_RV = pm.Normal('lighttime_b', mu=100., sd=100.0)
                self.rv_vrad_b = ((lighttime_RV / 86400) * (-2.0 * np.pi * (1 / self.period) * (1/tt.sqrt(1.0 - tt.square(self.eccen))) * (tt.cos(rv_true_anom + self.varpi) + self.eccen*tt.cos(self.varpi))))
                self.rv_vrad_b *= 299792.458  # c in km/s
                self.rv_vrad_b += self.gammav

                pm.Normal("obs_radial_velocity_"+lighttime, mu=self.rv_vrad_b, sd=tt.exp(logs_rv), observed=rv)