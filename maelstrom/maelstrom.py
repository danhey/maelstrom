# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .estimator import estimate_frequencies
from .utils import unique_colors, amplitude_spectrum, dft_phase

import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm
from pymc3.model import Model
import exoplanet as xo
import corner
from exoplanet.orbits import get_true_anomaly
from astropy.stats import LombScargle

__all__ = ["Maelstrom", "PB1Model", "PB2Model", "BaseOrbitModel"]


class BaseOrbitModel(Model):
    
    def __init__(self, time, flux, freq=None, is_flux=True, name='', model=None, **kwargs):
        """
        A base model from which all Maelstrom models inherit. This class should
        not be instantiated by itself unless you know what you're doing.

        The BaseOrbitModel can be used as a model on the context stack for PyMC3;
        custom_model = BaseOrbitModel(time, flux)
        with custom_model as model:
            your model code here.
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
        time -= self.time_mid
        self.time = time

        # Relative flux
        self.flux = flux - np.mean(flux)
        self.freq = freq
        
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
        optimize : bool, optional
            If set to True, the sampler will optimize the model before
            attempting to sample. If False (default), the sampler will
            initialise at the testpoints of your priors.
        target_accept : float, optional
            The target acceptance ratio of the NUTS sampler (default 0.9).

        Returns
        -------
        trace : `pm trace object?`

        """

        if start is None:
            start = self.optimize()

        sampler = xo.PyMC3Sampler(start=50, window=50, finish=300)

        with self as model:
            burnin = sampler.tune(tune=tune, start=start, **kwargs,
                                  step_kwargs=dict(target_accept=target_accept))
            self.trace = sampler.sample(draws=draws, **kwargs)
        return self.trace
        
    def evaluate(self, var, opt=None):
        """
        Convenience function with wraps exoplanet.utils.eval_in_model()
        """
        with self:
            return xo.utils.eval_in_model(var, opt)
    

    def corner_plot(self, trace, varnames=None):
        """

        """
        import corner
        if varnames is None:
            vvars = ["period", "t0","varpi", "eccen", "logs"]
            varnames=[self.name + i + "_" for i in vvars]
        samples = pm.trace_to_dataframe(trace, varnames=varnames)
        for k in samples.columns:
            if "_" in k:
                samples[k.replace(self.name+"_", "")] = samples[k]
                del samples[k]
        corner.corner(samples)
    
    def print_model(self):
        """ Convenience wrapper function for pm.model_to_trace """
        return pm.model_to_graphviz(self.model)

    def get_period_estimate(self):
        """
        Estimates the period by a weighted average of the segmented time delay

        Returns
        ----------
        period : float
            Estimate of the period from extracted time delays.
        """
        # This should really use a weighted average periodogram
        time_midpoint, time_delay = self.get_time_delay()
        ls_model = LombScargle(time_midpoint, time_delay[0])
        ls_frequencies = np.linspace(1e-3, 0.5/np.median(np.diff(time_midpoint)), 10000)
        power = ls_model.power(ls_frequencies, method="fast",
                               normalization="psd")
        period = 1/ls_frequencies[np.argmax(power)]
        return period

    def get_time_delay(self, segment_size=10):
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
            mean_phase = np.mean(ph)
            ph[np.where(ph - mean_phase > np.pi/2)] -= np.pi
            ph[np.where(ph - mean_phase < -np.pi/2)] += np.pi
            ph -= np.mean(ph)

            td = ph / (2*np.pi*(f / uHz_conv * 1e-6))
            time_delays.append(td)
        return time_midpoints, time_delays

    def first_look(self, segment_size=10, save_path=None, **kwargs):
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
            each.
        save_path : `string`
            If `save_path` is not `None`, will save a copy of the first_look
            plots into the given path.
        
        Returns
        -------
        """
        fig, axes = plt.subplots(3,1,figsize=[10,10])
        t, y = self.time, self.flux

        # Lightcurve
        ax = axes[0]
        ax.plot(t, y, "k", linewidth=0.5)
        ax.set_xlabel('Time (BJD)')
        ax.set_ylabel("Amplitude (rel. flux)")
        ax.set_xlim([t.min(), t.max()])
        ax.invert_yaxis()

        # Time delays
        ax = axes[2]
        time_midpoints, time_delays = self.get_time_delay(segment_size, **kwargs)
        colors = unique_colors(len(time_delays))
        for delay, color in zip(time_delays, colors):
            ax.scatter(time_midpoints, delay, alpha=1, s=8,color=color)
            ax.set_xlabel('Time (BJD)')
            ax.set_ylabel(r'$\tau [s]$')
        ax.set_xlim([t.min(), t.max()])

        # Periodogram
        ax = axes[1]
        periodogram_freq, periodogram_amp = amplitude_spectrum(self.time, self.flux)
        ax.plot(periodogram_freq, periodogram_amp, "k", linewidth=0.5)
        ax.set_xlabel("Frequency ($d^{-1}$)")
        ax.set_ylabel("Amplitude (rel. flux)")
        for freq, color in zip(self.freq, colors):
                ax.scatter(freq, np.max(periodogram_amp), color=color, marker='v')
        ax.set_xlim([periodogram_freq[0], periodogram_freq[-1]])
        ax.set_ylim([0,None])

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close("all")

    def plot_lc_model(self, opt):
        pass

    def plot_tau_model(self, opt):
        pass
        #self.time_mid

    def _assign_test_value(self, opt):
        """
        Some horrific code to update the test value of a model
        with optimization results. This saves us having to
        maintain an opt dict outside of the class.
        """
        with self as model:
            for x in opt:
                model[x].tag.test_value = opt[x]


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
        Optimizing this model will reveal which frequencies are associated
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
            t0 = pm.Normal("t0", mu=0.0, sd=100.0)
            varpi = xo.distributions.Angle("varpi")
            eccen = pm.Uniform("eccen", lower=1e-5, upper=1.0 - 1e-5,
                                    testval=0.5)
            logs = pm.Normal('logs', mu=np.log(np.std(self.flux)), sd=100)
            lighttime = pm.Normal('lighttime', mu=0.0, sd=100.0,
                                       shape=len(self.freq))
            
            # This parameter is only used if radial velocities are supplied
            gammav = pm.Normal('gammav', mu=0., sd=100.)
            
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
            self.tau = (lighttime / 86400.)[None, :] * psi[:, None]
            
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
            Segment size in which to separate the light curve, in units of
            the light curve time. For example, the default segment size of 10 
            will separate a 1000 d long light curve in 100 segments of 10 d
            each.
        
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
            return PB2Model(self.time, self.flux, nu_arr_positive, nu_arr_negative)
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
        # Let's be a little more clever about the optimization:
        with self as model:
            if vars is None:
                self.optimization_path = [
                    
                ]
                map_soln = xo.optimize(start=model.test_point, vars=[model.mean_flux, model.W_hat_cos, model.W_hat_sin], verbose=False)
                map_soln = xo.optimize(start=map_soln, vars=[model.logs, model.mean_flux, model.W_hat_cos, model.W_hat_sin], verbose=False)
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
        BaseOrbitModel class. If you want to define custom
        priors on your data, use that class instead.
        """
        super(PB1Model, self).__init__(time, flux, freq=freq, name=name, model=model)

    def init_params(self, period=None, eccen=None):
        with self:
            
            if period is None:
                period = self.get_period_estimate()

            # Parameters to sample
            logperiod = pm.Normal("logperiod", mu=np.log(period), sd=100)
            period = pm.Deterministic("period", tt.exp(logperiod))
            t0 = pm.Normal("t0", mu=0.0, sd=100.0)
            varpi = xo.distributions.Angle("varpi")
            eccen = pm.Uniform("eccen", lower=1e-5, upper=1.0 - 1e-5, testval=eccen)
            logs = pm.Normal('logs', mu=np.log(np.std(self.flux)), sd=100)
            lighttime_a = pm.Normal('lighttime_a', mu=0.0, sd=100.0)
            gammav = pm.Normal('gammav', mu=0., sd=100.)
            
    def init_orbit(self):
        with self:
            # Better parameterization for the reference time
            sinw = tt.sin(self.varpi)
            cosw = tt.cos(self.varpi)
            opsw = 1 + sinw
            E0 = 2 * tt.arctan2(tt.sqrt(1-self.eccen)*cosw, tt.sqrt(1+self.eccen)*opsw)
            M0 = E0 - self.eccen * tt.sin(E0)
            tref = pm.Deterministic("tref", self.t0 - M0 * self.period / (2*np.pi))

            # Mean anom
            M = 2.0 * np.pi * (self.time - tref) / self.period
            # True anom
            f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))
            psi = - (1 - tt.square(self.eccen)) * tt.sin(f+self.varpi) / (1 + self.eccen*tt.cos(f))
            
            # tau in d
            self.tau = (self.lighttime_a / 86400) * psi[:,None]

            # Sampling in the weights parameter is faster than solving the matrix.
            factor = 2. * np.pi * self.freq[None, :]
            arg = factor * self.time[:, None] - factor * self.tau
            mean_flux = pm.Normal("mean_flux", mu=0.0, sd=100.0)
            W_hat_cos = pm.Normal("W_hat_cos", mu=0.0, sd=100.0, shape=len(self.freq))
            W_hat_sin = pm.Normal("W_hat_sin", mu=0.0, sd=100.0, shape=len(self.freq))
            model_tensor = tt.dot(tt.cos(arg), W_hat_cos[:, None])
            model_tensor += tt.dot(tt.sin(arg), W_hat_sin[:, None])
            self.lc_model = tt.squeeze(model_tensor) + mean_flux

            # Condition on the observations
            pm.Normal("obs_photometry", mu=self.lc_model, sd=tt.exp(self.logs), observed=self.flux)

    def add_radial_velocity(self, time, rv, err=None, lighttime='a'):
        
        # Input validation for lighttime type
        if lighttime not in ('a', 'b'):
            raise ValueError("You must assign the lighttime to either star a or b")
            
        with self:
            # Account for uncertainties in RV data
            #logs_rv = pm.Normal('logs_RV_'+lighttime, mu=np.log(np.std(rv)), sd=100)
            if err is None:
                logs_rv = pm.Normal('logs_RV_'+lighttime, mu=np.log(np.median(rv)), sd=10)
            else:
                logs_rv = pm.Normal('logs_RV_'+lighttime, mu=np.log(np.median(err)), sd=10)
            
            # Solve Kepler's equation for the RVs
            rv_mean_anom = (2.0 * np.pi * (time - self.tref) / self.period)
            rv_true_anom = get_true_anomaly(rv_mean_anom, self.eccen +
                                            tt.zeros_like(rv_mean_anom))

            if lighttime=='a':
                rv_vrad = ((self.lighttime_a / 86400) * (-2.0 * np.pi * (1 / self.period) * (1/tt.sqrt(1.0 - tt.square(self.eccen))) * (tt.cos(rv_true_anom + self.varpi) + self.eccen*tt.cos(self.varpi))))
            elif lighttime=='b':
                # There's no second pulsating star in the PB1 model
                # In this case, we make a new lighttime solely used
                # by the radial velocity data.
                lighttime_RV = pm.Normal('lighttime_b', mu=0.0, sd=100.0)
                rv_vrad = ((lighttime_RV / 86400) * (-2.0 * np.pi * (1 / self.period) * (1/tt.sqrt(1.0 - tt.square(self.eccen))) * (tt.cos(rv_true_anom + self.varpi) + self.eccen*tt.cos(self.varpi))))

            rv_vrad *= 299792.458  # c in km/s
            rv_vrad += self.gammav

            pm.Normal("obs_radial_velocity_"+lighttime, mu=rv_vrad, sd=tt.exp(logs_rv), observed=rv)
    
    def optimize(self, vars=None):
        """
        Wrapper function for xo.optimize, if no vars are supplied will 
        try and optimize the model completely.
        """
        with self as model:
            if vars is None:
                map_soln = xo.optimize(start=model.test_point, vars=[model.mean_flux, model.W_hat_cos, model.W_hat_sin])
                map_soln = xo.optimize(start=map_soln, vars=[model.logs, model.mean_flux, model.W_hat_cos, model.W_hat_sin])
                map_soln = xo.optimize(start=map_soln, vars=[model.lighttime_a, model.t0])
                map_soln = xo.optimize(start=map_soln, vars=[model.logs, model.mean_flux, model.W_hat_cos, model.W_hat_sin])
                map_soln = xo.optimize(start=map_soln, vars=[model.logperiod, model.t0])
                map_soln = xo.optimize(start=map_soln, vars=[model.eccen, model.varpi])
                map_soln = xo.optimize(start=map_soln, vars=[model.logperiod, model.t0])
                map_soln = xo.optimize(start=map_soln, vars=[model.lighttime_a])
                map_soln = xo.optimize(start=map_soln, vars=[model.eccen, model.varpi])
                map_soln = xo.optimize(start=map_soln, vars=[model.logs, model.mean_flux, model.W_hat_cos, model.W_hat_sin])
                map_soln = xo.optimize(start=map_soln)
            else:
                map_soln = xo.optimize(vars=vars)  
        self._assign_test_value(map_soln)  
        return map_soln

    def summary(self, trace, varnames=["period", "lighttime", "t0", "varpi", "eccen",
                                "logspr", "mean_flux", "W_hat_cos", "W_hat_sin"]
                      ):
        """ Convenience wrapper function for pm.summary """
        return pm.summary(trace, varnames=varnames)

class PB2Model(BaseOrbitModel):

    def __init__(self, time, flux, freq_a, freq_b,
                 name='PB2', model=None, eccen=None):

        """
        Parameters:

        freq_ind (array):
        """

        freq = np.concatenate(freq_a, freq_b)

        super(PB2Model, self).__init__(time, flux, freq=freq, name=name, model=model)

        self.freq_a = freq_a
        self.freq_b = freq_b

        # Parameters to sample
        logperiod = pm.Normal("logperiod", mu=np.log(period), sd=100)
        period = pm.Deterministic("period", tt.exp(logperiod))
        t0 = pm.Normal("t0", mu=0.0, sd=100.0)
        varpi = xo.distributions.Angle("varpi")
        eccen = pm.Uniform("eccen", lower=1e-5, upper=1.0 - 1e-5, testval=0.5)
        logs = pm.Normal('logs', mu=np.log(np.std(self.flux)), sd=100)
        mean_flux = pm.Normal("mean_flux", mu=0.0, sd=100.0)
        lighttime_a = pm.Normal('lighttime_a', mu=0.0, sd=100.0, testval=0.)
        lighttime_b = pm.Normal('lighttime_b', mu=0.0, sd=100.0, testval=0.)

        # Better parameterization for the reference time
        sinw = tt.sin(varpi)
        cosw = tt.cos(varpi)
        opsw = 1 + sinw
        E0 = 2 * tt.arctan2(tt.sqrt(1-eccen)*cosw, tt.sqrt(1+eccen)*opsw)
        M0 = E0 - eccen * tt.sin(E0)
        tref = pm.Deterministic("tref", t0 - M0 * period / (2*np.pi))
        M = 2.0 * np.pi * (self.time - tref) / period
        f = get_true_anomaly(M, eccen + tt.zeros_like(M))
        psi = - (1 - tt.square(eccen)) * tt.sin(f+varpi) / (1 + eccen*tt.cos(f))

        # tau in d
        self.tau_a = (lighttime_a / 86400) * psi[:,None]
        self.tau_b = (lighttime_b / 86400) * psi[:,None]

        # Just sample in the weights parameters too. This seems to be faster
        factor_a = 2. * np.pi * self.freq_a[None, :]
        factor_b = 2. * np.pi * self.freq_b[None, :]

        arg_a = factor_a * self.time[:, None] - (factor_a * self.tau_a)
        arg_b = factor_b * self.time[:, None] - (factor_b * self.tau_b)

        W_hat_cos_a = pm.Normal("W_hat_cos_a", mu=0.0, sd=100.0, shape=len(self.freq_a))
        W_hat_sin_a = pm.Normal("W_hat_sin_a", mu=0.0, sd=100.0, shape=len(self.freq_a))
        W_hat_cos_b = pm.Normal("W_hat_cos_b", mu=0.0, sd=100.0, shape=len(self.freq_b))
        W_hat_sin_b = pm.Normal("W_hat_sin_b", mu=0.0, sd=100.0, shape=len(self.freq_b))

        model_tensor_a = tt.dot(tt.cos(arg_a), W_hat_cos_a[:, None]) + tt.dot(tt.sin(arg_a), W_hat_sin_a[:, None])
        self.lc_model_a = tt.squeeze(model_tensor_a) + mean_flux

        model_tensor_b = tt.dot(tt.cos(arg_b), W_hat_cos_b[:, None]) + tt.dot(tt.sin(arg_b), W_hat_sin_b[:, None])
        self.lc_model_b = tt.squeeze(model_tensor_b) + mean_flux

        # Condition on the observations
        pm.Normal("obs_photometry_a", mu=self.lc_model_a, sd=tt.exp(logs), observed=self.flux)
        pm.Normal("obs_photometry_b", mu=self.lc_model_b, sd=tt.exp(logs), observed=self.flux)
        
    def add_radial_velocity(self, time, rv, lighttime='a'):
        """
        Adds radial velocity measurements to constrain the orbital model. 

        Parameters
        ----------
        time : array-like
            Time measurements
        rv : array-like
            Radial velocity measurements (in km/s) for each time
        lighttime : `a` or `b`
            String denoting to which star the radial velocity model should
            be assigned. `a` corresponds to star a (lighttime_a), vice-versa
            for `b`. If `b` is chosen in a PB1Model object, the radial velocity
            is modelled independently of the time delay.

        """
        # Input validation for lighttime type
        if lighttime not in ('a', 'b'):
            raise ValueError("You must assign the radial velocity to either star a or b")
            
        with self:
            logs_rv = pm.Normal('logs_RV_'+lighttime, mu=np.log(np.std(rv)), sd=100)

            # Solve Kepler's equation for the RVs
            rv_mean_anom = (2.0 * np.pi * (time - self.tref) / self.period)
            rv_true_anom = get_true_anomaly(rv_mean_anom, self.eccen +
                                            tt.zeros_like(rv_mean_anom))

            if lighttime=='a':
                rv_vrad = ((self.lighttime_a / 86400) * (-2.0 * np.pi * (1 / self.period) * (1/tt.sqrt(1.0 - tt.square(self.eccen))) * (tt.cos(rv_true_anom + self.varpi) + self.eccen*tt.cos(self.varpi))))
            elif lighttime=='b':
                rv_vrad = ((self.lighttime_b / 86400) * (-2.0 * np.pi * (1 / self.period) * (1/tt.sqrt(1.0 - tt.square(self.eccen))) * (tt.cos(rv_true_anom + self.varpi) + self.eccen*tt.cos(self.varpi))))

            rv_vrad *= 299792.458  # c in km/s
            rv_vrad += self.gammav
            pm.Normal("obs_radial_velocity_"+lighttime, mu=rv_vrad, sd=tt.exp(logs_rv), observed=rv)
     
    def optimize(self, vars=None):
        """
        Wrapper function for xo.optimize, if no vars are supplied will 
        try and optimize the model completely.
        """
        with self as model:
            if vars is None:
                map_soln = xo.optimize()
            else:
                map_soln = xo.optimize(vars=vars)    
        return map_soln