#HELLO
from matplotlib import rcParams
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 300

import numpy as np
import corner
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
from maelstrom import Maelstrom
from astropy.stats import LombScargle
from astropy.convolution import convolve, Box1DKernel
import math
import matplotlib
from lightkurve import search_lightcurvefile
import lightkurve as lk

red = '#e41a1c'
blue = '#377eb8'
green = '#4daf4a'
purple = '#984ea3'
orange = '#ff7f00'

overleaf_path = '/Users/danielhey/Dropbox (Sydney Uni)/Apps/Overleaf/Maelstrom/figs/'
matplotlib.rcParams["font.size"] = 8.
matplotlib.rcParams['font.family'] = 'Arial'
#plt.rc('font', family='serif')
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7

# t, y = lc.time, lc.flux
time, mag = np.loadtxt('../../data/kic6780873_lc.txt', usecols=(0,1)).T
time += 2400000
time -= 2454833
time, mag = time, mag*1e3

from scipy.ndimage import gaussian_filter
from maelstrom.utils import amplitude_spectrum
y_low = gaussian_filter(mag,2)
y_high = mag - y_low

mag = y_high

freq = np.array([14.18764198, 13.43633836])

from exoplanet.orbits import get_true_anomaly
import pymc3 as pm
import theano.tensor as tt

rv = False

with pm.Model() as model:
    P = pm.Bound(pm.Normal, lower=0, upper=12)("P", mu=9.159153, sd=1,
                                     shape=1, testval=9.159153)

    # Wide log-normal prior for semi-amplitude
    logasini = pm.Bound(pm.Normal, lower=0, upper=25)("logasini", mu=np.log(17.441530), sd=5,
                                        shape=1, testval=np.log(17.441530))
    logs_lc = pm.Normal('logs_lc', mu=0.0001*np.log(np.std(mag)), sd=10, testval=0.)
    asini = pm.Deterministic('asini', tt.exp(logasini))
#     logs = pm.Normal("logs", mu=np.log(np.median(rv_err)), sd=5.0)
    ecc = xo.distributions.UnitUniform("ecc", shape=1, testval=0.27)
    omega = xo.distributions.Angle("omega", testval=2.306092)
    phi = xo.distributions.Angle('phi', testval=0.377081)
    lognu = pm.Normal("lognu", mu=np.log(freq), sd=0.1, shape=len(freq))
    nu = pm.Deterministic("nu", tt.exp(lognu))
    
    # LIGHTCURVE
    M = 2. * np.pi * time / P - phi
    f = get_true_anomaly(M, ecc + tt.zeros_like(M))
    psi = -((1 - tt.square(ecc)) * tt.sin(f+omega) / (1 + ecc*tt.cos(f)))
    tau = (asini / 86400.) * psi[:, None]
    arg = 2. * np.pi * nu * (time[:, None] - tau)
    D = tt.concatenate((tt.cos(arg), tt.sin(arg)), axis=-1)
    w = tt.slinalg.solve(tt.dot(D.T, D), tt.dot(D.T, mag))
    lc_model = tt.dot(D, w)
    full_lc = lc_model
    
#     logw0 = pm.Bound(pm.Normal,
#                      lower=np.log(2*np.pi/100.0),
#                      upper=np.log(2*np.pi/1))("logw0", mu=np.log(2*np.pi/10), sd=10,
#                                                 testval=np.log(2*np.pi/10))
#     logpower = pm.Normal("logpower", mu=np.log(np.var(mag)), sd=100)
#     logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)
#     kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
#     gp = xo.gp.GP(kernel, time, tt.exp(2*logs_lc) + tt.zeros(len(time)), J=2)
#     gp_l = gp.log_likelihood(mag - full_lc)
#     # Weight likelihood equally with RV data
#     pm.Potential("obs", gp_l#/ (len(time) / len(rv_jd))
#                 )
    
    pm.Normal('obs', mu=full_lc, sd=tt.exp(logs_lc), observed=mag)
    
    if rv:
        
        gammav = pm.Uniform('gammav', lower=-50, upper=50, testval=0.)
        logs_rv = pm.Normal('logs_rv', mu=np.log(np.std(rv_rv)), sd=10, testval=np.log(np.std(rv_rv)))

        M_RV = 2. * np.pi * rv_jd / P - phi
        # True anom
        f_RV = get_true_anomaly(M_RV, ecc + tt.zeros_like(M_RV))
        vrad = -2.0 * np.pi * ((tt.exp(logasini) / 86400) / P) * (1/tt.sqrt(1.0 - tt.square(ecc))) * (tt.cos(f_RV + omega) + ecc*tt.cos(omega))
        vrad *= 299792.458  # c in km/s
        vrad += gammav # Systemic velocity

        err = tt.sqrt(2*rv_err**2 + tt.exp(2*logs_rv))
        pm.Normal("obs_rv", mu=vrad, sd=err, observed=rv_rv)

        plt.plot(rv_jd, xo.eval_in_model(vrad))
        plt.scatter(rv_jd, rv_rv)
    
    
# t = np.linspace(x.min()-5, x.max()+5, 1000)
# with model:
#     M2 = 2. * np.pi * t / P - phi
#     f2 = get_true_anomaly(M2, ecc + tt.zeros_like(M2))
#     vrad2 = -2.0 * np.pi * ((tt.exp(logasini) / 86400) / P) * (1/tt.sqrt(1.0 - tt.square(ecc))) * (tt.cos(f2 + omega) + ecc*tt.cos(omega))
#     vrad2 *= 299792.458  # c in km/s|
#     vrad2 += gammav # Systemic velocity
#     pm.Deterministic("vrad_pred", vrad2)


with model:
    if rv:
#         map_soln = xo.optimize(start=model.test_point)
        map_soln = xo.optimize(start=model.test_point, vars=[gammav])
        map_soln = xo.optimize(start=map_soln, vars=[phi])
    
    all_but = [v for v in model.vars if v.name not in ["P_interval__"]]
    map_params = xo.optimize(start=None, vars=[logs_lc])
#     map_params = xo.optimize(start=map_params, vars=[logpower, logw0])
    map_params = xo.optimize(start=map_params, vars=[ecc, omega])
    map_params = xo.optimize(start=map_params, vars=[phi])
    map_params = xo.optimize(start=map_params, vars=[lognu])
    map_params = xo.optimize(start=map_params, 
                             vars=all_but
                            )
    
    map_params = xo.optimize(start=map_params, vars=[asini])
    map_params = xo.optimize(start=map_params,
                             vars=all_but
                            )

    map_params = xo.optimize(start=map_params, vars=[P])
    map_params = xo.optimize(start=map_params, 
                             vars=all_but
                            )
    
    
np.random.seed(42)
with model:
    trace = pm.sample(
        tune=1000, draws=1000, step=xo.get_dense_nuts_step(target_accept=0.9), start=map_params
    )
    
pm.save_trace(trace,'KIC 6780873/join_trace_X_NO_RV_NO_GP_FILTERED')