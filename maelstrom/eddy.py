# -*- coding: utf-8 -*-
from __future__ import division, print_function

from .maelstrom import BaseOrbitModel

from pymc3.model import Model

__all__ = ["Eddy"]


class Eddy(BaseOrbitModel):
    """
    A model for directly fitting the time delays of a light curve.
    This uses the old method of segmenting the light curve into equal
    sections and calculating the time delay within each section.
    Although much faster, this method suffers from undersampling
    near periastron when the eccentricity of the orbit is large.
    We do not recommend using this class, except as a "first look"
    for the orbital parameters

    Parameters
    ----------
    """
    def __init__(self, time, flux, freq=None, name='TD', model=None, **kwargs):
        super(Eddy, self).__init__(time, flux, freq=freq, name=name, model=model, **kwargs)
        
        self.time_midpoint, self.time_delay = self.get_time_delay()

    def init_params(self):
        pass

    def init_orbit(self):
        pass
