#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import numpy as np

from . import utils, sift, spectra


def abreu(nonlinearity_deg=.3, nonlinearity_phi=-np.pi / 4,
          seconds=10, freq=1, sample_rate=1000, noise=0):
    """

    Parameters
    ----------
    nonlinearity_deg :
         (Default value = .3)
    nonlinearity_phi :
         (Default value = -np.pi/4)
    seconds :
         (Default value = 10)
    freq :
         (Default value = 1)
    sample_rate :
         (Default value = 1000)
    noise :
         (Default value = 0)

    Returns
    -------


    """

    num_samples = sample_rate * seconds

    time_vect = np.linspace(0, seconds, num_samples)

    x = utils.abreu2010(freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds)
    x = x + np.random.randn(*x.shape) * noise

    imf = sift.sift(x)

    IP, IF, IA = spectra.frequency_stats(imf, sample_rate, 'quad', smooth_phase=31)

    return imf, time_vect, IP, IF, IA
