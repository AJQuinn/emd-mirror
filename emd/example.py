#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Routines used for generating examples.

Routines:
  abreu

"""

import numpy as np

from . import utils, sift, spectra


def abreu(nonlinearity_deg=.3, nonlinearity_phi=-np.pi / 4,
          seconds=10, freq=1, sample_rate=1000, noise=0):
    """Generate an example analysis of an Abreu2010 type wave [1]_.

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
    ndarray
        Set of IMFs
    ndarray
        Time vector
    narray
        Set of instantaneous phases
    narray
        Set of instantaneous frequencies
    narray
        Set of instantaneous amplitudes

    References
    ----------
    .. [1] Abreu, T., Silva, P. A., Sancho, F., & Temperville, A. (2010).
    Analytical approximate wave form for asymmetric waves. Coastal Engineering,
    57(7), 656–667. https://doi.org/10.1016/j.coastaleng.2010.02.005

    """
    num_samples = sample_rate * seconds

    time_vect = np.linspace(0, seconds, num_samples)

    x = utils.abreu2010(freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds)
    x = x + np.random.randn(*x.shape) * noise

    imf = sift.sift(x)

    IP, IF, IA = spectra.frequency_transform(imf, sample_rate, 'quad', smooth_phase=31)

    return imf, time_vect, IP, IF, IA
