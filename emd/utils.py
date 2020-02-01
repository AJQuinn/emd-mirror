#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Support functions.

Routines:

amplitude_normalise
get_padded_extrema
interp_envelope
find_extrema
zero_crossing_count
abreu2010
est_orthogonality
find_extrema_locked_epochs
apply_epochs
wrap_phase

"""

import numpy as np
from scipy import interpolate as interp
from scipy import signal


def amplitude_normalise(X, thresh=1e-10, clip=False, interp_method='pchip',
                        max_iters=3):
    """
    Normalise the amplitude envelope of an IMF to be 1. Multiple runs of
    normalisation are carried out until the desired threshold is reached.

    This uses the method described as part of the AM-FM transform [1]_

    Parameters
    ----------
    X : ndarray
        Input array of IMFs to be normalised
    thresh : scalar
         Threshold for stopping normalisation (Default value = 1e-10)
    clip : bool
         Whether to clip the output between -1 and 1 (Default value = False)
    interp_method : {'pchip','mono_pchip','splrep'}
         Method used to interpolate envelopes (Default value = 'pchip')
    max_iters : int
        Maximum number of iterations of normalisation to perform

    Returns
    -------
    ndarray
        Amplitude normalised IMFs

    References
    ----------
    .. [1] Huang, N. E., Wu, Z., Long, S. R., Arnold, K. C., Chen, X., & Blank,
       K. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096

    """

    # Don't normalise in place
    X = X.copy()

    orig_dim = X.ndim
    if X.ndim == 2:
        X = X[:, :, None]

    for iimf in range(X.shape[1]):
        for jimf in range(X.shape[2]):

            env = interp_envelope(X[:, iimf, jimf], mode='combined', interp_method=interp_method)

            if env is None:
                continue_norm = False
            else:
                continue_norm = True

            iters = 0
            while continue_norm and (iters < max_iters):
                iters += 1

                X[:, iimf, jimf] = X[:, iimf, jimf] / env
                env = interp_envelope(X[:, iimf, jimf], mode='combined',
                                      interp_method=interp_method)

                if env is None:
                    continue_norm = False
                else:
                    continue_norm = True

                    if np.abs(env.sum() - env.shape[0]) < thresh:
                        continue_norm = False

    if clip:
        # Make absolutely sure nothing daft is happening
        X = np.clip(X, -1, 1)

    if orig_dim == 2:
        X = X[:, :, 0]

    return X


def get_padded_extrema(X, combined_upper_lower=False):
    """
    Return a set of extrema from a signal including padded extrema at the edges
    of the signal.

    Parameters
    ----------
    X : ndarray
        Input signal
    combined_upper_lower : bool
         Flag to indicate whether both upper and lower extrema should be
         considered (Default value = False)

    Returns
    -------
    max_locs : ndarray
        location of extrema in samples
    max_pks : ndarray
        Magnitude of each extrema


    """

    if X.ndim == 2:
        X = X[:, 0]

    if combined_upper_lower:
        max_locs, max_pks = find_extrema(np.abs(X))
    else:
        max_locs, max_pks = find_extrema(X)

    # Return nothing we don't have enough extrema
    if max_locs.size <= 1:
        return None, None

    # Determine how much padding to use
    N = 2  # should make this analytic somehow
    if max_locs.size < N:
        N = max_locs.size

    # Pad peak locations
    ret_max_locs = np.pad(max_locs, N, 'reflect', reflect_type='odd')

    # Pad peak magnitudes
    ret_max_pks = np.pad(max_pks, N, 'median', stat_length=1)

    while max(ret_max_locs) < len(X) or min(ret_max_locs) >= 0:
        ret_max_locs = np.pad(ret_max_locs, N, 'reflect', reflect_type='odd')
        ret_max_pks = np.pad(ret_max_pks, N, 'median', stat_length=1)

    return ret_max_locs, ret_max_pks


def interp_envelope(X, mode='upper', interp_method='splrep'):
    """
    Interpolate the amplitude envelope of a signal.

    Parameters
    ----------
    X : ndarray
        Input signal
    mode : {'upper','lower','combined'}
         Flag to set which envelope should be computed (Default value = 'upper')
    interp_method : {'splrep','pchip','mono_pchip'}
         Flag to indicate which interpolation method should be used (Default value = 'splrep')

    Returns
    -------
    ndarray
        Interpolated amplitude envelope


    """

    if interp_method not in ['splrep', 'mono_pchip', 'pchip']:
        raise ValueError("Invalid interp_method value")

    if mode == 'upper':
        locs, pks = get_padded_extrema(X, combined_upper_lower=False)
    elif mode == 'lower':
        locs, pks = get_padded_extrema(-X, combined_upper_lower=False)
    elif mode == 'combined':
        locs, pks = get_padded_extrema(X, combined_upper_lower=True)
    else:
        raise ValueError('Mode not recognised. Use mode= \'upper\'|\'lower\'|\'combined\'')

    if locs is None:
        return None

    # Run interpolation on envelope
    t = np.arange(locs[0], locs[-1])
    if interp_method == 'splrep':
        f = interp.splrep(locs, pks)
        env = interp.splev(t, f)
    elif interp_method == 'mono_pchip':
        pchip = interp.PchipInterpolator(locs, pks)
        env = pchip(t)
    elif interp_method == 'pchip':
        pchip = interp.pchip(locs, pks)
        env = pchip(t)

    t_max = np.arange(locs[0], locs[-1])
    tinds = np.logical_and((t_max >= 0), (t_max < X.shape[0]))

    env = np.array(env[tinds])

    if env.shape[0] != X.shape[0]:
        raise ValueError('Envelope length does not match input data {0} {1}'.format(
            env.shape[0], X.shape[0]))

    if mode == 'lower':
        return -env
    else:
        return env


def find_extrema(X, ret_min=False):
    """
    Identify extrema within a time-course and reject extrema whose magnitude is
    below a set threshold.

    Parameters
    ----------
    X : ndarray
       Input signal
    ret_min : bool
         Flag to indicate whether maxima (False) or minima (True) should be identified(Default value = False)

    Returns
    -------
    locs : ndarray
        Location of extrema in samples
    extrema : ndarray
        Value of each extrema


    """

    if ret_min:
        ind = signal.argrelmin(X, order=1)[0]
    else:
        ind = signal.argrelmax(X, order=1)[0]

    # Only keep peaks with magnitude above machine precision
    if len(ind) / X.shape[0] > 1e-3:
        good_inds = ~(np.isclose(X[ind], X[ind - 1]) * np.isclose(X[ind], X[ind + 1]))
        ind = ind[good_inds]

    # if ind[0] == 0:
    #    ind = ind[1:]

    # if ind[-1] == X.shape[0]:
    #    ind = ind[:-2]

    return ind, X[ind]


def zero_crossing_count(X):
    """
    Count the number of zero-crossings within a time-course through
    differentiation of the sign of the signal.

    Parameters
    ----------
    X : ndarray
        Input array

    Returns
    -------
    int
        Number of zero-crossings

    """

    if X.ndim == 2:
        X = X[:, None]

    return (np.diff(np.sign(X), axis=0) != 0).sum(axis=0)


def abreu2010(f, nonlin_deg, nonlin_phi, sample_rate, seconds):
    """
    Simulate a non-linear waveform using equation 9 in [1]_.

    Parameters
    ----------
    f : scalar
        Fundamental frequency of generated signal
    nonlin_deg : scalar
        Degree of non-linearity in generated signal
    nonlin_phi : scalar
        Skew in non-linearity of generated signal
    sample_rate : scalar
        The sampling frequency of the generated signal
    seconds : scalar
        The number of seconds of data to generate

    Returns
    -------
    ndarray
        Simulated signal containing non-linear wave

    References
    ----------
    .. [1] Abreu, T., Silva, P. A., Sancho, F., & Temperville, A. (2010).
       Analytical approximate wave form for asymmetric waves. Coastal Engineering,
       57(7), 656–667. https://doi.org/10.1016/j.coastaleng.2010.02.005

    """

    time_vect = np.linspace(0, seconds, int(seconds * sample_rate))

    factor = np.sqrt(1 - nonlin_deg**2)
    num = nonlin_deg * np.sin(nonlin_phi) / 1 + np.sqrt(1 - nonlin_deg**2)
    num = num + np.sin(2 * np.pi * f * time_vect)

    denom = 1 - nonlin_deg * np.cos(2 * np.pi * f * time_vect + nonlin_phi)

    return factor * (num / denom)


def est_orthogonality(imf):
    """
    Compute the index of orthogonality as described in equation 6.5 of [1]_.

    Parameters
    ----------
    imf : ndarray
        Input array of IMFs

    Returns
    -------
    ndarray
        Matrix of orthogonality values [nimfs x nimfs]

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """

    ortho = np.ones((imf.shape[1], imf.shape[1])) * np.nan

    for ii in range(imf.shape[1]):
        for jj in range(imf.shape[1]):
            ortho[ii, jj] = np.abs(np.sum(imf[:, ii] * imf[:, jj])) \
                / (np.sqrt(np.sum(imf[:, jj] * imf[:, jj])) * np.sqrt(np.sum(imf[:, ii] * imf[:, ii])))

    return ortho


def find_extrema_locked_epochs(X, winsize, lock_to='max', percentile=None):
    """
    Helper function for defining epochs around peaks or troughs within the data

    Parameters
    ----------
    X : ndarray
        Input time-series
    winsize : integer
        Width of window to extract around each extrema
    lock_to : {'max','min'}
         Flag to select peak or trough locking (Default value = 'max')
    percentile : scalar
         Optional flag to selection only the upper percentile of extrema by
         magnitude (Default value = None)

    Returns
    -------
    ndarray
        Array of start and end indices for epochs around extrema.

    """
    if lock_to not in ['max', 'min']:
        raise ValueError("Invalid lock_to value")

    if lock_to == 'max':
        locs, pks = find_extrema(X, ret_min=False)
    else:
        locs, pks = find_extrema(X, ret_min=True)

    if percentile is not None:
        thresh = np.percentile(pks[:, 0], percentile)
        locs = locs[pks[:, 0] > thresh]
        pks = pks[pks > thresh]

    winstep = int(winsize / 2)

    trls = np.r_[np.atleast_2d(locs - winstep), np.atleast_2d(locs + winstep)].T

    # Reject trials which start before 0
    inds = trls[:, 0] < 0
    trls = trls[inds is False, :]

    # Reject trials which end after X.shape[0]
    inds = trls[:, 1] > X.shape[0]
    trls = trls[inds is False, :]

    return trls


def apply_epochs(X, trls):
    """
    Apply a set of epochs to a continuous dataset

    Parameters
    ----------
    X : ndarray
        Input dataset to be epoched
    trls : ndarray
        2D array of start and end indices for each epoch. The second dimension
        should be of len==2 and contain start and end indices in order.

    Returns
    -------
    ndarray
        Epoched time-series

    """

    Y = np.zeros((trls[0, 1] - trls[0, 0], X.shape[1], trls.shape[0]))
    for ii in np.arange(trls.shape[0]):

        Y[:, :, ii] = X[trls[ii, 0]:trls[ii, 1], :]

    return Y


def wrap_phase(IP, ncycles=1, mode='2pi'):
    """
    Wrap a phase time-course.

    Parameters
    ----------
    IP : ndarray
        Input array of unwrapped phase values
    ncycles : integer
         Number of cycles per wrap (Default value = 1)
    mode : {'2pi','-pi2pi'}
         Flag to indicate the values to wrap phase within (Default value = '2pi')

    Returns
    -------
    ndarray
        Wrapped phase time-course

    """
    if mode not in ['2pi', '-pi2pi']:
        raise ValueError("Invalid mode value")

    if mode == '2pi':
        phases = (IP) % (ncycles * 2 * np.pi)
    elif mode == '-pi2pi':
        phases = (IP + (np.pi * ncycles)) % (ncycles * 2 * np.pi) - (np.pi * ncycles)

    return phases
