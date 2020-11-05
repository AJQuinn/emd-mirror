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

import logging
import numpy as np
from scipy import signal

from .sift import interp_envelope, zero_crossing_count
from .support import ensure_2d, ensure_1d_with_singleton

# Housekeeping for logging
logger = logging.getLogger(__name__)


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
    logger.info('STARTED: Amplitude-Normalise')

    X = ensure_1d_with_singleton([X], ['X'], 'amplitude_normalise')

    if X.ndim == 2:
        logger.debug('Normalising {0} samples across {1} IMFs'.format(*X.shape))
    else:
        logger.debug('Normalising {0} samples across {1} first-level and {2} second-level IMFs'.format(*X.shape))
    logger.debug('Using {0} interpolation with threshold of {1} and max_iters {2}'.format(interp_method,
                                                                                          thresh,
                                                                                          max_iters))

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

                    iter_val = np.abs(env.sum() - env.shape[0])
                    if iter_val < thresh:
                        continue_norm = False

                        logger.info('Normalise of IMF-{0}-{1} complete in {2} iters (val={3})'.format(iimf,
                                                                                                      jimf,
                                                                                                      iters,
                                                                                                      iter_val))

    if clip:
        logger.debug('Clipping signal to -1:1 range')
        # Make absolutely sure nothing daft is happening
        X = np.clip(X, -1, 1)

    if orig_dim == 2:
        X = X[:, :, 0]

    logger.info('COMPLETED: Amplitude-Normalise')
    return X


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


def find_extrema_locked_epochs(X, winsize, lock_to='peaks', percentile=None):
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
    if lock_to not in ['peaks', 'troughs', 'combined']:
        raise ValueError("Invalid lock_to value")

    from .sift import get_padded_extrema
    locs, pks = get_padded_extrema(X, pad_width=0, mode=lock_to)

    if percentile is not None:
        thresh = np.percentile(pks, percentile)
        locs = locs[pks > thresh]
        pks = pks[pks > thresh]

    winstep = int(winsize / 2)

    # Get all trials
    trls = np.r_[np.atleast_2d(locs - winstep), np.atleast_2d(locs + winstep)].T

    # Reject trials which start before 0
    inds = trls[:, 0] < 0
    trls = trls[inds == False, :]  # noqa: E712

    # Reject trials which end after X.shape[0]
    inds = trls[:, 1] > X.shape[0]
    trls = trls[inds == False, :]  # noqa: E712

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


def ar_simulate(freq, sample_rate, seconds, r=.95, noise_std=None, random_seed=None):
    """
    Create a simulated oscillation using an autoregressive filter. A simple
    filter is defined by direct pole placement and applied to white noise to
    generate a random signal with a defined oscillatory peak frequency that
    exhibits random variability frequency, amplitude and waveform.

    """

    if random_seed is not None:
        np.random.seed(random_seed)

    if freq > 0:
        freq_rads = (2 * np.pi * freq) / sample_rate
        a1 = np.array([1, -2*r*np.cos(freq_rads), (r**2)])
    else:
        a1 = np.poly(r)

    num_samples = int(sample_rate * seconds)

    x = signal.filtfilt(1, a1, np.random.randn(1, num_samples)).T

    if noise_std is not None:
        noise = np.std(x)*noise_std*np.random.randn(1, num_samples).T
        x = x + noise

    if random_seed is not None:
        np.random.seed()  # restore defaults

    return x


def is_imf(imf, avg_tol=5e-2, envelope_opts=None, extrema_opts=None):
    """
    Run checks to validate whether a signal is a 'true IMF' according to two
    criteria. Firstly, the number of extrema and number of zero-crossings must
    differ by zero or one. Secondly,the mean of the upper and lower envelopes
    must be within a tolerance of zero.

    Parameters
    ----------
    imf : 2d array
        Array of signals to check [nsamples x nimfs]
    avg_tol : float
        Tolerance of acceptance for criterion two. The sum-square of the mean
        of the upper and lower envelope must be below avg_tol of the sum-square
        of the signal being checked.
    envelope_opts : dict
        Dictionary of envelope estimation options, must be identical to options
        used when estimating IMFs.
    extrema_opts : dict
        Dictionary of extrema estimation options, must be identical to options
        used when estimating IMFs.

    Parameters
    ----------
    array [2 x nimfs]
        Boolean array indicating whether each IMF passed each test.

    Notes
    -----
    These are VERY strict criteria to apply to real data. The tests may
    indicate a fail if the sift doesn't coverge well in a short segment of the
    signal when the majority of the IMF is well behaved.

    The tests are only valid if called with identical envelope_opts and
    extrema_opts as were used in the sift estimation.

    """

    imf = ensure_2d([imf], ['imf'], 'is_imf')

    if envelope_opts is None:
        envelope_opts = {}

    checks = np.zeros((imf.shape[1], 2), dtype=bool)

    for ii in range(imf.shape[1]):

        # Extrema and zero-crossings differ by <=1
        num_zc = zero_crossing_count(imf[:, ii])
        num_ext = signal.find_peaks(imf[:, ii])[0].shape[0] + signal.find_peaks(-imf[:, ii])[0].shape[0]

        # Mean of envelopes should be zero
        upper = interp_envelope(imf[:, ii], mode='upper',
                                **envelope_opts, extrema_opts=extrema_opts)
        lower = interp_envelope(imf[:, ii], mode='lower',
                                **envelope_opts, extrema_opts=extrema_opts)

        # If upper or lower are None we should stop sifting altogether
        if upper is None or lower is None:
            logger.debug('IMF-{0} False - no peaks detected')
            continue

        # Find local mean
        avg = np.mean([upper, lower], axis=0)[:, None]
        avg_sum = np.sum(np.abs(avg))
        imf_sum = np.sum(np.abs(imf[:,  ii]))
        diff = avg_sum / imf_sum

        checks[ii, 0] = np.abs(np.diff((num_zc, num_ext))) <= 1
        checks[ii, 1] = diff < avg_tol

        msg = 'IMF-{0} {1} - {2} extrema and {3} zero-crossings. Avg of envelopes is {4:.4}/{5:.4} ({6:.4}%)'
        msg = msg.format(ii, np.alltrue(checks[ii, :]),  num_ext, num_zc, avg_sum, imf_sum, 100*diff)
        logger.debug(msg)

    return checks
