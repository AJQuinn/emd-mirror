#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Implementations of the SIFT algorithm for Empirical Mode Decomposition.

Routines:

sift
ensemble_sift
complete_ensemble_sift
sift_second_layer
mask_sift_adaptive
mask_sift_specified
get_next_imf
get_next_imf_mask

"""


import yaml
import logging
import warnings
import numpy as np
import collections
from scipy import signal
from scipy import interpolate as interp

from . import spectra
from .logger import sift_logger

# Housekeeping for logging
logger = logging.getLogger(__name__)


##################################################################
# Basic SIFT

# Utilities

def get_next_imf(X, sd_thresh=.1, env_step_size=1, envelope_opts={}, extrema_opts={}):
    """
    Compute the next IMF from a data set. This is a helper function used within
    the more general sifting functions.

    Parameters
    ----------
    X : ndarray [nsamples x 1]
        1D input array containing the time-series data to be decomposed
    sd_thresh : scalar
        The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    env_step_size : float
        Scaling of envelope prior to removal at each iteration of sift. The
        average of the upper and lower envelope is muliplied by this value
        before being subtracted from the data. Values should be between
        0 > x >= 1 (Default value = 1)

    Returns
    -------
    proto_imf : ndarray
        1D vector containing the next IMF extracted from X
    continue_flag : bool
        Boolean indicating whether the sift can be continued beyond this IMF

    Other Parameters
    ----------------
    envelope_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.sift
    emd.sift.interp_envelope

    """

    proto_imf = X.copy()

    continue_imf = True
    continue_flag = True
    niters = 0
    while continue_imf:
        niters += 1

        upper = interp_envelope(proto_imf, mode='upper',
                                **envelope_opts, extrema_opts=extrema_opts)
        lower = interp_envelope(proto_imf, mode='lower',
                                **envelope_opts, extrema_opts=extrema_opts)

        # If upper or lower are None we should stop sifting altogether
        if upper is None or lower is None:
            continue_flag = False
            continue_imf = False
            logger.debug('Finishing sift: IMF has no extrema')
            continue

        # Find local mean
        avg = np.mean([upper, lower], axis=0)[:, None]

        # Remove local mean estimate from proto imf
        x1 = proto_imf - avg

        # Stop sifting if we pass threshold
        sd = np.sum((proto_imf - x1)**2) / np.sum(proto_imf**2)
        if sd < sd_thresh:
            proto_imf = x1
            continue_imf = False
            logger.debug('Completed in {0} iters with sd {1}'.format(niters, sd))
            continue

        proto_imf = proto_imf - (env_step_size*avg)

    if proto_imf.ndim == 1:
        proto_imf = proto_imf[:, None]

    return proto_imf, continue_flag


# SIFT implementation

@sift_logger('sift')
def sift(X, sift_thresh=1e-8, max_imfs=None,
         imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    original sift algorithm [1]_.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.get_next_imf

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """

    if not imf_opts:
        imf_opts = {'env_step_size': 1,
                    'sd_thresh': .1}

    if X.ndim == 1:
        # add dummy dimension
        X = X[:, None]

    continue_sift = True
    layer = 0

    proto_imf = X.copy()

    while continue_sift:

        next_imf, continue_sift = get_next_imf(proto_imf,
                                               envelope_opts=envelope_opts,
                                               extrema_opts=extrema_opts,
                                               **imf_opts)

        if layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:, None]
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            logger.info('Finishing sift: reached max number of imfs ({0})'.format(layer))
            continue_sift = False

        if np.abs(next_imf).sum() < sift_thresh:
            logger.info('Finishing sift: reached threshold {0}'.format(np.abs(next_imf).sum()))
            continue_sift = False

    return imf


##################################################################
# Ensemble SIFT variants

# Utilities

def _sift_with_noise(X, noise_scaling=None, noise=None, noise_mode='single',
                     sift_thresh=1e-8, max_imfs=None, job_ind=1,
                     imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Helper function for applying white noise to a signal prior to computing the
    sift.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    noise_scaling : scalar
         Standard deviation of noise to add to each ensemble (Default value =
         None)
    noise : ndarray
         array of noise values the same size as X to add prior to sift (Default value = None)
    noise_mode : {'single','flip'}
         Flag indicating whether to compute each ensemble with noise once or
         twice with the noise and sign-flipped noise (Default value = 'single')
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)
    job_ind : 1
        Optional job index value for display in logger (Default value = 1)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.ensemble_sift
    emd.sift.complete_ensemble_sift
    emd.sift.get_next_imf


    """
    if job_ind is not None:
        from multiprocessing import current_process
        p = current_process()
        logger.info('Starting SIFT Ensemble: {0} on process {1}'.format(job_ind, p._identity[0]))

    if noise is None:
        noise = np.random.randn(*X.shape)

    if noise_scaling is not None:
        noise = noise * noise_scaling

    ensX = X.copy() + noise
    imf = sift(ensX, sift_thresh=sift_thresh, max_imfs=max_imfs,
               imf_opts=imf_opts, envelope_opts=envelope_opts, extrema_opts=extrema_opts)

    if noise_mode == 'single':
        return imf
    elif noise_mode == 'flip':
        ensX = X.copy() - noise
        imf += sift(ensX, sift_thresh=sift_thresh, max_imfs=max_imfs,
                    imf_opts=imf_opts, envelope_opts=envelope_opts, extrema_opts=extrema_opts)
        return imf / 2


# Implementation

@sift_logger('ensemble_sift')
def ensemble_sift(X, nensembles=4, ensemble_noise=.2, noise_mode='single',
                  nprocesses=1, sift_thresh=1e-8, max_imfs=None,
                  imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    ensemble empirical model decomposition algorithm [1]_. This approach sifts
    an ensemble of signals with white-noise added and treats the mean IMFs as
    the result.

    The resulting IMFs from the ensemble sift resembles a dyadic filter [2]_.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    nensembles : int
        Integer number of different ensembles to compute the sift across.
    ensemble_noise : scalar
         Standard deviation of noise to add to each ensemble (Default value = .2)
    noise_mode : {'single','flip'}
         Flag indicating whether to compute each ensemble with noise once or
         twice with the noise and sign-flipped noise (Default value = 'single')
    nprocesses : integer
         Integer number of parallel processes to compute. Each process computes
         a single realisation of the total ensemble (Default value = 1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf : ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_next_imf.
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.get_next_imf

    References
    ----------
    .. [1] Wu, Z., & Huang, N. E. (2009). Ensemble Empirical Mode Decomposition:
       A Noise-Assisted Data Analysis Method. Advances in Adaptive Data Analysis,
       1(1), 1–41. https://doi.org/10.1142/s1793536909000047
    .. [2] Wu, Z., & Huang, N. E. (2004). A study of the characteristics of
       white noise using the empirical mode decomposition method. Proceedings of
       the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 460(2046), 1597–1611.
       https://doi.org/10.1098/rspa.2003.1221


    """

    if noise_mode not in ['single', 'flip']:
        raise ValueError(
            'noise_mode: {0} not recognised, please use \'single\' or \'flip\''.format(noise_mode))

    # Noise is defined with respect to variance in the data
    noise_scaling = X.std() * ensemble_noise

    import multiprocessing as mp
    p = mp.Pool(processes=nprocesses)

    noise = None
    args = [(X, noise_scaling, noise, noise_mode, sift_thresh, max_imfs, ii, imf_opts, envelope_opts, extrema_opts)
            for ii in range(nensembles)]

    res = p.starmap(_sift_with_noise, args)

    p.close()

    if max_imfs is None:
        max_imfs = res[0].shape[1]

    imfs = np.zeros((X.shape[0], max_imfs))
    for ii in range(max_imfs):
        imfs[:, ii] = np.array([r[:, ii] for r in res]).mean(axis=0)

    return imfs


@sift_logger('complete_ensemble_sift')
def complete_ensemble_sift(X, nensembles=4, ensemble_noise=.2,
                           noise_mode='single', nprocesses=1,
                           sift_thresh=1e-8, max_imfs=None,
                           imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    complete ensemble empirical model decomposition algorithm [1]_. This approach sifts
    an ensemble of signals with white-noise added taking a single IMF across
    all ensembles at before moving to the next IMF.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    nensembles : int
        Integer number of different ensembles to compute the sift across.
    ensemble_noise : scalar
         Standard deviation of noise to add to each ensemble (Default value = .2)
    noise_mode : {'single','flip'}
         Flag indicating whether to compute each ensemble with noise once or
         twice with the noise and sign-flipped noise (Default value = 'single')
    nprocesses : integer
         Integer number of parallel processes to compute. Each process computes
         a single realisation of the total ensemble (Default value = 1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.
    noise: array_like
        The Intrisic Mode Functions from the decomposition of X.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_next_imf.
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.get_next_imf

    References
    ----------
    .. [1] Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P.
       (2011). A complete ensemble empirical mode decomposition with adaptive
       noise. In 2011 IEEE International Conference on Acoustics, Speech and
       Signal Processing (ICASSP). IEEE.
       https://doi.org/10.1109/icassp.2011.5947265

    """

    import multiprocessing as mp
    p = mp.Pool(processes=nprocesses)

    if X.ndim == 1:
        # add dummy dimension
        X = X[:, None]

    # Noise is defined with respect to variance in the data
    noise_scaling = X.std() * ensemble_noise

    continue_sift = True
    layer = 0

    # Compute the noise processes - large matrix here...
    noise = np.random.random_sample((X.shape[0], nensembles)) * noise_scaling

    # Do a normal ensemble sift to obtain the first IMF
    args = [(X, noise_scaling, noise[:, ii, None], noise_mode, sift_thresh,
             1, ii, imf_opts, envelope_opts, extrema_opts)
            for ii in range(nensembles)]
    res = p.starmap(_sift_with_noise, args)
    imf = np.array([r for r in res]).mean(axis=0)

    args = [(noise[:, ii, None], sift_thresh, 1, imf_opts) for ii in range(nensembles)]
    res = p.starmap(sift, args)
    noise = noise - np.array([r[:, 0] for r in res]).T

    while continue_sift:

        proto_imf = X - imf.sum(axis=1)[:, None]

        args = [(proto_imf, None, noise[:, ii, None], noise_mode, sift_thresh,
                1, ii, imf_opts, envelope_opts, extrema_opts)
                for ii in range(nensembles)]
        res = p.starmap(_sift_with_noise, args)
        next_imf = np.array([r for r in res]).mean(axis=0)

        imf = np.concatenate((imf, next_imf), axis=1)

        args = [(noise[:, ii, None], sift_thresh, 1, imf_opts)
                for ii in range(nensembles)]
        res = p.starmap(sift, args)
        noise = noise - np.array([r[:, 0] for r in res]).T

        pks, locs = find_extrema(imf[:, -1])
        if len(pks) < 2:
            continue_sift = False

        if max_imfs is not None and layer == max_imfs:
            continue_sift = False

        if np.abs(next_imf).mean() < sift_thresh:
            continue_sift = False

        layer += 1

    p.close()

    return imf, noise


##################################################################
# Mask SIFT implementations

# Utilities

def get_next_imf_mask(X, z, amp, mask_type='all',
                      imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute the next IMF from a data set using the mask sift appraoch. This is
    a helper function used within the more general sifting functions.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    z : scalar
        Mask frequency as a proportion of the sampling rate, values between 0->z->.5
    amp : scalar
        Mask amplitude
    mask_type : {'all','sine','cosine'}
         Flag indicating whether to apply sine, cosine or all masks (Default value = 'all')

    Returns
    -------
    proto_imf : ndarray
        1D vector containing the next IMF extracted from X

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.mask_sift
    emd.sift.get_next_imf

    """
    if mask_type not in ['all', 'sine', 'cosine']:
        raise ValueError("Invalid mask type")

    z = z * 2 * np.pi

    if mask_type == 'all' or mask_type == 'cosine':
        mask = amp * np.cos(z * np.arange(X.shape[0]))[:, None]
        next_imf_up_c, continue_sift = get_next_imf(X + mask,
                                                    envelope_opts=envelope_opts,
                                                    extrema_opts=extrema_opts,
                                                    **imf_opts)
        next_imf_up_c -= mask
        next_imf_down_c, continue_sift = get_next_imf(X - mask,
                                                      envelope_opts=envelope_opts,
                                                      extrema_opts=extrema_opts,
                                                      **imf_opts)
        next_imf_down_c += mask

    if mask_type == 'all' or mask_type == 'sine':
        mask = amp * np.sin(z * np.arange(X.shape[0]))[:, None]
        next_imf_up_s, continue_sift = get_next_imf(X + mask,
                                                    envelope_opts=envelope_opts,
                                                    extrema_opts=extrema_opts,
                                                    **imf_opts)
        next_imf_up_s -= mask
        next_imf_down_s, continue_sift = get_next_imf(X - mask,
                                                      envelope_opts=envelope_opts,
                                                      extrema_opts=extrema_opts,
                                                      **imf_opts)
        next_imf_down_s += mask

    if mask_type == 'all':
        return (next_imf_up_c + next_imf_down_c + next_imf_up_s + next_imf_down_s) / 4.
    elif mask_type == 'sine':
        return (next_imf_up_s + next_imf_down_s) / 2.
    elif mask_type == 'cosine':
        return (next_imf_up_c + next_imf_down_c) / 2.


def get_mask_freqs(X, first_mask_mode='zc', imf_opts={}):

    if (first_mask_mode == 'zc') or (first_mask_mode == 'if'):
        logger.info('Computing first mask frequency with method {0}'.format(first_mask_mode))
        logger.info('Getting first IMF with no mask')
        # First IMF is computed normally
        imf, _ = get_next_imf(X, **imf_opts)

    # Compute first mask frequency from first IMF
    if first_mask_mode == 'zc':
        num_zero_crossings = zero_crossing_count(imf)[0, 0]
        z = num_zero_crossings / imf.shape[0] / 4
        logger.info('Found first mask frequency of {0}'.format(z))
    elif first_mask_mode == 'if':
        _, IF, IA = spectra.frequency_stats(imf[:, 0, None], 1, 'nht',
                                            smooth_phase=3)
        z = np.average(IF, weights=IA)
        logger.info('Found first mask frequency of {0}'.format(z))
    elif first_mask_mode < .5:
        if first_mask_mode <= 0 or first_mask_mode >= .5:
            raise ValueError("The frequency of the first mask must be 0<x<.5")
        logger.info('Using specified first mask frequency of {0}'.format(first_mask_mode))
        z = first_mask_mode

    return z


# Implementation

@sift_logger('mask_sift')
def mask_sift(X, mask_amp=1, mask_amp_mode='ratio_imf',
              mask_freqs='zc', mask_step_factor=2,
              mask_type='all', ret_mask_freq=False,
              max_imfs=9, sift_thresh=1e-8,
              imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute Intrinsic Mode Functions from a dataset using a set of masking
    signals to reduce mixing of components between modes [1]_.

    This function can either compute the mask frequencies based on the fastest
    dynamics in the data (the properties of the first IMF from a standard sift)
    or apply a pre-specified set of masks.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    mask_amp : scalar or array_like
         Amplitude of mask signals as specified by mask_amp_mode. If scalar the
         same value is applied to all IMFs, if an array is passed each value is
         applied to each IMF in turn (Default value = 1)
    mask_amp_mode : {'abs','ratio_imf','ratio_sig'}
         Method for computing mask amplitude. Either in absolute units ('abs'), or as a
         ratio of the amplitude of the input signal ('ratio_signal') or previous imf
         ('ratio_imf') (Default value = 'ratio_imf')
   mask_freqs : {'zc','if',float,,array_like}
        Define the set of mask frequencies to use. If 'zc' or 'if' are passed,
        the frequency of the first mask is taken from either the zero-crossings
        or instantaneous frequnecy the first IMF of a standard sift on the
        data. If a float is passed this is taken as the first mask frequency.
        Subsequent masks are defined by the mask_step_factor. If an array_like
        vector is passed, the values in the vector will specify the mask
        frequencies.
    mask_step_factor : scalar
         Step in frequency between successive masks (Default value = 2)
    mask_type : {'all','sine','cosine'}
        Which type of masking signal to use. 'sine' or 'cosine' options return
        the average of a +ve and -ve flipped wave. 'all' applies four masks:
        sine and cosine with +ve and -ve sign and returns the average of all
        four.
    ret_mask_freq : bool
         Boolean flag indicating whether mask frequencies are returned (Default value = False)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)

    Returns
    -------
    imf : ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.
    mask_freqs : ndarray
        1D array of mask frequencies, if ret_mask_freq is set to True.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    Notes
    -----
    Here are some example mask_sift variants you can run:

    A mask sift in which the mask frequencies are determined with
    zero-crossings and mask amplitudes by a ratio with the amplitude of the
    previous IMF (note - this is also the default):

    >> imf = emd.sift.mask_sift(X, mask_amp_mode='ratio_imf', mask_freqs='zc')

    A mask sift in which the first mask is set at .4 of the sampling rate and
    subsequent masks found by successive division of this mask_freq by 3:

    >> imf = emd.sift.mask_sift(X, mask_freqs=.4, mask_step_factor=3)

    A mask sift using user specified frequencies and amplitudes:

    >> mask_freqs = np.array([.4,.2,.1,.05,.025,0])
    >> mask_amps = np.array([2,2,1,1,.5,.5])
    >> imf = emd.sift.mask_sift(X, mask_freqs=mask_freqs, mask_amp=mask_amps, mask_amp_mode='abs')

    See Also
    --------
    emd.sift.get_next_imf
    emd.sift.get_next_imf_mask

    References
    ----------
    .. [1] Ryan Deering, & James F. Kaiser. (2005). The Use of a Masking Signal
       to Improve Empirical Mode Decomposition. In Proceedings. (ICASSP ’05). IEEE
       International Conference on Acoustics, Speech, and Signal Processing, 2005.
       IEEE. https://doi.org/10.1109/icassp.2005.1416051

    """

    # initialise
    if X.ndim == 1:
        # add dummy dimension
        X = X[:, None]

    # if first mask is if or zc - compute first imf as normal and get freq
    if isinstance(mask_freqs, (list, tuple, np.ndarray)):
        logger.info('Using user specified masks')
    elif mask_freqs in ['zc', 'if'] or isinstance(mask_freqs, float):
        z = get_mask_freqs(X, mask_freqs, imf_opts=imf_opts)
        mask_freqs = np.array([z/mask_step_factor**ii for ii in range(max_imfs)])

    # Initialise mask amplitudes
    if mask_amp_mode == 'ratio_imf':
        sd = X.std()  # Take ratio of input signal for first IMF
    elif mask_amp_mode == 'ratio_sig':
        sd = X.std()
    elif mask_amp_mode == 'abs':
        sd = 1

    continue_sift = True
    imf_layer = 0
    proto_imf = X.copy()
    imf = []
    while continue_sift:

        # Update mask amplitudes if needed
        if mask_amp_mode == 'ratio_imf' and imf_layer > 0:
            sd = imf[:, -1].std()

        if isinstance(mask_amp, int) or isinstance(mask_amp, float):
            amp = mask_amp * sd
        else:
            # Should be array_like if not a single number
            amp = mask_amp[imf_layer] * sd

        logging.info('Sift IMF-{0} with mask-freq {1} and amp {2}'.format(imf_layer, mask_freqs[imf_layer], amp))

        next_imf = get_next_imf_mask(proto_imf, mask_freqs[imf_layer], amp, mask_type=mask_type,
                                     imf_opts=imf_opts, envelope_opts=envelope_opts, extrema_opts=extrema_opts)

        if imf_layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:, None]

        if max_imfs is not None and imf_layer == max_imfs-1:
            continue_sift = False

        if np.abs(next_imf).sum() < sift_thresh:
            continue_sift = False

        imf_layer += 1

    if ret_mask_freq:
        return imf, mask_freqs
    else:
        return imf


@sift_logger('mask_sift_adaptive')
def mask_sift_adaptive(X, sift_thresh=1e-8, max_imfs=None,
                       mask_amp=1, mask_amp_mode='ratio_imf',
                       mask_step_factor=2, ret_mask_freq=False,
                       first_mask_mode='if',
                       imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute Intrinsic Mode Functions from a dataset using a set of masking
    signals to reduce mixing of components between modes.

    The simplest masking signal approach uses single mask for each IMF after
    the first is computed as normal [1]_. This has since been expanded to the
    complete mask sift which uses a set of positive and negative sign sine and
    cosine signals as masks for each IMF. The mean of the four is taken as the
    IMF.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)
    mask_amp : scalar or array_like
         Amplitude of mask signals as specified by mask_amp_mode. If scalar the
         same value is applied to all IMFs, if an array is passed each value is
         applied to each IMF in turn (Default value = 1)
    mask_amp_mode : {'abs','imf_ratio','sig_ratio'}
         Method for computing mask amplitude. Either in absolute units ('abs'), or as a
         ratio of the amplitude of the input signal ('ratio_signal') or previous imf
         ('ratio_imf') (Default value = 'ratio_imf')
    mask_step_factor : scalar
         Step in frequency between successive masks (Default value = 2)
    ret_mask_freq : bool
         Boolean flag indicating whether mask frequencies are returned (Default value = False)
    mask_initial_freq : scalar
         Frequency of initial mask as a proportion of the sampling frequency (Default value = None)
    interp_method : {'mono_pchip','splrep','pchip'}
         The interpolation method used when computing upper and lower envelopes (Default value = 'mono_pchip')

    Returns
    -------
    imf : ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.
    mask_freqs : ndarray
        1D array of mask frequencies, if ret_mask_freq is set to True.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    References
    ----------
    .. [1] Ryan Deering, & James F. Kaiser. (2005). The Use of a Masking Signal
       to Improve Empirical Mode Decomposition. In Proceedings. (ICASSP ’05). IEEE
       International Conference on Acoustics, Speech, and Signal Processing, 2005.
       IEEE. https://doi.org/10.1109/icassp.2005.1416051

    """

    warnings.warn("'emd.sift.mask_sift_adaptive' is deprecated and will be \
                   removed in the next version of EMD.\nPlease switch to use \
                   'emd.sift.mask_sift' to remove this warning", DeprecationWarning)

    if X.ndim == 1:
        # add dummy dimension
        X = X[:, None]

    continue_sift = True
    layer = 0

    # Initialise mask amplitudes
    if mask_amp_mode == 'ratio_imf':
        sd = X.std()  # Take ratio of input signal for first IMF
    elif mask_amp_mode == 'ratio_sig':
        sd = X.std()
    elif mask_amp_mode == 'abs':
        sd = 1

    if isinstance(mask_amp, int) or isinstance(mask_amp, float):
        amp = mask_amp * sd
    else:
        # Should be array_like if not a single number
        amp = mask_amp[layer] * sd

    if (first_mask_mode == 'zc') or (first_mask_mode == 'if'):
        logger.info('Sift IMF-{0} with no mask'.format(layer))
        # First IMF is computed normally
        imf, _ = get_next_imf(X, **imf_opts)

    # Compute first mask frequency from first IMF
    if first_mask_mode == 'zc':
        num_zero_crossings = zero_crossing_count(imf)[0, 0]
        w = X.shape[0] / (num_zero_crossings / 2)
        z = num_zero_crossings / imf.shape[0] / 4
        zs = [z]
    elif first_mask_mode == 'if':
        _, IF, IA = spectra.frequency_stats(imf[:, 0, None], 1, 'nht',
                                            smooth_phase=3)
        w = np.average(IF, weights=IA)
        z = 2 * np.pi * w / mask_step_factor
        z = w / 2
        zs = [z]
    elif first_mask_mode < .5:
        z = first_mask_mode
        amp = amp * X.std()
        logging.info('Sift IMF-{0} with mask-freq {1} and amp {2}'.format(layer, z, amp))
        imf = get_next_imf_mask(X, z, amp, mask_type='all',
                                imf_opts=imf_opts, envelope_opts=extrema_opts, extrema_opts=extrema_opts)
        zs = [z]
        z = z / mask_step_factor
        zs.append(z)

    layer = 1
    proto_imf = X.copy() - imf
    while continue_sift:

        # Update mask amplitude if needed
        if mask_amp_mode == 'ratio_imf':
            sd = imf[:, -1].std()

        if isinstance(mask_amp, int) or isinstance(mask_amp, float):
            amp = mask_amp * sd
        else:
            # Should be array_like if not a single number
            amp = mask_amp[layer] * sd

        logging.info('Sift IMF-{0} with mask-freq {1} and amp {2}'.format(layer, z, amp))

        next_imf = get_next_imf_mask(proto_imf, z, amp, mask_type='all',
                                     imf_opts=imf_opts, envelope_opts=extrema_opts, extrema_opts=extrema_opts)

        imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:, None]

        z = z / mask_step_factor
        zs.append(z)
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift = False

    if ret_mask_freq:
        return imf, np.array(zs)
    else:
        return imf


@sift_logger('mask_sift_specified')
def mask_sift_specified(X, sd_thresh=.1, max_imfs=None,
                        mask_amp=1, mask_amp_mode='ratio_imf',
                        mask_step_factor=2, ret_mask_freq=False,
                        mask_initial_freq=None, mask_freqs=None,
                        mask_amps=None,
                        imf_opts={}, envelope_opts={}, extrema_opts={}):
    """
    Compute Intrinsic Mode Functions from a dataset using a set of masking
    signals to reduce mixing of components between modes.

    The simplest masking signal approach uses single mask for each IMF after
    the first is computed as normal [1]_. This has since been expanded to the
    complete mask sift which uses a set of positive and negative sign sine and
    cosine signals as masks for each IMF. The mean of the four is taken as the
    IMF.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)
    mask_amp : scalar or array_like
         Amplitude of mask signals as specified by mask_amp_mode. If scalar the
         same value is applied to all IMFs, if an array is passed each value is
         applied to each IMF in turn (Default value = 1)
    mask_amp_mode : {'abs','imf_ratio','sig_ratio'}
         Method for computing mask amplitude. Either in absolute units ('abs'), or as a
         ratio of the amplitude of the input signal ('ratio_signal') or previous imf
         ('ratio_imf') (Default value = 'ratio_imf')
    mask_step_factor : scalar
         Step in frequency between successive masks (Default value = 2)
    ret_mask_freq : bool
         Boolean flag indicating whether mask frequencies are returned (Default value = False)
    mask_initial_freq : scalar
         Frequency of initial mask as a proportion of the sampling frequency (Default value = None)
    mask_freqs : array_like
         1D array, list or tuple of mask frequencies as a proportion of the
         sampling frequency (Default value = None)
    interp_method : {'mono_pchip','splrep','pchip'}
         The interpolation method used when computing upper and lower envelopes (Default value = 'mono_pchip')

    Returns
    -------
    imf : ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.
    mask_freqs : ndarray
        1D array of mask frequencies, if ret_mask_freq is set to True.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    References
    ----------
    .. [1] Ryan Deering, & James F. Kaiser. (2005). The Use of a Masking Signal
       to Improve Empirical Mode Decomposition. In Proceedings. (ICASSP ’05). IEEE
       International Conference on Acoustics, Speech, and Signal Processing, 2005.
       IEEE. https://doi.org/10.1109/icassp.2005.1416051

    """

    warnings.warn("'emd.sift.mask_sift_specified' is deprecated and will be \
                   removed in the next version of EMD.\nPlease switch to use \
                   'emd.sift.mask_sift' to remove this warning", DeprecationWarning)

    if X.ndim == 1:
        # add dummy dimension
        X = X[:, None]

    continue_sift = True
    layer = 0

    # First sift
    z = mask_freqs[0]
    zs = [z]  # Store mask freqs for return later

    # Initialise mask amplitudes
    if mask_amp_mode == 'ratio_imf':
        sd = X.std()  # Take ratio of input signal for first IMF
    elif mask_amp_mode == 'ratio_sig':
        sd = X.std()
    elif mask_amp_mode == 'abs':
        sd = 1

    if isinstance(mask_amp, int) or isinstance(mask_amp, float):
        amp = mask_amp * sd
    else:
        # Should be array_like if not a single number
        amp = mask_amp[layer] * sd

    logging.info('Sift IMF-{0} with mask-freq {1} and amp {2}'.format(1, z, amp))

    imf = get_next_imf_mask(X, z, amp, mask_type='all',
                            imf_opts=imf_opts, envelope_opts=extrema_opts, extrema_opts=extrema_opts)

    layer = 1
    proto_imf = X.copy() - imf
    while continue_sift:

        z = mask_freqs[layer]
        zs.append(z)
        layer += 1

        # Update mask amplitudes if needed
        if mask_amp_mode == 'ratio_imf':
            sd = imf[:, -1].std()

        if isinstance(mask_amp, int) or isinstance(mask_amp, float):
            amp = mask_amp * sd
        else:
            # Should be array_like if not a single number
            amp = mask_amp[layer] * sd

        logging.info('Sift IMF-{0} with mask-freq {1} and amp {2}'.format(layer, z, amp))

        next_imf = get_next_imf_mask(proto_imf, z, amp, mask_type='all',
                                     imf_opts=imf_opts, envelope_opts=extrema_opts, extrema_opts=extrema_opts)

        imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:, None]

        if max_imfs is not None and layer == max_imfs:
            continue_sift = False

    if ret_mask_freq:
        return imf, np.array(zs)
    else:
        return imf


##################################################################
# Second Layer SIFT


@sift_logger('second_layer_sift')
def sift_second_layer(IA, sift_func=sift, sift_args=None):
    """
    Compute second layer IMFs from the amplitude envelopes of a set of first
    layer IMFs [1]_.

    Parameters
    ----------
    IA : ndarray
        Input array containing a set of first layer IMFs
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf2 : ndarray
        3D array [samples x first layer imfs x second layer imfs ] containing
        the second layer IMFs

    References
    ----------
    .. [1] Huang, N. E., Hu, K., Yang, A. C. C., Chang, H.-C., Jia, D., Liang,
       W.-K., … Wu, Z. (2016). On Holo-Hilbert spectral analysis: a full
       informational spectral representation for nonlinear and non-stationary
       data. Philosophical Transactions of the Royal Society A: Mathematical,
       Physical and Engineering Sciences, 374(2065), 20150206.
       https://doi.org/10.1098/rsta.2015.0206

    """
    if (sift_args is None) or ('max_imfs' not in sift_args):
        max_imfs = IA.shape[1]
    elif 'max_imfs' in sift_args:
        max_imfs = sift_args['max_imfs']

    imf2 = np.zeros((IA.shape[0], IA.shape[1], max_imfs))

    for ii in range(max_imfs):
        tmp = sift_func(IA[:, ii], **sift_args)
        imf2[:, ii, :tmp.shape[1]] = tmp

    return imf2


##################################################################
# SIFT Estimation Utilities


def get_padded_extrema(X, pad_width=2, combined_upper_lower=False,
                       loc_pad_opts={}, mag_pad_opts={}, parabolic_extrema=False):
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

    if not loc_pad_opts:  # Empty dict evaluates to False
        loc_pad_opts = {'mode': 'reflect', 'reflect_type': 'odd'}
    else:
        loc_pad_opts = loc_pad_opts.copy()  # Don't work in place...
    loc_pad_mode = loc_pad_opts.pop('mode')

    if not mag_pad_opts:  # Empty dict evaluates to False
        mag_pad_opts = {'mode': 'median', 'stat_length': 1}
    else:
        mag_pad_opts = mag_pad_opts.copy()  # Don't work in place...
    mag_pad_mode = mag_pad_opts.pop('mode')

    if X.ndim == 2:
        X = X[:, 0]

    if combined_upper_lower:
        max_locs, max_pks = find_extrema(np.abs(X))
    else:
        max_locs, max_pks = find_extrema(X)

    if parabolic_extrema:
        y = np.c_[X[max_locs-1], X[max_locs], X[max_locs+1]].T
        max_locs, max_pks = compute_parabolic_extrema(y, max_locs)

    # Return nothing we don't have enough extrema
    if max_locs.size <= 1:
        return None, None

    # Determine how much padding to use
    if max_locs.size < pad_width:
        pad_width = max_locs.size

    # Pad peak locations
    ret_max_locs = np.pad(max_locs, pad_width, loc_pad_mode, **loc_pad_opts)

    # Pad peak magnitudes
    ret_max_pks = np.pad(max_pks, pad_width, mag_pad_mode, **mag_pad_opts)

    while max(ret_max_locs) < len(X) or min(ret_max_locs) >= 0:
        ret_max_locs = np.pad(ret_max_locs, pad_width, loc_pad_mode, **loc_pad_opts)
        ret_max_pks = np.pad(ret_max_pks, pad_width, mag_pad_mode, **mag_pad_opts)

    return ret_max_locs, ret_max_pks


def compute_parabolic_extrema(y, locs):
    """
    Compute a parabolic approximation of the extrema of in triplets of points
    based on section 3.2.1 from Rato 2008 [1]_.

    Parameters
    ----------
    y : array_like
        A [3 x nextrema] array containing the points immediately around the
        extrema in a time-series.
    locs : array_like
        A [nextrema] length vector containing x-axis positions of the extrema

    Returns
    -------
    numpy array
        The estimated y-axis values of the interpolated extrema
    numpy array
        The estimated x-axis values of the interpolated extrema

    References
    ----------
    .. [1] Rato, R. T., Ortigueira, M. D., & Batista, A. G. (2008). On the HHT,
    its problems, and some solutions. Mechanical Systems and Signal Processing,
    22(6), 1374–1394. https://doi.org/10.1016/j.ymssp.2007.11.028

    """

    # Parabola equation parameters for computing y from parameters a, b and c
    # w = np.array([[1, 1, 1], [4, 2, 1], [9, 3, 1]])
    # ... and its inverse for computing a, b and c from y
    w_inv = np.array([[.5, -1, .5], [-5/2, 4, -3/2], [3, -3, 1]])
    abc = w_inv.dot(y)

    # Find co-ordinates of extrema from parameters abc
    tp = - abc[1, :] / (2*abc[0, :])
    t = tp - 2 + locs
    y_hat = tp*abc[1, :]/2 + abc[2, :]

    return t, y_hat


def interp_envelope(X, mode='upper', interp_method='splrep', extrema_opts={},
                    ret_extrema=False):
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

    if not extrema_opts:  # Empty dict evaluates to False
        extrema_opts = {'pad_width': 2,
                        'loc_pad_opts': {},
                        'mag_pad_opts': {}}
    else:
        extrema_opts = extrema_opts.copy()  # Don't work in place...

    if interp_method not in ['splrep', 'mono_pchip', 'pchip']:
        raise ValueError("Invalid interp_method value")

    if mode == 'upper':
        locs, pks = get_padded_extrema(X, combined_upper_lower=False, **extrema_opts)
    elif mode == 'lower':
        locs, pks = get_padded_extrema(-X, combined_upper_lower=False, **extrema_opts)
    elif mode == 'combined':
        locs, pks = get_padded_extrema(X, combined_upper_lower=True, **extrema_opts)
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
        env = -env
        pks = -pks

    if ret_extrema:
        return env, (locs, pks)
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


##################################################################
# SIFT Config Utilities


class SiftConfig(collections.abc.MutableMapping):
    """
    A dictionary like object specifying keyword arguments configuring a sift.

    """

    def __init__(self, name='sift', *args, **kwargs):
        self.store = dict()
        self.name = name
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        key = self.__keytransform__(key)
        if isinstance(key, list):
            if len(key) == 2:
                return self.store[key[0]][key[1]]
            elif len(key) == 3:
                return self.store[key[0]][key[1]][key[2]]
        else:
            return self.store[key]

    def __setitem__(self, key, value):
        key = self.__keytransform__(key)
        if isinstance(key, list):
            if len(key) == 2:
                self.store[key[0]][key[1]] = value
            elif len(key) == 3:
                self.store[key[0]][key[1]][key[2]] = value
        else:
            self.store[key] = value

    def __delitem__(self, key):
        key = self.__keytransform__(key)
        if isinstance(key, list):
            if len(key) == 2:
                del self.store[key[0]][key[1]]
            elif len(key) == 3:
                del self.store[key[0]][key[1]][key[2]]
        else:
            del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __str__(self):
        out = []
        lower_level = ['imf_opts', 'envelope_opts', 'extrema_opts']
        for stage in self.store.keys():
            if stage not in lower_level:
                out.append('{0} : {1}'.format(stage, self.store[stage]))
            else:
                out.append(stage + ':')
                for key in self.store[stage].keys():
                    out.append('    {0} : {1}'.format(key, self.store[stage][key]))

        return '%s %s\n%s' % (self.name, self.__class__, '\n'.join(out))

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        key = key.split('/')
        if len(key) == 1:
            return key[0]
        else:
            if len(key) > 3:
                raise ValueError("Requested key is nested too deep. Should be a \
                                 maximum of three levels separated by '/'")
            return key

    def _get_yamlsafe_dict(self):
        conf = self.store.copy()
        return _array_to_list(conf)

    def to_yaml_text(self):
        return yaml.dump(self._get_yamlsafe_dict(), sort_keys=False)

    def to_yaml_file(self, fname):
        with open(fname, 'w') as f:
            yaml.dump(self._get_yamlsafe_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, fname):
        ret = cls()
        with open(fname, 'r') as f:
            ret.store = yaml.load(f, Loader=yaml.FullLoader)

        return ret

    @classmethod
    def from_yaml_stream(cls, stream):
        ret = cls()
        ret.store = yaml.load(stream, Loader=yaml.FullLoader)
        return ret

    def to_opts(self, stage='sift'):

        if stage == 'sift':
            out = self.store['sift']
            out['imf_opts'] = self.store['imf']
            out['imf_opts']['envelope_opts'] = self.store['envelope']
            out['imf_opts']['envelope_opts']['extrema_opts'] = self.store['extrema']
            out['imf_opts']['envelope_opts']['extrema_opts']['mag_pad_opts'] = self.store['mag_pad']
            out['imf_opts']['envelope_opts']['extrema_opts']['loc_pad_opts'] = self.store['loc_pad']
        elif stage == 'imf':
            out = self.store['imf']
            out['envelope_opts'] = self.store['envelope']
            out['envelope_opts']['extrema_opts'] = self.store['extrema']
            out['envelope_opts']['extrema_opts']['mag_pad_opts'] = self.store['mag_pad']
            out['envelope_opts']['extrema_opts']['loc_pad_opts'] = self.store['loc_pad']
        elif stage == 'envelope':
            out = self.store['envelope']
            out['extrema_opts'] = self.store['extrema']
            out['extrema_opts']['mag_pad_opts'] = self.store['mag_pad']
            out['extrema_opts']['loc_pad_opts'] = self.store['loc_pad']
        elif stage == 'extrema':
            out = self.store['extrema']
            out['mag_pad_opts'] = self.store['mag_pad']
            out['loc_pad_opts'] = self.store['loc_pad']
        else:
            raise TypeError("stage ({}) not recognised, use 'sift', 'imf', 'envelope' or 'extrema'".format(stage))
        return out.copy()


def get_config(siftname='sift'):
    """
    Helper function for specifying config objects specifying parameters to be
    used in a sift. The functions used during the sift areinspected
    automatically and default values are populated into a nested dictionary
    which can be modified and used as input to one of the sift functions.

    Parameters
    ----------
    siftname : str
        Name of the sift function to find configuration from

    Returns
    -------
    SiftConfig
        A modified dictionary containing the sift specification

    Notes
    -----

    The sift config acts as a nested dictionary which can be modified to
    specify parameters for different parts of the sift. This is initialised
    using this function:

    config = emd.sift.get_config()

    The first level of the dictionary contains six sub-dicts configuring
    different parts of the algorithm:

    config['sift'] - top level sift options, mostly specific to the particular sift algorithm
    config['imf'] - options for detecting IMFs
    config['envelope'] - options for upper and lower envelope interpolation
    config['extrema'] - options for extrema detection
    config['mag_pad'] - options for y-values of padded extrema at edges
    config['loc_pad'] - options for x-values of padded extrema at edges

    Specific values can be modified in the dictionary

    config['extrema']['parabolic_extrema'] = True

    or using this shorthand

    config['imf/env_step_factor'] = 1/3

    Finally, the SiftConfig dictionary should be nested before being passed as
    keyword arguments to a sift function.

    imfs = emd.sift.sift(X, **config.nest())

    """

    # Extrema padding opts are hard-coded for the moment, these run through
    # np.pad which has a complex signature
    mag_pad_opts = {'mode': 'median', 'stat_length': 1}
    loc_pad_opts = {'mode': 'reflect', 'reflect_type': 'odd'}

    # Get defaults for extrema detection and padding
    extrema_opts = _get_function_opts(get_padded_extrema, ignore=['X', 'mag_pad_opts',
                                                                  'loc_pad_opts',
                                                                  'combined_upper_lower'])

    # Get defaults for envelope interpolation
    envelope_opts = _get_function_opts(interp_envelope, ignore=['X', 'extrema_opts', 'mode', 'ret_extrema'])

    # Get defaults for computing IMFs
    imf_opts = _get_function_opts(get_next_imf, ignore=['X', 'envelope_opts', 'extrema_opts'])

    # Get defaults for the given sift variant
    sift_types = ['sift', 'ensemble_sift', 'complete_ensemble_sift',
                  'mask_sift', 'mask_sift_adaptive', 'mask_sift_specified']
    if siftname in sift_types:
        import sys
        mod = sys.modules[__name__]
        sift_opts = _get_function_opts(getattr(mod, siftname), ignore=['X', 'imf_opts'
                                                                       'envelope_opts',
                                                                       'extrema_opts'])
    else:
        raise AttributeError('Sift siftname not recognised: please use one of {0}'.format(sift_types))

    out = SiftConfig(siftname)
    for key in sift_opts:
        out[key] = sift_opts[key]
    out['imf_opts'] = imf_opts
    out['envelope_opts'] = envelope_opts
    out['extrema_opts'] = extrema_opts
    out['extrema_opts/mag_pad_opts'] = mag_pad_opts
    out['extrema_opts/loc_pad_opts'] = loc_pad_opts

    return out


def _get_function_opts(func, ignore=None):
    """
    Helper function for inspecting a function and extracting its keyword
    arguments and their default values

    Parameters
    ----------
    func : function
        handle for the function to be inspected
    ignore : {None or list}
        optional list of keyword argument names to be ignored in function
        signature

    Returns
    -------
    dict
        Dictionary of keyword arguments with keyword keys and default value
        values.

    """

    if ignore is None:
        ignore = []
    import inspect
    out = {}
    sig = inspect.signature(func)
    for p in sig.parameters:
        if p not in out.keys() and p not in ignore:
            out[p] = sig.parameters[p].default
    return out


def _array_to_list(conf):
    for key, val in conf.items():
        if isinstance(val, np.ndarray):
            conf[key] = val.tolist()
        elif isinstance(val, dict):
            conf[key] = _array_to_list(conf[key])
    return conf
