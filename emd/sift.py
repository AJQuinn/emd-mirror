#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Implimentations of the SIFT algorithm for Empirical Mode Decomposition.

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

import logging
import numpy as np

from . import spectra, utils
from .logger import sift_logger

# Housekeeping for logging
logger = logging.getLogger(__name__)


@sift_logger('sift')
def sift(X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None,
         interp_method='mono_pchip'):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    original sift algorithm [1]_.

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
    interp_method : {'mono_pchip','splrep','pchip'}
         The interpolation method used when computing upper and lower envelopes (Default value = 'mono_pchip')

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """

    if X.ndim == 1:
        # add dummy dimension
        X = X[:, None]

    continue_sift = True
    layer = 0

    proto_imf = X.copy()

    while continue_sift:

        next_imf, continue_sift = get_next_imf(proto_imf, sd_thresh=sd_thresh,
                                               interp_method=interp_method)

        if layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:, None]
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift = False

        if np.abs(next_imf).sum() < sift_thresh:
            continue_sift = False

    return imf


@sift_logger('ensemble_sift')
def ensemble_sift(X, nensembles, ensemble_noise=.2,
                  sd_thresh=.1, sift_thresh=1e-8,
                  max_imfs=None, nprocesses=1,
                  noise_mode='single'):
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
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)
    nprocesses : integer
         Integer number of parallel processes to compute. Each process computes
         a single realisation of the total ensemble (Default value = 1)
    noise_mode : {'single','flip'}
         Flag indicating whether to compute each ensemble with noise once or
         twice with the noise and sign-flipped noise (Default value = 'single')

    Returns
    -------
    imf : ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.

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

    if noise_mode == 'single':
        sift_func = _sift_with_noise
    else:
        sift_func = _sift_with_noise_flip

    # Noise is defined with respect to variance in the data
    noise_scaling = X.std() * ensemble_noise

    import multiprocessing as mp
    p = mp.Pool(processes=nprocesses)

    noise = None
    args = [(X, noise_scaling, noise, sd_thresh, sift_thresh, max_imfs, ii)
            for ii in range(nensembles)]

    res = p.starmap(sift_func, args)

    p.close()

    if max_imfs is None:
        max_imfs = res[0].shape[1]

    imfs = np.zeros((X.shape[0], max_imfs))
    for ii in range(max_imfs):
        imfs[:, ii] = np.array([r[:, ii] for r in res]).mean(axis=0)

    return imfs


@sift_logger('complete_ensemble_sift')
def complete_ensemble_sift(X, nensembles, ensemble_noise=.2,
                           sd_thresh=.1, sift_thresh=1e-8,
                           max_imfs=None, nprocesses=1):
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
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)
    nprocesses : integer
         Integer number of parallel processes to compute. Each process computes
         a single realisation of the total ensemble (Default value = 1)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.
    noise: array_like
        The Intrisic Mode Functions from the decomposition of X.

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
    noise = np.random.random_sample((X.shape[0], nensembles)) * ensemble_noise

    # Do a normal ensemble sift to obtain the first IMF
    args = [(X, noise_scaling, noise[:, ii, None], sd_thresh, sift_thresh, 1)
            for ii in range(nensembles)]
    res = p.starmap(_sift_with_noise, args)
    imf = np.array([r for r in res]).mean(axis=0)

    args = [(noise[:, ii, None], sd_thresh, sift_thresh, 1) for ii in range(nensembles)]
    res = p.starmap(sift, args)
    noise = noise - np.array([r[:, 0] for r in res]).T

    while continue_sift:

        proto_imf = X - imf.sum(axis=1)[:, None]

        args = [(proto_imf, None, noise[:, ii, None], sd_thresh, sift_thresh, 1)
                for ii in range(nensembles)]
        res = p.starmap(_sift_with_noise, args)
        next_imf = np.array([r for r in res]).mean(axis=0)

        imf = np.concatenate((imf, next_imf), axis=1)

        args = [(noise[:, ii, None], sd_thresh, sift_thresh, 1)
                for ii in range(nensembles)]
        res = p.starmap(sift, args)
        noise = noise - np.array([r[:, 0] for r in res]).T

        pks, locs = utils.find_extrema(imf[:, -1])
        if len(pks) < 2:
            continue_sift = False

        if max_imfs is not None and layer == max_imfs:
            continue_sift = False

        if np.abs(next_imf).mean() < sift_thresh:
            continue_sift = False

        layer += 1

    p.close()

    return imf, noise


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


@sift_logger('mask_sift_adaptive')
def mask_sift_adaptive(X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None,
                       mask_amp=1, mask_amp_mode='ratio_imf',
                       mask_step_factor=2, ret_mask_freq=False,
                       first_mask_mode='if', interp_method='mono_pchip'):
    """
    Compute Intrinsic Mode Functions from a dataset using a set of masking
    signals to reduce mixing of components between modes.

    The simplest masking signal approach uses single mask for each IMF after
    hte first is computed as normal [1]_. This has since been expanded to the
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

    References
    ----------
    .. [1] Ryan Deering, & James F. Kaiser. (2005). The Use of a Masking Signal
       to Improve Empirical Mode Decomposition. In Proceedings. (ICASSP ’05). IEEE
       International Conference on Acoustics, Speech, and Signal Processing, 2005.
       IEEE. https://doi.org/10.1109/icassp.2005.1416051

    """

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
        imf, _ = get_next_imf(X)

    # Compute first mask frequency from first IMF
    if first_mask_mode == 'zc':
        num_zero_crossings = utils.zero_crossing_count(imf)[0, 0]
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
        imf = get_next_imf_mask(X, z, amp,
                                sd_thresh=sd_thresh,
                                interp_method=interp_method,
                                mask_type='all')
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

        next_imf = get_next_imf_mask(proto_imf, z, amp,
                                     sd_thresh=sd_thresh,
                                     interp_method=interp_method,
                                     mask_type='all')

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
def mask_sift_specified(X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None,
                        mask_amp=1, mask_amp_mode='ratio_imf',
                        mask_step_factor=2, ret_mask_freq=False,
                        mask_initial_freq=None, mask_freqs=None,
                        mask_amps=None, interp_method='mono_pchip'):
    """
    Compute Intrinsic Mode Functions from a dataset using a set of masking
    signals to reduce mixing of components between modes.

    The simplest masking signal approach uses single mask for each IMF after
    hte first is computed as normal [1]_. This has since been expanded to the
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

    References
    ----------
    .. [1] Ryan Deering, & James F. Kaiser. (2005). The Use of a Masking Signal
       to Improve Empirical Mode Decomposition. In Proceedings. (ICASSP ’05). IEEE
       International Conference on Acoustics, Speech, and Signal Processing, 2005.
       IEEE. https://doi.org/10.1109/icassp.2005.1416051

    """

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

    imf = get_next_imf_mask(X, z, amp,
                            sd_thresh=sd_thresh,
                            interp_method=interp_method,
                            mask_type='all')

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

        next_imf = get_next_imf_mask(proto_imf, z, amp,
                                     sd_thresh=sd_thresh,
                                     interp_method=interp_method,
                                     mask_type='all')

        imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:, None]

        if max_imfs is not None and layer == max_imfs:
            continue_sift = False

    if ret_mask_freq:
        return imf, np.array(zs)
    else:
        return imf

# Sift Utils


def _sift_with_noise(X, noise_scaling=None, noise=None, sd_thresh=.1,
                     sift_thresh=1e-8, max_imfs=None, job_ind=None):
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
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.


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

    return sift(ensX, sd_thresh=sd_thresh, sift_thresh=sift_thresh, max_imfs=max_imfs)


def _sift_with_noise_flip(X, noise_scaling=None, noise=None,
                          sd_thresh=.1, sift_thresh=1e-8, max_imfs=None):
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
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.


    """

    if noise is None:
        noise = np.random.randn(*X.shape)

    if noise_scaling is not None:
        noise = noise * noise_scaling

    ensX = X.copy() + noise
    imf = sift(ensX, sd_thresh=sd_thresh, sift_thresh=sift_thresh, max_imfs=max_imfs)

    ensX = X.copy() - noise
    imf += sift(ensX, sd_thresh=sd_thresh, sift_thresh=sift_thresh, max_imfs=max_imfs)

    return imf / 2


def get_next_imf(X, sd_thresh=.1, interp_method='mono_pchip'):
    """
    Compute the next IMF from a data set. This is a helper function used within
    the more general sifting functions.

    Parameters
    ----------
    X : ndarray [nsamples x 1]
        1D input array containing the time-series data to be decomposed
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    interp_method : {'mono_pchip','splrep','pchip'}
         The interpolation method used when computing upper and lower envelopes (Default value = 'mono_pchip')

    Returns
    -------
    proto_imf : ndarray
        1D vector containing the next IMF extracted from X
    continue_flag : bool
        Boolean indicating whether the sift can be continued beyond this IMF

    """

    proto_imf = X.copy()

    continue_imf = True
    continue_flag = True
    niters = 0
    while continue_imf:
        niters += 1

        upper = utils.interp_envelope(proto_imf, mode='upper',
                                      interp_method=interp_method)
        lower = utils.interp_envelope(proto_imf, mode='lower',
                                      interp_method=interp_method)

        # If upper or lower are None we should stop sifting alltogether
        if upper is None or lower is None:
            continue_flag = False
            continue_imf = False
            logger.debug('Completed in {0} iters with no extrema'.format(niters))
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

        proto_imf = proto_imf - avg

    if proto_imf.ndim == 1:
        proto_imf = proto_imf[:, None]

    return proto_imf, continue_flag


def get_next_imf_mask(X, z, amp,
                      sd_thresh=.1, interp_method='mono_pchip', mask_type='all'):
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
    sd_thresh : scalar
         The threshold at which the sift of each IMF will be stopped. (Default value = .1)
    interp_method : {'mono_pchip','splrep','pchip'}
         The interpolation method used when computing upper and lower envelopes (Default value = 'mono_pchip')
    mask_type : {'all','sine','cosine'}
         Flag indicating whether to apply sine, cosine or all masks (Default value = 'all')

    Returns
    -------
    proto_imf : ndarray
        1D vector containing the next IMF extracted from X

    """
    z = z * 2 * np.pi

    if mask_type == 'all' or mask_type == 'cosine':
        mask = amp * np.cos(z * np.arange(X.shape[0]))[:, None]
        next_imf_up_c, continue_sift = get_next_imf(X + mask,
                                                    sd_thresh=sd_thresh, interp_method=interp_method)
        next_imf_up_c -= mask
        next_imf_down_c, continue_sift = get_next_imf(X - mask,
                                                      sd_thresh=sd_thresh, interp_method=interp_method)
        next_imf_down_c += mask

    if mask_type == 'all' or mask_type == 'sine':
        mask = amp * np.sin(z * np.arange(X.shape[0]))[:, None]
        next_imf_up_s, continue_sift = get_next_imf(X + mask,
                                                    sd_thresh=sd_thresh, interp_method=interp_method)
        next_imf_up_s -= mask
        next_imf_down_s, continue_sift = get_next_imf(X - mask,
                                                      sd_thresh=sd_thresh, interp_method=interp_method)
        next_imf_down_s += mask

    if mask_type == 'all':
        return (next_imf_up_c + next_imf_down_c + next_imf_up_s + next_imf_down_s) / 4.
    elif mask_type == 'sine':
        return (next_imf_up_s + next_imf_down_s) / 2.
    elif mask_type == 'cosine':
        return (next_imf_up_c + next_imf_down_c) / 2.
