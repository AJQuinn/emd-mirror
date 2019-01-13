import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
from . import utils,spectra

def sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None,
          interp_method='mono_pchip'):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    original sift algorithm [1].

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
        X = X[:,None]

    continue_sift = True
    layer = 0

    proto_imf = X.copy()

    while continue_sift:

        next_imf,continue_sift = get_next_imf( proto_imf, sd_thresh=sd_thresh,
                                               interp_method=interp_method )

        if layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate( (imf,next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:,None]
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

        if np.abs( next_imf ).sum() < sift_thresh:
            continue_sift=False

    return imf

def ensemble_sift( X, nensembles, ensemble_noise=.2,
                        sd_thresh=.1, sift_thresh=1e-8,
                        max_imfs=None, nprocesses=1,
                        noise_mode='single' ):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    ensemble empirical model decomposition algorithm [1]. This approach sifts
    an ensemble of signals with white-noise added and treats the mean IMFs as
    the result.

    The resulting IMFs from the ensemble sift resembles a dyadic filter [2].

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

    if noise_mode not in ['single','flip']:
        raise ValueError('noise_mode: {0} not recognised, please use \'single\' or \'flip\''.format(noise_mode))

    if noise_mode is 'single':
        sift_func = _sift_with_noise
    else:
        sift_func = _sift_with_noise_flip

    # Noise is defined with respect to variance in the data
    noise_scaling = X.std()*ensemble_noise

    import multiprocessing as mp
    p = mp.Pool(processes=nprocesses)

    noise = None
    args = [(X,noise_scaling,noise,sd_thresh,sift_thresh,max_imfs) for ii in range(nensembles)]

    res = p.starmap( sift_func, args )

    p.close()

    if max_imfs is None:
        max_imfs = res[0].shape[1]

    imfs = np.zeros( (X.shape[0], max_imfs) )
    for ii in range(max_imfs):
        imfs[:,ii] = np.array([ r[:,ii] for r in res]).mean(axis=0)

    return imfs

def complete_ensemble_sift( X, nensembles, ensemble_noise=.2,
                            sd_thresh=.1, sift_thresh=1e-8,
                            max_imfs=None, nprocesses=1 ):
    """
    Compute Intrinsic Mode Functions from an input data vector using the
    complete ensemble empirical model decomposition algorithm [1]. This approach sifts
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
        X = X[:,None]

    # Noise is defined with respect to variance in the data
    noise_scaling = X.std()*ensemble_noise

    continue_sift = True
    layer = 0

    # Compute the noise processes - large matrix here...
    noise = np.random.random_sample( (X.shape[0],nensembles) ) * ensemble_noise

    # Do a normal ensemble sift to obtain the first IMF
    args = [(X,noise_scaling,noise[:,ii,None],sd_thresh,sift_thresh,1) \
            for ii in range(nensembles)]
    res = p.starmap( _sift_with_noise, args )
    imf = np.array([ r for r in res]).mean(axis=0)

    args = [(noise[:,ii,None],sd_thresh,sift_thresh,1) for ii in range(nensembles)]
    res = p.starmap( sift, args )
    noise = noise - np.array([ r[:,0] for r in res]).T

    while continue_sift:

        proto_imf = X - imf.sum(axis=1)[:,None]

        args = [(proto_imf,None,noise[:,ii,None],sd_thresh,sift_thresh,1) \
                           for ii in range(nensembles)]
        res = p.starmap( _sift_with_noise, args )
        next_imf = np.array([ r for r in res]).mean(axis=0)

        imf = np.concatenate( (imf, next_imf), axis=1)

        args = [(noise[:,ii,None],sd_thresh,sift_thresh,1) \
               for ii in range(nensembles)]
        res = p.starmap( sift, args )
        noise = noise - np.array([ r[:,0] for r in res]).T

        pks,locs = utils.find_extrema( imf[:,-1] )
        if len(pks) < 2:
            continue_sift=False

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

        if np.abs( next_imf ).mean() < sift_thresh:
            continue_sift=False

        layer += 1

    p.close()

    return imf,noise

def sift_second_layer( IA, sift_func=sift, sift_args=None ):
    """
    Compute second layer IMFs from the amplitude envelopes of a set of first
    layer IMFs [1].

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

    imf2 = np.zeros( (IA.shape[0],IA.shape[1],max_imfs ) )

    for ii in range(max_imfs):
        tmp = sift_func( IA[:,ii], **sift_args )
        imf2[:,ii,:tmp.shape[1]] = tmp

    return imf2

def mask_sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None,
        mask_amp_ratio=1, mask_step_factor=2, ret_mask_freq=False,
        mask_initial_freq=None, mask_freqs=None,
        interp_method='mono_pchip'):
    """

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
    mask_amp_ratio : scalar
         Amplitude of mask signals relative to amplitude of previous IMF (Default value = 1)
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

    """

    if X.ndim == 1:
        # add dummy dimension
        X = X[:,None]

    continue_sift = True
    layer = 0


    # Compute mask frequency
    if (mask_initial_freq is None) and (mask_freqs is None):

        # First IMF is computed normally
        imf,_ = get_next_imf( X )

        # Compute first mask frequency from first IMF
        mask_method='if'
        if mask_method == 'zc':
            num_zero_crossings = utils.zero_crossing_count(imf)[0,0]
            w = num_zero_crossings / X.shape[0]
        elif mask_method == 'if':
            _,IF,IA = spectra.frequency_stats( imf[:,0,None], 1, 'quad', smooth_phase=31 )
            w = np.average(IF,weights=IA)
        z = 2 * np.pi * w / mask_step_factor

    else:
        # First sift
        if mask_initial_freq is not None:
            w = mask_initial_freq
        else:
            w = mask_freqs[0]

        z = w
        amp = mask_amp_ratio*X.std()
        imf = get_next_imf_mask( X, z, amp,
                                 sd_thresh=sd_thresh,
                                 interp_method=interp_method,
                                 mask_type='all' )

    zs = [z]

    layer = 1
    proto_imf = X.copy() - imf
    allmask = np.zeros_like( proto_imf )
    while continue_sift:

        sd = imf[:,-1].std()
        amp = mask_amp_ratio*sd

        next_imf = get_next_imf_mask( proto_imf, z, amp,
                                      sd_thresh=sd_thresh,
                                      interp_method=interp_method,
                                      mask_type='all' )

        imf = np.concatenate( (imf, next_imf), axis=1)
        #allmask = np.concatenate( (allmask, mask), axis=1)

        proto_imf = X - imf.sum(axis=1)[:,None]

        if mask_freqs is not None:
            w = mask_freqs[layer]
            z = w
        else:
            z = z / mask_step_factor
            #z = z / mask_step_factor
        zs.append(z)
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

        #if utils.is_trend( proto_imf ):
        #    imf = np.concatenate( (imf, proto_imf), axis=1)
        #    continue_sift=False

    if ret_mask_freq:
        return imf,np.array(zs)
    else:
        return imf

def adaptive_mask_sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None, mask_amp_ratio=1, ret_mask_freq=False ):
    """ Not sure really,againwenf

    Parameters
    ----------
    X :

    sd_thresh :
         (Default value = .1)
    sift_thresh :
         (Default value = 1e-8)
    max_imfs :
         (Default value = None)
    mask_amp_ratio :
         (Default value = 1)
    ret_mask_freq :
         (Default value = False)

    Returns
    -------


    """

    if X.ndim == 1:
        # add dummy dimension
        X = X[:,None]

    continue_sift = True
    layer = 0

    # First IMF is computed normally
    imf,_ = get_next_imf( X )

    # Compute mask frequency
    num_zero_crossings = utils.zero_crossing_count(imf)[0,0]
    w = num_zero_crossings / X.shape[0]
    z = np.pi * num_zero_crossings / X.shape[0]

    zs = [z]

    layer = 0
    proto_imf = X.copy()
    while continue_sift:

        sd = imf[:,-1].std()
        amp = mask_amp_ratio*sd

        next_imf = get_next_imf_mask( proto_imf, z, amp,
                                      sd_thresh=sd_thresh,
                                      interp_method=interp_method,
                                      mask_type='all' )

        if layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate( (imf,next_imf), axis=1)

        # Compute frequency for next mask
        num_zero_crossings = utils.zero_crossing_count(imf[:,-1])
        w = num_zero_crossings / X.shape[0]
        z = np.pi * num_zero_crossings / X.shape[0]
        zs.append(z)

        proto_imf = X - imf.sum(axis=1)[:,None]

        zs.append(z)
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

    if ret_mask_freq:
        return imf,zs
    else:
        return imf

## Sift Utils
def _sift_with_noise( X, noise_scaling=None, noise=None, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):
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
        noise = np.random.randn( *X.shape )

    if noise_scaling is not None:
        noise = noise * noise_scaling

    ensX = X.copy() + noise

    return sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)

def _sift_with_noise_flip( X, noise_scaling=None, noise=None, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):
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
        noise = np.random.randn( *X.shape )

    if noise_scaling is not None:
        noise = noise * noise_scaling

    ensX = X.copy() + noise
    imf = sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)

    ensX = X.copy() - noise
    imf += sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)

    return imf / 2

def get_next_imf( X, sd_thresh=.1, interp_method='mono_pchip' ):
    """Should be passed X as [nsamples,1]

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
    while continue_imf:

        upper = utils.interp_envelope( proto_imf, mode='upper',
                                       interp_method=interp_method )
        lower = utils.interp_envelope( proto_imf, mode='lower',
                                       interp_method=interp_method )

        # If upper or lower are None we should stop sifting alltogether
        if upper is None or lower is None:
            continue_flag=False
            continue_imf=False
            continue

        # Find local mean
        avg = np.mean([upper,lower],axis=0)[:,None]

        # Remove local mean estimate from proto imf
        x1 = proto_imf - avg

        # Stop sifting if we pass threshold
        sd = sum((proto_imf-x1)**2)/sum(proto_imf**2);
        if sd < sd_thresh:
            proto_imf = x1
            continue_imf=False
            continue

        proto_imf = proto_imf - avg

    if proto_imf.ndim == 1:
        proto_imf = proto_imf[:,None]

    return proto_imf, continue_flag

def get_next_imf_mask( X, z, amp,
                       sd_thresh=.1, interp_method='mono_pchip',mask_type='all' ):
    """

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

    if mask_type is 'all' or mask_type is 'sine':
        mask = amp*np.cos( z * np.arange(X.shape[0]) )[:,None]
        next_imf_up_c,continue_sift = get_next_imf( X+mask )
        next_imf_up_c -= mask
        next_imf_down_c,continue_sift = get_next_imf( X-mask )
        next_imf_down_c += mask

    if mask_type is 'all' or mask_type is 'cosine':
        mask = amp*np.sin( z * np.arange(X.shape[0]) )[:,None]
        next_imf_up_s,continue_sift = get_next_imf( X+mask )
        next_imf_up_s -= mask
        next_imf_down_s,continue_sift = get_next_imf( X-mask )
        next_imf_down_s += mask

    if mask_type is 'all':
        return (next_imf_up_c+next_imf_down_c+next_imf_up_s+next_imf_down_s)/4.
    elif mask_type is 'sine':
        return (next_imf_up_s+next_imf_down_s)/2.
    elif mask_type is 'cosine':
        return (next_imf_up_c+next_imf_down_c)/2.


