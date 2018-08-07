import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
from . import utils,spectra

def sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):
    """
    Basic sift

    """

    if X.ndim == 1:
        # add dummy dimension
        X = X[:,None]

    continue_sift = True
    layer = 0

    proto_imf = X.copy()

    while continue_sift:

        next_imf,continue_sift = get_next_imf( proto_imf )

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

# Some ensemble helper functions
def _sift_with_noise( X, noise_scaling=None, noise=None, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):

    if noise is None:
        noise = np.random.randn( *X.shape )

    if noise_scaling is not None:
        noise = noise * noise_scaling

    ensX = X.copy() + noise

    return sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)

def _sift_with_noise_flip( X, noise_scaling=None, noise=None, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):

    if noise is None:
        noise = np.random.randn( *X.shape )

    if noise_scaling is not None:
        noise = noise * noise_scaling

    ensX = X.copy() + noise
    imf = sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)

    ensX = X.copy() - noise
    imf += sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)

    return imf / 2

def ensemble_sift( X, nensembles, ensemble_noise=.2,
                        sd_thresh=.1, sift_thresh=1e-8,
                        max_imfs=None, nprocesses=1,
                        noise_mode='single' ):
    """
    Ensemble sifting, add noise n times and average the IMFs
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
    a bit more complicated

    need to extract the first IMF by EEMD as normal
    then compute the subsequent IMFs with the second IMFs of the noise added to the X-IMF1

    VERY memory intensive

    some references
    https://github.com/helske/Rlibeemd/blob/master/src/ceemdan.c
    http://bioingenieria.edu.ar/grupos/ldnlys/metorres/re_inter.htm#Codigos
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5947265

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

        pks,locs = utils.find_extrema( imf[:,-1,None] )
        if len(pks) < 2:
            continue_sift=False

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

        if np.abs( next_imf ).mean() < sift_thresh:
            continue_sift=False

        layer += 1

    p.close()

    return imf,noise

def sift_second_layer( imf, sd_thresh=.1, sift_thresh=1e8 ):

    imf2layer = np.ones( (imf.shape[0],imf.shape[1],imf.shape[1] ) ) * np.nan

    for ii in range(imf.shape[1]-1):

        envelope = utils.interp_envelope( imf[:,ii,None], mode='upper' )
        tmp = sift(envelope)
        imf2layer[:,ii,:tmp.shape[1]] = tmp

    return imf2layer

def mask_sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None, mask_amp_ratio=1, mask_step_factor=2, ret_mask_freq=False ):

    if X.ndim == 1:
        # add dummy dimension
        X = X[:,None]

    continue_sift = True
    layer = 0

    # First IMF is computed normally
    imf,_ = get_next_imf( X )

    # Compute mask frequency
    mask_method='if'
    if mask_method == 'zc':
        num_zero_crossings = utils.zero_crossing_count(imf)[0,0]
        w = num_zero_crossings / X.shape[0]
    elif mask_method == 'if':
        _,IF,IA = spectra.frequency_stats( imf[:,0,None], 1, 'quad', smooth_phase=31 )
        w = np.average(IF,weights=IA)

    z = 2 * np.pi * w / mask_step_factor
    zs = [z]

    layer = 1
    proto_imf = X.copy() - imf
    while continue_sift:

        sd = imf[:,-1].std()
        amp = mask_amp_ratio*sd

        mask = amp*np.cos( z * np.arange(X.shape[0]) )[:,None]
        next_imf_up_c,continue_sift = get_next_imf( proto_imf+mask )
        next_imf_down_c,continue_sift = get_next_imf( proto_imf-mask )

        mask = amp*np.sin( z * np.arange(X.shape[0]) )[:,None]
        next_imf_up_s,continue_sift = get_next_imf( proto_imf+mask )
        next_imf_down_s,continue_sift = get_next_imf( proto_imf-mask )

        next_imf = (next_imf_up_c+next_imf_down_c+next_imf_up_s+next_imf_down_s)/4.

        imf = np.concatenate( (imf, next_imf), axis=1)

        proto_imf = X - imf.sum(axis=1)[:,None]

        z = z / mask_step_factor
        zs.append(z)
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

        #if utils.is_trend( proto_imf ):
        #    imf = np.concatenate( (imf, proto_imf), axis=1)
        #    continue_sift=False

    if ret_mask_freq:
        return imf,zs
    else:
        return imf


def adaptive_mask_sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None, mask_amp_ratio=1, ret_mask_freq=False ):

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

        # Sine wave Masks
        mask = amp*np.cos( z * np.arange(X.shape[0]) )[:,None]
        next_imf_up_c,continue_sift = get_next_imf( proto_imf+mask )
        next_imf_down_c,continue_sift = get_next_imf( proto_imf-mask )

        # Cosine wave Masks
        mask = amp*np.sin( z * np.arange(X.shape[0]) )[:,None]
        next_imf_up_s,continue_sift = get_next_imf( proto_imf+mask )
        next_imf_down_s,continue_sift = get_next_imf( proto_imf-mask )

        next_imf = (next_imf_up_c+next_imf_down_c+next_imf_up_s+next_imf_down_s)/4.

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

def get_next_imf( X, sd_thresh=.1 ):
    """
    Should be passed X as [nsamples,1]
    """

    proto_imf = X.copy()

    continue_imf = True
    continue_sift = True
    while continue_imf:

        upper = utils.interp_envelope( proto_imf, mode='upper',
        interp_method='mono_pchip' )
        lower = utils.interp_envelope( proto_imf, mode='lower',
        interp_method='mono_pchip' )

        # If upper or lower are None we should stop sifting alltogether
        if upper is None or lower is None:
            continue_sift=False
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

    return proto_imf, continue_sift
