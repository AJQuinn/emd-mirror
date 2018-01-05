import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
from . import utils

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

    return imf

def ensemble_sift( X, nensembles, ensemble_noise=.2, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):
    """
    Ensemble sifting, add noise n times and average the IMFs
    """

    skips = 0
    for ii in range(nensembles):
        ensX = X.copy() + np.random.randn( *X.shape )*ensemble_noise

        if ii == 0:
            imf = sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)
        else:
            ens_imf = sift(ensX,sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=max_imfs)
            if ens_imf.shape[1] != imf.shape[1]:
                skips += 1
                continue
            # update mean
            imf = imf + (1./(ii+1))*(ens_imf-imf)

    print(('%d ensembles skipped' % skips))
    return imf

def complete_ensemble_sift( X, nensembles, ensemble_noise=.2, sd_thresh=.1, sift_thresh=1e-8 ):
    """
    a bit more complicated

    need to extract the first IMF by EEMD as normal
    then compute the subsequent IMFs with the second IMFs of the noise added to the X-IMF1

    VERY memory intensive

    not convinced by this implementation yet

    some references
    https://github.com/helske/Rlibeemd/blob/master/src/ceemdan.c
    http://bioingenieria.edu.ar/grupos/ldnlys/metorres/re_inter.htm#Codigos
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5947265

    """

    if X.ndim == 1:
        # add dummy dimension
        X = X[:,None]

    cimf = np.zeros_like( X )
    continue_sift = True

    # Compute the noise processes - large matrix here...
    noise = np.random.random_sample( (X.shape[0],nensembles) ) * ensemble_noise

    imf = np.zeros_like( X )

    # Do a normal ensemble sift to obtain the first IMF
    for ii in range(nensembles):
        # Compute IMF
        ens_imf = sift(X+noise[:,ii,None],sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=1)
        # running average of the IMF over ensembles
        imf = imf + (1./(ii+1))*(ens_imf-imf)
        # remove first IMF of noise
        tmp,_ = get_next_imf( noise[:,ii,None] )
        noise[:,ii] = noise[:,ii] - tmp[:,0]

    while continue_sift:

        proto_imf = X - imf.sum(axis=1)[:,None]

        next_imf = np.zeros_like( X )

        # Do a normal ensemble sift to obtain the first IMF
        for ii in range(nensembles):
            # Compute IMF
            ens_imf = sift(proto_imf+noise[:,ii,None],sd_thresh=sd_thresh,sift_thresh=sift_thresh,max_imfs=1)
            # running average of the IMF over ensembles
            next_imf = next_imf + (1./(ii+1))*(ens_imf-next_imf)
            # remove first IMF of noise
            tmp,_ = get_next_imf( noise[:,ii,None] )
            noise[:,ii] = noise[:,ii] - tmp[:,0]

        imf = np.concatenate( (imf, next_imf), axis=1)

        if utils.is_trend( imf[:,-1,None] ):
            continue_sift=False

    return imf

def sift_second_layer( imf, sd_thresh=.1, sift_thresh=1e8 ):

    imf2layer = np.ones( (imf.shape[0],imf.shape[1],imf.shape[1] ) ) * np.nan

    for ii in range(imf.shape[1]-1):

        envelope = utils.get_envelope( imf[:,ii,None], N=4 )
        tmp = sift(envelope)
        imf2layer[:,ii,:tmp.shape[1]] = tmp

    return imf2layer

def mask_sift( X, sd_thresh=.1, sift_thresh=1e-8, max_imfs=None ):

    from .frequency_transforms import instantaneous_stats
    if X.ndim == 1:
        # add dummy dimension
        X = X[:,None]

    continue_sift = True
    layer = 0

    # First IMF is computed normally
    imf,continue_sift = get_next_imf( X )
    IF,IA = instantaneous_stats( imf, 50., 'hilbert' )

    layer = 1
    proto_imf = X.copy()
    while continue_sift:

        mask_freq = (IA * IF**2).sum() / (IA*IF).sum()
        print(IF.mean())
        upz,downz = utils.count_zero_crossings( imf[:,-1] )
        mask_freq = ( np.pi*(upz+downz) ) / X.shape[0]
        mask_freq = .5**(1+layer)

        print(mask_freq)
        mask = np.sin( 2*np.pi*mask_freq*np.arange(X.shape[0]) )[:,None]
        mask = mask * .5*np.mean(IA)

        next_imf_up,continue_sift = get_next_imf( proto_imf+mask )
        next_imf_down,continue_sift = get_next_imf( proto_imf-mask )
        next_imf = (next_imf_up+next_imf_down)/2.
        #next_imf,continue_sift = get_next_imf( proto_imf )

        imf = np.concatenate( (imf, next_imf), axis=1)
        print(imf.shape)

        IF,IA = instantaneous_stats( next_imf, 50., 'hilbert' )

        proto_imf = X - imf.sum(axis=1)[:,None]

        #import matplotlib.pyplot as plt
        #plt.plot(next_imf)
        #plt.plot(mask)
        #plt.show()

        layer += 1

        if max_imfs is not None and layer == max_imfs:
            continue_sift=False

        if utils.is_trend( proto_imf ):
            imf = np.concatenate( (imf, proto_imf), axis=1)
            continue_sift=False

    return imf



def get_next_imf( X, sd_thresh=.1 ):
    """
    Should be passed X as [nsamples,1]
    """

    proto_imf = X.copy()

    continue_imf = True
    continue_sift = True
    while continue_imf:

        upper,lower = utils.find_envelopes( proto_imf )

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
