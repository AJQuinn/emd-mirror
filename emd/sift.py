import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
import utils

def sift_second_layer( imf, sd_thresh=.1, sift_thresh=1e8 ):

    imf2layer = np.ones( (imf.shape[0],imf.shape[1],imf.shape[1] ) ) * np.nan

    for ii in range(imf.shape[1]-1):

        envelope = utils.get_envelope( imf[:,ii,None], N=4 )
        tmp = sift(envelope)
        imf2layer[:,ii,:tmp.shape[1]] = tmp

    return imf2layer

def ensemble_sift( X, nensembles, ensemble_noise=.2, sd_thresh=.1, sift_thresh=1e-8 ):

    skips = 0
    for ii in range(nensembles):
        ensX = X.copy() + np.random.randn( *X.shape )*ensemble_noise

        if ii == 0:
            imf = sift(ensX)
        else:
            ens_imf = sift(ensX)
            if ens_imf.shape[1] != imf.shape[1]:
                skips += 1
                continue
            # update mean
            imf = imf + (1./(ii+1))*(ens_imf-imf)

    print('%d ensembles skipped' % skips)
    return imf

def ensemble_sift2( X, nensembles, ensemble_noise=.2, sd_thresh=.1, sift_thresh=1e-8 ):

    continue_sift = True

    imf = get_next_imf( X[:,None], t, sd_thresh=sd_thresh )[:,None]

    layer = 1
    while continue_sift:
        print layer

        proto_imf = X[:,None] - np.sum(imf,axis=1)[:,None]
        if np.nansum( np.power( X - proto_imf, 2) ) < sift_thresh:
            continrue_sift=False
            return np.concatenate( (imf,proto_imf), axis=1 )

        ens_imf = get_next_imf( proto_imf + np.random.randn( *proto_imf.shape ), t )[:,None]
        imf = np.concatenate( (imf,ens_imf), axis=1 )

        for ii in range(1,nensembles):
            ens_imf = get_next_imf( proto_imf + np.random.randn( *proto_imf.shape ), t )[:,None]
            imf[:,layer] = imf[:,layer] + (1./(ii+1))*(ens_imf[:,0]-imf[:,layer])

        layer += 1

    return imf


def get_next_imf( X, t, sd_thresh=.1 ):

    proto_imf = X.copy()

    continue_imf = True
    while continue_imf:

        upper,lower = utils.find_envelopes( t, proto_imf )

        # If upper or lower are None we should stop sifting alltogether
        if upper is None or lower is None:
            continue_sift=False
            return proto_imf

        # Find average envelope
        avg = np.mean([upper,lower],axis=0)

        x1 = proto_imf[:,0] - avg
        sd = sum((proto_imf[:,0]-x1)**2)/sum(proto_imf[:,0]**2);

        # If the envelope has less power than our threshold, stop this imf
        #if np.nansum(np.power(avg,2)) < imf_thresh:
        if sd < sd_thresh:
            return x1

        proto_imf = proto_imf - avg[:,None]

    return imf



def sift( X, sd_thresh=.1, sift_thresh=1e-8 ):

    continue_sift = True
    layer = 0
    while continue_sift:

        # initialise
        if layer == 0:
            proto_imf = X[:,None].copy()
            imf = np.zeros_like( proto_imf )
        else:
            proto_imf = X[:,None] - np.sum(imf,axis=1)[:,None]

        continue_imf = True
        itera = 0

        if np.nansum( np.power(proto_imf,2) ) < sift_thresh:
            continue_sift = False

        start_proto_imf = proto_imf.copy()

        while continue_imf:

            upper,lower = utils.find_envelopes( proto_imf )

            # If upper or lower are None we should stop sifting alltogether
            if upper is None or lower is None:
                continue_sift=False

                if layer == 0:
                    imf = proto_imf
                else:
                    imf = np.concatenate( (imf,proto_imf), axis=1 )
                break

            # Find average envelope
            avg = np.mean([upper,lower],axis=0)

            x1 = proto_imf[:,0] - avg
            sd = sum((proto_imf[:,0]-x1)**2)/sum(proto_imf[:,0]**2);

            # If the envelope has less power than our threshold, stop this imf
            #if np.nansum(np.power(avg,2)) < imf_thresh:
            if sd < sd_thresh:
                continue_imf=False

                if layer == 0:
                    imf = proto_imf
                else:
                    imf = np.concatenate( (imf,proto_imf), axis=1 )
                layer += 1
                break

            # We're going to keep sifting if we made it this far
            proto_imf = proto_imf - avg[:,None]
            itera += 1

    return imf
