import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp


## a peak is a sign change in the differential of a signal, we want to find
# adjacent samples in the diff with different signs

def find_extrema( X, ret_min=False ):

    dX = np.diff( X )

    # Finds anywhere there is a sign change in the diff
    inds = np.where(np.sign(dX[1:]*dX[:-1]) < 0)[0] + 1

    # Find sign differenes between adjacent extrema
    d_ext = np.sign(np.diff(X[inds]))

    # ID max or min
    if ret_min:
        #ext_ind = inds[np.sign( X[inds] ) == -1]
        ext_ind = inds[ np.where(d_ext == 1)[0] ]
    else:
        #ext_ind = inds[np.sign( X[inds] ) == 1]
        ext_ind = inds[ np.where(d_ext == -1)[0] ]

    if ext_ind.size == 0:
        return [],[]
    else:
        return ext_ind, X[ext_ind]

def interpolate_envelope( locs, peaks, nsamples ):
    """ Designed to work on indices not time values """

    #f = interp.interp1d( locs, peaks, kind='cubic')
    f = interp.InterpolatedUnivariateSpline( locs, peaks, ext=0 )

    #ret = np.nan * np.zeros( nsamples, )
    ret = np.zeros( nsamples, )

    ret[locs[0]:locs[-1]] = f(range(locs[0],locs[-1]))

    #return ret
    return f(range(0,nsamples))

def find_envelopes( time_vect, X ):

    [max_loc,max_pk] = find_extrema( X[:,0] )
    [min_loc,min_pk] = find_extrema( X[:,0], ret_min=True)

    if max_loc is None or min_loc is None:
        return None,None

    max_env = interpolate_envelope( max_loc, max_pk, X.shape[0] )
    min_env = interpolate_envelope( min_loc, min_pk, X.shape[0] )

    return max_env,min_env


t = np.linspace(0,10,15000)
X = np.sin( 2*np.pi*4*t ) + np.random.randn(15000,)*.1

# Find envelopes


sift_thresh = 1e-8
imf_thresh = 1e-4

X2 = X.copy()

continue_sift = True
layer = 0
while continue_sift:
    print layer

    # initialise
    if layer == 0:
        proto_imf = X2[:,None].copy()
        imf = np.zeros_like( proto_imf )
    else:
        proto_imf = X2[:,None] - np.sum(imf,axis=1)[:,None]

    continue_imf = True
    itera = 0
    while continue_imf:
        #print layer,itera

        upper,lower = find_envelopes( t, proto_imf )
        if upper is None or lower is None:
            continue_sift=False

        avg = np.mean([upper,lower],axis=0)

        #print np.nansum(np.power(avg,2))

        proto_imf = proto_imf - avg[:,None]

        if np.nansum(np.power(avg,2)) < imf_thresh:
            if layer == 0:
                imf = proto_imf
            else:
                imf = np.concatenate( (imf,proto_imf), axis=1 )
            continue_imf = False
            layer += 1
        else:
            itera += 1
