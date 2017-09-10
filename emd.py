import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp


## a peak is a sign change in the differential of a signal, we want to find
# adjacent samples in the diff with different signs

def find_extrema( X, ret_min=False ):

    dX = np.diff( X )

    # Finds anywhere there is a sign change in the diff
    inds = np.where(np.sign(dX[1:]*dX[:-1]) < 0)[0] + 1

    # Find sign differences between adjacent extrema
    #d_ext = np.sign(np.diff(X[inds]))
    #if d_ext[-1] == 1:
    #    d_ext = np.concatenate( (d_ext,[1]) )
    #else:
    #    d_ext = np.concatenate( (d_ext,[-1]) )
    d_ext = np.sign(dX[inds])

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

def interpolate_envelope( locs, peaks, pad, nsamples ):
    """ Designed to work on indices not time values """

    #f = interp.interp1d( locs, peaks, kind='cubic')
    #f = interp.InterpolatedUnivariateSpline( locs, peaks, ext=3 )

    f = interp.splrep( locs, peaks )
    spl = interp.splev(range(locs[0],locs[-1]), f)

    print spl.shape
    print pad,nsamples

    #ret = np.nan * np.zeros( nsamples, )
    #ret = np.zeros( nsamples, )

    #ret[locs[0]:locs[-1]] = f(range(locs[0],locs[-1]))

    #return f(range(0,nsamples))
    return spl[pad:pad+nsamples]

def find_envelopes( time_vect, X ):

    [max_loc,max_pk] = find_extrema( X[:,0] )
    [min_loc,min_pk] = find_extrema( X[:,0], ret_min=True)

    if np.array(max_loc).size < 2 or np.array(min_loc).size < 2:
        return None,None

    # extend maxima
    if len(max_loc) > 3:
        max_loc_ext,max_pk_ext = extend_extrema( max_loc,max_pk )
    else:
        max_loc_ext,max_pk_ext = extend_extrema( max_loc,max_pk, N=1 )

    if len(min_loc) > 3:
        min_loc_ext,min_pk_ext = extend_extrema( min_loc,min_pk )
    else:
        min_loc_ext,min_pk_ext = extend_extrema( min_loc,min_pk, N=1 )

    max_pad = max_loc[0] + np.abs(max_loc_ext[0])
    min_pad = min_loc[0] + np.abs(min_loc_ext[0])

    max_env = interpolate_envelope( max_loc_ext, max_pk_ext, max_pad, X.shape[0] )
    min_env = interpolate_envelope( min_loc_ext, min_pk_ext, min_pad, X.shape[0] )

    print max_env.shape
    print min_env.shape

    return max_env,min_env

def extend_extrema( locs, pks, N=3 ):

    # peaks to prepend
    d = np.diff(locs)
    p_locs = np.array(locs[0] - np.cumsum( d[:N][::-1] ))
    print locs
    print p_locs
    p_pks = pks[1:N+1]

    # peaks to append
    a_locs = locs[-1] + np.cumsum(d[-N:][::-1])
    a_pks = pks[-N-1:-1][::-1]

    # full lists
    ret_locs = np.concatenate( (p_locs[::-1],locs,a_locs) )
    ret_pks = np.concatenate( (p_pks[::-1],pks,a_pks) )

    return ret_locs,ret_pks

t = np.linspace(0,10,15000)
X = np.sin( 2*np.pi*4*t ) + np.random.randn(15000,)*.001

sift_thresh = 1e-8
imf_thresh = 1e-4

X2 = X.copy()

continue_sift = True
layer = 0
while continue_sift:
    #print layer

    # initialise
    if layer == 0:
        proto_imf = X2[:,None].copy()
        imf = np.zeros_like( proto_imf )
    else:
        proto_imf = X2[:,None] - np.sum(imf,axis=1)[:,None]

    continue_imf = True
    itera = 0
    while continue_imf:
        print layer,itera

        upper,lower = find_envelopes( t, proto_imf )

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

        # If the envelope has less power than our threshold, stop this imf
        if np.nansum(np.power(avg,2)) < imf_thresh:
            print 'stopping this imf'
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
