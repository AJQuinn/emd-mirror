import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp


## a peak is a sign change in the differential of a signal, we want to find
# adjacent samples in the diff with different signs

def not_retarded_find_envelopes( t, X ):

    max_locs,max_pks = find_extrema( X[:,0] )
    min_locs,min_pks = find_extrema( X[:,0], ret_min=True)

    if max_locs.size < 2 or min_locs.size < 2:
        return [],[]

    N = 50 # should make this analytic somehow

    ret_max_locs = np.pad( max_locs,N,'reflect',reflect_type='odd')
    ret_min_locs = np.pad( min_locs,N,'reflect',reflect_type='odd')
    #print ret_max_locs[0],ret_min_locs[0]
    #print ret_max_locs[-1],ret_min_locs[-1]

    ret_max_pks = np.pad( max_pks,N,'reflect',reflect_type='even')
    ret_min_pks = np.pad( min_pks,N,'reflect',reflect_type='even')

    f = interp.splrep( ret_max_locs, ret_max_pks )
    upper = interp.splev(range(ret_max_locs[0],ret_max_locs[-1]), f)

    t = np.arange(ret_max_locs[0],ret_max_locs[-1])
    tinds = np.logical_and((t >= 0), (t < X.shape[0]))
    upper = upper[tinds]

    f = interp.splrep( ret_min_locs, ret_min_pks )
    lower = interp.splev(range(ret_min_locs[0],ret_min_locs[-1]), f)

    t = np.arange(ret_min_locs[0],ret_min_locs[-1])
    tinds = np.logical_and((t >= 0), (t < X.shape[0]))
    lower = lower[tinds]

    return upper, lower

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

##

# reflect upper and lower points and locs around axis
# append to existing upper and lower points and locs

def extend_extrema2( min_loc, min_pk, max_loc, max_pk, N=3 ):

    # Find the axis of symmetry at start and end of extrema
    start_symm = np.min( (min_loc[0],max_loc[0]) )
    end_symm = np.max( (min_loc[-1],max_loc[-1]) )

    # Check whether the final point is in the upper or lower extrema, we don't
    # want to reflect the final point
    start_minima = False
    if start_symm == min_loc[0]:
        start_minima = True

    end_minima = False
    if end_symm == min_loc[0]:
        end_minima = True

    # Minima

    # If the first extrema is a trough, skip it here
    if start_minima:
        inds = range(1,N+1)
    else:
        inds = range(0,N)
    print inds
    # Do the reflection
    p_locs = reflect_points( min_loc[inds], origin=start_symm )
    p_pks = reflect_points( min_pk[inds], origin=start_symm )

    # If the final extream is a trough, don't try to reflect it
    if end_minima:
        inds = range(min_loc.shape[0]-N-2,min_loc.shape[0]-2)
    else:
        inds = range(min_loc.shape[0]-N-1,min_loc.shape[0]-1)
    print inds
    # Do the reflection
    a_locs = reflect_points( min_loc[inds], origin=end_symm, direction='right' )
    a_pks = reflect_points( min_pk[inds], origin=end_symm, direction='right' )
    print end_symm
    print a_pks

    ret_min_locs = np.concatenate( (p_locs,min_loc,a_locs) )
    ret_min_pks = np.concatenate( (p_pks,min_pk,a_pks) )

    # Maxima

    # If the first extrema is a trough, skip it here
    if start_minima:
        inds = range(1,N+1)
    else:
        inds = range(0,N)
    print inds
    # Do the reflection
    p_locs = reflect_points( max_loc[inds], origin=start_symm )
    p_pks = reflect_points( max_pk[inds], origin=start_symm )

    # If the final extream is a trough, don't try to reflect it
    if end_minima:
        inds = range(max_loc.shape[0]-N-2,max_loc.shape[0]-2)
    else:
        inds = range(max_loc.shape[0]-N-1,max_loc.shape[0]-1)
    print inds
    # Do the reflection
    a_locs = reflect_points( max_loc[inds], origin=end_symm, direction='right' )
    a_pks = reflect_points( max_pk[inds], origin=end_symm, direction='right' )

    ret_max_locs = np.concatenate( (p_locs,max_loc,a_locs) )
    ret_max_pks = np.concatenate( (p_pks,max_pk,a_pks) )

    return ret_min_locs,ret_min_pks,ret_max_locs,ret_min_pks

def find_envelopes( time_vect, X ):

    [max_loc,max_pk] = find_extrema( X[:,0] )
    [min_loc,min_pk] = find_extrema( X[:,0], ret_min=True)

    if np.array(max_loc).size < 2 or np.array(min_loc).size < 2:
        return None,None

    # extend maxima
    #if len(max_loc) > 3:
    #    max_loc_ext,max_pk_ext = extend_extrema( max_loc,max_pk )
    #else:
    #    max_loc_ext,max_pk_ext = extend_extrema( max_loc,max_pk, N=1 )

    #if len(min_loc) > 3:
    #    min_loc_ext,min_pk_ext = extend_extrema( min_loc,min_pk )
    #else:
    #    min_loc_ext,min_pk_ext = extend_extrema( min_loc,min_pk, N=1 )

    if len(min_loc) > 3:
         min_loc_ext,min_pk_ext,max_loc_ext,max_pk_ext = extend_extrema2( min_loc, min_pk, max_loc, max_pk )
    else:
         min_loc_ext,min_pk_ext,max_loc_ext,max_pk_ext = extend_extrema2( min_loc, min_pk, max_loc, max_pk, N=1 )

    max_pad = max_loc[0] + np.abs(max_loc_ext[0])
    min_pad = min_loc[0] + np.abs(min_loc_ext[0])

    max_env = interpolate_envelope( max_loc_ext, max_pk_ext, max_pad, X.shape[0] )
    min_env = interpolate_envelope( min_loc_ext, min_pk_ext, min_pad, X.shape[0] )

    print max_env.shape
    print min_env.shape

    return max_env,min_env

def reflect_points( points, direction='left', origin=0 ):

    if direction == 'left':
        # Compute step to points from origin
        steps =  np.cumsum(np.diff(np.concatenate( ([origin],points) )))
        return np.array(origin-steps-1)[::-1]

    elif direction == 'right':
        # Compute step to points from origin
        a = np.concatenate( (points,[origin]) )
        print a
        steps =  np.cumsum(np.diff(np.concatenate( (points,[origin]) )[::-1]))
        return np.array(origin-steps+1)

    else:
        print 'Direction %s not recognised' % direction

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

if __name__ == '__main__':

    t = np.linspace(0,10,15000)
    X = np.sin( 2*np.pi*4*t) + np.sin( 2*np.pi*14*t) + np.random.randn(15000,)*.02 + np.linspace(-1,1,15000)
    
    sift_thresh = 1e-12
    imf_thresh = 1e-6
    
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

        if np.nansum( np.power(proto_imf,2) ) < sift_thresh:
            continue_sift = False

        while continue_imf:
    
            upper,lower = not_retarded_find_envelopes( t, proto_imf )
    
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
