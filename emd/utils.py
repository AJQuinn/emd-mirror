
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal

def amplitude_normalise( X, thresh=1e-10, clip=False ):

    # Don't normalise in place
    X = X.copy()

    for iimf in range(X.shape[1]):

        env = interp_envelope( X[:,iimf,None], mode='combined' )

        if env is None:
            continue_norm = False
        else:
            continue_norm = True
            env = env[...,None]

        while continue_norm:

            X[:,iimf,None] = X[:,iimf,None] / env
            env = interp_envelope( X[:,iimf,None], mode='combined' )

            if env is None:
                continue_norm = False
            else:
                continue_norm = True
                env = env[...,None]

                if np.abs(env.sum()-env.shape[0]) < thresh:
                    continue_norm = False

    if clip:
        # Make absolutely sure nothing daft is happening
        X = np.clip( X, -1, 1)

    return X

def get_padded_extrema( X, combined_upper_lower=False ):

    if combined_upper_lower:
        max_locs,max_pks = find_extrema( np.abs(X[:,0]) )
    else:
        max_locs,max_pks = find_extrema( X[:,0] )

    # Find maxima and minima
    max_locs,max_pks = find_extrema( X[:,0] )

    # Return nothing we don't have enough extrema
    if max_locs.size <= 2:
        return None,None

    # Determine how much padding to use
    N = 2 # should make this analytic somehow
    if max_locs.size < N:
        N = max_locs.size

    # Pad peak locations
    ret_max_locs = np.pad( max_locs,N,'reflect',reflect_type='odd' )

    # Pad peak magnitudes
    ret_max_pks = np.pad( max_pks,N,'reflect',reflect_type='odd' )

    while max(ret_max_locs) < len(X) or min(ret_max_locs) >= 0:
        ret_max_locs = np.pad( ret_max_locs,N,'reflect',reflect_type='odd' )
        ret_max_pks = np.pad( ret_max_pks,N,'reflect',reflect_type='odd' )

    return ret_max_locs,ret_max_pks

def interp_envelope( X, to_plot=False, ret_all=False, mode='upper' ):

    if mode == 'upper':
        locs,pks = get_padded_extrema( X, combined_upper_lower=False)
    elif mode == 'lower':
        locs,pks = get_padded_extrema( -X, combined_upper_lower=False)
    elif mode == 'combined':
        locs,pks = get_padded_extrema( X, combined_upper_lower=True)
    else:
        raise ValueError('Mode not recognised. Use mode= \'upper\'|\'lower\'|\'combined\'')

    if locs is None:
        return None

    # Run interpolation on upper envelope
    f = interp.splrep( locs, pks )
    t = np.arange(locs[0],locs[-1])
    env = interp.splev(t, f)

    t_max = np.arange(locs[0],locs[-1])
    tinds = np.logical_and((t_max >= 0), (t_max < X.shape[0]))

    env = np.array(env[tinds])

    if env.shape[0] != X.shape[0]:
        raise ValueError('Envelope length does not match input data {0} {1}'.format(upper.shape[0],X.shape[0]))

    if mode == 'lower':
        return -env
    else:
        return env

def find_extrema( X, ret_min=False ):

    if ret_min:
        #ind = signal.argrelextrema( X, np.less)[0]
        ind = signal.argrelmin( X, order=1 )[0]
    else:
        #ind = signal.argrelextrema( X, np.greater)[0]
        ind = signal.argrelmax( X, order=1)[0]

    #if ind[0] == 0:
    #    ind = ind[1:]

    #if ind[-1] == X.shape[0]:
    #    ind = ind[:-2]

    return ind, X[ind]

def abreu2010( f, nonlin_deg, nonlin_phi, sample_rate, seconds ):

    time_vect = np.linspace(0,seconds,seconds*sample_rate)

    factor = np.sqrt( 1- nonlin_deg**2 )
    num = nonlin_deg*np.sin(nonlin_phi) / 1+np.sqrt( 1-nonlin_deg**2 )
    num = num + np.sin( 2*np.pi*f*time_vect)

    denom = 1 - nonlin_deg * np.cos( 2*np.pi*f*time_vect + nonlin_phi )

    return factor * ( num / denom )

def est_orthogonality( imf ):


    ortho = np.ones( (imf.shape[1],imf.shape[1]) ) * np.nan

    for ii in range(imf.shape[1]):
        for jj in range(imf.shape[1]):
            ortho[ii,jj] = np.abs( np.sum(imf[:,ii]*imf[:,jj]) ) / \
                        ( np.sqrt(np.sum(imf[:,jj]*imf[:,jj])) * np.sqrt(np.sum(imf[:,ii]*imf[:,ii])) );

    return ortho

def find_peaks( X, winsize, lock_to='max', percentile=None ):
    """
    Helper function for defining trials around peaks within the data
    """

    if lock_to=='max':
        locs,pks = find_extrema( X, ret_min=False )
    else:
        locs,pks = find_extrema( X, ret_min=True )

    if percentile is not None:
        thresh = np.percentile(pks[:,0],percentile)
        locs = locs[pks[:,0]>thresh]
        pks = pks[pks>thresh]

    winstep = int(winsize/2)

    trls = np.r_[np.atleast_2d(locs-winstep), np.atleast_2d(locs+winstep)].T

    # Reject trials which start before 0
    inds = trls[:,0] < 0
    trls = trls[inds==False,:]

    # Reject trials which end after X.shape[0]
    inds = trls[:,1] > X.shape[0]
    trls = trls[inds==False,:]

    return trls

def apply_epochs( X, trls ):
    """
    Helper function which applies a set of epochs to a continuous dataset
    """

    Y = np.zeros( (trls[0,1]-trls[0,0],X.shape[1],trls.shape[0]) )
    for ii in np.arange(trls.shape[0]):

        Y[:,:,ii] = X[trls[ii,0]:trls[ii,1],:]

    return Y

def bin_by_phase( ip, x, nbins=24, weights=None, variance_metric='variance', phase_bins=None ):
    """
    Compute distribution of x by phase-bins in ip

    """

    if phase_bins is None:
        phase_bins = np.linspace(-np.pi, np.pi, nbins)
    else:
        nbins = len(phase_bins)

    bin_inds = np.digitize( ip, phase_bins )

    avg = np.zeros( (nbins,) )*np.nan
    var = np.zeros( (nbins,) )*np.nan
    for ii in range( nbins ):
        inds = bin_inds==ii
        if weights is None:
            avg[ii] = np.average( x[inds] )
            v = np.average( (x[inds]-avg[ii])**2 )
        else:
            if inds.sum() > 0:
                avg[ii] = np.average( x[inds], weights=weights[inds] )
                v = np.average( (x[inds]-avg[ii])**2, weights=weights[inds] )
            else:
                v = np.nan

        if variance_metric=='variance':
            var[ii] = v
        elif variance_metric=='std':
            var[ii] = np.sqrt( v )
        elif variance_metric=='sem':
            var[ii] = np.sqrt( v ) / np.sqrt( inds.sum() )

    return avg,var,phase_bins


def wrap_phase( IP, ncycles=1 ):

    phases = ( IP + (np.pi*ncycles)) % (ncycles * 2 * np.pi ) - (np.pi*ncycles)

    return phases
