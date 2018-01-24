
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal

def amplitude_normalise( X, thresh=1e-10 ):

    # We're ignoring the trend IMF for now...
    for iimf in np.arange(X.shape[1]-1):

        env = get_envelope( X[:,iimf,None], combined_upper_lower=True )[...,None]

        if env is None:
            continue_norm = False
        else:
            continue_norm = True

        while continue_norm:

            X[:,iimf,None] = X[:,iimf,None] / env
            env = get_envelope( X[:,iimf,None], combined_upper_lower=True )[...,None]

            if np.abs(env.sum()-env.shape[0]) < thresh:
                continue_norm = False

    return X

def get_envelope( X, N=10, combined_upper_lower=False ):

    if combined_upper_lower:
        max_locs,max_pks = find_extrema( np.abs(X[:,0]) )
    else:
        max_locs,max_pks = find_extrema( X[:,0] )
    ret_max_locs = np.pad( max_locs,N,'reflect',reflect_type='odd')
    ret_max_pks = np.pad( max_pks,N,'reflect',reflect_type='even')

    f = interp.splrep( ret_max_locs, ret_max_pks )
    envelope = interp.splev(list(range(ret_max_locs[0],ret_max_locs[-1])), f)

    t = np.arange(ret_max_locs[0],ret_max_locs[-1])
    tinds = np.logical_and((t >= 0), (t < X.shape[0]))
    envelope = np.array(envelope[tinds])

    return envelope

def find_envelopes( X, to_plot=False, ret_all=False ):

    # Find maxima and minima
    max_locs,max_pks = find_extrema( X[:,0] )
    min_locs,min_pks = find_extrema( X[:,0], ret_min=True)

    # Return nothing we don't have enough extrema
    if max_locs.size <= 1 or min_locs.size <= 1:
        return None,None

    # Determine how much padding to use
    N = 14 # should make this analytic somehow
    if max_locs.size < N or min_locs.size < N:
        N = max_locs.size

    # Pad peak locations
    ret_max_locs = np.pad( max_locs,N,'reflect',reflect_type='odd' )
    ret_min_locs = np.pad( min_locs,N,'reflect',reflect_type='odd' )

    # Pad peak magnitudes
    ret_max_pks = np.pad( max_pks,N,'reflect',reflect_type='odd' )
    ret_min_pks = np.pad( min_pks,N,'reflect',reflect_type='odd' )

    if max(ret_max_locs) < len(X) or min(ret_max_locs) >= 0:
        ret_max_locs = np.pad( ret_max_locs,N,'reflect',reflect_type='odd' )
        ret_max_pks = np.pad( ret_max_pks,N,'reflect',reflect_type='odd' )

    if max(ret_min_locs) < len(X) or min(ret_min_locs) >= 0:
        ret_min_locs = np.pad( ret_min_locs,N,'reflect',reflect_type='odd' )
        ret_min_pks = np.pad( ret_min_pks,N,'reflect',reflect_type='odd' )

    # Run interpolation on upper envelope
    f = interp.splrep( ret_max_locs, ret_max_pks )
    t = np.arange(ret_max_locs[0],ret_max_locs[-1])
    upper = interp.splev(t, f)

    t_max = np.arange(ret_max_locs[0],ret_max_locs[-1])
    tinds_max = np.logical_and((t_max >= 0), (t_max < X.shape[0]))

    # Run interpolation on lower envelope
    f = interp.splrep( ret_min_locs, ret_min_pks )
    t = np.arange(ret_min_locs[0],ret_min_locs[-1])
    lower = interp.splev(t, f)

    t_min = np.arange(ret_min_locs[0],ret_min_locs[-1])
    tinds_min = np.logical_and((t_min >= 0), (t_min < X.shape[0]))

    if to_plot:

        plt.figure(figsize=(12,4))
        plt.plot(X[:,0],'k')
        plt.plot(ret_max_locs,ret_max_pks,'*')
        #plt.plot(max_locs,max_pks,'o')
        plt.plot(ret_min_locs,ret_min_pks,'*')
        #plt.plot(min_locs,min_pks,'o')
        plt.plot(t_max,upper)
        plt.plot(t_min,lower)

    upper = np.array(upper[tinds_max])
    lower = np.array(lower[tinds_min])
    t = np.array(t_min[tinds_min])

    if to_plot:
        plt.plot( (upper+lower) / 2 )
        locs, labels = plt.xticks()
        plt.xticks(locs, ('-.5','0','.5','1','1.5','2','2.5','3'))

    if ret_all:
        return upper, lower, t_max, t_min, ret_max_locs,ret_max_pks,ret_min_locs,ret_min_pks
    else:
        return upper, lower

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

    print(locs.shape)
    print(pks.shape)
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

