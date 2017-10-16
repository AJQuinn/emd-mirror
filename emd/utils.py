
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal

def amplitude_normalise( X, thresh=1e-6 ):

    env = get_envelope( X )[...,None]

    continue_norm = True
    while continue_norm:

        X = X / env
        env = get_envelope( X )[...,None]

        if env.sum()-env.shape[0] < thresh:
            continue_norm = False

    return X

def get_envelope( X, N=10 ):

    max_locs,max_pks = find_extrema( X[:,0] )
    ret_max_locs = np.pad( max_locs,N,'reflect',reflect_type='odd')
    ret_max_pks = np.pad( max_pks,N,'reflect',reflect_type='even')

    f = interp.splrep( ret_max_locs, ret_max_pks )
    envelope = interp.splev(range(ret_max_locs[0],ret_max_locs[-1]), f)

    t = np.arange(ret_max_locs[0],ret_max_locs[-1])
    tinds = np.logical_and((t >= 0), (t < X.shape[0]))
    envelope = np.array(envelope[tinds])

    return envelope

def find_envelopes( X, to_plot=False, ret_all=False ):

    max_locs,max_pks = find_extrema( X[:,0] )
    min_locs,min_pks = find_extrema( X[:,0], ret_min=True)

    if max_locs.size <= 1 or min_locs.size <= 1:
        return None,None

    N = 4 # should make this analytic somehow
    if max_locs.size < N or min_locs.size < N:
        N = max_locs.size

    ret_max_locs = np.pad( max_locs,N,'reflect',reflect_type='odd')
    ret_min_locs = np.pad( min_locs,N,'reflect',reflect_type='odd')

    ret_max_pks = np.pad( max_pks,N,'reflect',reflect_type='odd')
    ret_min_pks = np.pad( min_pks,N,'reflect',reflect_type='odd')

    if max(ret_max_locs) < len(X) or min(ret_max_locs) >= 0:
        ret_max_locs = np.pad( ret_max_locs,N,'reflect',reflect_type='odd')
        ret_max_pks = np.pad( ret_max_pks,N,'reflect',reflect_type='odd')

    if max(ret_min_locs) < len(X) or min(ret_min_locs) >= 0:
        ret_min_locs = np.pad( ret_min_locs,N,'reflect',reflect_type='odd')
        ret_min_pks = np.pad( ret_min_pks,N,'reflect',reflect_type='odd')

    f = interp.splrep( ret_max_locs, ret_max_pks )
    upper = interp.splev(range(ret_max_locs[0],ret_max_locs[-1]), f)

    t_max = np.arange(ret_max_locs[0],ret_max_locs[-1])
    tinds_max = np.logical_and((t_max >= 0), (t_max < X.shape[0]))

    f = interp.splrep( ret_min_locs, ret_min_pks )
    lower = interp.splev(range(ret_min_locs[0],ret_min_locs[-1]), f)

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


