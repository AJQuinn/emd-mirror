import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal


## a peak is a sign change in the differential of a signal, we want to find
# adjacent samples in the diff with different signs

def find_envelopes( t, X, to_plot=False ):

    max_locs,max_pks = find_extrema( X[:,0] )
    min_locs,min_pks = find_extrema( X[:,0], ret_min=True)

    if max_locs.size < 2 or min_locs.size < 2:
        return [],[]

    N = 2 # should make this analytic somehow

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
    if to_plot:
        print tinds.sum()
        print t.shape
        print upper.shape
    upper = upper[tinds]

    f = interp.splrep( ret_min_locs, ret_min_pks )
    lower = interp.splev(range(ret_min_locs[0],ret_min_locs[-1]), f)

    t = np.arange(ret_min_locs[0],ret_min_locs[-1])
    tinds = np.logical_and((t >= 0), (t < X.shape[0]))
    if to_plot: print tinds.sum()
    lower = lower[tinds]

    if to_plot:
        plt.plot(X[:,0])
        plt.plot(ret_max_locs,ret_max_pks,'*')
        plt.plot(max_locs,max_pks,'o')
        plt.plot(ret_min_locs,ret_min_pks,'*')
        plt.plot(min_locs,min_pks,'o')
        plt.plot(upper)
        plt.plot(lower)
        plt.show()

    return upper, lower

def find_extrema( X, ret_min=False ):

    if ret_min:
        ind = signal.argrelextrema( X, np.less_equal)[0]
    else:
        ind = signal.argrelextrema( X, np.greater_equal)[0]

    return ind, X[ind]

def find_extrema3( X, ret_min=False ):

    dX = np.diff( X )

    # Finds anywhere there is a sign change in the diff
    inds = np.where(np.sign(dX[1:]*dX[:-1]) < 0)[0] + 1

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

def sift( X, imf_thresh=1e-4, sift_thresh=1e-8 ):

    continue_sift = True
    layer = 0
    while continue_sift:
        print layer

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

        while continue_imf:

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

    return imf

def instantaneous_stats( imf, sample_rate, method ):

    infreq = np.zeros_like( imf )

    if method == 'hilbert':

        analytic_signal = signal.hilbert( imf, axis=0 )
        instantaneous_phase = np.unwrap(np.angle(analytic_signal),axis=0)
        instantaneous_frequency = (np.diff(instantaneous_phase,axis=0) / (2.0*np.pi) * sample_rate)

        instantaneous_amp = np.abs(analytic_signal)

    else:

        print('Method not recognised')

    return instantaneous_frequency, instantaneous_amp

def hilberthuang( infr, inam, fbins ):

    # add this later...
    #tinds = np.digitize( time_vect, tbins )

    hht = np.zeros( (infr.shape[0],len(fbins)+1) )

    for ii in range(infr.shape[0]):

        # Add frequency info for this time bin
        finds = np.digitize( infr[ii,:], fbins )
        hht[ii,finds] = inam[ii,:]

    return hht


if __name__ == '__main__':

    sample_rate = 1000
    seconds = 10
    num_samples = seconds*sample_rate
    t = np.linspace(0,seconds,num_samples)
    #X = np.sin( 2*np.pi*4*t) + np.sin( 2*np.pi*14*t) + np.random.randn(num_samples,)*.02 + np.linspace(-1,1,num_samples)
    X = np.r_[signal.chirp( t[:5000], 5, 5, 25 ), signal.chirp( t[5000:],25,10,5)] + np.sin( 2*np.pi*50*t )

    imf = sift( X )

    plt.figure()
    plt.plot(imf)

    infr, inam = instantaneous_stats( imf, sample_rate, 'hilbert' )
    hht = hilberthuang( infr, inam, np.linspace(0,75,125) )

    plt.figure()
    plt.contourf( np.sqrt(hht.T) )

    plt.show()
