import numpy as np
from scipy import signal

def holospectrum_am( infr, infr2, inam2, fbins, fbins2 ):
    """ Not so sure where to start"""

    # carrier x am x time
    holo = np.zeros( ( len(fbins)+1, len(fbins2)+1, infr.shape[0], infr.shape[1] ) )

    for t_ind in range(infr.shape[0]):

         # carrier freq inds
         finds_carrier = np.digitize( infr[t_ind,:], fbins )

         for imf2_ind in range(infr2.shape[2]):

             # am freq inds
             finds_am = np.digitize( infr2[t_ind,:,imf2_ind], fbins2 )
             tmp = inam2[t_ind,:,:]
             tmp[tmp==np.nan] = 0
             holo[finds_carrier,finds_am,t_ind,:] += tmp

    return holo

def instantaneous_stats( imf, sample_rate, method ):

    infreq = np.zeros_like( imf )

    if method == 'hilbert':

        analytic_signal = signal.hilbert( imf, axis=0 )
        instantaneous_phase = np.unwrap(np.angle(analytic_signal),axis=0)
        instantaneous_frequency = (np.diff(instantaneous_phase,axis=0) / (2.0*np.pi) * sample_rate)
        instantaneous_frequency = np.r_[ instantaneous_frequency[None,0,:], instantaneous_frequency]

        instantaneous_amp = np.abs(analytic_signal)

    else:

        print('Method not recognised')

    return instantaneous_frequency, instantaneous_amp

def hilberthuang( infr, inam, fbins, time_vect, tbins ):

    # add this later...
    tinds = np.digitize( time_vect, tbins )

    hht = np.zeros( (len(tbins),len(fbins)+1) )

    # for each time bin....
    for ii in range(len(tbins)-1):

        # Add frequency info for this time bin
        finds = np.digitize( infr[tinds==ii,:].mean(axis=0), fbins )
        hht[ii,finds] = inam[tinds==ii,:].sum(axis=0)

    return hht

