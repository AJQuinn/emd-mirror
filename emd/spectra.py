import numpy as np
from scipy import signal
from . import utils

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

def frequency_stats( imf, sample_rate, method,smooth_phase=None ):

    infreq = np.zeros_like( imf )

    # Each case here should compute the analytic form of the imfs and the
    # instantaneous amplitude.
    if method == 'hilbert':

        analytic_signal = signal.hilbert( imf, axis=0 )

        # Estimate instantaneous amplitudes
        iamp = np.abs(analytic_signal)

    elif method == 'quad':

        print(imf.shape)
        n_imf = utils.amplitude_normalise( imf.copy() )

        imag_imf = np.lib.scimath.sqrt(1-np.power( n_imf,2 )).real

        mask = ((np.diff(n_imf,axis=0)>0) * -2) + 1
        mask[mask==0] = -1
        mask = np.r_[mask,mask[-1,None,:]]

        q = imag_imf * mask
        analytic_signal = n_imf + 1j * q

        iamp = np.zeros_like(imf)
        # Not computing amplitude for the trend for now
        print(imf.shape)
        for ii in range(imf.shape[1]):
            iamp[:,ii] = utils.get_envelope( imf[:,ii,None], combined_upper_lower=True )

    elif method == 'direct_quad':

        n_imf = utils.amplitude_normalise( imf.copy() )
        iphase = np.unwrap(phase_angle( n_imf ))

        # Not computing amplitude for the trend for now
        iamp = np.zeros_like(imf)
        for ii in range(imf.shape[1]-1):
            iamp[:,ii] = utils.get_envelope( imf[:,ii,None], combined_upper_lower=True )

    else:
        print('Method not recognised')

    if method == 'hilbert' or method == 'quad':
        # Estimate instantaneous frequencies
        iphase = np.unwrap(np.angle(analytic_signal),axis=0)
        if smooth_phase is not None:
            print('smoothing phase')
            iphase = signal.savgol_filter(iphase,smooth_phase,3,axis=0)

    ifrequency = (np.diff(iphase,axis=0) / (2.0*np.pi) * sample_rate)
    ifrequency = np.r_[ ifrequency[None,0,:], ifrequency]

    return iphase,ifrequency, iamp

def direct_quadrature( fm ):
    """
    Section 3.2 of 'on instantaneous frequency'
    """
    ph = phase_angle( fm )

    # We'll have occasional nans where fm==1 or -1
    inds = np.argwhere(np.isnan(ph))

    vals = (ph[inds[:,0]-1,:] + ph[inds[:,0]+1,:] ) / 2
    ph[inds[:,0]] = vals

    return ph

def phase_angle( fm ):
    """
    Eqn 35 in 'On Instantaneous Frequency'
    ... with additional factor of 2 to make the range [-pi, pi]
    """

    return np.arctan( fm / np.lib.scimath.sqrt( 1 - np.power(fm,2) ) )

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

