import numpy as np
from scipy import signal,sparse
from . import utils

##
def frequency_stats( imf, sample_rate, method,
                     smooth_phase=31 ):

    # Each case here should compute the analytic form of the imfs and the
    # instantaneous amplitude.
    if method == 'hilbert':

        analytic_signal = analytic_signal_from_hilbert( imf )

        # Estimate instantaneous amplitudes directly from analytic signal
        iamp = np.abs(analytic_signal)

    elif method == 'quad':

        analytic_signal = analytic_signal_from_quadrature( imf )

        orig_dim = imf.ndim
        if imf.ndim == 2:
            imf = imf[:,:,None]

        # Estimate inst amplitudes with spline interpolation
        iamp = np.zeros_like(imf)
        for ii in range(imf.shape[1]):
            for jj in range(imf.shape[2]):
                #iamp[:,ii,jj] = utils.interp_envelope( imf[:,ii,jj], mode='combined' )
                iamp[:,ii,jj] = utils.interp_envelope( imf[:,ii,jj],
                                    mode='upper' )

        if orig_dim == 2:
            iamp = iamp[:,:,0]

    elif method == 'direct_quad':
        raise ValueError('direct_quad method is broken!')

        n_imf = utils.amplitude_normalise( imf.copy() )
        iphase = np.unwrap(phase_angle( n_imf ))

        iamp = np.zeros_like(imf)
        for ii in range(imf.shape[1]):
            iamp[:,ii] = utils.interp_envelope( imf[:,ii,None], mode='combined' )

    else:
        print('Method not recognised')

    # Compute unwrapped phase for frequency estimation
    iphase = phase_from_analytic_signal( analytic_signal, smoothing=smooth_phase, ret_phase='unwrapped' )
    ifreq = freq_from_phase( iphase, sample_rate )

    # Return wrapped phase
    iphase = utils.wrap_phase( iphase )

    return iphase,ifreq,iamp

# Frequency stat utils

def analytic_signal_from_hilbert( X ):

    return signal.hilbert( X, axis=0 )

def analytic_signal_from_quadrature( X ):

    nX = utils.amplitude_normalise( X.copy(), clip=True )

    imagX = np.lib.scimath.sqrt(1-np.power( nX,2 )).real

    mask = ((np.diff(nX,axis=0)>0) * -2) + 1
    mask[mask==0] = -1
    mask = np.r_[mask,mask[-1,None,:]]

    q = imagX * mask

    return  nX + 1j * q

def phase_from_analytic_signal( analytic_signal, smoothing=None,
                                ret_phase='wrapped', phase_jump='ascending' ):

    # Compute unwrapped phase
    iphase = np.unwrap(np.angle(analytic_signal),axis=0)

    # Apply smoothing if requested
    if smoothing is not None:
        iphase = signal.savgol_filter(iphase,smoothing,1,axis=0)

    # Set phase jump point to requested part of cycle
    if phase_jump=='ascending':
        iphase = iphase + np.pi/2
    elif phase_jump=='peak':
        pass # do nothing
    elif phase_jump=='descending':
        iphase = iphase - np.pi/2
    elif phase_jump=='trough':
        iphase = iphase + np.pi

    if ret_phase=='wrapped':
        return utils.wrap_phase( iphase )
    elif ret_phase=='unwrapped':
        return iphase

def freq_from_phase( iphase, sample_rate ):

    # Differential of instantaneous phase
    iphase = np.gradient( iphase, axis=0 )

    # Convert to freq
    ifrequency = iphase / (2.0*np.pi) * sample_rate

    return ifrequency

def phase_from_freq( ifrequency, sample_rate, phase_start=-np.pi):

    iphase_diff = (ifrequency/sample_rate) * (2*np.pi)

    iphase = phase_start + np.cumsum(iphase_diff,axis=0)

    return iphase

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

## Time-frequency spectra


def holospectrum( infr, infr2, inam2, freq_edges, freq_edges2, mode='energy',
        return_time=True ):

    if mode == 'energy':
        inam2 = inam2**2

    IA_inds = np.digitize( infr2, freq_edges2 )
    infr_inds = np.digitize( infr, freq_edges )

    new_shape = (infr_inds.shape[0],infr_inds.shape[1],imf2.shape[2])
    infr_inds = np.broadcast_to( infr_inds[:,:,None], new_shape )

    fold_dim1 = len(freq_edges)+1
    fold_dim2 = len(freq_edges2)+1

    infr_inds = infr_inds + IA_inds *  fold_dim1

    T_inds = np.arange(infr.shape[0])[:,None,None]
    T_inds = np.broadcast_to( T_inds, new_shape )

    coords =(T_inds.reshape(-1),infr_inds.reshape(-1))
    holo = sparse.coo_matrix( (inam2.reshape(-1),coords),
            shape=(infr.shape[0],fold_dim1*fold_dim2) )

    # Always returns full matrix until someone implements ND sparse in scipy
    if return_time:
        # Return the full matrix
        holo = holo.toarray().reshape(new_shape[0],fold_dim2,fold_dim1)
        return holo[:,1:-1,1:-1]
    else:
        # Collapse time dimension while we're still sparse
        holo = holo.sum(axis=0)
        holo = holo.reshape(fold_dim2,fold_dim1)
        return np.array(holo[1:-1,1:-1]) # don't return a matrix

def hilberthuang( infr, inam, freq_edges, mode='energy', return_sparse=False ):

    if mode == 'energy':
        inam = inam**2

    # Create sparse co-ordinates
    yinds = np.digitize(infr,freq_edges)
    xinds = np.tile( np.arange(yinds.shape[0]),(yinds.shape[1],1) ).T

    coo_data = (inam.reshape(-1),(yinds.reshape(-1),xinds.reshape(-1)))

    # Remove values outside our bins
    goods = np.any( np.c_[coo_data[1][0]<len(freq_edges)-1, (coo_data[1][0]==0)],axis=1 )
    coo_data = (coo_data[0][goods], (coo_data[1][0][goods], coo_data[1][1][goods]))

    # Create sparse matrix
    hht = sparse.coo_matrix( coo_data,shape=(len(freq_edges)-1,xinds.shape[0]))

    if return_sparse:
        return hht
    else:
        return hht.toarray()

def hilberthuang_1d( infr, inam, fbins, mode='energy'):

    specs = np.zeros( (len(fbins)-1,infr.shape[1]) )

    # Remove values outside the bin range
    infr = infr.copy()
    infr[infr<fbins[0]] = np.nan
    infr[infr>fbins[-1]] = np.nan

    finds = np.digitize( infr, fbins )

    for ii in range( len(fbins)-1 ):
        for jj in range( infr.shape[1] ):

            if mode == 'power':
                specs[ii,jj] = np.nansum(inam[finds[:,jj]==ii,jj])
            elif mode == 'energy':
                specs[ii,jj] = np.nansum(np.power(inam[finds[:,jj]==ii,jj],2))

    return specs


def define_hist_bins( data_min, data_max, nbins, scale='linear' ):
    """
    Find the bin edges and centre frequencies for use in a histogram
    """

    if scale == 'log':
        p = np.log([data_min,data_max])
        edges = np.linspace(p[0],p[1],nbins+1)
        edges = np.exp(edges)
    elif scale == 'linear':
        edges = np.linspace( data_min, data_max, nbins+1 )
    else:
        raise ValueError( 'scale \'{0}\' not recognised. please use \'log\' or \'linear\'.')

    # Get centre frequecy for the bins
    centres = np.array( [ (edges[ii]+edges[ii+1])/2 for ii in range(len(edges)-1) ] )

    return edges, centres

def define_hist_bins_from_data( X, nbins=None, mode='sqrt', scale='linear' ):
    """
    Find the bin edges and centre frequencies for use in a histogram

    if nbins is defined, mode is ignored
    """

    data_min = X.min()
    data_max = X.max()

    if nbins is None:
        if mode == 'sqrt':
            nbins = np.sqrt(X.shape[0]).astype(int)
        else:
            raise ValueError('mode {0} not recognised, please use \'sqrt\'')

    return define_hist_bins( data_min, data_max, nbins, scale=scale )

def mean_vector( IP, IA, mask=None ):

    phi = np.sin(IP) + 1j*np.cos(IP)
    mv = phi[:,None] * IA
    return mv.mean(axis=0)
