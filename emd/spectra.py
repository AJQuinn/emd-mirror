#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Compute instantanous spectral metrics (Phase,Amplitude and Frequency) and
compute frequency or time frequency spectra.

Routines:

frequency_stats
quadrature_transform
phase_from_complex_signal
freq_from_phase
phase_from_freq
direct_quadrature
phase_angle
holospectrum
hilberthuang
hilberthuang_1d
define_hist_bins
define_hist_bins_from_data

"""

import logging
import numpy as np
from scipy import signal, sparse

from . import utils

# Housekeeping for logging
logger = logging.getLogger(__name__)

##


def frequency_stats(imf, sample_rate, method,
                    smooth_phase=31):
    """
    Compute instantaneous phase, frequency and amplitude from a set of IMFs.
    Several approaches are implemented from [1]_ and [2]_.

    Parameters
    ----------
    imf : ndarray
        Input array of IMFs.
    sample_rate : scalar
        Sampling frequency of the signal in Hz
    method : {'hilbert','quad','direct_quad','nht'}
        The method for computing the frequency stats
    smooth_phase : integer
         Length of window when smoothing the unwrapped phase (Default value = 31)

    Returns
    -------
    IP : ndarray
        Array of instantaneous phase estimates
    IF : ndarray
        Array of instantaneous frequency estimates
    IA : ndarray
        Array of instantaneous amplitude estimates

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193
    .. [2] Huang, N. E., Wu, Z., Long, S. R., Arnold, K. C., Chen, X., & Blank,
       K. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096

    """
    logger.info('STARTED: compute frequency stats')
    logger.debug('computing on {0} samples over {1} imfs at sample rate {2}'.format(imf.shape[0],
                                                                                    imf.shape[1],
                                                                                    sample_rate))

    # Each case here should compute the analytic form of the imfs and the
    # instantaneous amplitude.
    if method == 'hilbert':
        logger.info('Using Hilbert transform')

        analytic_signal = signal.hilbert(imf, axis=0)

        # Estimate instantaneous amplitudes directly from analytic signal
        iamp = np.abs(analytic_signal)

    elif method == 'nht':
        logger.info('Using Amplitude-Normalised Hilbert transform')

        n_imf = utils.amplitude_normalise(imf)
        analytic_signal = signal.hilbert(n_imf, axis=0)

        orig_dim = imf.ndim
        if imf.ndim == 2:
            imf = imf[:, :, None]

        # Estimate inst amplitudes with spline interpolation
        iamp = np.zeros_like(imf)
        for ii in range(imf.shape[1]):
            for jj in range(imf.shape[2]):
                iamp[:, ii, jj] = utils.interp_envelope(imf[:, ii, jj],
                                                        mode='upper')
        if orig_dim == 2:
            iamp = iamp[:, :, 0]

    elif method == 'quad':
        logger.info('Using Quadrature transform')

        analytic_signal = quadrature_transform(imf)

        orig_dim = imf.ndim
        if imf.ndim == 2:
            imf = imf[:, :, None]

        # Estimate inst amplitudes with spline interpolation
        iamp = np.zeros_like(imf)
        for ii in range(imf.shape[1]):
            for jj in range(imf.shape[2]):
                iamp[:, ii, jj] = utils.interp_envelope(imf[:, ii, jj],
                                                        mode='upper')

        if orig_dim == 2:
            iamp = iamp[:, :, 0]

    elif method == 'direct_quad':
        logger.info('Using Direct-Quadrature transform')
        raise ValueError('direct_quad method is broken!')

        n_imf = utils.amplitude_normalise(imf.copy())
        iphase = np.unwrap(phase_angle(n_imf))

        iamp = np.zeros_like(imf)
        for ii in range(imf.shape[1]):
            iamp[:, ii] = utils.interp_envelope(imf[:, ii, None], mode='combined')

    else:
        logger.error("Method '{0}' not recognised".format(method))
        raise ValueError("Method '{0}' not recognised\nPlease use one of 'hilbert','nht' or 'quad'".format(method))

    # Compute unwrapped phase for frequency estimation
    iphase = phase_from_complex_signal(
        analytic_signal, smoothing=smooth_phase, ret_phase='unwrapped')
    ifreq = freq_from_phase(iphase, sample_rate)

    # Return wrapped phase
    iphase = utils.wrap_phase(iphase)

    logger.info('COMPLETED: compute frequency stats. Returning {0} imfs'.format(iphase.shape[1]))
    return iphase, ifreq, iamp

# Frequency stat utils


def quadrature_transform(X):
    """
    Compute the quadrature transform on a set of time-series as defined in
    equation 34 of [1]_. The return is a complex array with the input data as
    the real part and the quadrature transform as the imaginary part.

    Parameters
    ----------
    X : ndarray
        Array containing time-series to transform

    Returns
    -------
    quad_signal : ndarray
        Complex valued array containing the quadrature transformed signal

    References
    ----------
    .. [1] Huang, N. E., Wu, Z., Long, S. R., Arnold, K. C., Chen, X., & Blank,
       K. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096

    """

    nX = utils.amplitude_normalise(X.copy(), clip=True)

    imagX = np.lib.scimath.sqrt(1 - np.power(nX, 2)).real

    mask = ((np.diff(nX, axis=0) > 0) * -2) + 1
    mask[mask == 0] = -1
    mask = np.r_[mask, mask[-1, None, :]]

    q = imagX * mask

    return nX + 1j * q


def phase_from_complex_signal(complex_signal, smoothing=None,
                              ret_phase='wrapped', phase_jump='ascending'):
    """
    Compute the instantaneous phase from a complex signal obtained from either
    the Hilbert Transform or by Direct Quadrature.

    Parameters
    ----------
    complex_signal : complex ndarray
        Complex valued input array
    smoothing : int
         Integer window length used in phase smoothing (Default value = None)
    ret_phase : {'wrapped','unwrapped'}
         Flag indicating whether to return the wrapped or unwrapped phase (Default value = 'wrapped')
    phase_jump : {'ascending','peak','descending','trough'}
         Flag indicating where in the cycle the phase jump should be (Default value = 'ascending')

    Returns
    -------
    IP : ndarray
        Array of instantaneous phase values

    """

    # Compute unwrapped phase
    iphase = np.unwrap(np.angle(complex_signal), axis=0)

    orig_dim = iphase.ndim
    if iphase.ndim == 2:
        iphase = iphase[:, :, None]

    # Apply smoothing if requested
    # if smoothing is not None:
    #    iphase = signal.savgol_filter(iphase,smoothing,1,axis=0)
    if smoothing is not None:
        for ii in range(iphase.shape[1]):
            for jj in range(iphase.shape[2]):
                iphase[:, ii, jj] = signal.medfilt(iphase[:, ii, jj], 5)

    if orig_dim == 2:
        iphase = iphase[:, :, 0]

    # Set phase jump point to requested part of cycle
    if phase_jump == 'ascending':
        iphase = iphase + np.pi / 2
    elif phase_jump == 'peak':
        pass  # do nothing
    elif phase_jump == 'descending':
        iphase = iphase - np.pi / 2
    elif phase_jump == 'trough':
        iphase = iphase + np.pi

    if ret_phase == 'wrapped':
        return utils.wrap_phase(iphase)
    elif ret_phase == 'unwrapped':
        return iphase


def freq_from_phase(iphase, sample_rate):
    """
    Compute the instantaneous frequency from the differential of the
    instantaneous phase.

    Parameters
    ----------
    iphase : ndarray
        Input array containing the unwrapped instantaneous phase time-course
    sample_rate : scalar
        The sampling frequency of the data

    Returns
    -------
    IF : ndarray
        Array containing the instantaneous frequencies

    """

    # Differential of instantaneous phase
    iphase = np.gradient(iphase, axis=0)

    # Convert to freq
    ifrequency = iphase / (2.0 * np.pi) * sample_rate

    return ifrequency


def phase_from_freq(ifrequency, sample_rate, phase_start=-np.pi):
    """
    Compute the instantaneous phase of a signal from its instantaneous phase.

    Parameters
    ----------
    ifrequency : ndarray
        Input array containing the instantaneous frequencies of a signal
    sample_rate : scalar
        The sampling frequency of the data
    phase_start : scalar
         Start value of the phase output (Default value = -np.pi)

    Returns
    -------
    IP : ndarray
        The instantaneous phase of the signal

    """

    iphase_diff = (ifrequency / sample_rate) * (2 * np.pi)

    iphase = phase_start + np.cumsum(iphase_diff, axis=0)

    return iphase


def direct_quadrature(fm):
    """Section 3.2 of 'on instantaneous frequency'
    Compute the quadrature transform on a set of time-series as defined in
    equation 35 of [1].

    THIS IS IN DEVELOPMENT

    Parameters
    ----------
    fm :


    Returns
    -------

    References
    ----------
    .. [1] Huang, N. E., Wu, Z., Long, S. R., Arnold, K. C., Chen, X., & Blank,
       K. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096

    """
    ph = phase_angle(fm)

    # We'll have occasional nans where fm==1 or -1
    inds = np.argwhere(np.isnan(ph))

    vals = (ph[inds[:, 0] - 1, :] + ph[inds[:, 0] + 1, :]) / 2
    ph[inds[:, 0]] = vals

    return ph


def phase_angle(fm):
    """
    Compute the quadrature transform on a set of time-series as defined in
    equation 35 of [1]_.

    THIS IS IN DEVELOPMENT

    Parameters
    ----------
    X : ndarray
        Array containing time-series to transform

    Returns
    -------
    quad_signal : ndarray
        Complex valued array containing the quadrature transformed signal

    References
    ----------
    .. [1] Huang, N. E., Wu, Z., Long, S. R., Arnold, K. C., Chen, X., & Blank,
       K. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096


    """

    return np.arctan(fm / np.lib.scimath.sqrt(1 - np.power(fm, 2)))

# Time-frequency spectra


def holospectrum(infr, infr2, inam2, freq_edges, freq_edges2, mode='energy',
                 squash_time='sum'):
    """
    Compute the Holospectrum from the first and second layer frequecy
    statistics of a dataset. The Holospectrum represents the energy of a signal
    across time, carrier frequency and amplitude-modulation frequency [1]_.

    Parameters
    ----------
    infr : ndarray
        2D first level instantaneous frequencies
    infr2 : ndarray
        3D second level instantaneous frequencies
    inam2 : ndarray
        3D second level instantaneous amplitudes
    freq_edges : ndarray
        Vector of frequency bins for carrier frequencies
    freq_edges2 :
        Vector of frequency bins for amplitude-modulation frequencies
    mode : {'energy','amplitude'}
         Flag indicating whether to sum the energy or amplitudes (Default value = 'energy')
    squash_time : {'sum','mean',False}
         Flag indicating whether to marginalise over the time dimension
         (Default value = 'sum')

    Returns
    -------
    holo : ndarray
        Holospectrum of input data.

    Notes
    -----
    Output will be a 3D [samples x am_freq x carrier_freq] array if squash_time
    is False and a 2D [ am_freq x carrier_freq ] array if squash_time is true.


    References
    ----------
    .. [1] Huang, N. E., Hu, K., Yang, A. C. C., Chang, H.-C., Jia, D., Liang,
       W.-K., … Wu, Z. (2016). On Holo-Hilbert spectral analysis: a full
       informational spectral representation for nonlinear and non-stationary
       data. Philosophical Transactions of the Royal Society A: Mathematical,
       Physical and Engineering Sciences, 374(2065), 20150206.
       https://doi.org/10.1098/rsta.2015.0206

    """
    logger.info('STARTED: compute Holospectrum')
    logger.debug('computing on {0} samples over {1} first-level \
                  IMFs and {2} second level IMFs'.format(infr2.shape[0],
                                                         infr2.shape[1],
                                                         infr2.shape[2]))
    logger.debug('First level freq bins: {0} to {1} in {2} steps'.format(freq_edges[0],
                                                                         freq_edges[1],
                                                                         len(freq_edges)))
    logger.debug('Second level freq bins: {0} to {1} in {2} steps'.format(freq_edges2[0],
                                                                          freq_edges2[1],
                                                                          len(freq_edges2)))

    if mode == 'energy':
        inam2 = inam2**2

    IA_inds = np.digitize(infr2, freq_edges2)
    infr_inds = np.digitize(infr, freq_edges)

    new_shape = (infr_inds.shape[0], infr_inds.shape[1], infr2.shape[2])
    infr_inds = np.broadcast_to(infr_inds[:, :, None], new_shape)

    fold_dim1 = len(freq_edges) + 1
    fold_dim2 = len(freq_edges2) + 1

    infr_inds = infr_inds + IA_inds * fold_dim1

    T_inds = np.arange(infr.shape[0])[:, None, None]
    T_inds = np.broadcast_to(T_inds, new_shape)

    coords = (T_inds.reshape(-1), infr_inds.reshape(-1))
    holo = sparse.coo_matrix((inam2.reshape(-1), coords),
                             shape=(infr.shape[0], fold_dim1 * fold_dim2))

    # Reduce time-dimension if specified
    if squash_time is False:
        # Return the full matrix
        holo = holo.toarray().reshape(new_shape[0], fold_dim2, fold_dim1)
        logger.info('Returning full 3D Holospectrum')
    elif squash_time == 'mean':
        # Collapse time dimension while we're still sparse
        holo = holo.mean(axis=0)
        holo = holo.reshape(fold_dim2, fold_dim1)
        logger.info('Returning 2D Holospectrum averaged over time')
    elif squash_time == 'sum':
        # Collapse time dimension while we're still sparse
        holo = holo.sum(axis=0)
        holo = holo.reshape(fold_dim2, fold_dim1)
        logger.info('Returning 2D Holospectrum summed over time')

    if squash_time is False:
        holo = np.array(holo[:, 1:-1, 1:-1])
    else:
        # Alays returns full-array until someone implements N-D sparse in scipy
        holo = np.array(holo[1:-1, 1:-1])  # don't return a matrix

    logger.info('COMPLETED: Holospectrum - output size {0}'.format(holo.shape))
    return holo


def hilberthuang(infr, inam, freq_edges, mode='energy', return_sparse=False):
    """
    Compute the Hilbert-Huang transform from the instataneous frequency
    statistics of a dataset. The Hilbert-Huang Transform represents the energy
    of a signal across time and frequency [1]_.

    Parameters
    ----------
    infr : ndarray
        2D first level instantaneous frequencies
    inam : ndarray
        2D first level instantaneous amplitudes
    freq_edges : ndarray
        Vector of frequency bins for carrier frequencies
    mode : {'energy','amplitude'}
         Flag indicating whether to sum the energy or amplitudes (Default value = 'energy')
    return_sparse : bool
         Flag indicating whether to return the full or sparse form(Default value = True)

    Returns
    -------
    hht : ndarray
        2D array containing the Hilbert-Huang Transform

    Notes
    -----
    If return_sparse is set to True the returned array is a sparse matrix in
    COOrdinate form (scipy.sparse.coo_matrix), also known as 'ijv' or 'triplet'
    form. This is much more memory efficient than the full form but may not
    behave as expected in functions expecting full arrays.

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """

    if infr.ndim == 1:
        infr = infr[:, np.newaxis]

    if inam.ndim == 1:
        inam = inam[:, np.newaxis]

    logger.info('STARTED: compute Hilbert-Huang Transform')
    logger.debug('computing on {0} samples over {1} IMFs '.format(infr.shape[0],
                                                                  infr.shape[1]))
    logger.debug('Freq bins: {0} to {1} in {2} steps'.format(freq_edges[0],
                                                             freq_edges[1],
                                                             len(freq_edges)))

    if mode == 'energy':
        inam = inam**2

    # Create sparse co-ordinates
    yinds = np.digitize(infr, freq_edges) - 1
    yinds[yinds < 0] = 0
    xinds = np.tile(np.arange(yinds.shape[0]), (yinds.shape[1], 1)).T

    coo_data = (inam.reshape(-1), (yinds.reshape(-1), xinds.reshape(-1)))

    # Remove values outside our bins
    goods = np.any(np.c_[coo_data[1][0] < len(freq_edges) - 1, (coo_data[1][0] == 0)], axis=1)
    coo_data = (coo_data[0][goods], (coo_data[1][0][goods], coo_data[1][1][goods]))

    # Create sparse matrix
    hht = sparse.coo_matrix(coo_data, shape=(len(freq_edges) - 1, xinds.shape[0]))

    logger.info('COMPLETED: Hilbert-Huang Transform - output size {0}'.format(hht.shape))
    if return_sparse:
        return hht
    else:
        return hht.toarray()


def hilberthuang_1d(infr, inam, freq_edges, mode='energy'):
    """
    Compute the Hilbert-Huang transform from the instataneous frequency
    statistics of a dataset. The 1D Hilbert-Huang Transform represents the
    energy in a signal across frequencies and IMFs [1]_.

    Parameters
    ----------
    infr : ndarray
        2D first level instantaneous frequencies
    inam : ndarray
        2D first level instantaneous amplitudes
    freq_edges : ndarray
        Vector of frequency bins for carrier frequencies
    mode : {'energy','amplitude'}
         Flag indicating whether to sum the energy or amplitudes (Default value = 'energy')

    Returns
    -------
    specs : ndarray
        2D array containing Hilbert-Huang Spectrum [ frequencies x imfs ]

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """

    specs = np.zeros((len(freq_edges) - 1, infr.shape[1]))

    # Remove values outside the bin range
    infr = infr.copy()  # Don't work in place on input freqs
    outside_inds = (infr < freq_edges[0]) + (infr > freq_edges[-1])
    infr[outside_inds] = np.nan

    finds = np.digitize(infr, freq_edges)

    for ii in range(1, len(freq_edges)):
        for jj in range(infr.shape[1]):

            if mode == 'amplitude':
                specs[ii - 1, jj] = np.nansum(inam[finds[:, jj] == ii, jj])
            elif mode == 'energy':
                specs[ii - 1, jj] = np.nansum(np.power(inam[finds[:, jj] == ii, jj], 2))

    return specs


def define_hist_bins(data_min, data_max, nbins, scale='linear'):
    """
    Define the bin edges and centre values for use in a histogram

    Parameters
    ----------
    data_min : scalar
        Value for minimum edge
    data_max : scalar
        Value for maximum edge
    nbins : integer
        Number of bins to create
    scale : {'linear','log'}
         Flag indicating whether to use a linear or log spacing between bins (Default value = 'linear')

    Returns
    -------
    edges : ndarray
        1D array of bin edges
    centres : ndarray
        1D array of bin centres

    Notes
    >> edges,centres = emd.spectra.define_hist_bins( 1, 5, 3 )
    >> print(edges)
    [1. 2. 3. 4. 5.]
    >> print(centres)
    [1.5 2.5 3.5 4.5]

    """

    if scale == 'log':
        p = np.log([data_min, data_max])
        edges = np.linspace(p[0], p[1], nbins + 1)
        edges = np.exp(edges)
    elif scale == 'linear':
        edges = np.linspace(data_min, data_max, nbins + 1)
    else:
        raise ValueError('scale \'{0}\' not recognised. please use \'log\' or \'linear\'.')

    # Get centre frequecy for the bins
    centres = np.array([(edges[ii] + edges[ii + 1]) / 2 for ii in range(len(edges) - 1)])

    return edges, centres


def define_hist_bins_from_data(X, nbins=None, mode='sqrt', scale='linear'):
    """Find the bin edges and centre frequencies for use in a histogram

    if nbins is defined, mode is ignored

    Parameters
    ----------
    X : ndarray
        Dataset whose summary stats will define the histogram
    nbins : int
         Number of bins to create, if undefined this is derived from the data (Default value = None)
    mode : {'sqrt'}
         Method for deriving number of bins if nbins is undefined (Default value = 'sqrt')
    scale : {'linear','log'}
         (Default value = 'linear')

    Returns
    -------
    edges : ndarray
        1D array of bin edges
    centres : ndarray
        1D array of bin centres

    """

    data_min = X.min()
    data_max = X.max()

    if nbins is None:
        if mode == 'sqrt':
            nbins = np.sqrt(X.shape[0]).astype(int)
        else:
            raise ValueError('mode {0} not recognised, please use \'sqrt\'')

    return define_hist_bins(data_min, data_max, nbins, scale=scale)
