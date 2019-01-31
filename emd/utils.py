
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
from . import spectra

def amplitude_normalise(X, thresh=1e-10, clip=False, interp_method='pchip',
        max_iters=3):
    """
    Normalise the amplitude envelope of an IMF to be 1. Mutiple runs of
    normalisation are carried out until the desired threshold is reached.

    This uses the method described as part of the AM-FM transform [1]_

    Parameters
    ----------
    X : ndarray
        Input array of IMFs to be normalised
    thresh : scalar
         Threshold for stopping normalisation (Default value = 1e-10)
    clip : bool
         Whether to clip the output between -1 and 1 (Default value = False)
    interp_method : {'pchip','mono_pchip','splrep'}
         Method used to interpolate envelopes (Default value = 'pchip')

    Returns
    -------
    ndarray
        Amplitude normalised IMFs

    References
    ----------
    .. [1] Huang, N. E., Wu, Z., Long, S. R., Arnold, K. C., Chen, X., & Blank,
       K. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096

    """

    # Don't normalise in place
    X = X.copy()

    orig_dim = X.ndim
    if X.ndim == 2:
        X = X[:, :, None]

    container_dim = (X.shape[0], 1, 1)

    for iimf in range(X.shape[1]):
        for jimf in range(X.shape[2]):

            env = interp_envelope(X[:, iimf, jimf], mode='combined', interp_method=interp_method)

            if env is None:
                continue_norm = False
            else:
                continue_norm = True
                #env = env.reshape(*container_dim)

            iters = 0
            while continue_norm and (iters<max_iters):
                iters += 1

                X[:, iimf, jimf] = X[:, iimf, jimf] / env
                env = interp_envelope(X[:, iimf, jimf], mode='combined', interp_method=interp_method)

                if env is None:
                    continue_norm = False
                else:
                    continue_norm = True
                    #env = env.reshape(*container_dim)

                    if np.abs(env.sum()-env.shape[0]) < thresh:
                        continue_norm = False

    if clip:
        # Make absolutely sure nothing daft is happening
        X = np.clip(X, -1, 1)

    if orig_dim == 2:
        X = X[:, :, 0]

    return X

def get_padded_extrema(X, combined_upper_lower=False):
    """
    Return a set of extrema from a signal including padded extrema at the edges
    of the signal.

    Parameters
    ----------
    X : ndarray
        Input signal
    combined_upper_lower : bool
         Flag to indicate whether both upper and lower extrema should be
         considered (Default value = False)

    Returns
    -------
    max_locs : ndarray
        location of extrema in samples
    max_pks : ndarray
        Magnitude of each extrema


    """

    if X.ndim == 2:
        X = X[:, 0]

    if combined_upper_lower:
        max_locs, max_pks = find_extrema(np.abs(X))
    else:
        max_locs, max_pks = find_extrema(X)

    # Return nothing we don't have enough extrema
    if max_locs.size <= 1:
        return None, None

    # Determine how much padding to use
    N = 2 # should make this analytic somehow
    if max_locs.size < N:
        N = max_locs.size

    # Pad peak locations
    ret_max_locs = np.pad(max_locs, N, 'reflect', reflect_type='odd')

    # Pad peak magnitudes
    ret_max_pks = np.pad(max_pks, N, 'median', stat_length=1)

    while max(ret_max_locs) < len(X) or min(ret_max_locs) >= 0:
        ret_max_locs = np.pad(ret_max_locs, N, 'reflect', reflect_type='odd')
        ret_max_pks = np.pad(ret_max_pks, N, 'median', stat_length=1)

    return ret_max_locs, ret_max_pks

def interp_envelope(X, mode='upper', interp_method='splrep'):
    """
    Interpolate the amplitude envelope of a signal.

    Parameters
    ----------
    X : ndarray
        Input signal
    mode : {'upper','lower','combined'}
         Flag to set which envelope should be computed (Default value = 'upper')
    interp_method : {'splrep','pchip','mono_pchip'}
         Flag to indicate which interpolation method should be used (Default value = 'splrep')

    Returns
    -------
    ndarray
        Interpolated amplitude envelope


    """

    if mode == 'upper':
        locs, pks = get_padded_extrema(X, combined_upper_lower=False)
    elif mode == 'lower':
        locs, pks = get_padded_extrema(-X, combined_upper_lower=False)
    elif mode == 'combined':
        locs, pks = get_padded_extrema(X, combined_upper_lower=True)
    else:
        raise ValueError('Mode not recognised. Use mode= \'upper\'|\'lower\'|\'combined\'')

    if locs is None:
        return None

    # Run interpolation on envelope
    t = np.arange(locs[0], locs[-1])
    if interp_method == 'splrep':
        f = interp.splrep(locs, pks)
        env = interp.splev(t, f)
    elif interp_method == 'mono_pchip':
        pchip = interp.PchipInterpolator(locs, pks)
        env = pchip(t)
    elif interp_method == 'pchip':
        pchip = interp.pchip(locs, pks)
        env = pchip(t)

    t_max = np.arange(locs[0], locs[-1])
    tinds = np.logical_and((t_max >= 0), (t_max < X.shape[0]))

    env = np.array(env[tinds])

    if env.shape[0] != X.shape[0]:
        raise ValueError('Envelope length does not match input data {0} {1}'.format(env.shape[0], X.shape[0]))

    if mode == 'lower':
        return -env
    else:
        return env

def find_extrema(X, ret_min=False):
    """
    Identify extrema within a time-course and reject extrema whose magnitude is
    below a set threshold.

    Parameters
    ----------
    X : ndarray
       Input signal
    ret_min : bool
         Flag to indicate whether maxima (False) or minima (True) should be identified(Default value = False)

    Returns
    -------
    locs : ndarray
        Location of extrema in samples
    extrema : ndarray
        Value of each extrema


    """

    if ret_min:
        #ind = signal.argrelextrema( X, np.less)[0]
        ind = signal.argrelmin(X, order=1)[0]
    else:
        #ind = signal.argrelextrema( X, np.greater)[0]
        ind = signal.argrelmax(X, order=1)[0]

    # Only keep peaks with magnitude above machine precision
    if len(ind) / X.shape[0] > 1e-3:
        good_inds = ~(np.isclose(X[ind], X[ind-1]) * np.isclose(X[ind], X[ind+1]))
        ind = ind[good_inds]

    #if ind[0] == 0:
    #    ind = ind[1:]

    #if ind[-1] == X.shape[0]:
    #    ind = ind[:-2]

    return ind, X[ind]

def zero_crossing_count(X):
    """
    Count the number of zero-crossings within a time-course through
    differentiation of the sign of the signal.

    Parameters
    ----------
    X : ndarray
        Input array

    Returns
    -------
    int
        Number of zero-crossings

    """

    if X.ndim == 2:
        X = X[:, None]

    return (np.diff(np.sign(X), axis=0) != 0).sum(axis=0)


def abreu2010(f, nonlin_deg, nonlin_phi, sample_rate, seconds):
    """
    Simulate a non-linear waveform using equation 9 in [1]_.

    Parameters
    ----------
    f : scalar
        Fundamental frequency of generated signal
    nonlin_deg : scalar
        Degree of non-linearity in generated signal
    nonlin_phi : scalar
        Skew in non-linearity of generated signal
    sample_rate : scalar
        The sampling frequency of the generated signal
    seconds : scalar
        The number of seconds of data to generate

    Returns
    -------
    ndarray
        Simulated signal containing non-linear wave

    References
    ----------
    .. [1] Abreu, T., Silva, P. A., Sancho, F., & Temperville, A. (2010).
       Analytical approximate wave form for asymmetric waves. Coastal Engineering,
       57(7), 656–667. https://doi.org/10.1016/j.coastaleng.2010.02.005

    """

    time_vect = np.linspace(0, seconds, seconds*sample_rate)

    factor = np.sqrt(1- nonlin_deg**2)
    num = nonlin_deg*np.sin(nonlin_phi) / 1+np.sqrt(1-nonlin_deg**2)
    num = num + np.sin(2*np.pi*f*time_vect)

    denom = 1 - nonlin_deg * np.cos(2*np.pi*f*time_vect + nonlin_phi)

    return factor * (num / denom)

def est_orthogonality(imf):
    """
    Compute the index of orthogonality as described in equation 6.5 of [1]_.

    Parameters
    ----------
    imf : ndarray
        Input array of IMFs

    Returns
    -------
    ndarray
        Matrix of orthogonality values [nimfs x nimfs]

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """


    ortho = np.ones((imf.shape[1], imf.shape[1])) * np.nan

    for ii in range(imf.shape[1]):
        for jj in range(imf.shape[1]):
            ortho[ii, jj] = np.abs(np.sum(imf[:, ii]*imf[:, jj])) / \
                        (np.sqrt(np.sum(imf[:, jj]*imf[:, jj])) * np.sqrt(np.sum(imf[:, ii]*imf[:, ii])));

    return ortho

def find_extrema_locked_epochs(X, winsize, lock_to='max', percentile=None):
    """
    Helper function for defining epochs around peaks or troughs within the data

    Parameters
    ----------
    X : ndarray
        Input time-series
    winsize : integer
        Width of window to extract around each extrema
    lock_to : {'max','min'}
         Flag to select peak or trough locking (Default value = 'max')
    percentile : scalar
         Optional flag to selection only the upper percentile of extrema by
         magnitude (Default value = None)

    Returns
    -------
    ndarray
        Array of start and end indices for epochs around extrema.

    """

    if lock_to=='max':
        locs, pks = find_extrema(X, ret_min=False)
    else:
        locs, pks = find_extrema(X, ret_min=True)

    if percentile is not None:
        thresh = np.percentile(pks[:, 0], percentile)
        locs = locs[pks[:, 0]>thresh]
        pks = pks[pks>thresh]

    winstep = int(winsize/2)

    trls = np.r_[np.atleast_2d(locs-winstep), np.atleast_2d(locs+winstep)].T

    # Reject trials which start before 0
    inds = trls[:, 0] < 0
    trls = trls[inds==False, :]

    # Reject trials which end after X.shape[0]
    inds = trls[:, 1] > X.shape[0]
    trls = trls[inds==False, :]

    return trls

def apply_epochs(X, trls):
    """
    Apply a set of epochs to a continuous dataset

    Parameters
    ----------
    X : ndarray
        Input dataset to be epoched
    trls : ndarray
        2D array of start and end indices for each epoch. The second dimension
        should be of len==2 and contain start and end indices in order.

    Returns
    -------
    ndarray
        Epoched time-series

    """

    Y = np.zeros((trls[0, 1]-trls[0, 0], X.shape[1], trls.shape[0]))
    for ii in np.arange(trls.shape[0]):

        Y[:, :, ii] = X[trls[ii, 0]:trls[ii, 1], :]

    return Y

def wrap_phase(IP, ncycles=1, mode='2pi'):
    """
    Wrap a phase time-course.

    Parameters
    ----------
    IP : ndarray
        Input array of unwrapped phase values
    ncycles : integer
         Number of cycles per wrap (Default value = 1)
    mode : {'2pi','-pi2pi'}
         Flag to indicate the values to wrap phase within (Default value = '2pi')

    Returns
    -------
    ndarray
        Wrapped phase time-course

    """

    if mode == '2pi':
        phases = (IP) % (ncycles * 2 * np.pi)
    elif mode == '-pi2pi':
        phases = (IP + (np.pi*ncycles)) % (ncycles * 2 * np.pi) - (np.pi*ncycles)

    return phases


## Cycle Metrics

def bin_by_phase(ip, x, nbins=24, weights=None, variance_metric='variance',
                 bin_edges=None):
    """
    Compute distribution of x by phase-bins in the Instantaneous Frequency.

    Parameters
    ----------
    ip : ndarray
        Input vector of instataneous phase values
    x : ndarray
        Input array of values to be binned, first dimension much match length of
        IP
    nbins : integer
         number of phase bins to define (Default value = 24)
    weights : ndarray (optional)
         Optional set of linear weights to apply before averaging (Default value = None)
    variance_metric : {'variance','std','sem'}
         Flag to select whether the variance, standard deviation or standard
         error of the mean in computed across cycles (Default value = 'variance')
    bin_edges : ndarray (optional)
         Optional set of bin edges to override automatic bin specification (Default value = None)

    Returns
    -------
    avg : ndarray
        Vector containing the average across cycles as a function of phase
    var : ndarray
        Vector containing the selected variance metric across cycles as a
        function of phase
    bin_centres : ndarray
        Vector of bin centres

    """

    if bin_edges is None:
        bin_edges, bin_centres = spectra.define_hist_bins(0, 2*np.pi, nbins)
    else:
        nbins = len(bin_edges) - 1
        bin_centres = bin_edges[:-1] + np.diff(bin_edges)/2

    bin_inds = np.digitize(ip, bin_edges)[:, 0]

    out_dims = list((nbins, *x.shape[1:]))
    avg = np.zeros(out_dims)*np.nan
    var = np.zeros(out_dims)*np.nan
    for ii in range(1, nbins):
        inds = bin_inds==ii
        if weights is None:
            avg[ii-1, ...] = np.average(x[inds, ...], axis=0)
            v = np.average((x[inds, ...] - np.repeat(avg[None, ii-1, ...], np.sum(inds), axis=0))**2, axis=0)
        else:
            if inds.sum() > 0:
                avg[ii-1, ...] = np.average(x[inds, ...], axis=0, weights=weights[inds].dot(np.ones((1, x.shape[1]))))
                v = np.average((x[inds, ...] - np.repeat(avg[None, ii-1, ...], np.sum(inds), axis=0)**2),
                               weights=weights[inds].dot(np.ones((1, x.shape[1]))), axis=0)
            else:
                v = np.nan

        if variance_metric=='variance':
            var[ii-1, ...] = v
        elif variance_metric=='std':
            var[ii-1, ...] = np.sqrt(v)
        elif variance_metric=='sem':
            var[ii-1, ...] = np.sqrt(v) / np.repeat(np.sqrt(inds.sum()[None, ...]), x.shape[0], axis=0)

    return avg, var, bin_centres

def phase_align_cycles(ip, x, cycles=None):
    """
    Compute phase alignment of a vector of observed values across a set of cycles.

    Parameters
    ----------
    ip : ndarray
        Input array of Instantaneous Phase values to base alignment on
    x : ndarray
        Input array of observed values to phase align
    cycles : ndarray (optional)
         Optional set of cycles within IP to use (Default value = None)

    Returns
    -------
    ndarray :
        array containing the phase aligned observations

    """

    phase_edges, phase_bins = spectra.define_hist_bins(0, 2*np.pi, 48)

    if cycles is None:
        cycles = get_cycle_inds(ip)

    ncycles = cycles.max()
    avg = np.zeros((48, ncycles))
    for ii in range(1, ncycles+1):

        phase_data = ip[cycles[:, 0]==ii, 0]
        x_data = x[cycles[:, 0]==ii]

        f = interp.interp1d(phase_data, x_data,
                            bounds_error=False, fill_value='extrapolate')

        avg[:, ii-1] = f(phase_bins)

    return avg

def get_cycle_inds(phase, return_good=True, mask=None, imf=None):
    """
    Identify cycles within a instantaneous phase time-course and, optionally,
    remove 'bad' cycles by a number of criteria.

    Parameters
    ----------
    phase : ndarray
        Input vector of Instantaneous Phase values
    return_good : bool
         Boolen indicating whether 'bad' cycles should be removed (Default value = True)
    mask : ndarray
         Vector of mask values that should be ignored (Default value = None)
    imf : ndarray
         Optional array of IMFs to used for control point identification when
         identifying good/bad cycles (Default value = None)

    Returns
    -------
    ndarray
        Vector of integers indexing the location of each cycle

    Notes
    -----
    Good cycles are those with
    1 : A strictly positively increasing phase
    2 : A phase starting within pi/24 of zero
    3 : A phase ending within pi/24 of 2pi
    4 : A set of 4 unqiue control points
            (ascending zero, peak, descending zero & trough)

    Good cycles can be idenfied with:
    >> good_cycles = emd.utils.get_cycle_inds( phase )

    The total number of cycles is then
    >> good_cycles.max()

    Indices where good cycles is zero do not contain a valid cycle
    bad_segments = good_cycles>0

    A single cycle can be isolated by matching its index, eg for the 5th cycle
    cycle_5_inds = good_cycles==5


    """

    if phase.max() > 2*np.pi:
        print('Wrapping phase')
        phase = wrap_phase(phase)

    cycles = np.zeros_like(phase, dtype=int)

    for ii in range(phase.shape[1]):

        inds = np.where(np.abs(np.diff(phase[:, ii])) > 6)[0] + 1
        unwrapped = np.unwrap(phase[:, ii], axis=0)

        count = 1
        for jj in range(len(inds)-1):

            if mask is not None:
                # Ignore cycle if a part of it is masked out
                if any(~mask[inds[jj]:inds[jj+1]]):
                    continue

            dat = unwrapped[inds[jj]:inds[jj+1]];

            if return_good:
                cycle_checks = np.zeros((4,), dtype=bool)

                # Check for postively increasing phase
                if all(np.diff(dat) > 0):
                    cycle_checks[0] = True

                # Check that start of cycle is close to 0
                if (phase[inds[jj], ii] >= 0 and
                    phase[inds[jj], ii] <= np.pi/24):
                    cycle_checks[1] = True

                # Check that end of cycle is close to pi
                if (phase[inds[jj+1]-1, ii] <= 2*np.pi and
                    phase[inds[jj+1]-1, ii] >= 2*np.pi-np.pi/24):
                    cycle_checks[2] = True

                if imf is not None:
                    # Check we find 5 sensible control points if imf is provided
                    try:
                        cycle = imf[inds[jj]:inds[jj+1]]
                        # Should extend this to cope with multiple peaks etc
                        ctrl = (0, find_extrema(cycle)[0][0],
                                np.where(np.gradient(np.sign(cycle))==-1)[0][0],
                                find_extrema(-cycle)[0][0],
                                len(cycle))
                        if len(ctrl) == 5 and np.all(np.sign(np.diff(ctrl))):
                            cycle_checks[3] = True
                    except IndexError:
                        # Sometimes we don't find any candidate for a control point
                        cycle_checks[3] = False
                else:
                    # No time-series so assume everything is fine
                    cycle_checks[3] = True

            else:
                # Pretend eveything is ok
                cycle_checks = np.ones((3,), dtype=bool)

            # Add cycle to list if the checks are good
            if all(cycle_checks):
                cycles[inds[jj]:inds[jj+1], ii] = count;
                count += 1

    return cycles

def get_cycle_vals(cycles, values, mode='compressed'):
    """
    Compute the average of a set of observations for each cycle.

    Parameters
    ----------
    cycles : ndarray
        array whose content index cycle locations
    values : ndarray
        array of observations to average within each cycle
    mode : {'compressed','full'}
         Flag to indicate whether to return a single value per cycle or the
         average values filled within a vector of the same size as values
         (Default value = 'compressed')

    Returns
    -------
    ndarray
        Array containing the cycle-averaged values


    """

    #https://stackoverflow.com/a/39598529
    unq, ids, count = np.unique(cycles, return_inverse=True, return_counts=True)
    vals = np.bincount(ids, values)/count

    if mode == 'full':
        ret = np.zeros_like(cycles, dtype=float)
        ret.fill(np.nan)
        for ii in range(1, cycles.max()+1):
            ret[cycles==ii] = vals[ii]
        vals = ret

    return vals

def get_control_points(x, good_cycles):
    """
    Indentify sets of control points from identified cycles. The control points
    are the ascending zero, peak, descending zero & trough.

    Parameters
    ----------
    x : ndarray
        Input array of oscillatory data
    good_cycles : ndarray
        array whose content index cycle locations

    Returns
    -------
    ndarray
        The control points for each cycle in x


    """

    ctrl = list()
    for ii in range(1, good_cycles.max()):
        cycle = x[good_cycles==ii]

        # Note! we're currently just taking the first peak or trough if there
        # are more than one. This is dumb.
        ctrl.append((0, find_extrema(cycle)[0][0],
                     np.where(np.gradient(np.sign(cycle))==-1)[0][0],
                     find_extrema(-cycle)[0][0],
                     len(cycle)))

    return np.array(ctrl)

def get_cycle_chain(cycles, min_chain=1):
    """
    Indentify chains of valid cycles in a set of cycles.

    Parameters
    ----------
    cycles : ndarray
        array whose content index cycle locations
    min_chain : integer
         Minumum length of chain to return (Default value = 1)

    Returns
    -------
    list
        nested  list of cycle numbers within each chain


    """

    chains = list()
    new_chain = True
    chn = None
    # get diff to next cycle for each cycle
    for ii in range(1, cycles.max()+1):

        di = theta_cycles[np.where(theta_cycles==ii)[0][-1]+1][0] - ii

        if di < 1 or ii == 1:
            if chn is not None:
                if len(chn) >= min_chain:
                    chains.append(chn)
            # Start New chain
            chn = [ii]
        else:
            # Extend current chain
            chn.append(ii)

    return chains
