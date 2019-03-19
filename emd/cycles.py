
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
from . import spectra

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

def phase_align(ip, x, cycles=None, npoints=48, interp_kind='linear'):
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
    npoints : int
        Number of points in the phase cycle to align to (Default = 48)
    interp_kind : {'linear','nearest','zero','slinear',
                   'quadratic','cubic','previous', 'next'}
        Type of interpolation to perform. Argument is passed onto
        scipy.interpolate.interp1d. (Default = 'linear')

    Returns
    -------
    ndarray :
        array containing the phase aligned observations

    """

    phase_edges, phase_bins = spectra.define_hist_bins(0, 2*np.pi, npoints)

    if cycles is None:
        cycles = get_cycle_inds(ip)

    ncycles = cycles.max()
    avg = np.zeros((npoints, ncycles))
    for ii in range(1, ncycles+1):

        phase_data = ip[cycles[:, 0]==ii, 0]
        x_data = x[cycles[:, 0]==ii]

        f = interp.interp1d(phase_data, x_data, kind=interp_kind,
                            bounds_error=False, fill_value='extrapolate')

        avg[:, ii-1] = f(phase_bins)

    return avg

def get_cycle_inds(phase, return_good=True, mask=None,
                          imf=None, phase_step=1.5*np.pi,
                          phase_edge=np.pi/12):
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
    phase_step : scalar
        Minimum value in the differential of the wrapped phase to identify a
        cycle transition (Default value = 1.5*np.pi)
    phase_edge : scalar
        Maximum distance from 0 or 2pi for the first and last phase value in a
        good cycle. Only used when return_good is True
        (Default value = np.pi/12)

    Returns
    -------
    ndarray
        Vector of integers indexing the location of each cycle

    Notes
    -----
    Good cycles are those with
    1 : A strictly positively increasing phase
    2 : A phase starting within phase_step of zero (ie 0 < x < phase_step)
    3 : A phase ending within phase_step of 2pi (is 2pi-phase_step < x < 2pi)
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

        inds = np.where(np.abs(np.diff(phase[:, ii])) > phase_step)[0] + 1

        # No Cycles to be found
        if len(inds) == 0:
            continue

        # Include first and last cycles,
        # These are likely to be bad/incomplete in real data but we should
        # check anyway
        if inds[0] >= 1:
            inds = np.r_[0,inds]
        if inds[-1] <= phase.shape[0]-1:
            inds = np.r_[inds,phase.shape[0]-1]
        print(inds)

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
                    phase[inds[jj], ii] <= phase_edge):
                    cycle_checks[1] = True

                # Check that end of cycle is close to pi
                if (phase[inds[jj+1]-1, ii] <= 2*np.pi and
                    phase[inds[jj+1]-1, ii] >= 2*np.pi-phase_edge):
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
            print(cycle_checks)
            if all(cycle_checks):
                cycles[inds[jj]:inds[jj+1], ii] = count;
                count += 1

    return cycles

def get_cycle_stat(cycles, values, mode='compressed',metric='mean'):
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
    vals = np.bincount(ids, values)

    if metric is 'mean':
        vals = vals / count
    elif metric is not 'sum':
        # We already have the sum so just check the argument is as expected
        raise ValueError('Metric not recognise, please use either \'mean\' or \'sum\'')

    if mode == 'full':
        ret = np.zeros_like(cycles, dtype=float)
        ret.fill(np.nan)
        for ii in range(1, cycles.max()+1):
            ret[cycles==ii] = vals[ii-1]
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
    for ii in range(1, good_cycles.max()+1):
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

def mean_vector(IP, X, mask=None):
    """
    Compute the mean vector of a set of values wrapped around the unit circle.

    Parameters
    ----------
    IP : ndarray
        Instantaneous Phase values
    X : ndarray
        Observations corresponding to IP values
    mask :
         (Default value = None)

    Returns
    -------
    mv : ndarray
        Set of mean vectors


    """

    phi = np.sin(IP) + 1j*np.cos(IP)
    mv = phi[:, None] * IA
    return mv.mean(axis=0)

def kdt_match( x, y, N=15 ):
    """
    Find unique nearest-neighbours between two n-dimensional feature sets.
    Useful for matching two sets of cycles on one or more features (ie
    amplitude and average frequency).

    Rows in x are matched to rows in y. As such - it is good to have (many)
    more rows in y than x if possible.

    This uses a k-dimensional tree to query for the N nearest neighbours and
    returns the closest unique neighbour. If no unique match is found - the row
    is not returned. Increasing N will find more matches but allow matches
    between more distant observations.

    Not advisable for use with more than a handful of features.

    Parameters
    ----------
    x : ndarray
        [ num observations x num features ] array to match to
    y : ndarray
        [ num observations x num features ] array of potential matches
    N : int
        number of potential nearest-neigbours to query

    Returns
    -------
    ndarray
        indices of matched observations in x
    ndarray
        indices of matched observations in y

    """

    kdt = spatial.cKDTree(y)
    D,inds = kdt.query(x,k=N)
    uni,cnt = np.unique(inds,return_counts=True,axis=0)

    II = np.zeros_like(inds)
    selected = []
    for ii in range(N):
        # Find unique values in this column
        uni,cnt = np.unique(inds[:,ii],return_counts=True)
        # Remove duplicates
        uni = uni[cnt==1]
        # Remove previously selected
        bo = np.array([u in selected for u in uni])
        uni = uni[bo==False]
        # Find indices of matches between uniques and values in col
        uni_matches = np.sum(inds[:,ii,None] == uni,axis=1)
        # Remove matches which are selected in previous columns
        uni_matches[II[:,:ii].sum(axis=1)>0] = 0
        # Mark remaining matches with 1s in this col
        II[np.where(uni_matches)[0],ii] = 1
        selected.extend( inds[np.where(uni_matches)[0],ii] )

    # Find column index of left-most choice per row (ie closest unique neighbour)
    winner = np.argmax(II,axis=1)
    # Find row index of winner
    final = np.zeros( (512,), dtype=int)
    for ii in range(512):
        final[ii] = inds[ii,winner[ii]]

    # Still have to remove duplicates
    uni,cnt = np.unique(final,return_counts=True)
    x_inds = np.where(cnt==1)[0]
    y_inds = uni[np.where(cnt==1)[0]]

    return x_inds,y_inds
