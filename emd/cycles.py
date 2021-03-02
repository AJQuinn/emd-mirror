#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Identification and analysis of cycles in an oscillatory signal.

Routines:
  bin_by_phase
  phase_align
  get_cycle_vector
  get_cycle_stat
  get_chain_stat
  get_control_points
  get_cycle_chain
  mean_vector
  basis_project


"""

import re
import warnings
import numpy as np
from scipy import interpolate as interp

from . import spectra, utils, sift, _cycles_support
from .support import ensure_equal_dims, ensure_vector, ensure_2d, ensure_1d_with_singleton

# Housekeeping for logging
import logging
logger = logging.getLogger(__name__)


###################################################
# CYCLE IDENTIFICATION


def get_cycle_inds(*args, **kwargs):
    msg = "WARNING: 'emd.cycles.get_cycle_inds' is deprecated and " + \
          "will be removed in a future version of EMD. Please change to use " + \
          "'emd.cycles.get_cycle_vector' to remove this warning and " + \
          "future-proof your code"

    warnings.warn(msg)
    logger.warning(msg)
    return get_cycle_vector(*args, **kwargs)


def get_cycle_vector(phase, return_good=True, mask=None,
                     imf=None, phase_step=1.5 * np.pi,
                     phase_edge=np.pi / 12):
    """
    Identify cycles within a instantaneous phase time-course and, optionally,
    remove 'bad' cycles by a number of criteria.

    Parameters
    ----------
    phase : ndarray
        Input vector of Instantaneous Phase values
    return_good : bool
        Boolean indicating whether 'bad' cycles should be removed (Default value = True)
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
    2 : A phase starting within phase_step of zero (ie 0 < x < phase_edge)
    3 : A phase ending within phase_step of 2pi (is 2pi-phase_edge < x < 2pi)
    4 : A set of 4 unique control points (asc-zero, peak, desc-zero & trough)

    Good cycles can be idenfied with:
    >> good_cycles = emd.utils.get_cycle_vector( phase )

    The total number of cycles is then
    >> good_cycles.max()

    Indices where good cycles is zero do not contain a valid cycle
    bad_segments = good_cycles>0

    A single cycle can be isolated by matching its index, eg for the 5th cycle
    cycle_5_inds = good_cycles==5


    """

    # Preamble
    logger.info('STARTED: get cycle indices')
    if mask is not None:
        phase, mask = ensure_2d([phase, mask], ['phase', 'mask'], 'get_cycle_vector')
        ensure_equal_dims((phase, mask), ('phase', 'mask'), 'get_cycle_vector', dim=0)
    else:
        phase = ensure_2d([phase], ['phase'], 'get_cycle_vector')

    logger.debug('computing on {0} samples over {1} IMFs '.format(phase.shape[0],
                                                                  phase.shape[1]))
    if mask is not None:
        logger.debug('{0} ({1}%) samples masked out'.format(mask.sum(), np.round(100*(mask.sum()/phase.shape[0]), 2)))

    # Main body

    if phase.max() > 2 * np.pi:
        print('Wrapping phase')
        phase = utils.wrap_phase(phase)

    cycles = np.zeros_like(phase, dtype=int) - 1

    for ii in range(phase.shape[1]):

        inds = np.where(np.abs(np.diff(phase[:, ii])) > phase_step)[0] + 1

        # No Cycles to be found
        if len(inds) == 0:
            continue

        # Include first and last cycles,
        # These are likely to be bad/incomplete in real data but we should
        # check anyway
        if inds[0] >= 1:
            inds = np.r_[0, inds]
        if inds[-1] <= phase.shape[0] - 1:
            inds = np.r_[inds, phase.shape[0] - 1]

        unwrapped = np.unwrap(phase[:, ii], axis=0)

        count = 0
        for jj in range(len(inds) - 1):

            if mask is not None:
                # Ignore cycle if a part of it is masked out
                if any(~mask[inds[jj]:inds[jj + 1]]):
                    continue

            phase = unwrapped[inds[jj]:inds[jj + 1]]

            if return_good:
                cycle_checks = is_good(phase, ret_all_checks=True, phase_edge=phase_edge)
            else:
                # Pretend eveything is ok
                cycle_checks = np.ones((4,), dtype=bool)

            # Add cycle to list if the checks are good
            if all(cycle_checks):
                cycles[inds[jj]:inds[jj + 1], ii] = count
                count += 1

        logger.info('found {0} cycles in IMF-{1}'.format(cycles[:, ii].max(), ii))

    logger.info('COMPLETED: get cycle indices')
    return cycles


def get_subset_vector(valids):
    subset_vect = np.zeros_like(valids).astype(int) - 1
    count = 0
    for ii in range(len(valids)):
        if valids[ii] == 0:
            subset_vect[ii] = -1
        else:
            subset_vect[ii] = count
            count += 1
    return subset_vect


def get_chain_vector(subset_vect):

    chain_inds = np.where(subset_vect > -1)[0]
    dchain_inds = np.r_[1, np.diff(chain_inds)]
    chainv = np.zeros_like(chain_inds)-1

    count = 0
    for ii in range(len(chain_inds)):
        if dchain_inds[ii] == 1:
            chainv[ii] = count
        elif dchain_inds[ii] > 1:
            count += 1
            chainv[ii] = count
    return chainv


def get_cycle_vector_from_waveform(imf, cycle_start='peaks'):
    """
    ASSUMING LOCALLY SYMMETRICAL SIGNALS!!
    """
    imf = ensure_1d_with_singleton([imf], ['imf'], 'get_cycle_vector_from_waveform')

    if cycle_start == 'desc':
        print("'desc' is Not implemented yet")
        raise ValueError

    cycles = np.zeros_like(imf)
    for ii in range(imf.shape[1]):
        peak_loc, peak_mag = sift._find_extrema(imf[:, ii])
        trough_loc, trough_mag = sift._find_extrema(-imf[:, ii])
        trough_mag = -trough_mag

        for jj in range(len(peak_loc)-1):
            if cycle_start == 'peaks':
                start = peak_loc[jj]
                cycles[peak_loc[jj]:peak_loc[jj+1], ii] = jj+1
            elif cycle_start == 'asc':
                pk = peak_loc[jj]
                tr_ind = np.where(trough_loc - peak_loc[jj] < 0)[0][-1]
                tr = trough_loc[tr_ind]
                if (imf[tr, ii] > 0) or (imf[pk, ii] < 0):
                    continue
                start = np.where(np.diff(np.sign(imf[tr:pk, ii])) == 2)[0][0] + tr

                pk = peak_loc[jj+1]
                tr_ind = np.where(trough_loc - peak_loc[jj+1] < 0)[0][-1]
                tr = trough_loc[tr_ind]
                if (imf[tr, ii] > 0) or (imf[pk, ii] < 0):
                    continue
                stop = np.where(np.diff(np.sign(imf[tr:pk, ii])) == 2)[0][0] + tr

                cycles[start:stop, ii] = jj+1
            elif cycle_start == 'troughs':
                start = trough_loc[jj]
                cycles[trough_loc[jj]:trough_loc[jj+1], ii] = jj+1
            elif cycle_start == 'desc':
                pass

    return cycles.astype(int)


def is_good(phase, waveform=None, ret_all_checks=False, phase_edge=np.pi/12, mode='cycle'):

    cycle_checks = np.zeros((4,), dtype=bool)

    if mode == 'augmented':
        phase = np.unwrap(phase) - 2*np.pi
        phase_min = -np.pi/2
    else:
        phase_min = 0

    # Check for postively increasing phase
    if np.all(np.diff(phase) > 0):
        cycle_checks[0] = True

    # Check that start of cycle is close to 0
    if (phase[0] >= phase_min and phase[0] <= phase_min + phase_edge):
        cycle_checks[1] = True

    # Check that end of cycle is close to pi
    if (phase[- 1] <= 2 * np.pi) and (phase[- 1] >= 2 * np.pi - phase_edge):
        cycle_checks[2] = True

    if waveform is not None:
        # Check we find 5 sensible control points if imf is provided
        try:
            # Should extend this to cope with multiple peaks etc
            ctrl = (0, sift._find_extrema(waveform)[0][0],
                    np.where(np.gradient(np.sign(waveform)) == -1)[0][0],
                    sift._find_extrema(-waveform)[0][0],
                    len(waveform))
            if len(ctrl) == 5 and np.all(np.sign(np.diff(ctrl))):
                cycle_checks[3] = True
        except IndexError:
            # Sometimes we don't find any candidate for a control point
            cycle_checks[3] = False
    else:
        # No time-series so assume everything is fine
        cycle_checks[3] = True

    if ret_all_checks:
        return cycle_checks
    else:
        return np.all(cycle_checks)


###################################################
# CYCLE FEATURES

def get_cycle_stat(cycles, values, mode='compressed', func=np.mean):
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
    func : function
        Function to call on the data in values for each cycle (Default
        np.mean). This can be any function, built-in or user defined, that
        processes a single vector of data returning a single value.

    Returns
    -------
    ndarray
        Array containing the cycle-averaged values


    """
    # Preamble
    logger.info('STARTED: get cycle stats')
    cycles = _ensure_cycle_inputs(cycles)
    values = ensure_vector([values], ['values'], 'get_cycle_stat')

    if cycles.nsamples != values.shape[0]:
        raise ValueError("Mismatched inputs between 'cycles' and 'values'")

    logger.debug('computing stats for {0} cycles over {1} samples'.format(cycles.ncycles, values.shape[0]))
    logger.debug('computing metric {0} and returning {1}-array'.format(func, mode))

    # Main Body

    if mode == 'compressed':
        out = np.zeros((cycles.ncycles, )) * np.nan
    elif mode == 'full':
        out = np.zeros_like(values) * np.nan

    for cind, cycle_inds in cycles:
        if cycle_inds is None:
            continue
        stat = func(values[cycle_inds])
        if mode == 'compressed':
            out[cind] = stat
        elif mode == 'full':
            out[cycle_inds] = stat

    logger.info('COMPLETED: get cycle stats')
    return out


def get_chain_stat(chains, var, func=np.mean):
    """
    Compute a given function for observations across each chain of cycles.

    Parameters
    ----------
    chains : list
        Nested list of cycle indices. Output of emd.cycles.get_cycle_chain.
    var : ndarray
        1d array properties across all good cycles. Compressed output
        of emd.cycles.get_cycle_stat
    func : function
        Function to call on the data in values for each cycle (Default
        np.mean). This can be any function, built-in or user defined, that
        processes a single vector of data returning a single value.

    Returns
    -------
    stat : ndarray
        1D array of evaluated function on property var across each chain.

    """
    # Preamble
    logger.info('STARTED: get cycle stats')

    logger.debug('computing stats for {0} cycles over {1} chains'.format(len(var), len(chains)))
    logger.debug('computing metric {0}'.format(func))

    # Actual computation
    stat = np.array([func(var[x]) for x in chains])

    logger.info('COMPLETED: get chain stat')
    return stat


def phase_align(ip, x, cycles=None, npoints=48, interp_kind='linear', ii=None, mode='cycle'):
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
    interp_kind : {'linear','nearest','zero','slinear', 'quadratic','cubic','previous', 'next'}
        Type of interpolation to perform. Argument is passed onto
        scipy.interpolate.interp1d. (Default = 'linear')

    Returns
    -------
    ndarray :
        array containing the phase aligned observations

    """

    # Preamble
    logger.info('STARTED: phase-align cycles')

    out = ensure_vector((ip, x), ('ip', 'x'), 'phase_align')
    ip, x = out
    ensure_equal_dims((ip, x), ('ip', 'x'), 'phase_align')

    if cycles is None:
        cycles = get_cycle_vector(ip, return_good=False)
    cycles = _ensure_cycle_inputs(cycles)

    cycles.mode = mode

    if cycles.nsamples != ip.shape[0]:
        raise ValueError("Mismatched inputs between 'cycles' and 'ip'")

    # Main Body

    if mode == 'cycle':
        phase_edges, phase_bins = spectra.define_hist_bins(0, 2 * np.pi, npoints)
    elif mode == 'augmented':
        phase_edges, phase_bins = spectra.define_hist_bins(-np.pi / 2, 2 * np.pi, npoints)

    msg = 'aligning {0} cycles over {1} phase points with {2} interpolation'
    logger.debug(msg.format(cycles.niters,
                 npoints,
                 interp_kind))

    avg = np.zeros((npoints, cycles.niters))
    for cind, cycle_inds in cycles:
        if (ii is not None) and (cind is not ii):
            continue
        if cycle_inds is None:
            continue
        phase_data = ip[cycle_inds].copy()

        if mode == 'augmented':
            phase_data = np.unwrap(phase_data) - 2 * np.pi
            #if phase_data[0] < -1.6  or phase_data[0] > -1.5:
            #    print(phase_data[0])
            #else:
            #    print('ok')

        x_data = x[cycle_inds]

        f = interp.interp1d(phase_data, x_data, kind=interp_kind,
                            bounds_error=False, fill_value='extrapolate')

        avg[:, cind] = f(phase_bins)
        if np.any(avg[:, cind] > 50):
            print(cind)

    logger.info('COMPLETED: phase-align cycles')
    return avg, phase_bins


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

    # Preamble
    ip = ensure_vector([ip], ['ip'], 'bin_by_phase')
    if weights is not None:
        weights = ensure_1d_with_singleton([weights], ['weights'], 'bin_by_phase')
        ensure_equal_dims((ip, x, weights), ('ip', 'x', 'weights'), 'bin_by_phase', dim=0)
    else:
        ensure_equal_dims((ip, x), ('ip', 'x'), 'bin_by_phase', dim=0)

    # Main body

    if bin_edges is None:
        bin_edges, bin_centres = spectra.define_hist_bins(0, 2 * np.pi, nbins)
    else:
        nbins = len(bin_edges) - 1
        bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

    bin_inds = np.digitize(ip, bin_edges)

    out_dims = list((nbins, *x.shape[1:]))
    avg = np.zeros(out_dims) * np.nan
    var = np.zeros(out_dims) * np.nan
    for ii in range(1, nbins):
        inds = bin_inds == ii
        if weights is None:
            avg[ii - 1, ...] = np.average(x[inds, ...], axis=0)
            v = np.average(
                (x[inds, ...] - np.repeat(avg[None, ii - 1, ...], np.sum(inds), axis=0))**2, axis=0)
        else:
            if inds.sum() > 0:
                avg[ii - 1, ...] = np.average(x[inds, ...], axis=0,
                                              weights=weights[inds].dot(np.ones((1, x.shape[1]))))
                v = np.average((x[inds, ...] - np.repeat(avg[None, ii - 1, ...], np.sum(inds), axis=0)**2),
                               weights=weights[inds].dot(np.ones((1, x.shape[1]))), axis=0)
            else:
                v = np.nan

        if variance_metric == 'variance':
            var[ii - 1, ...] = v
        elif variance_metric == 'std':
            var[ii - 1, ...] = np.sqrt(v)
        elif variance_metric == 'sem':
            var[ii - 1, ...] = np.sqrt(v) / np.repeat(np.sqrt(inds.sum()
                                                              [None, ...]), x.shape[0], axis=0)

    return avg, var, bin_centres


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

    phi = np.cos(IP) + 1j * np.sin(IP)
    mv = phi[:, None] * X
    return mv.mean(axis=0)


def basis_project(X, ncomps=1, ret_basis=False):
    """
    Express a set of signals in a simple sine-cosine basis set

    Parameters
    ----------
    IP : ndarray
        Instantaneous Phase values
    X : ndarray
        Observations corresponding to IP values
    ncomps : int
        Number of sine-cosine pairs to express signal in (default=1)
    ret_basis : bool
        Flag indicating whether to return basis set (default=False)

    Returns
    -------
    basis : ndarray
        Set of values in basis dimensions


    """
    nsamples = X.shape[0]
    basis = np.c_[np.cos(np.linspace(0, 2 * np.pi, nsamples)),
                  np.sin(np.linspace(0, 2 * np.pi, nsamples))]

    if ncomps > 1:
        for ii in range(1, ncomps + 1):
            basis = np.c_[basis,
                          np.cos(np.linspace(0, 2 * (ii + 1) * np.pi, nsamples)),
                          np.sin(np.linspace(0, 2 * (ii + 1) * np.pi, nsamples))]
    basis = basis.T

    if ret_basis:
        return basis.dot(X), basis
    else:
        return basis.dot(X)


def get_chain_position(chains, mode='compressed', cycles=None):
    cp = np.concatenate([np.arange(len(c)) for c in chains])
    if mode == 'compressed':
        return cp
    elif mode == 'full':
        return _cycles_support.map_cycles_to_samples(cp, cycles)


###################################################
# CONTROL POINT FEATURES


def get_control_points(x, cycles, interp=False, mode='cycle'):
    """
    Identify sets of control points from identified cycles. The control points
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

    if isinstance(cycles, np.ndarray) and mode == 'augmented':
        raise ValueError

    # Preamble
    x = ensure_vector([x], ['x'], 'get_control_points')
    cycles = _ensure_cycle_inputs(cycles)
    print(type(cycles))  # This should be an IterateCycles instance - ALWAYS!!
    if mode == 'augmented':
        cycles.mode = 'augmented'

    if cycles.nsamples != x.shape[0]:
        raise ValueError("Mismatched inputs between 'cycles' and 'values'")

    # Main Body

    ctrl = list()
    for cind, cycle_inds in cycles:
        if (cycle_inds is None) or (len(cycle_inds) < 5):
            # We need at least 5 samples to compute control points...
            if mode == 'augmented':
                ctrl.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            else:
                ctrl.append((np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        cycle = x[cycle_inds]

        if mode == 'augmented':
            asc = cf_ascending_zero_sample(cycle, interp=interp)
        else:
            asc = None

        pk = cf_peak_sample(cycle, interp=interp)

        desc = cf_descending_zero_sample(cycle, interp=interp)

        tr = cf_trough_sample(cycle, interp=interp)

        # Append to list
        if mode == 'cycle':
            ctrl.append((0, pk, desc, tr, len(cycle)-1))
        elif mode == 'augmented':
            ctrl.append((0, asc, pk, desc, tr, len(cycle)-1))

    # Return as array
    ctrl = np.array(ctrl)
    ctrl[ctrl == None] = np.nan  # noqa: E711

    return ctrl


def get_control_point_metrics(ctrl, normalise=True):

    # Peak to trough ratio
    p2t = (ctrl[:, 2] - (ctrl[:, 4]-ctrl[:, 2]))
    # Ascending to Descending ratio
    a2d = (ctrl[:, 1]+(ctrl[:, 4]-ctrl[:, 3])) - (ctrl[:, 3]-ctrl[:, 1])

    if normalise:
        p2t = p2t / ctrl[:, 4]
        a2d = a2d / ctrl[:, 4]

    return p2t, a2d


def get_control_point_metrics_aug(ctrl):
    """inputs are
    (start, asc, peak, desc, trough, end)
    """

    # Peak to trough ratio ( P / P+T )
    p2t = (ctrl[:, 3] - ctrl[:, 1]) / (ctrl[:, 5]-ctrl[:, 1])
    # Ascending to Descending ratio ( A / A+D )
    a2d = ctrl[:, 2] / ctrl[:, 4]

    return p2t, a2d


###################################################
# FEATURE MATCHING


def kdt_match(x, y, K=15, distance_upper_bound=np.inf):
    """
    Find unique nearest-neighbours between two n-dimensional feature sets.
    Useful for matching two sets of cycles on one or more features (ie
    amplitude and average frequency).

    Rows in x are matched to rows in y. As such - it is good to have (many)
    more rows in y than x if possible.

    This uses a k-dimensional tree to query for the K nearest neighbours and
    returns the closest unique neighbour. If no unique match is found - the row
    is not returned. Increasing K will find more matches but allow matches
    between more distant observations.

    Not advisable for use with more than a handful of features.

    Parameters
    ----------
    x : ndarray
        [ num observations x num features ] array to match to
    y : ndarray
        [ num observations x num features ] array of potential matches
    K : int
        number of potential nearest-neigbours to query

    Returns
    -------
    ndarray
        indices of matched observations in x
    ndarray
        indices of matched observations in y

    """

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    #
    logger.info('Starting KD-Tree Match')
    msg = 'Matching {0} features from y ({1} observations) to x ({2} observations)'
    logger.info(msg.format(x.shape[1], y.shape[0], x.shape[0]))
    logger.debug('K: {0}, distance_upper_bound: {1}'.format(K, distance_upper_bound))

    # Initialise Tree and find nearest neighbours
    from scipy import spatial
    kdt = spatial.cKDTree(y)
    D, inds = kdt.query(x, k=K, distance_upper_bound=distance_upper_bound)

    II = np.zeros_like(inds)
    selected = []
    for ii in range(K):
        # Find unique values and their indices in this column
        uni, uni_inds = _unique_inds(inds[:, ii])
        # Get index of lowest distance match amongst occurrences of each unique value
        ix = [np.argmin(D[uni_inds[jj], ii]) for jj in range(len(uni))]
        # Map closest match index to full column index
        closest_uni_inds = [uni_inds[jj][ix[jj]] for jj in range(len(uni))]
        # Remove duplicates and -1s (-1 indicates distance to neighbour is
        # above threshold)
        uni = uni[(uni != np.inf)]
        # Remove previously selected
        bo = np.array([u in selected for u in uni])
        uni = uni[bo == False]  # noqa: E712
        # Find indices of matches between uniques and values in col
        uni_matches = np.zeros((inds.shape[0],))
        uni_matches[closest_uni_inds] = np.sum(inds[closest_uni_inds, ii, None] == uni, axis=1)
        # Remove matches which are selected in previous columns
        uni_matches[II[:, :ii].sum(axis=1) > 0] = 0
        # Mark remaining matches with 1s in this col
        II[np.where(uni_matches)[0], ii] = 1
        selected.extend(inds[np.where(uni_matches)[0], ii])

        msg = '{0} Matches in layer {1}'
        logger.debug(msg.format(np.sum(uni_matches), ii))

    # Find column index of left-most choice per row (ie closest unique neighbour)
    winner = np.argmax(II, axis=1)
    # Find row index of winner
    final = np.zeros((II.shape[0],), dtype=int)
    for ii in range(II.shape[0]):
        if (np.sum(II[ii, :]) == 1) and (winner[ii] < y.shape[0]) and \
           (inds[ii, winner[ii]] < y.shape[0]):
            final[ii] = inds[ii, winner[ii]]
        else:
            final[ii] = -1  # No good match

    # Remove failed matches
    uni, cnt = np.unique(final, return_counts=True)
    x_inds = np.where(final > -1)[0]
    y_inds = final[x_inds]

    #
    logger.info('Returning {0} matched observations'.format(x_inds.shape[0]))

    return x_inds, y_inds


def _unique_inds(ar):
    """
    Find the unique elements of an array, ignoring shape.
    Adapted from numpy.lib.arraysetops._unique1d
        Original function only returns index of first occurrence of unique value

    """
    ar = np.asanyarray(ar).flatten()
    ar.sort()
    aux = ar

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ar_inds = [np.where(ar == ii)[0] for ii in ar[mask]]

    return ar[mask], ar_inds


###################################################
# CYCLE FEATURE FUNCS

def cf_start_value(x):
    """Return first value in a cycle"""
    return x[0]


def cf_end_value(x):
    """Return last value in a cycle"""
    return x[-1]


def cf_peak_sample(x, interp=True):
    """Compute index of peak in a single cycle."""
    locs, pks = sift._find_extrema(x, parabolic_extrema=interp)
    if len(pks) == 0:
        return None
    else:
        return locs[np.argmax(pks)]


def cf_peak_value(x, interp=True):
    """Compute value at peak in a single cycle."""
    locs, pks = sift._find_extrema(x, parabolic_extrema=interp)
    if len(pks) == 0:
        return None
    else:
        return pks[np.argmax(pks)]


def cf_trough_sample(x, interp=True):
    """Compute index of trough in a single cycle."""
    locs, trs = sift._find_extrema(-x, parabolic_extrema=interp)
    trs = -trs
    if len(trs) == 0:
        return None
    else:
        return locs[np.argmin(trs)]


def cf_trough_value(x, interp=True):
    """Compute value at trough in a single cycle."""
    locs, trs = sift._find_extrema(-x, parabolic_extrema=interp)
    trs = -trs
    if len(trs) == 0:
        return None
    else:

        return trs[np.argmin(trs)]


def cf_descending_zero_sample(x, interp=True):
    """Compute index of descending zero-crossing in a single cycle."""
    desc = np.where(np.diff(np.sign(x)) == -2)[0]
    if len(desc) == 0:
        return None
    else:
        desc = desc[0]
    if interp:
        interp_ind = np.argmin(np.abs(np.linspace(x[desc], x[desc+1], 1000)))
        desc = desc + np.linspace(0, 1, 1000)[interp_ind]
    return desc


def cf_ascending_zero_sample(x, interp=True):
    """Compute index of ascending zero-crossing in a single cycle."""
    asc = np.where(np.diff(np.sign(x)) == 2)[0]
    if len(asc) == 0:
        return None
    else:
        asc = asc[0]
    if interp:
        interp_ind = np.argmin(np.abs(np.linspace(x[asc], x[asc+1], 1000)))
        asc = asc + np.linspace(0, 1, 1000)[interp_ind]
    return asc


###################################################
# ITERATING OVER CYCLES


def _ensure_cycle_inputs(invar):
    """Take a variable and return a valid iterable cycles class if possible"""
    if isinstance(invar, np.ndarray):
        # Assume we have a cycles vector
        invar = ensure_vector([invar], ['cycles'], '_check_cycle_inputs')
        return IterateCycles(cycle_vect=invar)
    elif isinstance(invar, Cycles):
        return invar.iterate()
    elif isinstance(invar, IterateCycles):
        return invar
    else:
        raise ValueError("'cycles' input not recognised, must be either a cycle-vector or Cycles class")


class IterateCycles:
    def __init__(self, iter_through='cycles', mode='cycle', valids=None,
                 cycle_vect=None, subset_vect=None, chain_vect=None, phase=None):
        self.cycle_vect = cycle_vect
        self.subset_vect = subset_vect
        self.chain_vect = chain_vect
        self.phase = phase
        self.valids = valids

        self.mode = mode
        self.iter_through = iter_through

        if self.cycle_vect is not None:
            self.ncycles = cycle_vect.max() + 1
            self.nsamples = cycle_vect.shape[0]
        if self.subset_vect is not None:
            self.nsubset = subset_vect.max() + 1
        if self.chain_vect is not None:
            self.nchain = chain_vect.max() + 1

    @property
    def niters(self):
        if self.iter_through == 'cycles':
            return self.cycle_vect.max() + 1
        elif self.iter_through == 'valids':
            return self.valids.sum() + 1
        elif self.iter_through == 'subset':
            return self.subset_vect.max() + 1
        elif self.iter_through == 'chains':
            return self.chain_vect.max() + 1

    def __iter__(self):
        if self.iter_through == 'cycles':
            return self.iterate_cycles()
        elif self.iter_through == 'valids':
            return self.iterate_valids()
        elif self.iter_through == 'subset':
            return self.iterate_subset()
        elif self.iter_through == 'chains':
            return self.iterate_chains()
        else:
            raise ValueError

    def iterate_cycles(self):
        for ii in range(self.ncycles):
            if self.mode == 'cycle':
                inds = _cycles_support.map_cycle_to_samples(self.cycle_vect, ii)
                yield ii, inds
            elif self.mode == 'augmented':
                inds = _cycles_support.map_cycle_to_samples_augmented(self.cycle_vect, ii, self.phase)
                yield ii, inds
            else:
                raise ValueError

    def iterate_valids(self):
        for idx, ii in enumerate(np.where(self.valids)[0]):
            if self.mode == 'cycles':
                inds = _cycles_support.map_cycle_to_samples(self.cycle_vect, ii)
                yield idx, inds
            elif self.mode == 'augmented':
                inds = _cycles_support.map_cycle_to_samples_augmented(self.cycle_vect, ii, self.phase)
                if inds is None:
                    continue
                yield idx, inds
            else:
                raise ValueError

    def iterate_subset(self):
        for ii in range(self.nsubset):
            if self.mode == 'cycle':
                inds = _cycles_support.map_subset_to_sample(self.subset_vect, self.cycle_vect, ii)
                yield ii, inds
            elif self.mode == 'augmented':
                inds = _cycles_support.map_subset_to_sample_augmented(self.subset_vect, self.cycle_vect, ii, self.phase)
                yield ii, inds
            else:
                raise ValueError

    def iterate_chains(self):
        for ii in range(self.nchain):
            inds = _cycles_support.map_chain_to_samples(self.chain_vect, self.subset_vect, self.cycle_vect, ii)
            yield ii, inds


###################################################
# THE CYCLES CLASS

class Cycles:

    def __init__(self, IP, phase_step=1.5 * np.pi, phase_edge=np.pi / 12,
                 compute_timings=False, mode='cycle'):
        self.phase = IP
        self.phase_step = phase_step
        self.phase_edge = phase_edge

        self.phase = ensure_1d_with_singleton([IP], ['IP'], 'Cycles')
        self.cycle_vect = get_cycle_vector(self.phase, return_good=False,
                                           phase_step=phase_step, phase_edge=phase_edge)
        self.subset_vect = None
        self.chain_vect = None
        self.mask_conditions = None

        self.ncycles = self.cycle_vect.max() + 1
        self.nsamples = self.phase.shape[0]

        self.metrics = dict()
        self.compute_cycle_metric('is_good', self.phase, is_good, dtype=int)
        if compute_timings:
            self.compute_cycle_timings()

    def __repr__(self):
        if self.subset_vect is None:
            return "{0} ({1} cycles {2} metrics) ".format(type(self),
                                                          self.ncycles,
                                                          len(self.metrics.keys()))
        else:
            msg = "{0} ({1} cycles {2} subset {3} chains - {4} metrics) "
            return msg.format(type(self),
                              self.ncycles,
                              self.subset_vect.max()+1,
                              self.chain_vect.max(),
                              len(self.metrics.keys()))

    def __iter__(self):
        return self.iterate().__iter__()

    def iterate(self, through='cycles', conditions=None, mode='cycle'):
        if conditions is not None:
            valids = self.get_valid_cycles(conditions)
        else:
            valids = None

        looper = IterateCycles(iter_through=through, mode=mode, valids=valids,
                               cycle_vect=self.cycle_vect, subset_vect=self.subset_vect,
                               chain_vect=self.chain_vect, phase=self.phase)
        return looper

    def get_inds_of_cycle(self, ii, mode='cycle'):
        if mode == 'cycle':
            inds = _cycles_support.map_cycle_to_samples(self.cycle_vect, ii)
            return inds
        elif mode == 'augmented':
            inds = _cycles_support.map_cycle_to_samples_augmented(self.cycle_vect, ii, self.phase)
            return inds

    def get_cycle_vector(self, ii, mode='cycle'):
        """Get sample indices of a single cycle"""
        if mode == 'cycle':
            return _cycles_support.map_cycle_to_samples(self.cycle_vect, ii)
        elif mode == 'augmented':
            return _cycles_support.map_cycle_to_samples_augmented(self.cycle_vect, ii, self.phase)
        else:
            raise ValueError

    def compute_cycle_metric(self, name, vals, func, dtype=None, mode='cycle'):
        """Compute a statistic for all cycles and store the result in the Cycle
        object for later use.
        """
        if mode == 'cycle':
            vals = _cycles_support.get_cycle_stat_from_samples(vals, self.cycle_vect, func=func)
        elif mode == 'augmented':
            vals = _cycles_support.get_augmented_cycle_stat_from_samples(vals, self.cycle_vect, self.phase, func=func)
        else:
            raise ValueError

        if dtype is not None:
            vals = vals.astype(dtype)
        self.add_cycle_metric(name, vals)

    def add_cycle_metric(self, name, cycle_vals, dtype=None):
        """Add an externally computed per-cycle metric"""
        if len(cycle_vals) != self.ncycles:
            msg = "Input metrics ({0}) mismatched to existing metrics ({1})"
            return ValueError(msg.format(cycle_vals.shape, self.ncyclee))

        if dtype is not None:
            cycle_vals = cycle_vals.astype(dtype)

        self.metrics[name] = cycle_vals

    def compute_chain_metric(self, name, vals, func, dtype=None):
        """Compute a metric for each chain and store the result in the cycle object"""

        if self.mask_conditions is None:
            raise ValueError

        vals = _cycles_support.get_chain_stat_from_samples(vals, self.chain_vect,
                                                           self.subset_vect, self.cycle_vect, func=func)
        vals = _cycles_support.project_chain_to_cycles(vals, self.chain_vect, self.subset_vect)

        if dtype is not None:
            vals = vals.astype(dtype)

        self.add_cycle_metric(name, vals)

    def compute_cycle_timings(self):
        self.compute_cycle_metric('start_sample',
                                  np.arange(len(self.cycle_vect)),
                                  cf_start_value,
                                  dtype=int)
        self.compute_cycle_metric('stop_sample',
                                  np.arange(len(self.cycle_vect)),
                                  cf_end_value,
                                  dtype=int)
        self.compute_cycle_metric('duration',
                                  self.cycle_vect,
                                  len,
                                  dtype=int)

    def compute_chain_timings(self):
        self.compute_chain_metric('chain_start', np.arange(0, len(self.cycle_vect)), cf_start_value, dtype=int)
        self.compute_chain_metric('chain_end', np.arange(0, len(self.cycle_vect)), cf_end_value, dtype=int)
        self.compute_chain_metric('chain_len_samples', self.cycle_vect, len, dtype=int)

        def _get_chain_len(x):
            return len(np.unique(x))
        self.compute_chain_metric('chain_len_cycles', self.cycle_vect, _get_chain_len, dtype=int)

    def get_metric_dataframe(self, subset=False, conditions=None):
        import pandas as pd
        d = pd.DataFrame.from_dict(self.metrics)

        if subset and (conditions is not None):
            raise ValueError("Please specify either 'subset=True' or a set of conditions")
        elif subset:
            conditions = self.mask_conditions

        if conditions is not None:
            inds = self.get_valid_cycles(conditions) == False  # noqa: E712
            d = d.drop(np.where(inds)[0])
            d = d.reset_index()

        return d

    def get_valid_cycles(self, conditions, ret_separate=False):
        """Find subset of cycles matching specified conditions
        """

        if isinstance(conditions, str):
            conditions = [conditions]

        out = np.zeros((len(self.metrics['is_good']), len(conditions)))
        for idx, c in enumerate(conditions):
            name, func, val = self._parse_condition(c)
            out[:, idx] = func(self.metrics[name], val)

        if ret_separate:
            return out
        else:
            return np.all(out, axis=1)

    def apply_cycle_mask(self, conditions):
        """Set conditions to define subsets + chains"""
        self.mask_conditions = conditions

        valids = self.get_valid_cycles(conditions)
        self.subset_vect = get_subset_vector(valids)
        self.chain_vect = get_chain_vector(self.subset_vect)

        vals = _cycles_support.project_chain_to_cycles(np.arange(self.chain_vect.max()+1),
                                                       self.chain_vect, self.subset_vect)
        self.add_cycle_metric('chain_ind', vals, dtype=int)

    def _parse_condition(self, cond):
        """Parse strings defining conditional statements.
        """
        name = re.split(r'[=<>!]', cond)[0]
        comp = cond[len(name):]

        if comp[:2] == '==':
            func = np.equal
        elif comp[:2] == '!=':
            func = np.not_equal
        elif comp[:2] == '<=':
            func = np.less_equal
        elif comp[:2] == '>=':
            func = np.greater_equal
        elif comp[0] == '<':
            func = np.less
        elif comp[0] == '>':
            func = np.greater
        else:
            print('Comparator not recognised!')

        val = float(comp.lstrip('!=<>'))

        return (name, func, val)
