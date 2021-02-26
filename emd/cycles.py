#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Identification and analysis of cycles in an oscillatory signal.

Routines:
  bin_by_phase
  phase_align
  get_cycle_inds
  get_cycle_stat
  get_chain_stat
  get_control_points
  get_cycle_chain
  mean_vector
  basis_project


"""
import re
import logging
import numpy as np
from functools import partial
from scipy import interpolate as interp
from scipy import spatial

from . import spectra, utils, sift
from .support import ensure_equal_dims, ensure_vector, ensure_2d, ensure_1d_with_singleton

# Housekeeping for logging
logger = logging.getLogger(__name__)


def bin_by_phase(ip, x, nbins=24, weights=None, variance_metric='variance',
                 bin_edges=None):
    """Compute distribution of x by phase-bins in the Instantaneous Frequency.

    Parameters
    ----------
    ip : ndarray
        Input vector of instataneous phase values
    x : ndarray
        Input array of values to be binned, first dimension much match length of
        IP
    nbins : int
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
    for ii in range(1, nbins+1):
        inds = bin_inds == ii
        if weights is None:
            avg[ii - 1, ...] = np.average(x[inds, ...], axis=0)
            v = np.average(
                (x[inds, ...] - np.repeat(avg[None, ii - 1, ...], np.sum(inds), axis=0))**2, axis=0)
        else:
            if inds.sum() > 0:
                avg[ii - 1, ...] = np.average(x[inds, ...], axis=0,
                                              weights=weights[inds].dot(np.ones((1, x.shape[1]))))
                v = np.average((x[inds, ...] - np.repeat(avg[None, ii - 1, ...],
                                                         np.sum(inds), axis=0)**2),
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


def phase_align(ip, x, cycles=None, npoints=48, interp_kind='linear'):
    """Compute phase alignment of a vector of observed values across a set of cycles.

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

    if cycles is None:
        out = ensure_vector((ip, x), ('ip', 'x'), 'phase_align')
        ip, x = out
        ensure_equal_dims((ip, x), ('ip', 'x'), 'phase_align')
        cycles = get_cycle_inds(ip)
    else:
        out = ensure_vector((ip, x, cycles), ('ip', 'x', 'cycles'), 'phase_align')
        ip, x, cycles = out
        ensure_equal_dims((ip, x, cycles), ('ip', 'x', 'cycles'), 'phase_align')

    # Main Body

    _, phase_bins = spectra.define_hist_bins(0, 2 * np.pi, npoints)

    msg = 'aligning {0} cycles over {1} phase points with {2} interpolation'
    logger.debug(msg.format(cycles.max(), npoints, interp_kind))

    ncycles = cycles.max()
    avg = np.zeros((npoints, ncycles+1))
    for ii in range(ncycles+1):

        phase_data = ip[cycles[:] == ii]
        x_data = x[cycles[:] == ii]

        f = interp.interp1d(phase_data, x_data, kind=interp_kind,
                            bounds_error=False, fill_value='extrapolate')

        avg[:, ii] = f(phase_bins)

    logger.info('COMPLETED: phase-align cycles')
    return avg


def get_cycle_inds(phase, return_good=True, mask=None,
                   imf=None, phase_step=1.5 * np.pi,
                   phase_edge=np.pi / 12):
    """Identify cycles within a instantaneous phase time-course.

    Cycles are located by phase jumps and optionally assessed to remove 'bad'
    cycles by criteria specified in Notes.

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
    phase_step : float
        Minimum value in the differential of the wrapped phase to identify a
        cycle transition (Default value = 1.5*np.pi)
    phase_edge : float
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
    >> good_cycles = emd.utils.get_cycle_inds( phase )

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
        phase, mask = ensure_2d([phase, mask], ['phase', 'mask'], 'get_cycle_inds')
        ensure_equal_dims((phase, mask), ('phase', 'mask'), 'get_cycle_inds', dim=0)
    else:
        phase = ensure_2d([phase], ['phase'], 'get_cycle_inds')

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

            dat = unwrapped[inds[jj]:inds[jj + 1]]

            if return_good:
                cycle_checks = np.zeros((4,), dtype=bool)

                # Check for postively increasing phase
                if all(np.diff(dat) > 0):
                    cycle_checks[0] = True

                # Check that start of cycle is close to 0
                if (phase[inds[jj], ii] >= 0 and phase[inds[jj], ii] <= phase_edge):
                    cycle_checks[1] = True

                # Check that end of cycle is close to pi
                if (phase[inds[jj + 1] - 1, ii] <= 2 * np.pi) and \
                        (phase[inds[jj + 1] - 1, ii] >= 2 * np.pi - phase_edge):
                    cycle_checks[2] = True

                if imf is not None:
                    # Check we find 5 sensible control points if imf is provided
                    try:
                        cycle = imf[inds[jj]:inds[jj + 1]]
                        # Should extend this to cope with multiple peaks etc
                        ctrl = (0, sift._find_extrema(cycle)[0][0],
                                np.where(np.gradient(np.sign(cycle)) == -1)[0][0],
                                sift._find_extrema(-cycle)[0][0],
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
                cycles[inds[jj]:inds[jj + 1], ii] = count
                count += 1

        logger.info('found {0} cycles in IMF-{1}'.format(cycles[:, ii].max(), ii))

    logger.info('COMPLETED: get cycle indices')
    return cycles


def get_cycle_inds_from_waveform(imf, cycle_start='peaks'):
    """
    ASSUMING LOCALLY SYMMETRICAL SIGNALS!!
    """
    imf = ensure_1d_with_singleton([imf], ['imf'], 'get_cycle_inds_from_waveform')

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


def get_control_points(x, cycles):
    """Identify sets of control points from identified cycles.

    The control points are the ascending zero, peak, descending zero & trough.

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
    # Preamble
    x = ensure_vector([x], ['x'], 'get_control_points')
    cycles = _ensure_cycle_inputs(cycles)

    if cycles.nsamples != x.shape[0]:
        raise ValueError("Mismatched inputs between 'cycles' and 'values'")

    # Main Body

    ctrl = list()
    for cind, cycle_inds in cycles:
        cycle = x[cycle_inds]

        # Peak
        pk = sift._find_extrema(cycle, parabolic_extrema=interp)[0]
        # Ascending-zero crossing
        asc = np.where(np.diff(np.sign(cycle)) == -2)[0]
        if interp:  # Note sure what's going wrong in the indexing here, need a cleaner solution
            aa = asc.copy()
            asc = []
            for idx, a in enumerate(aa):
                interp_ind = np.argmin(np.abs(np.linspace(cycle[a], cycle[a+1], 1000)))
                asc.append(a + np.linspace(0, 1, 1000)[interp_ind])
        # Trough
        tr = sift._find_extrema(-cycle, parabolic_extrema=interp)[0]

        # Replace values with nan if more or less than 1 ctrl point is found
        if len(pk) == 1:
            pk = pk[0]
        else:
            pk = np.nan

        if len(tr) == 1:
            tr = tr[0]
        else:
            tr = np.nan

        if len(asc) == 1:
            asc = asc[0]
        else:
            asc = np.nan

        # Append to list
        ctrl.append((0, pk, asc, tr, len(cycle)-1))

    # Return as array
    return np.array(ctrl)


def get_control_point_metrics(ctrl, normalise=True):

    # Peak to trough ratio
    p2t = (ctrl[:, 2] - (ctrl[:, 4]-ctrl[:, 2]))
    # Ascending to Descending ratio
    a2d = (ctrl[:, 1]+(ctrl[:, 4]-ctrl[:, 3])) - (ctrl[:, 3]-ctrl[:, 1])

    if normalise:
        p2t = p2t / ctrl[:, 4]
        a2d = a2d / ctrl[:, 4]

    return p2t, a2d


def get_cycle_chain(cycles, min_chain=1, drop_first=False, drop_last=False):
    """
    Identify chains of valid cycles in a set of cycles.

    Parameters
    ----------
    cycles : ndarray
        array whose content index cycle locations
    min_chain : int
        Minimum length of chain to return (Default value = 1)
    drop_first : {bool, int}
        Number of cycles to remove from start of chain (default is False)
    drop_last : {bool, int}
        Number of cycles to remove from end of chain (default is False)

    Returns
    -------
    list
        nested list of cycle numbers within each chain


    """
    # Preamble
    cycles = ensure_2d([cycles], ['cycles'], 'get_cycle_chain')

    # Main Body

    if drop_first is True:
        drop_first = 1

    if drop_last is True:
        drop_last = 1

    chains = list()
    chn = None
    # get diff to next cycle for each cycle
    for ii in range(cycles.max()+1):

        if chn is None:
            chn = [ii]  # Start new chain if there isn't one
        else:
            # We're currently in a chain - test whether current cycle is directly after previous cycle
            if cycles[np.where(cycles == ii)[0][0] - 1][0] == -1:
                # Start of new chain - store previous chain and start new one
                if len(chn) >= min_chain:  # Drop chains which are too short
                    if drop_first > 0:
                        chn = chn[drop_first:]
                    if drop_last > 0:
                        chn = chn[:-drop_last]
                    chains.append(chn)
                # Initialise next chain
                chn = [ii]

            else:
                # Continuation of previous chain
                chn.append(ii)

    # If we're at the end - store what we have
    if len(chn) >= min_chain:  # Drop chains which are too short
        if drop_first > 0:
            chn = chn[drop_first:]
        if drop_last > 0:
            chn = chn[:-drop_last]
        chains.append(chn)
    return chains


def mean_vector(IP, X):
    """Compute the mean vector of a set of values wrapped around the unit circle.

    Parameters
    ----------
    IP : ndarray
        Instantaneous Phase values
    X : ndarray
        Observations corresponding to IP values

    Returns
    -------
    mv : ndarray
        Set of mean vectors

    """
    phi = np.cos(IP) + 1j * np.sin(IP)
    mv = phi[:, None] * X
    return mv.mean(axis=0)


def basis_project(X, ncomps=1, ret_basis=False):
    """Express a set of signals in a simple sine-cosine basis set.

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


def kdt_match(x, y, K=15, distance_upper_bound=np.inf):
    """Find unique nearest-neighbours between two n-dimensional feature sets.

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
    uni, _ = np.unique(final, return_counts=True)
    x_inds = np.where(final > -1)[0]
    y_inds = final[x_inds]

    #
    logger.info('Returning {0} matched observations'.format(x_inds.shape[0]))

    return x_inds, y_inds


def _unique_inds(ar):
    """Find the unique elements of an array, ignoring shape.

    Adapted from numpy.lib.arraysetops._unique1d - Original function only
    returns index of first occurrence of unique value

    """
    ar = np.asanyarray(ar).flatten()
    ar.sort()
    aux = ar

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ar_inds = [np.where(ar == ii)[0] for ii in ar[mask]]

    return ar[mask], ar_inds


# ------------------------------------------------

def _ensure_cycle_inputs(cycles):
    """Take a variable and return a valid iterable cycles class if possible"""
    if isinstance(cycles, np.ndarray):
        cycles = ensure_vector([cycles], ['cycles'], '_check_cycle_inputs')
        return IterateCycles(cycles)
    elif isinstance(cycles, Cycles):
        return cycles
    elif isinstance(cycles, IterateCycles):
        return cycles
    elif isinstance(cycles, IterateCyclesFromPhase):
        return cycles
    else:
        raise ValueError("'cycles' input not recognised, must be either a cycle-vector or Cycles class")


class IterateCycles:
    """
    Iterate through pre-defined set of cycles
    """
    def __init__(self, cycles):
        # Preamble
        self.cycles = ensure_vector([cycles], ['cycles'], 'cycle_generator')
        self.ncycles = self.cycles.max() + 1
        self.nsamples = cycles.shape[0]

    def __iter__(self):
        for ii in range(self.ncycles):
            yield ii, np.where(self.cycles == ii)[0]


class IterateCyclesFromPhase:
    """
    Find and iterate through all cycles based on phase.
    """
    def __init__(self, phase, phase_step=1.5 * np.pi,
                 phase_edge=np.pi / 12, mode='cycle'):
        # Preamble
        self.phase = ensure_1d_with_singleton([phase], ['phase'], 'cycle_generator')
        self.mode = mode

        # Main body
        if self.phase.max() > 2 * np.pi:
            self.phase = utils.wrap_phase(self.phase)

        self.inds = np.where(np.abs(np.diff(self.phase[:, 0])) > phase_step)[0] + 1
        self.ncycles = len(self.inds) + 1
        self.nsamples = self.phase.shape[0]

    def __iter__(self):
        for jj in range(-1, len(self.inds)):
            if jj == -1:
                # First cycle is normally a stump
                yield np.arange(0, self.inds[0])
            elif jj == len(self.inds) - 1:
                yield np.arange(self.inds[jj], len(self.phase))
            elif self.mode == 'cycle':
                yield np.arange(self.inds[jj], self.inds[jj + 1])
            elif self.mode == 'augmented_cycle':
                if jj == 0:
                    start = np.where(self.phase[0:self.inds[jj]] > 1.5 * np.pi)[0][0]
                    idx = 0 + start
                else:
                    start = np.where(self.phase[self.inds[jj-1]:self.inds[jj]] > 1.5*np.pi)[0][0]
                    idx = self.inds[jj-1] + start
                yield np.arange(idx, self.inds[jj+1])


# ------------------------------------------------


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


def get_chain_position(chains, mode='compressed', cycles=None):
    cp = np.concatenate([np.arange(len(c)) for c in chains])
    if mode == 'compressed':
        return cp
    elif mode == 'full':
        return _map_cycles_to_samples(cp, cycles)


def _map_cycles_to_samples(cycle_stats, cycles):
    """
    Move from vector of len(cycles) to vector of length(samples)
    """
    full = np.zeros_like(cycles, dtype=float) - 1
    for ii in range(len(cycle_stats)):
        full[cycles == ii] = cycle_stats[ii]
    return full


def _map_chains_to_cycles(chain_stats, chains, cycles):
    """
    Move from vector of len(chains) to vector of length(cycles)
    """
    out = np.zeros((cycles.max()+1,))
    for ii in range(len(chains)):
        out[chains[ii]] = chain_stats[ii]
    return out


def _map_chains_to_samples(chain_stats, chains, cycles):
    """
    Move from vector of len(chains) to vector of length(samples)
    """
    cycle_stats = _map_chains_to_cycles(chain_stats, chains, cycles)
    return _map_cycles_to_samples(cycle_stats, cycles)


def _get_start_sample(x):
    return x[0]


def _get_stop_sample(x):
    return x[-1]


class Cycles:
    """
    Class defining Cycle analysis API. Some rules:

    A set of cycles for a time series is defined by a vector with the following properties.
    1) cycle vector is same length as time series
    2) cycle vector contains integer denoting which cycle each sample belongs to
    3) cycle vector has -1 for samples with no cycle OR an excluded cycle
    4) integer values in cycle vector are sequential

    So,
    get_cycle_stat should drop -1 automatically. Then values in cycle vector match indices into compressed cycle stat.
    _map_cycles_to_samples can map between the two.
    """

    def __init__(self, IP,
                 phase_step=1.5 * np.pi, phase_edge=np.pi / 12,
                 compute_timings=False, mode='cycle'):

        if mode not in ['cycle', 'augmented']:
            raise ValueError("mode '{0}' not recognised".format(mode))

        self.phase = ensure_1d_with_singleton([IP], ['IP'], 'Cycles')

        self.cycle_starts = np.where(np.abs(np.diff(self.phase[:, 0])) > phase_step)[0] + 1

        self._all_cycles = get_cycle_inds(self.phase, return_good=False, phase_step=phase_step, phase_edge=phase_edge)
        self.ncycles = self._all_cycles.max()
        self.nsamples = IP.shape[0]
        self.mode = mode

        self._metrics = dict()
        if self.mode == 'cycle':
            self.add_cycle_stat('is_good', self.phase, is_good)
        elif self.mode == 'augmented':
            is_good_aug = partial(is_good, mode='augmented')
            self.add_cycle_stat('is_good', self.phase, is_good_aug)

        if compute_timings:
            self.compute_timing_metrics()

    def _get_cycle(self, ii):
        if self.mode == 'cycle':
            return np.where(self._all_cycles == ii)[0]
        elif self.mode == 'augmented':
            if ii == 0:
                # We don't have a previous cycle to use, return normal
                return np.where(self._all_cycles == ii)[0]
            else:
                prev = np.where(self._all_cycles == ii-1)[0]
                trough_in_prev = prev[np.where(self.phase[prev] > 1.5*np.pi)[0][0]]
                stop = np.where(self._all_cycles == ii)[0][-1] + 1
                return np.arange(trough_in_prev, stop)

    def __iter__(self):
        for ii in range(self.ncycles):
            yield ii, self._get_cycle(ii)

    def iter_subset(self, conditions=None):
        indx = self.get_cycle_index(conditions)
        for ii in range(self.ncycles):
            if indx[ii] == True:  # noqa: E712
                yield ii, self._get_cycle(ii)
            else:
                continue

    @classmethod
    def load(cls, cycle_vect, metrics_dataframe):
        """
        Load cycle stats from an existing analyses into a new Cycles instance.
        """

        ret = cls()
        ret._all_cycles = cycle_vect
        ret._metric = metrics_dataframe.to_dict('list')

        return ret

    def compute_timing_metrics(self):
        self.add_cycle_stat('start_sample',
                            np.arange(len(self._all_cycles)),
                            _get_start_sample,
                            dtype=int)
        self.add_cycle_stat('stop_sample',
                            np.arange(len(self._all_cycles)),
                            _get_stop_sample,
                            dtype=int)
        self.add_cycle_stat('duration',
                            self._all_cycles,
                            len,
                            dtype=int)

    def get_subset(self, conditions=None):
        """
        Return the cycle vector and metrics dataframe for a subset of cycles
        defined by a set of specified conditions.
        """

        vect = self.get_cycle_vector(conditions=conditions)
        d = self.get_metric_dataframe(conditions=conditions)
        return vect, d

    def get_cycle_vector(self, conditions=None):
        """
        Get standalone cycles vector for cycles meeting specified conditions.
        Non-cycle samples are set to -1.
        """
        if conditions is None:
            return self._all_cycles
        else:
            inds = self.get_cycle_index(conditions)
            out = self._all_cycles.copy()
            for ii in range(len(inds)):
                if inds[ii] == False:  # noqa: E712
                    out[self._all_cycles == ii] = -1
            return self._shakedown(out)

    def get_metric_dataframe(self, conditions=None):
        """
        Get standalone dataframe of cycle metrics for cycles meeting specified conditions.
        This method requires that pandas is installed.
        """
        import pandas as pd
        d = pd.DataFrame.from_dict(self._metrics)

        if conditions is not None:
            inds = self.get_cycle_index(conditions) == False
            d = d.drop(np.where(inds)[0])
            d = d.reset_index()

        return d

    def add_cycle_stat(self, name, X, func, dtype=None):
        """
        Compute a statistic for all cycles and store the result in the Cycle
        object for later use.
        """
        vals = get_cycle_stat(self, X,
                              mode='compressed',
                              func=func)

        if dtype is not None:
            vals = vals.astype(dtype)

        self._metrics[name] = vals

    def add_cycle_metric(self, name, metric):
        """
        Store a precomputed metric for all cycle
        """
        if len(metric) != len(self._metrics['is_good']):
            raise ValueError("Input '{0}' ({1}) doesn't match cycles ({2})".format(name,
                                                                                   len(metric),
                                                                                   len(self._metrics['is_good'])))
        else:
            self._metrics[name] = metric

    def get_cycle_index(self, conditions, ret_separate=False):
        """
        Return the indices for a subset of cycles defined by the specified conditions
        """

        if isinstance(conditions, str):
            conditions = [conditions]

        out = np.zeros((len(self._metrics['is_good']), len(conditions)))
        for idx, c in enumerate(conditions):
            name, func, val = self._parse_condition(c)
            out[:, idx] = func(self._metrics[name], val)

        if ret_separate:
            return out
        else:
            return np.all(out, axis=1)

    def _parse_condition(self, cond):
        """
        Helper method to parse strings defining conditional statements.
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

    def _sanitise(self, cycles):
        """
        replace any nans with zeros
        """
        cycles[np.isnan(cycles)] = 0
        return cycles.astype(int)

    def _shakedown(self, cycles):
        """
        Make cycle inds properly sequential again
        """
        out = self._sanitise(cycles)
        cycles_in = np.unique(cycles)
        if not np.in1d(-1, cycles_in):
            cycles_in = np.r_[-1, cycles_in]
        cycles_out = np.arange(-1, len(cycles_in)-1)
        for ii in range(len(cycles_in)):
            out[cycles == cycles_in[ii]] = cycles_out[ii]
        return out
