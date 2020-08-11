#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Identification and analysis of cycles in an oscillatory signal.

Routines:

bin_by_phase
phase_align
get_cycle_inds
get_cycle_stat
get_control_points
get_cycle_chain
mean_vector
kdt_match

"""

import numpy as np
from scipy import interpolate as interp

from . import spectra, utils, sift

# Housekeeping for logging
import logging
logger = logging.getLogger(__name__)


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
        bin_edges, bin_centres = spectra.define_hist_bins(0, 2 * np.pi, nbins)
    else:
        nbins = len(bin_edges) - 1
        bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

    bin_inds = np.digitize(ip, bin_edges)[:, 0]

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
    logger.info('STARTED: phase-align cycles')

    phase_edges, phase_bins = spectra.define_hist_bins(0, 2 * np.pi, npoints)

    if cycles is None:
        cycles = get_cycle_inds(ip)

    if ip.ndim == 2 and ip.shape[1] > 1:
        # too many imfs - error
        msg = 'emd.cycles.phase_align only works on a single IMF - input IP dims are {0}'.format(ip.shape)
        logger.warning(msg)
        raise ValueError(msg)
    elif ip.ndim == 1:
        ip = ip[:, None]

    if x.ndim == 2 and x.shape[1] > 1:
        # too many imfs - error
        msg = 'emd.cycles.phase_align only works on a single IMF - input x dims are {0}'.format(x.shape)
        logger.warning(msg)
        raise ValueError(msg)
    elif x.ndim == 1:
        x = x[:, None]

    if cycles.ndim == 2 and cycles.shape[1] > 1:
        # too many imfs - error
        msg = 'emd.cycles.phase_align only works on a single IMF - input cycles dims are {0}'.format(cycles.shape)
        logger.warning(msg)
        raise ValueError(msg)
    elif cycles.ndim == 1:
        cycles = cycles[:, None]

    logger.debug('aligning {0} cycles over {1} phase points with {2} interpolation'.format(cycles.max(),
                                                                                           npoints,
                                                                                           interp_kind))

    ncycles = cycles.max()
    avg = np.zeros((npoints, ncycles))
    for ii in range(1, ncycles + 1):

        phase_data = ip[cycles[:] == ii]
        x_data = x[cycles[:] == ii]

        f = interp.interp1d(phase_data, x_data, kind=interp_kind,
                            bounds_error=False, fill_value='extrapolate')

        avg[:, ii - 1] = f(phase_bins)

    logger.info('COMPLETED: phase-align cycles')
    return avg


def get_cycle_inds(phase, return_good=True, mask=None,
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
    4 : A set of 4 unique control points
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

    logger.info('STARTED: get cycle indices')
    logger.debug('computing on {0} samples over {1} IMFs '.format(phase.shape[0],
                                                                  phase.shape[1]))
    if mask is not None:
        logger.debug('{0} ({1}%) samples masked out'.format(mask.sum(), np.round(100*(mask.sum()/phase.shape[0]), 2)))

    if phase.max() > 2 * np.pi:
        print('Wrapping phase')
        phase = utils.wrap_phase(phase)

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
            inds = np.r_[0, inds]
        if inds[-1] <= phase.shape[0] - 1:
            inds = np.r_[inds, phase.shape[0] - 1]

        unwrapped = np.unwrap(phase[:, ii], axis=0)

        count = 1
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
                        ctrl = (0, sift.find_extrema(cycle)[0][0],
                                np.where(np.gradient(np.sign(cycle)) == -1)[0][0],
                                sift.find_extrema(-cycle)[0][0],
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
    if (cycles.ndim > 1) and (cycles.shape[1] > 1):
        raise ValueError('Cycles for {0} IMFs passed in, \
                          please input the cycles for a single IMF'.format(cycles.shape[1]))

    logger.info('STARTED: get cycle stats')
    logger.debug('computing stats for {0} cycles over {1} samples'.format(cycles.max(), cycles.shape[0]))
    logger.debug('computing metric {0} and returning {1}-array'.format(func, mode))

    if mode == 'compressed':
        out = np.zeros((cycles.max() + 1, )) * np.nan
    elif mode == 'full':
        out = np.zeros_like(values) * np.nan

    for cind in range(1, cycles.max() + 1):
        stat = func(values[cycles == cind])

        if mode == 'compressed':
            out[cind] = stat
        elif mode == 'full':
            out[cycles == cind] = stat

    # Currently including the first value as the stat for 'non-cycles' in
    # compressed mode for backwards compatibility with earlier work, might be
    # confusing overall - should probably rethink this whether this makes any
    # sense
    if mode == 'compressed':
        out[0] = func(values[cycles == 0])

    logger.info('COMPLETED: get cycle stats')
    return out


def get_control_points(x, good_cycles):
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

    ctrl = list()
    for ii in range(1, good_cycles.max() + 1):
        cycle = x[good_cycles == ii]

        # Peak
        pk = sift.find_extrema(cycle)[0]
        # Ascending-zero crossing
        asc = np.where(np.diff(np.sign(cycle)) == -2)[0]
        # Trough
        tr = sift.find_extrema(-cycle)[0]

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


def get_cycle_chain(cycles, min_chain=1, drop_first=False, drop_last=False):
    """
    Identify chains of valid cycles in a set of cycles.

    Parameters
    ----------
    cycles : ndarray
        array whose content index cycle locations
    min_chain : integer
        Minimum length of chain to return (Default value = 1)
    drop_first : {bool, integer}
        Number of cycles to remove from start of chain (default is False)
    drop_last : {bool, integer}
        Number of cycles to remove from end of chain (default is False)

    Returns
    -------
    list
        nested list of cycle numbers within each chain


    """

    if cycles.ndim == 1:
        cycles = cycles[:, None]

    if drop_first is True:
        drop_first = 1

    if drop_last is True:
        drop_last = 1

    chains = list()
    chn = None
    # get diff to next cycle for each cycle
    for ii in range(1, cycles.max() + 1):

        if chn is None:
            chn = [ii]  # Start new chain if there isn't one
        else:
            # We're currently in a chain - test whether current cycle is directly after previous cycle
            if cycles[np.where(cycles == ii)[0][0] - 1][0] == 0:
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

    ##
    logging.info('Starting KD-Tree Match')
    msg = 'Matching {0} features from y ({1} observations) to x ({2} observations)'
    logging.info(msg.format(x.shape[1], y.shape[0], x.shape[0]))
    logging.debug('K: {0}, distance_upper_bound: {1}'.format(K, distance_upper_bound))

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
        logging.debug(msg.format(np.sum(uni_matches), ii))

    # Find column index of left-most choice per row (ie closest unique neighbour)
    winner = np.argmax(II, axis=1)
    # Find row index of winner
    final = np.zeros((II.shape[0],), dtype=int)
    for ii in range(II.shape[0]):
        if (np.sum(II[ii, :]) == 1) and (winner[ii] < y.shape[0]) and \
           (inds[ii, winner[ii]] < y.shape[0]):
            final[ii] = inds[ii, winner[ii]]
        else:
            final[ii] = -1  #  No good match

    # Remove failed matches
    uni, cnt = np.unique(final, return_counts=True)
    x_inds = np.where(final > -1)[0]
    y_inds = final[x_inds]

    ##
    logging.info('Returning {0} matched observations'.format(x_inds.shape[0]))

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
