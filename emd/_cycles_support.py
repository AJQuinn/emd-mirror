#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Low-level functions for handling single cycle analyses.


"""

import numpy as np

# --------------------------------------
#
# This module contains functions to support analysis on individual and groups
# of cycles. Indexing between samples, cycles and chains/sequences of cycles
# quickly gets complicated. The functions here handle this mapping, implement
# some options for projecting information between levels and compute metrics on
# specific levels
#
# We have four levels of information to work with:
#
# 1 - samples    A time-series containing oscillations                           [ time x 1 ]
# 2 - cycles     Every conceivable cycle within a time series                    [ cycles x 1 ]
# 3 - subset     A user-defined masked subset of all cycles                      [ subset x 1 ]
# 4 - chains     A sequency of continuously occurring cycles within the subset   [ chains x 1 ]
#
# The 'subset' level is worth clarifying - whilst all cycles contains every
# single possible cycle even if distorted or incomplete, the subset is any
# restricted set of cycles which are going to be analysed further. This choice
# of subset is user-defined and may be as simple, complex or arbitrary as
# necessary. All we need here is the indices of cycles to be retained.
#
# Information in each level is stored in vectors of the lengths defined above.
# We can move this information between levels if we have a set of mapping
# vectors.
#
# 1 - samples <-> cycles  Which cycle does each sample belong to                         [ time x 1 ]
# 2 - cycles  <-> subset  Which subset-cycle does each cycle correspond to (-1 for None) [ cycles x 1 ]
# 3 - subset  <-> chains  Which chain does each subset cycle belong to                   [ subset x 1 ]
#
# There are two possible definitions of a 'cycle' - a standard form and an
# augmented form. These are illustrated below.#
#
# standard:       abcd
# augmented:     12345
#                 /\  /\
#               \/  \/  \/
#
# The standard cycle contains a single period of an oscillation spanning the
# full 2pi range of phase. This can be split into four quadrants around its
# control points illustrated by abcd in the schematic above.
#
# The augmented cycle includes an additional 5th segment which overlaps with
# the previous cycle - this is illustrated by the 12345 above.
#
# Metrics can be computed from either definition and stored in a
# cycles/subset/chain vector in the same way.


def make_slice_cache(cycle_vect):
    """Create a list of slice objects from a cycle_vect."""
    starts = np.where(np.diff(cycle_vect, axis=0) == 1)[0] + 1
    stops = starts

    starts = np.r_[0, starts]
    stops = np.r_[stops, len(cycle_vect)]

    slice_cache = [slice(starts[ii], stops[ii]) for ii in range(len(starts))]

    return slice_cache


def _slice_len(sli):
    """Find the length of array returned by a slice."""
    return sli.stop - sli.start + 1


def augment_slice(s, phase):
    """Augment a slice to include the closest trough to the left."""
    xx = np.where(np.flipud(phase[:s.start]) < 1.5*np.pi)[0]
    if len(xx) == 0:
        return None
    start_diff = xx[0]
    s2 = slice(s.start - start_diff, s.stop)
    return s2


def make_aug_slice_cache(slice_cache, phase, func=augment_slice):
    """Build a slice cache of augmented slices defined by some function."""
    return [func(s, phase) for s in slice_cache]


# --------------------------------------


def get_slice_stat_from_samples(vals, slices, func=np.mean):
    """Compute a stat from each slice in a list."""
    if isinstance(vals, tuple):
        out = np.zeros((len(slices), ))
        for idx, s in enumerate(slices):
            args = [v[s] for v in vals]
            out[idx] = func(*args)
        return out
    else:
        return np.array([func(vals[s]) if s is not None else np.nan for s in slices])


def get_cycle_stat_from_samples(vals, cycle_vect, func=np.mean):
    """Compute a metric across all samples from each cycle."""
    ncycles = np.max(cycle_vect) + 1
    out = np.zeros((ncycles,))

    for ii in range(ncycles):
        inds = map_cycle_to_samples(cycle_vect, ii)
        if isinstance(vals, tuple):
            args = [v[inds] for v in vals]
            out[ii] = func(*args)
        else:
            out[ii] = func(vals[inds])
    return out


def get_augmented_cycle_stat_from_samples(vals, cycle_vect, phase, func=np.mean):
    """Compute a metric across all augmented samples from each cycle."""
    ncycles = np.max(cycle_vect) + 1
    out = np.zeros((ncycles,))

    for ii in range(ncycles):
        inds = map_cycle_to_samples_augmented(cycle_vect, ii, phase)
        if isinstance(vals, tuple):
            args = [v[inds] for v in vals]
            out[ii] = func(*args)
        else:
            out[ii] = func(vals[inds])
    return out


def get_subset_stat_from_samples(vals, subset_vect, cycle_vect, func=np.mean):
    """Compute a metric across all samples from each cycle in a subset."""
    ncycles = np.max(subset_vect) + 1
    out = np.zeros((ncycles,))
    for ii in range(ncycles):
        out[ii] = func(vals[map_subset_to_sample(subset_vect, cycle_vect, ii)])
    return out


def get_chain_stat_from_samples(vals, chain_vect, subset_vect, cycle_vect, func=np.mean):
    """Compute a metric across all samples from each chain."""
    nchains = np.max(chain_vect) + 1
    out = np.zeros((nchains,))
    for ii in range(nchains):
        out[ii] = func(vals[map_chain_to_samples(chain_vect, subset_vect, cycle_vect, ii)])
    return out

# --------------------------------------


def project_cycles_to_samples(vals, cycle_vect):
    """Transform per-cycle data to full sample vector."""
    out = np.zeros_like(cycle_vect).astype(float) * np.nan
    for ii in range(len(vals)):
        inds = map_cycle_to_samples(cycle_vect, ii)
        out[inds] = vals[ii]
    return out


def project_subset_to_cycles(vals, subset_vect):
    """Transform per-cycle data from a subset of cycles to a vector of all cycles."""
    out = np.zeros_like(subset_vect).astype(float) * np.nan
    for ii in range(len(vals)):
        inds = map_subset_to_cycle(subset_vect, ii)
        out[inds] = vals[ii]
    return out


def project_subset_to_samples(vals, subset_vect, cycle_vect):
    """Transform per-cycle data from a subset of cycles to full sample vector."""
    cycle_vals = project_subset_to_cycles(vals, subset_vect)
    out = np.zeros_like(cycle_vect).astype(float) * np.nan
    for ii in range(len(cycle_vals)):
        inds = map_cycle_to_samples(cycle_vect, ii)
        out[inds] = cycle_vals[ii]
    return out


def project_chain_to_subset(vals, chain_vect):
    """Transform per-chain data to a subset vector."""
    out = np.zeros_like(chain_vect).astype(float) * np.nan
    for ii in range(len(vals)):
        inds = map_chain_to_subset(chain_vect, ii)
        out[inds] = vals[ii]
    return out


def project_chain_to_cycles(vals, chain_vect, subset_vect):
    """Transform per-chain data to an all cycles vector."""
    subset_vals = project_chain_to_subset(vals, chain_vect)
    return project_subset_to_cycles(subset_vals, subset_vect)


def project_chain_to_samples(vals, chain_vect, subset_vect, cycle_vect):
    """Transform per-chain data to an all samples vector."""
    cycle_vals = project_chain_to_cycles(vals, chain_vect, subset_vect)
    return project_cycles_to_samples(cycle_vals, cycle_vect)


# -------------------------------------

def map_cycle_to_samples(cycle_vect, ii):
    """which samples does the iith cycle contain?"""
    sample_inds = np.where(cycle_vect == ii)[0]
    if np.all(np.diff(sample_inds) == 1) is False:
        # Mapped samples are not continuous!
        raise ValueError
    return sample_inds


def map_cycle_to_samples_augmented(cycle_vect, ii, phase):
    """which samples does the augmented iith cycle contain?"""
    prev = np.where(cycle_vect == ii-1)[0]
    prev_segment_inds = np.where(phase[prev] > 1.5*np.pi)[0]
    if len(prev_segment_inds) == 0:
        # No candidate trough in previous cycle
        return None
    trough_in_prev = prev[prev_segment_inds[0]]
    stop = np.where(cycle_vect == ii)[0][-1] + 1
    return np.arange(trough_in_prev, stop)


def map_sample_to_cycle(cycle_vect, ii):
    """Which cycle is the ii-th sample in?"""
    return cycle_vect[ii]


def map_subset_to_cycle(subset_vect, ii):
    """Which of all cycles does the ii-th subset cycle correspond to?"""
    return np.where(subset_vect == ii)[0]


def map_cycle_to_subset(subset_vect, ii):
    """Which subset cycle does ii-th overall cycle correspond to? else None"""
    # answer might be -1...
    subset_ind = subset_vect[ii]
    return subset_ind if subset_ind > -1 else None


def map_subset_to_sample(subset_vect, cycle_vect, ii):
    """Which samples does ii-th subset cycle contain?"""
    all_cycle_ind = map_subset_to_cycle(subset_vect, ii)
    return map_cycle_to_samples(cycle_vect, all_cycle_ind)


def map_subset_to_sample_augmented(subset_vect, cycle_vect, ii, phase):
    """Which samples does ii-th subset cycle contain?"""
    all_cycle_ind = map_subset_to_cycle(subset_vect, ii)
    return map_cycle_to_samples_augmented(cycle_vect, all_cycle_ind, phase)


def map_sample_to_subset(subset_vect, cycle_vect, ii):
    """Which subset cycle does the ii-th sample belong to?"""
    all_cycle_ind = map_sample_to_cycle(cycle_vect, ii)
    if all_cycle_ind is None:
        return None
    return map_cycle_to_subset(subset_vect, all_cycle_ind)


def map_chain_to_subset(chain_vect, ii):
    """Which subset cycles does the ii-th chain contain?"""
    subset_inds = np.where(chain_vect == ii)[0]
    if (len(subset_inds) > 1) and (np.all(np.diff(subset_inds) == 1) is False):
        # Mapped subset is not continuous
        raise ValueError
    return subset_inds


def map_subset_to_chain(chain_vect, ii):
    """Which chain does the iith subset cycle belong to? else None"""
    return chain_vect[ii]


def map_cycle_to_chain(chain_vect, subset_vect, ii):
    """Which chain does the ii-th overall cycle belong to? else None"""
    subset_cycle_ind = map_cycle_to_subset(subset_vect, ii)
    if subset_cycle_ind is None:
        return None
    return map_subset_to_chain(chain_vect, subset_cycle_ind)


def map_chain_to_cycle(chain_vect, subset_vect, ii):
    """Which of all cycles does the ii-th chain contain"""
    subset_ind = map_chain_to_subset(chain_vect, ii)
    cycle_ind = np.squeeze([map_subset_to_cycle(subset_vect, jj) for jj in subset_ind])
    if (len(cycle_ind) > 1) and (np.all(np.diff(cycle_ind) == 1) is False):
        # Mapped cycles are not continuous!
        raise ValueError
    return cycle_ind


def map_chain_to_samples(chain_vect, subset_vect, cycle_vect, ii):
    """Which samples does the ii-th chain contain?"""
    subset_inds = map_chain_to_subset(chain_vect, ii)
    sample_inds = [map_subset_to_sample(subset_vect, cycle_vect, jj) for jj in subset_inds]
    return np.hstack(sample_inds)


def map_sample_to_chain(chain_vect, subset_vect, cycle_vect, ii):
    """Which chain does the ii-th sample belong to? else None"""
    subset_ind = map_sample_to_subset(subset_vect, cycle_vect, ii)
    if subset_ind is None:
        return None
    return map_subset_to_chain(chain_vect, subset_ind)
