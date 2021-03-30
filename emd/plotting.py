#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Routines for plotting results of EMD analyses.

Main Routines:
  plot_imfs
  plot_hilberthuang
  plot_holospectrum

Utilities:
  _get_log_tickpos

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_imfs(imfs, time_vect=None, sample_rate=1, scale_y=False, freqs=None, cmap=None, fig=None):
    """Create a quick summary plot for a set of IMFs.

    Parameters
    ----------
    imfs : ndarray
        2D array of IMFs to plot
    time_vect : ndarray
         Optional 1D array specifying time values (Default value = None)
    sample_rate : float
        Optional sample rate to determine time axis values if time_vect is not
        specified if time_vect is given.
    scale_y : Boolean
         Flag indicating whether the y-axis should be adative to each mode
         (False) or consistent across modes (True) (Default value = False)
    freqs : array_like
        Optional vector of frequencies for each IMF
    cmap : {None,True,matplotlib colormap}
        Optional colourmap to use. None will plot each IMF in black and True will
        use the plt.cm.Dark2 colormap as default. A different colormap may also
        be passed in.

    """
    nplots = imfs.shape[1] + 1
    if time_vect is None:
        time_vect = np.linspace(0, imfs.shape[0]/sample_rate, imfs.shape[0])

    mx = np.abs(imfs).max()
    mx_sig = np.abs(imfs.sum(axis=1)).max()

    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(nplots, 1, 1)
    if scale_y:
        ax.yaxis.get_major_locator().set_params(integer=True)
    for tag in ['top', 'right', 'bottom']:
        ax.spines[tag].set_visible(False)
    ax.plot((time_vect[0], time_vect[-1]), (0, 0), color=[.5, .5, .5])
    ax.plot(time_vect, imfs.sum(axis=1), 'k')
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_xlim(time_vect[0], time_vect[-1])
    ax.set_ylim(-mx_sig * 1.1, mx_sig * 1.1)
    ax.set_ylabel('Signal', rotation=0, labelpad=10)

    if cmap is True:
        # Use default colormap
        cmap = plt.cm.Dark2
        cols = cmap(np.linspace(0, 1, imfs.shape[1] + 1))
    elif isinstance(cmap, Colormap):
        # Use specified colormap
        cols = cmap(np.linspace(0, 1, imfs.shape[1] + 1))
    else:
        # Use all black lines - this is overall default
        cols = np.array([[0, 0, 0] for ii in range(imfs.shape[1] + 1)])

    for ii in range(1, nplots):
        ax = fig.add_subplot(nplots, 1, ii + 1)
        for tag in ['top', 'right', 'bottom']:
            ax.spines[tag].set_visible(False)
        ax.plot((time_vect[0], time_vect[-1]), (0, 0), color=[.5, .5, .5])
        ax.plot(time_vect, imfs[:, ii - 1], color=cols[ii, :])
        ax.set_xlim(time_vect[0], time_vect[-1])
        if scale_y:
            ax.set_ylim(-mx * 1.1, mx * 1.1)
            ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel('IMF {0}'.format(ii), rotation=0, labelpad=10)

        if ii < nplots - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Time')
        if freqs is not None:
            ax.set_title(freqs[ii - 1], fontsize=8)

    fig.subplots_adjust(top=.95, bottom=.1, left=.2, right=.99)


def plot_hilberthuang(hht, time_vect, freq_vect,
                      time_lims=None, freq_lims=None, log_y=False,
                      vmin=0, vmax=None,
                      fig=None, ax=None, cmap='hot_r'):
    """Create a quick summary plot for a Hilbert-Huang Transform.

    Parameters
    ----------
    hht : 2d array
        Hilbert-Huang spectrum to be plotted - output from emd.spectra.hilberthuang
    time_vect : vector
        Vector of time samples
    freq_vect : vector
        Vector of frequency bins
    time_lims : optional tuple or list (start_val, end_val)
        Optional time-limits to zoom in time on the x-axis
    freq_lims : optional tuple or list (start_val, end_val)
        Optional time-limits to zoom in frequency on the y-axis
    fig : optional figure handle
        Figure to plot inside
    ax : optional axis handle
        Axis to plot inside
    cmap : optional str or matplotlib.cm
        Colormap specification

    Returns
    -------
    ax
        Handle of plot axis

    """
    # Make figure if no fig or axis are passed
    if (fig is None) and (ax is None):
        fig = plt.figure()

    # Create axis if no axis is passed.
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    # Get time indices
    if time_lims is not None:
        tinds = np.logical_and(time_vect >= time_lims[0], time_vect <= time_lims[1])
    else:
        tinds = np.ones_like(time_vect).astype(bool)

    # Get frequency indices
    if freq_lims is not None:
        finds = np.logical_and(freq_vect >= freq_lims[0], freq_vect <= freq_lims[1])
    else:
        finds = np.ones_like(freq_vect).astype(bool)
        freq_lims = (freq_vect[0], freq_vect[-1])

    # Make space for colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if vmax is None:
        vmax = np.max(hht[np.ix_(finds, tinds)])

    # Make main plot
    pcm = ax.pcolormesh(time_vect[tinds], freq_vect[finds], hht[np.ix_(finds, tinds)],
                        vmin=vmin, vmax=vmax, cmap=cmap, shading='nearest')

    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Hilbert-Huang Transform')

    # Scale axes if requestedd
    if log_y:
        ax.set_yscale('log')
        ax.set_yticks((_get_log_tickpos(freq_lims[0], freq_lims[1])))
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # Add colourbar
    plt.colorbar(pcm, cax=cax, orientation='vertical')

    return ax


def plot_holospectrum(holo, freq_vect, am_freq_vect,
                      freq_lims=None, am_freq_lims=None,
                      log_x=False, log_y=False,
                      vmin=0, vmax=None,
                      fig=None, ax=None, cmap='hot_r', mask=True):
    """Create a quick summary plot for a Holospectrum.

    Parameters
    ----------
    holo : 2d array
        Hilbert-Huang spectrum to be plotted - output from emd.spectra.holospectrum
    freq_vect : vector
        Vector of frequency values for first-layer
    am_freq_vect : vector
        Vector of frequency values for amplitude modulations in second--layer
    freq_lims : optional tuple or list (start_val, end_val)
        Optional time-limits to zoom in frequency on the y-axis
    am_freq_lims : optional tuple or list (start_val, end_val)
        Optional time-limits to zoom in amplitude modulation frequency on the x-axis
    log_x : bool
        Flag indicating whether to set log-scale on x-axis
    log_y : bool
        Flag indicating whether to set log-scale on y-axis
    fig : optional figure handle
        Figure to plot inside
    ax : optional axis handle
        Axis to plot inside
    cmap : optional str or matplotlib.cm
        Colormap specification

    Returns
    -------
    ax
        Handle of plot axis

    """
    # Make figure if no fig or axis are passed
    if (fig is None) and (ax is None):
        fig = plt.figure()

    # Create axis if no axis is passed.
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    # Get frequency indices
    if freq_lims is not None:
        finds = np.logical_and(freq_vect > freq_lims[0], freq_vect < freq_lims[1])
    else:
        finds = np.ones_like(freq_vect).astype(bool)

    # Get frequency indices
    if am_freq_lims is not None:
        am_finds = np.logical_and(am_freq_vect > am_freq_lims[0], am_freq_vect < am_freq_lims[1])
    else:
        am_finds = np.ones_like(am_freq_vect).astype(bool)

    plot_holo = holo.copy()
    if mask:
        for ii in range(len(freq_vect)):
            for jj in range(len(am_freq_vect)):
                if freq_vect[ii] < am_freq_vect[jj]:
                    plot_holo[jj, ii] = np.nan

    # Set colourmap
    if isinstance(cmap, str):
        cmap = getattr(plt.cm, cmap)
    elif cmap is None:
        cmap = getattr(plt.cm, cmap)

    # Set mask values in colourmap
    cmap.set_bad([0.8, 0.8, 0.8])

    # Make space for colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if vmax is None:
        vmax = np.max(plot_holo[np.ix_(am_finds, finds)])

    # Make main plot
    pcm = ax.pcolormesh(am_freq_vect[am_finds], freq_vect[finds], plot_holo[np.ix_(am_finds, finds)].T,
                        cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')

    # Set labels
    ax.set_xlabel('Amplitude Modulation Frequency')
    ax.set_ylabel('Carrier Wave Frequency')
    ax.set_title('Holospectrum')

    # Scale axes if requestedd
    if log_y:
        ax.set_yscale('log')
        ax.set_yticks((_get_log_tickpos(freq_lims[0], freq_lims[1])))
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    if log_x:
        ax.set_xscale('log')
        ax.set_xticks((_get_log_tickpos(am_freq_lims[0], am_freq_lims[1])))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Add colourbar
    plt.colorbar(pcm, cax=cax, orientation='vertical')

    return ax


def _get_log_tickpos(lo, hi, tick_rate=5, round_vals=True):
    """Generate tick positions for log-scales.

    Parameters
    ----------
    lo : float
        Low end of frequency range
    hi : float
        High end of frequency range
    tick_rate : int
        Number of ticks per order-of-magnitude
    round_vals : bool
        Flag indicating whether ticks should be rounded to first non-zero value.

    Returns
    -------
    ndarray
        Vector of tick positions

    """
    lo_oom = np.floor(np.log10(lo)).astype(int)
    hi_oom = np.ceil(np.log10(hi)).astype(int) + 1
    ticks = []
    log_tick_pos_inds = np.round(np.logspace(1, 2, tick_rate)).astype(int) - 1
    for ii in range(lo_oom, hi_oom):
        tks = np.linspace(10**ii, 10**(ii+1), 100)[log_tick_pos_inds]
        if round_vals:
            ticks.append(np.round(tks / 10**ii)*10**ii)
        else:
            ticks.append(tks)
        #ticks.append(np.logspace(ii, ii+1, tick_rate))

    ticks = np.unique(np.r_[ticks])
    inds = np.logical_and(ticks > lo, ticks < hi)
    return ticks[inds]
