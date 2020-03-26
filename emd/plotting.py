#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


def plot_imfs(imfs, time_vect=None, scale_y=False, freqs=None, cmap=None, fig=None):
    """
    Create a quick summary plot for a set of IMFs.

    Parameters
    ----------
    imfs : ndarray
        2D array of IMFs to plot
    time_vect : ndarray
         Optional 1D array specifying time values (Default value = None)
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
        time_vect = np.arange(imfs.shape[0])

    mx = np.abs(imfs).max()

    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(nplots, 1, 1)
    for tag in ['top', 'right', 'bottom']:
        ax.spines[tag].set_visible(False)
    ax.plot((time_vect[0], time_vect[-1]), (0, 0), color=[.5, .5, .5])
    ax.plot(time_vect, imfs.sum(axis=1), 'k')
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_xlim(time_vect[0], time_vect[-1])
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
        ax.set_ylabel('IMF {0}'.format(ii), rotation=0, labelpad=10)

        if ii < nplots:
            ax.tick_params(axis='x', labelbottom=False)
        if freqs is not None:
            ax.set_title(freqs[ii - 1], fontsize=8)

    fig.subplots_adjust(top=.95, bottom=.05, right=.95)
