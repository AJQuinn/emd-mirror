.. emd documentation master file, created by
   sphinx-quickstart on Sun Jan 27 23:11:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EMD: Empirical Mode Decomposition
=================================

EMD is a python package implementing the Empirical Mode Decomposition and
functionality for ananlysis of instantaneous frequency.


Features
========

  * A range of sift algorithms including: sift, ensemble sift, complete ensemble sift, mask sift
  * Instantaneous phase, frequency and amplitude computation
  * Cycle detection and analysis
  * Hilbert-Huang spectrum estimation (1d frequency spectrum or 2d time-frequency spectrum)
  * Second layer sift to quantify structure in amplitude modulations
  * Holospectrum estimation (3d instantaneous frequency x amplitude modulation frequency x time spectrum)


Quick Start
===========

EMD can be install from `PyPI <https://pypi.org/project/emd/>`_ using pip::

    pip install emd

and used to decompose and describe non-linear timeseries.::

    # Imports
    import emd
    import numpy as np
    import matplotlib.pyplot as plt

    # Definitions
    sample_rate = 1000
    seconds = 3
    time_vect = np.linspace(0,seconds,seconds*sample_rate)

    # A non-linear oscillation
    x = emd.utils.abreu2010( 5, .25, -np.pi/4, sample_rate, seconds )
    # ...plus a linear oscillation
    x += np.cos( 2*np.pi*1*time_vect )

    # Sift
    imf = emd.sift.sift( x )

    # Visualise Intrinsic Mode Functions
    emd.plotting.plot_imfs( imf, scale_y=True, cmap=True )

    # Compute instantaneous spectral stats
    IP,IF,IA = emd.spectra.frequency_stats( imf, sample_rate ,'nht' )

    # Compute Hilbert-Huang transform
    edges,centres = emd.spectra.define_hist_bins(0,10,32)
    hht = emd.spectra.hilberthuang( IF, IA, edges )

    # Visualise time-frequency spectrum
    plt.figure()
    plt.pcolormesh( time_vect, centres, hht, cmap='hot_r')
    plt.colorbar()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Instantaneous  Frequency (Hz)')



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials.rst
   reference.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
