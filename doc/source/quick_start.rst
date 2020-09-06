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
