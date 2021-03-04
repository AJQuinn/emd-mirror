"""
Code speed and efficiency
=========================
EMD analysis can be time-consuming. This tutorial outlines some basic
information about how long different computations may take and what features
can be used to speed this up.

"""

#%%
# Sift Speed
#^^^^^^^^^^^

#%%
# The sift can be time-consuming for two reasons. Firstly, it is an iterative
# process which can vary in how long it takes to converge. Though many signals
# can be sifted in a handful of iterations some may take tens or hundreds of
# iterations before an IMF is identified - unfortunately we can't tell before
# the process is running. Secondly, the sift is sequential in that we can't
# compute the second IMF until the first IMF has been identified.

#%%
# The default settings in the sift are selected to operate reasonably well and
# reasonable quickly on a signal. Here we include some a very rough, order of
# magnitude illustration of timings based on running speeds on a modern
# computer (the readthedocs server generating this website).

import emd
import time
import numpy as np


# ---- Ten thousand sample example
x = np.random.randn(10000,)

t = time.process_time()
imf = emd.sift.sift(x)
elapsed = 1000 * (time.process_time() - t)
print('{0} samples sifted in {1} milliseconds'.format(10000, elapsed))

# ---- Five thousand samples example
x = np.random.randn(5000,)

t = time.process_time()
imf = emd.sift.sift(x)
elapsed = 1000 * (time.process_time() - t)
print('{0} samples sifted in {1} milliseconds'.format(5000, elapsed))

# ---- Five hundred samples example
x = np.random.randn(500,)

t = time.process_time()
imf = emd.sift.sift(x)
elapsed = 1000 * (time.process_time() - t)
print('{0} samples sifted in {1} milliseconds'.format(500, elapsed))

#%%
# The sift executes in well less than a second for all examples. Computation
# time increases with input array size linearly for relatively short input but
# exponentially but larger ones (>1 million samples, not computed here...).

#%%
# Some options can noticeably slow down the sift. For example, the imf option
# ``imf_opts/stop_method='rilling'`` is tends to use more iterations than the
# default ``imf_opts/stop_method='sd'``. Similarly changing the thresholds for
# either stopping method can increase the number of iterations computed. the
# envelope interpolation method ``envelope_opes/interp_method='mono_pchip'`` is
# much slower than the default ``envelope_opes/interp_method='splrep'``

#%%
# Sift Variants
#^^^^^^^^^^^^^^

#%%
# Compared to the classic sift, the ensemble and mask sift are slower but have
# more options for speeding up computation. The computation speed of
# ``emd.sift.ensemble_sift`` and ``emd.sift.complete_ensemble_sift`` is most
# strongly determined by the number of ensembles that are computed - however,
# these can be parallelised by setting the ``nprocesses`` option to be greater
# than 1.

# Run an ensemble sift with 24 ensembles
imf = emd.sift.ensemble_sift(x, nensembles=24, max_imfs=6)

# Run an ensemble sift with the 24 ensembles splits across 6 parallel threads
imf = emd.sift.ensemble_sift(x, nensembles=24, max_imfs=6, nprocesses=6)


#%%
# Similarly, the timing of ``emd.sift.mask_sift`` is strongly determined by the
# number of separate masks applied to each IMF - specified by ``nphases``.
# Again this can be parallelised by setting ``nprocesses`` to speed up
# computation time.

# Compute a mask sift, applying four masks per IMF
imf = emd.sift.mask_sift(x, nphases=4)

# Compute a mask sift, applying four masks per IMF split across 4 parallel processes
imf = emd.sift.mask_sift(x, nphases=4, nprocesses=4)

#%%
# Sparse Time-Frequency Transforms.
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# Another potentially slow computation during an EMD analysis is generating
# Hilbert-Huang and Holospectrum arrays. Both of these algorithms make use of
# nested looping to form the output. As this can be very slow, these operations
# are accelerated internally by using sparse arrays. This allows the
# Hilbert-Huang transform and Holospectrum arrays to be formed in one shot
# without looping.
#
# By default, these outputs are cast to normal numpy arrays before being
# returned to the user. If you are working with a very large transform, it is
# far more memory and computationally efficient to work with the sparse form of
# these arrays. These can be returned by specifying ``return_sparse=True`` in
# the options in either ``emd.spectra.hilberthuang`` or
# ``emd.spectra.holospectrum``.

IP, IF, IA = emd.spectra.frequency_transform(imf, 1, 'hilbert')
freq_edges, freq_bins = emd.spectra.define_hist_bins(0, .5, 75)

msg = 'Output is a {0} of size {1} using {2}Kb of memory'

hht = emd.spectra.hilberthuang(IF, IA, freq_edges)
print(msg.format(type(hht), hht.shape, hht.nbytes/1024))

hht = emd.spectra.hilberthuang(IF, IA, freq_edges, return_sparse=True)
print(msg.format(type(hht), hht.shape, hht.data.nbytes/1024))
