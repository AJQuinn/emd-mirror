"""
The 'Cycles' class
================================
EMD provides a Cycles class to help with more complex cycle comparisons. This
class is based on the `emd.cycles.get_cycle_inds` and
`eemd.cycles.get_cycle_stat` functions we used in the previous tutorial, but it
does some additional hard work for you. For example, the Cycles class is a good
way to compute and store many different stats from the same cycles and for
dynamically working with different subsets of cycles based on user specified
conditions. Lets take a closer look...

"""

#%%
# Simulating a noisy signal
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Firstly we will import emd and simulate a signal.

import emd
import numpy as np
import matplotlib.pyplot as plt

# Define and simulate a simple signal
peak_freq = 12
sample_rate = 512
seconds = 60
noise_std = .5
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99) * 1e-4
t = np.linspace(0, seconds, seconds*sample_rate)

# Plot the first 5 seconds of data
plt.figure(figsize=(10, 2))
plt.plot(t[:sample_rate*4], x[:sample_rate*4], 'k')

# sphinx_gallery_thumbnail_number = 5

#%%
# We next run a mask sift with the default parameters to isolate the 12Hz
# oscillation. There is only one clear oscillatory signal in this simulation.
# This is extracted in IMF-2 whilst the remaining IMFs contain low-amplitude
# noise.

# Run a mask sift
imf = emd.sift.mask_sift(x)

# Computee frequenecy transforms
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')

#%%
# The Cycles class

#%%
# We next initialise the 'Cycles' class with the instantaneous phase of the second IMF.

C = emd.cycles.Cycles(IP[:, 2])

#%%
# This calls `emd.cycles.get_cycle_inds` on the phase time course to identify
# individual cycles. The cycle vector is stored in the class instance as
# `cycle_vect`. Here we plot the cycle vector for the first four seconds of
# our signal.

plt.figure(figsize=(10, 6))
plt.plot(t[:sample_rate*8], C.cycle_vect[:sample_rate*8], 'k')

#%%
# Note that `cycle_vect` does not exclude any cycles from the analysis. During
# the `emd.cycles.Cycles` class initialisation, `emd.cycles.get_cycle_inds`
# does not perform the good_cycle detection. Instead, the Cycles class contains
# a dictionary which can store different stats and metrics associated with our
# cycles.
#
# The results of the good cycle detection are stored in this dictionary under
# the key `is_good`. This is a vector containing a ones for cycles which pass
# the good cycle detection and zeros for the rest.

print(C.metrics)

#%%
# We can extract a cycle vector for only the good cycles using the
# `get_matching_cycles` method attached to the `Cycles` class. This function
# takes a list of one or more conditions and returns a booleaen vector
# indiciating which cycles match the conditions.  These conditions specify the
# name of a cycle metric, a standard comparator (such as ==, > or <) and a
# comparison value.
#
# Here, we will identify which cycles are passing our good cycle checks.

good_cycles = C.get_matching_cycles(['is_good==1'])
print(good_cycles)

#%%
# and which cycles are failing...

bad_cycles = C.get_matching_cycles(['is_good==0'])
print(bad_cycles)

#%%
# Now we see that several cycles have been excluded in the `good_cycles`
# vector. We can now use this to run other analyses on our subset of good
# cycles. For instance, here we compute the cycle control points on the good
# cycles,

ctrl = emd.cycles.get_control_points(imf[:, 2], C)
print(ctrl.shape)
print(ctrl)

#%%

ctrl = emd.cycles.get_control_points(imf[:, 2], C.iterate(conditions='is_good==1'))
print(ctrl.shape)
print(ctrl)

#%%
# This is convenient, but not so different from the standard analyses in the
# previous tutorial. The real utility of the `Cycles` class in is computing
# lots of custom metrics and selecting subsets using complex combinations of
# comparitors.

#%%
# Adding custom metrics
#^^^^^^^^^^^^^^^^^^^^^^

#%%
# We can add the metrics stored in a `Cycles` class instance using the
# `add_cycle_stat` function. This is a wrapper around
# `emd.cycles.get_cycle_stat` function which computes a stat for each cycle and
# stores the result in the metrics dictionary. `add_cycle_stat` takes a name
# for the metric, the time-series to compute the metric from and the function
# to evaluate for each cycle. This is always computed for every cycle in the
# dataset, we can include or exclude cycles based on different conditions
# later.
#
# Here we compute the maximum amplitude for each cycle and the length of each
# cycle in samples.

# Compute the maximum instantaneous amplitude per cycle
C.compute_cycle_metric('max_amp', IA[:, 2], np.max)

# Compute the length of each cycle
C.compute_cycle_metric('duration', IA[:, 2], len)

#%%
# These values are now stored in the `metrics` dictionary along with the good cycle values.

print(C.metrics.keys())

print(C.metrics['is_good'])
print(C.metrics['max_amp'])
print(C.metrics['duration'])

#%%
# We can also store arbitrary cycle stats in the dictionary - as long as there
# is one value for every cycle. This might include external values or more
# complex stats that are beyond the scope of `emd.cycles.get_cycle_stat`. These
# can be stored using the `Cycles.add_cycle_metric` method.
#
# Let's compute and store the time of the peak and trough in each cycle in milliseconds.

ctrl = emd.cycles.get_control_points(imf[:, 2], C)

peak_time_ms = ctrl[:, 1]/sample_rate * 1000
trough_time_ms = ctrl[:, 3]/sample_rate * 1000

C.add_cycle_metric('peak_time_ms', peak_time_ms)
C.add_cycle_metric('trough_time_ms', trough_time_ms)

#%%
# Once we have this many cycle metrics, the dictionary storage can be tricky to
# visualise (though it works well in the internal code). If you have
# python-pandas installed, you can export the metrics into a DataFrame which is
# easier to summarise and visualise.

d = C.get_metric_dataframe()
print(d)

#%%
# The summary table for the DataFrame gives a convenient summary description of
# the cycle metrics.

print(d.describe())

#%%
# Extracting subsets of cycles
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
# At the start of this tutorial, we extracted the cycle vector for the good
# cycles using `get_cycle_vector` with a simple comparator. We can use this
# function to specify a range of comparators to select subsets of cycles based
# on the computed metrics.
#
# For example, lets get the cycle vector for cycles whose duration is longer than 40 samples.

long_cycles = C.get_cycle_vector('duration>40')

plt.figure(figsize=(10, 6))
plt.plot(t[:sample_rate*8], long_cycles[:sample_rate*8], 'k')

#%%
# If we want to use more than one camparator, these can be passed as a list.

# Now, we find only long cycles which also pass the good cycle detection
long_good_cycles = C.get_cycle_vector(['duration>40', 'is_good==1'])

# Finally, find long-good cycles which also have amplitudes above 1.25
big_long_good_cycles = C.get_cycle_vector(['max_amp>1.25', 'duration>40', 'is_good==1'])

plt.figure(figsize=(10, 6))
plt.plot(t[:sample_rate*8], big_long_good_cycles[:sample_rate*8], 'k')

#%%
# The cycle subset extraction can also be applied to the pandas dataframes

big_long_good_cyclemetrics = C.get_metric_dataframe(['max_amp>1.25', 'duration>40', 'is_good==1'])
print(big_long_good_cyclemetrics)

#%%
# Or both and cycle vector and summary data frame can be extracted togther
# using `Cycles.get_subset`.

cycle_vect, cycle_df = C.get_subset(['max_amp>1.25', 'duration>40', 'is_good==1'])

#%%
# Cycle chain analysis
#^^^^^^^^^^^^^^^^^^^^^

#%%
# Finally, we can use the `emd.cycles.Cycles` class to help with cycle chain
# analyses. This illustrates one of the most complex use-cases for the `Cycles`
# object! Computing metrics from groups of cycles and mapping these back to
# cycle-level metrics can involve some difficult indexing.
#
# Lets extract the big-long-good cycles and compute the continuous
# chains of cycles within this subset.

cycle_vect, cycle_df = C.get_subset(['max_amp>1', 'duration>30', 'is_good==1'])

cycle_chains = emd.cycles.get_cycle_chain(cycle_vect)

#%%
# Next we compute the maxumum amplitude and duration of each chain.

chain_amp = emd.cycles.get_chain_stat(cycle_chains, cycle_df['max_amp'], np.max)
chain_len = emd.cycles.get_chain_stat(cycle_chains, cycle_df['max_amp'], len)

#%%
# We next map these cycle-chain values back to a vector in which each sample
# contains the chain stat corresponding to the

cycle_amp_vect = emd.cycles._map_cycles_to_samples(cycle_df['max_amp'], cycle_vect)

chain_amp_vect = emd.cycles._map_chains_to_samples(chain_amp, cycle_chains, cycle_vect)
chain_len_vect = emd.cycles._map_chains_to_samples(chain_len, cycle_chains, cycle_vect)

# Plot the first 5 seconds of data
plt.figure(figsize=(10, 6))
plt.plot(t[:sample_rate*5], imf[:sample_rate*5, 2], 'k')
plt.plot(t[:sample_rate*5], IA[:sample_rate*5, 2], 'b')
plt.plot(t[:sample_rate*5], cycle_amp_vect[:sample_rate*5], 'g')
plt.plot(t[:sample_rate*5], chain_amp_vect[:sample_rate*5], 'r')
plt.legend(['IMF', 'InstAmp', 'CycleMaxAmp', 'ChainMaxAmp'])


#%%
# Finally, we use these chain metric vectors as inputs to
# `Cycles.add_cycle_stat` to get a measure of the amplitude or duration of the
# cycle-chain that each cycle belongs to  (and values of -1 for cycles which
# don't belong to any chain).

C.add_cycle_stat('chain_amp', chain_amp_vect, np.mean)
C.add_cycle_stat('chain_len', chain_len_vect, np.mean)

#%%
# The pandas dataframe can now summarise these new metrics

full_df = C.get_metric_dataframe()
print(full_df)
print(full_df.describe())
