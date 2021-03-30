"""
The 'Cycles' class
================================
EMD provides a Cycles class to help with more complex cycle comparisons. This
class is based on the `emd.cycles.get_cycle_vector` and
`emd.cycles.get_cycle_stat` functions we used in the previous tutorial, but it
does some additional work for you. For example, the Cycles class is a good way
to compute and store many different stats from the same cycles and for
dynamically working with different subsets of cycles based on user specified
conditions. Lets take a closer look...

"""

#%%
# Getting started
# ^^^^^^^^^^^^^^^
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

# sphinx_gallery_thumbnail_number = 4

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
# ^^^^^^^^^^^^^^^^

#%%
# We next initialise the 'Cycles' class with the instantaneous phase of the second IMF.

C = emd.cycles.Cycles(IP[:, 2])

#%%
# This calls `emd.cycles.get_cycle_vect` on the phase time course to identify
# individual cycles and then stores a load of relevant information which we can
# use later. The cycle vector is stored in the class instance as `cycle_vect`.
# Here we plot the cycle vector for the first four seconds of our signal.

plt.figure(figsize=(10, 6))
plt.plot(t[:sample_rate*8], C.cycle_vect[:sample_rate*8], 'k')

#%%
# The Cycles class has an attached function to help identify when specific
# cycles occurred in a dataset. The ``C.get_inds_of_cycle`` function finds and
# returns the samples in which the Nth cycle occurred. Here, we run this find
# and plot three cycles from our simulation. The cycle in the original
# time-series is plotted in grey and the cycle from the second IMF is in
# colour.

cycles_to_plot = [5, 23, 42]
plt.figure()
for ii in range(len(cycles_to_plot)):
    inds = C.get_inds_of_cycle(cycles_to_plot[ii])
    xinds = np.arange(len(inds)) + 55*ii
    plt.plot(xinds, x[inds], color=[0.8, 0.8, 0.8])
    plt.plot(xinds, imf[inds, 2])

#%%
# These cycles contain one complete period of an oscillation and form the basis
# for a lot of the computations in this tutorial. However there are a couple of
# shortcomings with this standard cycle. For example, we may want to separately
# analyse the ascending and descending edges of the oscillation, whilst the
# descending edge is continuous - the cycles above contain two halves of two
# separate ascending edges at the start and end of the cycle.
#
# We could adjust the phase to make our cycle identification start at the peak
# to ensure the ascending edge is continuous, but this will just split another
# part of the cycle... One way around this is to consider an 'augmented' cycle
# which contains a whole standard cycle plus the last quadrant of the
# proceeding cycle. These five quarters of a cycle mean that all sections of
# the cycle are continuously represented, though it does meaen that some parts
# of the data may be present in more than one cycle.
#
# We can work with augmented cycles by specifying ``mode='augmented'`` when
# finding our cycle indices.

cycles_to_plot = [5, 23, 42]
plt.figure()
for ii in range(len(cycles_to_plot)):
    inds = C.get_inds_of_cycle(cycles_to_plot[ii], mode='augmented')
    xinds = np.arange(len(inds)) + 60*ii
    plt.plot(xinds, x[inds], color=[0.8, 0.8, 0.8])
    plt.plot(xinds, imf[inds, 2])

#%%
# The cycles class can be used as an input to several other ``emd.cycles``
# functions to specify which cycles a particular computation should run across.
#
# For example, here we compute the control points across from IMF-3 for each of
# our cycles.

ctrl = emd.cycles.get_control_points(imf[:, 2], C)

#%%
# ...and here we run phase-alignment.

pa = emd.cycles.phase_align(IP[:, 2], IF[:, 2], C)

#%%
# Computing cycle metrics
# ^^^^^^^^^^^^^^^^^^^^^^^

#%%
# We can loop through our cycles using the ``C.get_inds_of_cycle`` function to
# identify a each cycle in turn. Here we run a loop to compute the maximum
# amplitude of each cycle.

amps = np.zeros((C.ncycles,))
for ii in range(C.ncycles):
    inds = C.get_inds_of_cycle(ii)
    amps[ii] = np.max(IA[inds, 2])

print(amps)

#%%
# The Cycles class has a handy method to help automate this process. Simply
# specify a metric name, some values to compute a per-cycle metric on and a
# function and ``C.compute_cycle_metric`` will loop across all cycles and store
# the result for you.

C.compute_cycle_metric('max_amp', IA[:, 2], func=np.max)

#%%
# This is always computed for every cycle in the dataset, we can include or
# exclude cycles based on different conditions later.
#
# For another example we compute the length of each cycle in samples.

# Compute the length of each cycle
C.compute_cycle_metric('duration', IA[:, 2], len)

#%%
# Cycle metrics can also be computed on the augmented cycles. Lets compute the
# standard deviation of amplitude values for each augmented cycle.

C.compute_cycle_metric('ampSD', IA[:, 2], np.std, mode='augmented')

#%%
# We have now computed four different metrics across our cycles.

print(C)

#%%
# These values are now stored in the `metrics` dictionary along with the good cycle values.

print(C.metrics.keys())

print(C.metrics['is_good'])
print(C.metrics['max_amp'])
print(C.metrics['duration'])

#%%
# These values can be accessed and used for further analyses as needed. The
# metrics can be copied into a pandas dataframe for further analysis if
# convenient.

df = C.get_metric_dataframe()
print(df)

#%%
# We can extract a cycle vector for only the good cycles using the
# `get_matching_cycles` method attached to the `Cycles` class. This function
# takes a list of one or more conditions and returns a booleaen vector
# indicating which cycles match the conditions.  These conditions specify the
# name of a cycle metric, a standard comparator (such as ==, > or <) and a
# comparison value.
#
# Here, we will identify which cycles are passing our good cycle checks.

good_cycles = C.get_matching_cycles(['is_good==1'])
print(good_cycles)

#%%
# This returns a boolean array indicating which cycles meet the specified conditions.

print('{0} matching cycles found'.format(np.sum(good_cycles)))

#%%
# and which cycles are failing...

bad_cycles = C.get_matching_cycles(['is_good==0'])
print('{0} matching cycles found'.format(np.sum(bad_cycles)))
print(bad_cycles)

#%%
# Several conditions can be specified in a list

good_cycles = C.get_matching_cycles(['is_good==1', 'duration>40', 'max_amp>1'])
print('{0} matching cycles found'.format(np.sum(good_cycles)))
print(good_cycles)

#%%
# The conditions can also be used to specify  which cycles to include in a
# pandas dataframe.

df = C.get_metric_dataframe(conditions=['is_good==1', 'duration>40', 'max_amp>1'])
print(df)

#%%
# Adding custom metrics
#^^^^^^^^^^^^^^^^^^^^^^

#%%
# Any function that takes a vector input and returns a single value can be used
# to compute cycle metrics. Here we make a complex user-defined function which
# computes the degree-of-nonlinearity of each cycle.


def degree_nonlinearity(x):
    """Compute degree of nonlinearity. Eqn 3 in
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168108"""
    y = np.sum(((x-x.mean()) / x.mean())**2)
    return np.sqrt(y / len(x))


C.compute_cycle_metric('DoN', IF[:, 2], degree_nonlinearity)


#%%
# Custom metrics which take multiple time-series arguments can also be defined.
# In these cases a tuple of vectors is passed into `compute_cycle_metric` and
# the samples for each cycle are indexed out of each vector and passed to the
# function. For example, here we compute the correlation between the IMF-3 and
# the raw time-course for each cycle.


def my_corr(x, y):
    return np.corrcoef(x, y)[0, 1]


C.compute_cycle_metric('raw_corr', (imf[:, 2], x[:, 0]), my_corr)


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

df = C.get_metric_dataframe()
print(df)

#%%
# The summary table for the DataFrame gives a convenient summary description of
# the cycle metrics.

print(df.describe())


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

C.pick_cycle_subset(['max_amp>1', 'duration>30', 'is_good==1'])

#%%
# This computes two additional variables. Firstly, a ``subset_vect`` which maps
# cycles into 'good' cycles matching our conditions with -1 indicating a cycle
# which was removed.

print(C.subset_vect)

#%%
# Secondly, a ``chain_vect`` defines which groups of cycles in the subset form
# continuous chains.

print(C.chain_vect)

#%%
# There is a helper function in the Cycles object which computes a set of
# simple chain timing metrics. These are 'chain_ind', `chain_start`,
# `chain_end`, `chain_len_samples`, `chain_len_cycles` and `chain_position`.
# Each metric is computed and a value saved out for each cycle.

C.compute_chain_timings()

df = C.get_metric_dataframe(subset=True)
print(df)

#%%
# We can also compute chain specific metrics similar to how we compute cycle
# metrics. Each chain metric is saved out for each cycle within the chain. Here
# we compute the maximum amplitude for each chain and plot its relationship
# with chain length.

C.compute_chain_metric('chain_max_amp', IA[:, 2], np.max)
df = C.get_metric_dataframe(subset=True)

plt.figure()
plt.plot(df['chain_len_samples'], df['chain_max_amp'], '.')
plt.xlabel('Chain Length (samples)')
plt.ylabel('Chain Max-Amplitude')

#%%
# We can then select the cycle metrics from the cycles in a single chain by
# specifying the chain index as a condition.

df = C.get_metric_dataframe(conditions='chain_ind==42')
print(df)
