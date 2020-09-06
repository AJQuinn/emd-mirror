"""
Cycle detection from IMFs
=========================
Here we will use the 'cycle' submodule of EMD to identify and analyse individual cycles of an oscillatory signal

"""

#%%
# Simulating a noisy signal
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Firstly we will import emd and simulate a signal

import emd
import numpy as np
import matplotlib.pyplot as plt

# Define and simulate a simple signal
peak_freq = 12
sample_rate = 512
seconds = 10
noise_std = .5
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99)
t = np.linspace(0, seconds, seconds*sample_rate)

# Plot the first 5 seconds of data
plt.figure(figsize=(10, 2))
plt.plot(t[:sample_rate*3], x[:sample_rate*3], 'k')

# sphinx_gallery_thumbnail_number = 5


#%%
# Extract IMFs & find cycles
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# We next run a mask sift with the default parameters to isolate the 12Hz
# oscillation. There is only one clear oscillatory signal in this simulation.
# This is extracted in IMF-2 whilst the remaining IMFs contain low-amplitude
# noise.

# Run a mask sift
imf = emd.sift.mask_sift(x)

# Visualise the IMFs
emd.plotting.plot_imfs(imf[:sample_rate*4, :], cmap=True, scale_y=True)


#%%
# Next, we want to identify single cycles of any oscillations that are present
# in our IMFs. There are many ways to do this depending on the  signal and
# signal-features of interest. The EMD package provides a method which extract
# cycle locations based on the instantaneous phase of an IMF. In its simplest
# form this will detect successive cycles within the IMF, though we can also
# run some additional checks to reject 'bad' cycles or specify time-periods to
# exclude from the cycle detection.
#
# Will will run through some examples of these in the next couple of sections.
# First, we compute the instantaneous phase of our IMFs using the
# ``frequency_stats`` function before detecting cycle indices from the IP
# using ``get_cycle_inds``.
#
# The detection is based on finding large jumps in the instantaneous phase of
# each IMF. By default, we will consider any phase jump greater than 1.5*pi as
# a boundary between two  cycles. This can be customised using the
# ``phase_step`` argument in ``get_cycle_inds``.
#
# We can optionally run a set of tests on the detected cycles to remove 'bad'
# cycles from the analysis at an early stage. We will go into more detail into
# this later, for now we will run the function to get ``all_cycles``.

# Compute frequency domain features using the normalised-Hilbert transform
IP, IF, IA = emd.spectra.frequency_stats(imf, sample_rate, 'nht')

# Extract cycle locations
all_cycles = emd.cycles.get_cycle_inds(IP, return_good=False)

# ``all_cycles`` is an array of the same size as the input instantaneous phase.
# Each row contains a vector of itegers indexing the location of successive
# cycles for that IMF.

print('Input IMF shape is - {0}'.format(IP.shape))
print('Input all_cycles shape is - {0}'.format(all_cycles.shape))

#%%
# For each IMF, ``all_cycles`` stores either a zero or an integer greater than
# zero for each time sample. A value zero indicates that no  cycle is occurring
# at that time (perhaps as it has been excluded by our cycle quality checks in
# the next section) whilst a non-zero value indicates that that time-sample
# belongs to a specific cycle.

# Firstly, cycles  are detected by looking for phase 'jumps' where the phase
# resets between cycles.  The default threshold for a jump is 1.5*pi. Let's
# take a look at this in IMF-2

plt.figure(figsize=(8, 6))
plt.subplots_adjust(hspace=0.3)
plt.subplot(311)
plt.plot(t[:sample_rate*2], imf[:sample_rate*2, 2])
plt.gca().set_xticklabels([])
plt.title('IMF-2')
plt.subplot(312)
plt.plot(t[:sample_rate*2], IP[:sample_rate*2, 2])
plt.title('IMF-2 Instantaneous Phase')
plt.gca().set_xticklabels([])
plt.subplot(313)
plt.plot(t[1:sample_rate*2], np.abs(np.diff(IP[:sample_rate*2, 2])))
plt.plot((0, 2), (1.5*np.pi, 1.5*np.pi), 'k:')
plt.xlabel('Time (seconds)')
plt.title('IMF-2 Instantaneous Phase Abs-Differential')
plt.legend(['IMF-2 IP Differential', 'Jump threshold'], loc='upper right')

#%%
# We can see that the large phase jumps occur at the ascending zero-crossing of
# each cycle. In a clear signal, these are very simple to detect using a blunt
# threshold.

#%%
# What makes a 'good' cycle?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# There are many methods for detecting oscillatory cycles within a signal. Here
# we provide one approach for identifying whether a signal contains clear and
# interpretable cycles based on analysis of its instantaneous phase. This
# process takes the cycles detected by the phase jumps and runs four
# additional checks.
#
# We define a 'good' cycle as one with:
#
# 1. A strictly positively increasing phase
# 2. A phase starting within phase_step of zero ie the lowest value of IP must be less than phase_step
# 3. A phase ending within phase_step of 2pi the highest value of IP must be between 2pi and 2pi-phase_step
# 4. A set of 4 unique control points (ascending zero, peak, descending zero & trough)
#
# Lets take a look at these checks in IMF-2. Firstly, we run test 1:

# We use the unwrapped phased so we don't have to worry about jumps between cycles
unwrapped_phase = np.diff(np.unwrap(IP[:sample_rate*2, 2]))

# Plot the differential of the unwrapped phasee
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t[:sample_rate*2], IP[:sample_rate*2, 2])
plt.legend(['IMF-2 Instantaneous Phase'])
plt.title('Test-1: Check phase is strictly increasing')
plt.subplot(212)
plt.plot(t[1:sample_rate*2], unwrapped_phase)
plt.plot((0, 2), (0, 0), 'k:')
plt.ylim(-.2, .4)
plt.legend(['IMF-2 Instantaneous Phase Differential'])

#%%
# We can see that the instantaneous phase of most cycles is positive throughout
# the cycle. Only one cycle (around 1.6 seconds into the simulation) has
# negative values which correspond to a reversal in the normal wrapped IP.
#
# The second test looks to make sure that each cycles phase covers the whole
# 2pi range. If the phase  doesn't reach these limits it indicates that a phase
# jump occurred early or late in the cycle (for instance we might have a peak
# which is below zero in the raw time-course) leaving an incomplete
# oscillation.

plt.figure(figsize=(8, 4))
plt.title('Tests 2+3: Check phase covers the full 2pi range')
plt.plot(t[:sample_rate*2], IP[:sample_rate*2, 2], label='IP')
plt.plot((0, 2), (0, 0), label='0')
plt.plot((0, 2), (np.pi*2, np.pi*2), label='2pi')

plt.plot((0, 2), (np.pi*2-np.pi/12, np.pi*2-np.pi/12), ':', label='Upper Thresh')
plt.plot((0, 2), (np.pi/12, np.pi/12), ':', label='Lower Thresh')
plt.legend()

#%%
# We can see that most cycles have instantaneous phase values crossing both the
# upper and lower threshold. Only the first and last cycles in these segments
# are missing these thresholds (as they are incomplete cycles cutt-off at the
# edges of this segment)
#
# Finally, we check that we can detect a complete set of control points from
# each cycle. The control points are the peak, trough, ascending zero-crossing
# and descending zero-crossing. These can be computed from the IMF and a cycles
# vector using ``emd.cycles.get_control_points``

ctrl = emd.cycles.get_control_points(imf[:, 2], all_cycles[:, 2])

# Define some marker styles and legend labels for the control points.
markers = ['og', '^b', 'oc', 'vb', 'or']
label = ['Asc-Start', 'Peak', 'Desc', 'Trough', 'Asc-End']

# Plot the first 10 cycles with control points
ncycles = 21
start = 0

plt.figure()
plt.plot(111)
plt.title('Test 4: Control points')
for ii in range(1, ncycles):
    print('Cycle {0:2d} - {1}'.format(ii, ctrl[ii, :]))
    cycle = imf[all_cycles[:, 2] == ii + 1, 2]
    plt.plot(np.arange(len(cycle))+start, cycle, 'k', label='Cycle')
    for jj in range(5):
        if np.isfinite(ctrl[ii, jj]):
            plt.plot(ctrl[ii, jj]+start, cycle[int(ctrl[ii, jj])], markers[jj], label=label[jj])
    start += len(cycle)

    # Only plot the legend for the first cycle
    if ii == 1:
        plt.legend()

#%%
# Most of these cycles have the full set of control points present. Only ones
# cycle (cycle-20 - close to the end) is missing an indicator for its
# peak or trough. This is as a distortion in the cycle means that there are two
# peaks and troughs present. In this case, ``get_control_points`` will return a
# ``np.nan`` as the value for that peak.

#%%
# We run these checks together automatically by setting the ``return_good``
# option to ``True``. This is also the default option in the code. Here we run
# cycle detection with the quality checks on and look at the first four seconds
# of signal.

good_cycles = emd.cycles.get_cycle_inds(IP, return_good=True, phase_step=np.pi)

plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=.3)
plt.subplot(311)
plt.plot(t[:sample_rate*4], imf[:sample_rate*4, 2], 'k')
plt.gca().set_xticklabels([])
plt.title('IMF-2')
plt.subplot(312)
plt.plot(t[:sample_rate*4], IP[:sample_rate*4, 2], 'b')
plt.gca().set_xticklabels([])
plt.plot((0, 4), (0, 0), label='0')
plt.plot((0, 4), (np.pi*2, np.pi*2), label='2pi')
plt.plot((0, 4), (np.pi*2-np.pi/12, np.pi*2-np.pi/12), ':', label='Upper Thresh')
plt.plot((0, 4), (np.pi/12, np.pi/12), ':', label='Lower Thresh')
plt.title('Instantanous Phase')
plt.subplot(313)
plt.plot(t[:sample_rate*4], good_cycles[:sample_rate*4, 2])
plt.title('Good cycles')
plt.xlabel('Time (seconds)')


#%%
# Most cycles pass the checks but a few do fail. Closer inspection shows that
# these cycles tend to have large distortions or have very low amplitudes.
# Either way, the sift has not found a clear oscillation so these cycles
# should be interpreted with caution.

#%%
# We can use the information in ``all_cycles`` to find explore the cycle
# content of each IMF. For instance, this section prints the number of cycles
# identified in each IMF

for ii in range(all_cycles.shape[1]):
    all_count = all_cycles[:, ii].max()
    good_count = good_cycles[:, ii].max()
    percent = np.round(100*(good_count/all_count), 1)
    print('IMF-{0} contains {1:3d} cycles of which {2:3d} ({3}%) are good'.format(ii,
                                                                                  all_count,
                                                                                  good_count,
                                                                                  percent))

#%%
# IMF-2 contains our simulated oscillation with a spectral peak around 12Hz.
# As we would expect, the cycle detection finds around 120-130 cycles in this
# 10 second segment. Most of these cycles (94.4%) pass our cycle-quality checks
# indicating that they have well behaved instantaneous phase profiles that can
# be interpreted in detail.
#
# The lower frequency cycles (IMF 3+) have fewer and fewer cycles reflecting
# their slowing frequency content (each successive IMF extracts slower dynamics
# than the previous one). Again, most of these cycles pass the quality checks
# on their instantaneous phase.
# However, we also see that the the
# higher frequency IMFs  0 and 1 seem to contain fewer cycles than IMF-2. We
# would expect these IMFs to capture faster dynamics with more cycles in each
# IMF - so why are there fewer here?
#
# The answer is in the quality of the instantaneous phase estimation in very
# fast oscillations. Lets plot the IMF and IP for IMF-1

plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=.3)
plt.subplot(311)
plt.plot(t[:sample_rate//2], imf[:sample_rate//2, 0], 'k')
plt.gca().set_xticklabels([])
plt.title('IMF-0')

plt.subplot(312)
plt.plot(t[:sample_rate//2], IP[:sample_rate//2, 0], 'b')

plt.plot((0, .5), (0, 0), label='0')
plt.plot((0, .5), (np.pi*2, np.pi*2), label='2pi')
plt.plot((0, .5), (np.pi*2-np.pi/12, np.pi*2-np.pi/12), ':', label='Upper Thresh')
plt.plot((0, .5), (np.pi/12, np.pi/12), ':', label='Lower Thresh')

plt.gca().set_xticklabels([])
plt.title('Instantanous Phase')

plt.subplot(313)
plt.plot(t[:sample_rate//2], good_cycles[:sample_rate//2, 0])
plt.title('Good cycles')
plt.xlabel('Time (seconds)')

#%%
# The oscillations here are much faster than in IMF2. We only have a handful of
# samples for each potential cycle in IMF-1 compared to ~40 for IMF-2. As such,
# more cycles are showing distortions and failing the quality checks. In this
# case it is ok as there is no signal in IMF-1 in our simulation. Much of IMF-1
# is noisy
