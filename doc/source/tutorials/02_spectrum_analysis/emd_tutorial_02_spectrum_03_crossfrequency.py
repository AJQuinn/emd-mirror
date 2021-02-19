"""
Cross-Frequency Coupling
========================
This tutorial shows how we can compute a holospectrum to characterise the
distribution of power in a signal as a function of both frequency of the
carrier wave and the frequency of any amplitude modulations

"""

#%%
# Simulating a signal with amplitude modulations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First of all, we import EMD alongside numpy and matplotlib. We will also use
# scipy's ndimage module to smooth our results for visualisation later.

# sphinx_gallery_thumbnail_number = 5

import matplotlib.pyplot as plt
import numpy as np
import emd

#%%
# First we create a simulated signal to analyse. This signal will be composed
# of a  linear trend and two oscillations, each with a different amplitude
# modulation.

seconds = 60
sample_rate = 200
t = np.linspace(0, seconds, seconds*sample_rate)

# First we create a slow 5Hz oscillation and a fast 37Hz oscillation
slow = np.sin(2*np.pi*5*t)

fast = np.cos(2*np.pi*37*t)

# Next we create three different amplitude modulation signals for the fast
# oscillation. One sinusoidal, one wide modulation and one narrow modulation.
# These cases differ by the duration of the high frequency burst.

fast_am = 0.5*slow + 0.5
fast_am_narrow = fast_am**4
fast_am_wide = 1 - (0.5*-slow + 0.5)**4

# We create our signal by summing the oscillation and adding some noise
x = slow + fast * fast_am + np.random.randn(*t.shape) * .05
x_wide = slow + fast * fast_am_wide + np.random.randn(*t.shape) * .05
x_narrow = slow + fast * fast_am_narrow + np.random.randn(*t.shape) * .05


plt.figure()
plt.subplot(311)
plt.plot(t[:seconds*3], x_narrow[:seconds*3])
plt.subplot(312)
plt.plot(t[:seconds*3], x[:seconds*3])
plt.subplot(313)
plt.plot(t[:seconds*3], x_wide[:seconds*3])

#%%


config = emd.sift.get_config('mask_sift')
config['max_imfs'] = 7
config['mask_freqs'] = 50/sample_rate
config['mask_amp_mode'] = 'ratio_sig'
config['imf_opts/sd_thresh'] = 0.05

imf = emd.sift.mask_sift(x, **config)
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')

imf_wide = emd.sift.mask_sift(x_wide, **config)
IP_wide, IF_wide, IA_wide = emd.spectra.frequency_transform(imf_wide, sample_rate, 'hilbert')

imf_narrow = emd.sift.mask_sift(x_narrow, **config)
IP_narrow, IF_narrow, IA_narrow = emd.spectra.frequency_transform(imf_narrow, sample_rate, 'hilbert')

#%%
# 1D Phase-amplitude coupling with instantaneous amplitude
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cycles = emd.cycles.get_cycle_inds(IP[:, 4])
phase = np.linspace(0, np.pi*2, 48)
pa = emd.cycles.phase_align(IP[:, 4], IA[:, 0], cycles=cycles)
pa_wide = emd.cycles.phase_align(IP[:, 4], IA_wide[:, 0], cycles=cycles)
pa_narrow = emd.cycles.phase_align(IP[:, 4], IA_narrow[:, 0], cycles=cycles)

plt.figure()
plt.plot(phase, pa.mean(axis=1))
plt.plot(phase, pa_wide.mean(axis=1))
plt.plot(phase, pa_narrow.mean(axis=1))
plt.legend(['Fast Amp', 'Fast Amp Wide', ' Fast Amp Narrow'])
plt.xlabel('Slow Phase')

#%%


plt.figure()
plt.subplot(311)
plt.plot(t[:seconds*3], imf_narrow[:seconds*3, 0])
plt.plot(t[:seconds*3], IA_narrow[:seconds*3, 0])
plt.subplot(312)
plt.plot(t[:seconds*3], imf[:seconds*3, 0])
plt.plot(t[:seconds*3], IA[:seconds*3, 0])
plt.subplot(313)
plt.plot(t[:seconds*3], imf_wide[:seconds*3, 0])
plt.plot(t[:seconds*3], IA_wide[:seconds*3, 0])


#%%
# 2D Phase-amplitude coupling with the HHT
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


freq_edges, freq_centres = emd.spectra.define_hist_bins(10, 75, 75, 'log')

hht = emd.spectra.hilberthuang(IF, IA, freq_edges, mode='amplitude')
hht_wide = emd.spectra.hilberthuang(IF_wide, IA_wide, freq_edges, mode='amplitude')
hht_narrow = emd.spectra.hilberthuang(IF_narrow, IA_narrow, freq_edges, mode='amplitude')

#%%

plt.figure()
plt.subplot(411)
plt.plot(t[:seconds*3], x[:seconds*3])
plt.subplot(412)
plt.pcolormesh(t[:seconds*3], freq_centres, hht_narrow[:, :seconds*3], cmap='ocean_r')
plt.subplot(413)
plt.pcolormesh(t[:seconds*3], freq_centres, hht[:, :seconds*3], cmap='ocean_r')
plt.subplot(414)
plt.pcolormesh(t[:seconds*3], freq_centres, hht_wide[:, :seconds*3], cmap='ocean_r')

#%%

hht_by_phase, _, phase_centres = emd.cycles.bin_by_phase(IP[:, 4], hht.T)
hht_by_phase_wide, _, _ = emd.cycles.bin_by_phase(IP[:, 4], hht_wide.T)
hht_by_phase_narrow, _, _ = emd.cycles.bin_by_phase(IP[:, 4], hht_narrow.T)


plt.figure()
plt.subplot(131)
plt.pcolormesh(phase_centres, freq_centres, hht_by_phase_narrow.T, vmax=0.25)
plt.subplot(132)
plt.pcolormesh(phase_centres, freq_centres, hht_by_phase.T, vmax=0.25)
plt.subplot(133)
plt.pcolormesh(phase_centres, freq_centres, hht_by_phase_wide.T, vmax=0.25)

#%%
# Quantifying the frequency of amplitude modulations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#%%
config['mask_freqs'] = [10/sample_rate/ii for ii in range(1, 10)]
imf2 = emd.sift.mask_sift_second_layer(IA[:, 0], config['mask_freqs'], sift_args=config)
IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, sample_rate, 'hilbert')

imf2_wide = emd.sift.mask_sift_second_layer(IA_wide[:, 0], config['mask_freqs'], sift_args=config)
IP2_wide, IF2_wide, IA2_wide = emd.spectra.frequency_transform(imf2_wide, sample_rate, 'hilbert')

imf2_narrow = emd.sift.mask_sift_second_layer(IA_narrow[:, 0], config['mask_freqs'], sift_args=config)
IP2_narrow, IF2_narrow, IA2_narrow = emd.spectra.frequency_transform(imf2_narrow, sample_rate, 'hilbert')

#%%

freq_edges, freq_centres = emd.spectra.define_hist_bins(1, 75, 75, 'log')
freq_edges_low, freq_centres_low = emd.spectra.define_hist_bins(2, 10, 20, 'linear')

holo = emd.spectra.holospectrum(IF[:, 0, None], IF2, IA2, freq_edges, freq_edges_low)
holo_narrow = emd.spectra.holospectrum(IF_narrow[:, 0, None], IF2_narrow, IA2_narrow, freq_edges, freq_edges_low)
holo_wide = emd.spectra.holospectrum(IF_wide[:, 0, None], IF2_wide, IA2_wide, freq_edges, freq_edges_low)

#%%

plt.figure()
plt.subplot(231)
plt.pcolormesh(freq_centres_low, freq_centres, holo_narrow.T)
plt.subplot(234)
plt.plot(freq_centres_low, holo_narrow[:, 62])

plt.subplot(232)
plt.pcolormesh(freq_centres_low, freq_centres, holo.T)
plt.subplot(235)
plt.plot(freq_centres_low, holo[:, 62])

plt.subplot(233)
plt.pcolormesh(freq_centres_low, freq_centres, holo_wide.T)
plt.subplot(236)
plt.plot(freq_centres_low, holo_wide[:, 62])


#%%
# 3D phase-amplitude coupling with the Holospectrum
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%

freq_edges_low, freq_centres_low = emd.spectra.define_hist_bins(.5, 15, 32, 'linear')

holot = emd.spectra.holospectrum(IF[:, 0, None], IF2, IA2,
                                 freq_edges, freq_edges_low, squash_time=False)
holot_wide = emd.spectra.holospectrum(IF_wide[:, 0, None], IF2_wide, IA2_wide,
                                      freq_edges, freq_edges_low, squash_time=False)
holot_narrow = emd.spectra.holospectrum(IF_narrow[:, 0, None], IF2_narrow, IA2_narrow,
                                        freq_edges, freq_edges_low, squash_time=False)

holo_by_phase, _, _ = emd.cycles.bin_by_phase(IP[:, 4], holot)
holo_by_phase_wide, _, _ = emd.cycles.bin_by_phase(IP[:, 4], holot_wide)
holo_by_phase_narrow, _, _ = emd.cycles.bin_by_phase(IP[:, 4], holot_narrow)

plt.figure(figsize=(8, 8))
plt.subplot(331)
plt.plot(phase, pa_narrow.mean(axis=1))
plt.ylim(0, 1)
plt.subplot(332)
plt.plot(phase, pa.mean(axis=1))
plt.ylim(0, 1)
plt.subplot(333)
plt.plot(phase, pa_wide.mean(axis=1))
plt.ylim(0, 1)

plt.subplot(334)
plt.pcolormesh(phase_centres, freq_centres, holo_by_phase_narrow.sum(axis=1).T)
plt.ylabel('Frequency (Hz)')
plt.subplot(335)
plt.pcolormesh(phase_centres, freq_centres, holo_by_phase.sum(axis=1).T)
plt.subplot(336)
plt.pcolormesh(phase_centres, freq_centres, holo_by_phase_wide.sum(axis=1).T)

plt.subplot(337)
plt.pcolormesh(phase_centres, freq_centres_low, holo_by_phase_narrow.sum(axis=2).T)
plt.ylabel('Amplitude Modulation\nFrequency (Hz)')
plt.xlabel('Slow phase')
plt.subplot(338)
plt.pcolormesh(phase_centres, freq_centres_low, holo_by_phase.sum(axis=2).T)
plt.subplot(339)
plt.pcolormesh(phase_centres, freq_centres_low, holo_by_phase_wide.sum(axis=2).T)
