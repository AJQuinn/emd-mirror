"""
The Sifting algorithm
=====================
This tutorial introduces the sifting algorithm in detail.

"""

#%%
# Creating an example signal
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First of all, we import EMD alongside numpy and matplotlib. We will also use
# scipy's ndimage module to smooth our results for visualisation later.

import matplotlib.pyplot as plt
import numpy as np
import emd

#%%
# We then define a simulated waveform containing a non-linear wave at 5Hz and a
# sinusoid at 1Hz - this is the same signal as we used in tutorial 1.

sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds

time_vect = np.linspace(0, seconds, num_samples)

freq = 5

# Change extent of deformation from sinusoidal shape [-1 to 1]
nonlinearity_deg = .25

# Change left-right skew of deformation [-pi to pi]
nonlinearity_phi = -np.pi/4

# Compute the signal
x = emd.utils.abreu2010( freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds )
x += np.cos( 2*np.pi*1*time_vect )

#%%
# The first stage in the sift detects local extrema (maxima and minima) in the time-series

max_locs, max_mag = emd.utils.find_extrema( x )
min_locs, min_mag = emd.utils.find_extrema( x, ret_min=True )

plt.figure(figsize=(12,3))
plt.plot( x, 'k' )
plt.plot( max_locs, max_mag, 'or' )
plt.plot( min_locs, min_mag, 'ob' )
plt.legend(['Signal','Maxima','Minima'])


#%%
# Extrema padding is used to stablise the envelope at the edges of the time-series.

max_locs, max_mag = emd.utils.get_padded_extrema( x )
min_locs, min_mag = emd.utils.get_padded_extrema( -x )
min_mag = -min_mag

plt.figure(figsize=(12,3))
plt.plot( x, 'k' )
plt.plot( max_locs, max_mag, 'or' )
plt.plot( min_locs, min_mag, 'ob' )
plt.legend(['Signal','Maxima','Minima'])

#%%
# We next interpolate the upper and lower extrema to get envelopes and take the average of those

from scipy import interpolate as interp

max_time = np.arange(max_locs[0],max_locs[-1])
max_func = interp.splrep(max_locs, max_mag)
upper_env = interp.splev(max_time, max_func)

min_time = np.arange(min_locs[0],min_locs[-1])
min_func = interp.splrep(min_locs, min_mag)
lower_env = interp.splev(min_time, min_func)

plt.figure(figsize=(12,3))
plt.plot( x, 'k' )
plt.plot( max_locs, max_mag, 'or' )
plt.plot( max_time, upper_env, 'r' )
plt.plot( min_locs, min_mag, 'ob' )
plt.plot( min_time, lower_env, 'b' )
plt.legend(['Signal','Maxima','Upper Envelope','Minima','Lower Envelope'])

#%%

upper_env = emd.utils.interp_envelope( X, mode='upper' )
lower_env = emd.utils.interp_envelope( X, mode='lower' )
avg_env = (upper_env+lower_env) / 2

proto_imf = x.copy()

plt.figure(figsize=(12,12))
for ii in range(4):

    upper_env = emd.utils.interp_envelope( proto_imf, mode='upper' )
    lower_env = emd.utils.interp_envelope( proto_imf, mode='lower' )
    avg_env = (upper_env+lower_env) / 2

    plt.subplot(4,1,ii+1)
    plt.plot( proto_imf, 'k' )
    plt.plot( upper_env, 'r' )
    plt.plot( lower_env, 'b' )
    plt.plot( avg_env, 'g')
    plt.legend(['Signal','Maxima','Upper Envelope','Minima','Lower Envelope'])

    proto_imf = proto_imf - arg_env
