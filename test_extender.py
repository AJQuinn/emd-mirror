
from scipy import interpolate as interp
import numpy as np

sample_rate = 1000
time_vect = np.linspace(0, 2, 2*sample_rate)

X = np.sin( 2*np.pi*10*time_vect) + .6*np.sin(2*np.pi*4*time_vect)

import emd
max_locs,max_pks = emd.find_extrema( X )
min_locs,min_pks = emd.find_extrema( X, ret_min=True)

ret_max_locs = np.pad( max_locs,4,'reflect',reflect_type='odd')
ret_min_locs = np.pad( min_locs,4,'reflect',reflect_type='odd')

ret_max_pks = np.pad( max_pks,4,'reflect',reflect_type='even')
ret_min_pks = np.pad( min_pks,4,'reflect',reflect_type='even')

f = interp.splrep( ret_max_locs, ret_max_pks )
upper = interp.splev(range(0,2*sample_rate), f)
f = interp.splrep( ret_min_locs, ret_min_pks )
lower = interp.splev(range(0,2*sample_rate), f)

import matplotlib.pyplot as plt
plt.plot(X)
plt.plot(max_locs,max_pks,'*')
plt.plot(min_locs,min_pks,'o')

plt.plot(ret_max_locs,ret_max_pks,'*')
plt.plot(ret_min_locs,ret_min_pks,'o')
