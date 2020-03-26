A python package for Empirical Mode Decomposition and related spectral analyses.

Please note that this project is in active development for the moment - the API may change relatively quickly between releases!

# Installation

You can install the latest stable release from the PyPI repository

```
pip install emd
```

or clone and install the source code.

```
git clone https://gitlab.com/emd-dev/emd.git
cd emd
pip install .
```

Requirements are specified in requirements.txt. Main functionality only depends
on numpy and scipy for computation and matplotlib for visualisation.

# Quick Start

Full documentation can be found at https://emd.readthedocs.org and development/issue tracking at gitlab.com/emd-dev/emd

Import emd

```python
import emd
```

Define a simulated waveform containing a non-linear wave at 5Hz and a sinusoid at 1Hz.

```python
sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds

import numpy as np
time_vect = np.linspace(0, seconds, num_samples)

freq = 5
nonlinearity_deg = .25 # change extent of deformation from sinusoidal shape [-1 to 1]
nonlinearity_phi = -np.pi/4 # change left-right skew of deformation [-pi to pi]
x = emd.utils.abreu2010( freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds )
x += np.cos( 2*np.pi*1*time_vect )
```

Estimate IMFs

```python
imf = emd.sift.sift( x )
```

Compute instantaneous frequency, phase and amplitude using the Normalised Hilbert Transform Method.

```python
IP,IF,IA = emd.spectra.frequency_stats( imf, sample_rate, 'nht' )
```
Compute Hilbert-Huang spectrum

```python
freq_edges,freq_bins = emd.spectra.define_hist_bins(0,10,100)
hht = emd.spectra.hilberthuang( IF, IA, freq_edges )
```
Make a summary plot

```python
import matplotlib.pyplot as plt
plt.figure( figsize=(16,8) )
plt.subplot(211,frameon=False)
plt.plot(time_vect,x,'k')
plt.plot(time_vect,imf[:,0]-4,'r')
plt.plot(time_vect,imf[:,1]-8,'g')
plt.plot(time_vect,imf[:,2]-12,'b')
plt.xlim(time_vect[0], time_vect[-1])
plt.grid(True)
plt.subplot(2,1,2)
plt.pcolormesh( time_vect, freq_bins, hht, cmap='ocean_r' )
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (secs)')
plt.grid(True)
plt.show()
```

