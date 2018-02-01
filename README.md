# emd
Empirical Mode Decomposition

# Installation

To copy emd into chosen directory /path/to/directory in the unix shell:

```shell
cd /path/to/directory
git clone https://github.com/AJQuinn/emd.git
```
and add /path/to/directory to your $PYTHONPATH

```shell
export PYTHONPATH=/path/to/directory/emd:$PYTHONPATH
```

# Requirements

Requirements are detailed in requirements.txt

```
matplotlib==2.0.2
numpy==1.13.1
scipy==0.19.1
```

# Quick Start

Import emd

```python
import emd
```

Define a simulated waveform

```python
sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds

import numpy as np
time_vect = np.linspace(0, seconds, num_samples)

freq = 1
nonlinearity_deg = .75 # change extent of deformation from sinusoidal shape [-1 to 1]
nonlinearity_phi = -np.pi/4 # change left-right skew of deformation [-pi to pi]
x = emd.utils.abreu2010( freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds )
```

Estimate IMFs

```python
imf = emd.sift.sift( x )
```

Compute instantaneous frequency, phase and amplitude

```python
IP,IF,IA = emd.frequency_transforms.instantaneous_stats( imf, sample_rate, 'quad', smooth_phase=129)
```
Compute Hilbert-Huang spectrum

```python
freq_vect = np.linspace(0,5,128)
hht = emd.frequency_transforms.hilberthuang( IF, IA, freq_vect, time_vect, time_vect)
```
Make a summary plot

```python
import matplotlib.pyplot as plt
plt.figure( figsize=(16,8) )
plt.subplot(311)
plt.plot(time_vect,x,'k')
plt.xlim(time_vect[0], time_vect[-1])
plt.grid(True)
plt.subplot(3,1,(2,3))
plt.contourf( time_vect, freq_vect, hht[:,1:].T,np.linspace(.1,1.5,32) )
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (secs)')
plt.grid(True)
plt.show()
```

