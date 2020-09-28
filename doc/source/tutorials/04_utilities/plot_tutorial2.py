"""
Using the logger
================
EMD has a built in logger which can be tuned to print out the progress of an
analysis to the console, to a file or both.

"""

#%%
# To get started, lets define a simple analysis which we can run with different
# logging levels.

import emd
emd.logger.set_up()

#%%
#

emd.logger.set_up(level='DEBUG')

#%%
#

emd.logger.set_up(level='DEBUG', prefix='Subj001')

#%%
#

import numpy as np

# Generate a simulation
peak_freq = 12
sample_rate = 512
seconds = 60
noise_std = .5
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99)

imf = emd.sift.sift(x)

#%%
#

imf = emd.sift.sift(x, verbose='WARNING')

#%%

emd.logger.disable()
imf = emd.sift.sift(x)
emd.logger.enable()

#%%

import logging
logger = logging.getLogger('emd')

import time
tic = time.perf_counter()

logger.info('Now starting my new analysis')
imf = emd.sift.sift(x)

toc = time.perf_counter()
elapsed = toc - tic
logger.info('My new analysis finished in {0:4f} seconds'.format(elapsed))

#%%


def my_analysis(x):
    tic = time.perf_counter()
    import logging
    logger = logging.getLogger('emd')
    logger.info('Starting my example analysis')
    logger.info('----------------------------')

    # Run a mask-sift
    imf = emd.sift.mask_sift(x)

    # Compute frequency stats
    IP, IF, IA = emd.spectra.frequency_stats(imf, sample_rate, 'nht')
    logger.info('Avg frequency of IMF-2 is {0:2f}Hz'.format(np.average(IF[:, 2], weights=IA[:, 2])))

    # Find cycles in IMF-2
    mask = IA[:, 2] > .05
    cycles = emd.cycles.get_cycle_inds(IP, return_good=True, mask=mask)

    # Compute cycle stats
    cycle_freq = emd.cycles.get_cycle_stat(cycles[:, 2], IF[:, 2], mode='compressed', func=np.mean)[1:]
    cycle_freq_std = emd.cycles.get_cycle_stat(cycles[:, 2], IF[:, 2], mode='compressed', func=np.std)[1:]
    cycle_amp = emd.cycles.get_cycle_stat(cycles[:, 2], IA[:, 2], mode='compressed', func=np.mean)[1:]

    logger.info('Freq-Amp correlation: r={0:2f}'.format(np.corrcoef(cycle_freq, cycle_amp)[0, 1]))
    logger.info('Amp-FreqStd correlation: r={0:2f}'.format(np.corrcoef(cycle_freq, cycle_freq_std)[0, 1]))

    toc = time.perf_counter()
    elapsed = toc - tic
    logger.info('My new analysis finished in {0:4f} seconds'.format(elapsed))

    return cycle_freq, cycle_amp


#%%
#
import tempfile
log_file = tempfile.NamedTemporaryFile(prefix="ExampleEMDLog", suffix='.log').name

emd.logger.set_up(level='INFO', prefix='Subj001', log_file=log_file)
freq, amp = my_analysis(x)

#%%
#

# Open the text file and print its contents
with open(log_file, 'r') as f:
    txt = f.read()
print(txt)

#%%

emd.logger.set_up(level='DEBUG', prefix='Subj001', console_format='verbose')
freq, amp = my_analysis(x)


#%%
