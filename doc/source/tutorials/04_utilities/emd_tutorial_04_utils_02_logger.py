"""
Using the logger
================
EMD has a built in logger which can be tuned to print out the progress of an
analysis to the console, to a file or both.

"""

#%%
# Logger Basics
#^^^^^^^^^^^^^^

#%%
# The logger must be initialised by calling ``emd.logger.set_up``. All
# subsequent calls to functions in the EMD library will then print output
# messages according to the logger specification.

# sphinx_gallery_thumbnail_path = '_static/emd_logger_thumb.png'

# Import numpy for later
import numpy as np

# Import EMD and initialise the logger
import emd
emd.logger.set_up()

#%%
# The detail of logger output can be tuned by changing the logger level. The
# available levels are ``CRITICAL`` (only print output when the program is about to
# crash), ``WARNING`` (only print output when something unusual is happening or an
# analysis is potentially wrong), ``INFO`` (print general statements about which
# processes are running) and ``DEBUG`` (print loads of info including details of
# computations).
#
#
# The default level is ``INFO``, so if we  re-initialise the logger on
# ``DEBUG`` we get more detailed outputs

emd.logger.set_up(level='DEBUG')

#%%
# Lets explore the logger by running some a few sifts. Here we create a simple
# simulated oscillation and run a standard sift with default options.

# Initialise the logger with default settings (level=INFO)
emd.logger.set_up()

# Generate a simulation
peak_freq = 12
sample_rate = 512
seconds = 10
noise_std = .5
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds, noise_std=noise_std, random_seed=42, r=.99)

# Run a standard sift
imf = emd.sift.sift(x)

#%%
# With the level on ``INFO`` the logger tells us that the sift is running but
# not much else. If we change the logger level to ``DEBUG`` we get more output
# about how the sift is performing.
#
# The level of an initialised logger can be changed using ``emd.logger.set_level``.

emd.logger.set_level('DEBUG')

# Run a standard sift
imf = emd.sift.sift(x)

#%%
# If we don't want to change the logger level for our whole script, some
# functions allow us to override the logger level for a single call to the
# function.
#
# Any functions with the ``verbose`` option in 'Other Parameters' can override
# the logger level. Here we run a sift on ``WARNING`` and should see no outputs.

imf = emd.sift.sift(x, verbose='WARNING')

#%%
# We can also disable the logger altogether for a section of code using
# ``emd.logger.disable`` and then restart it using ``emd.logger.enable``.
#
# Here we disable logging for a sift call and renable it afterwards.

emd.logger.disable()
imf = emd.sift.sift(x)
emd.logger.enable()

#%%
# Advanced Logging
#^^^^^^^^^^^^^^^^^

#%%
# This section contains some logger optionality for more advanced use cases.
#
# Firstly, we can supplement the EMD logger output from a script by loading the
# EMD logger into a script and adding our own logger calls. In this example, we
# load the logger and add some custom updates. To print some output at the
# ``INFO`` level through the logger, we call ``logger.info``.

# Initialised EMD logger
emd.logger.set_up(level='DEBUG')

# Load logger into this script
import logging
logger = logging.getLogger('emd')

# Check the time
import time
tic = time.perf_counter()

# Run a sift
logger.info('Now starting my new analysis')
imf = emd.sift.sift(x)

# Check time again
toc = time.perf_counter()

# Print sift run-time
elapsed = toc - tic
logger.info('My new analysis finished in {0:4f} seconds'.format(elapsed))

#%%
# This output respects the overall logger level, so info statements will be
# printed at levels ``INFO`` and ``DEBUG`` but suppressed if the overall logger
# level is at ``WARNING`` or ``CRITICAL``.
#
# Try changing the logger level in the example above to see the effect on the
# following output.

#%%
# Next, we define an analysis function which runs a sift followed by a
# frequency transform and simple cycle analysis. We'll run this example a few
# times in the next sections.
#
# Note that we've included some custom logger calls and a temporary logger
# override on the ``mask_sift``


def my_analysis(x):

    # Check time, and load the logger into the session
    tic = time.perf_counter()
    import logging
    logger = logging.getLogger('emd')

    # Print start-up message
    logger.info('Starting my example analysis')
    logger.info('----------------------------')

    # Run a mask-sift with detailed logging
    imf = emd.sift.mask_sift(x, verbose='DEBUG')

    # Compute frequency stats
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
    logger.info('Avg frequency of IMF-2 is {0:2f}Hz'.format(np.average(IF[:, 2], weights=IA[:, 2])))

    # Find cycles in IMF-2
    mask = IA[:, 2] > .05
    cycles = emd.cycles.get_cycle_inds(IP, return_good=True, mask=mask)

    # Compute cycle stats
    cycle_freq = emd.cycles.get_cycle_stat(cycles[:, 2], IF[:, 2], func=np.mean)[1:]
    cycle_amp = emd.cycles.get_cycle_stat(cycles[:, 2], IA[:, 2], func=np.mean)[1:]

    # Print some cycle correlations
    logger.info('Freq-Amp correlation: r={0:2f}'.format(np.corrcoef(cycle_freq, cycle_amp)[0, 1]))

    # Print the elapsed time of the analysis
    toc = time.perf_counter()
    elapsed = toc - tic
    logger.info('My new analysis finished in {0:4f} seconds'.format(elapsed))

    return cycle_freq, cycle_amp


#%%
# We can run this function as normal and inspect the logger outputs as they
# appear in the console.

# Run the analysis
freq, amp = my_analysis(x)

#%%
# We can see a lot of information about which processes were running as the
# logger is set to ``INFO``. If we configure the logger to run on ``WARNING``
# level, we should only see output about potential errors. This can be useful
# when running familiar code where you only need output when something
# potentially strange is going on.
#
# So in this case, after changing to ``WARNING`` we should only see output from
# the ``mask_sift`` call (as this has a logger override to ``DEBUG``). All
# other output is suppressed.

# Change logger level
emd.logger.set_level('WARNING')

# Run the analysis
freq, amp = my_analysis(x)

#%%
# If we're dealing with logger output from multiple sources or perhaps from
# multiple analyses running in parallel, then we can add a prefix to the logger
# to help distinguish the output coming from each. This can be specified when
# initialising the logger.
#
# For example, here we reun the analysis function with a logger prefix
# indicating that we're processing Subject 001.

# Initialise logger with a prefix
emd.logger.set_up(level='DEBUG', prefix='Subj001')

# Run the analysis
freq, amp = my_analysis(x)

#%%
# Finally, we can direct the logger output into a text file as well as the console.

# Define a temporary file
import tempfile
log_file = tempfile.NamedTemporaryFile(prefix="ExampleEMDLog", suffix='.log').name
# OR uncomment this line and define your own filepath
# log_file = /path/to/my/log_file.log

# Initialise the logger with a prefix and a text file
emd.logger.set_up(level='INFO', prefix='Subj001', log_file=log_file)

# Run the analysis
freq, amp = my_analysis(x)

#%%
# The log file is a simple text file containing very detailed outputs of which
# functions were executed and when. Here we read the log file and print its
# contents to the console.
#
# Note that the log file contains a much more detailed output that the console!

# Open the text file and print its contents
with open(log_file, 'r') as f:
    txt = f.read()
print(txt)

#%%
# If we want this extra detailed output in the console as well, we can specify
# the ``console_format`` when setting up the logger.

# Initialise logger with a verbose console format
emd.logger.set_up(level='DEBUG', prefix='Subj001', console_format='verbose')

# Run the analysis
freq, amp = my_analysis(x)
