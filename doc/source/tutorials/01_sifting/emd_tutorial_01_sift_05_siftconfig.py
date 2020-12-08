"""
SiftConfig Specification
========================
Here we look at how to customise the different parts of the sift algorithm.
There are many options which can be customised from top level sift parameters
all the way down to extrema detection.

"""

#%%
# The SiftConfig object
# ^^^^^^^^^^^^^^^^^^^^^
# EMD can create a config dictionary which contains all the options that can be
# customised for a given sift function. This can be created using the
# get_config function in the sift submodule. Lets import emd and create the
# config for a standard sift - we can view the options by calling print on the
# config.
#
# The SiftConfig dictionary contains all the arguments for functions that
# are used in the sift algorithm.
#
# - "sift" contains arguments for the high level sift functions such as ``emd.sift.sift`` or ``emd.sift.ensemble_sift``
# - "imf" contains arguments for ``emd.sift.get_next_imf``
# - "envelope" contains arguments for ``emd.sift.interpolate_envelope``
# - "extrema", "mag_pad" and  "loc_pad" have arguments for extrema detection and padding

# sphinx_gallery_thumbnail_path = '_static/emd_siftconfig_thumb.png'

import emd
import numpy as np

config = emd.sift.get_config('sift')
print(config)

#%%
# These arguments are specific for the each type of sift (particularly at the top "sift" level).

config = emd.sift.get_config('complete_ensemble_sift')
print(config)

#%%
# The SiftConfig dictionary contains arguments and default values for functions
# which are called internally within the different sift implementations. The
# dictionary can be used for viewing and editing the options before they are
# passed into the sift function.
#
# The SiftConfig dictionary is nested, in that some items in the dictionary
# store further dictionaries of options. This hierarchy of options reflects
# where the options are used in the sift process. The top-level of the
# dictionary contains arguments which may be passed directly to the sift
# functions, whilst options needed for internal function calls are stored in
# nested subdictionaries.
#
# The parameters in the config can be changed in the same way we would change
# the key-value pairs in a nested dictionary or using a h5py inspiried shorthand.

# This is a top-level argument used directly by ensemble_sift
config['nensembles'] = 20
config['nprocesses'] = 4
config['max_imfs'] = 5

# This is a sub-arguemnt used by interp_envelope, which is called within
# ensemble_sift.

# Standard
config['extrema_opts']['pad_width'] = 4
# Shorthard
config['extrema_opts/pad_width'] = 4

print(config)

#%%
# This nested shorthand can be used to customise the low level extrema padding options

# Standard
#config['extrema_opts']['loc_pad_opts']['reflect_type'] = 'even'
# Shorthand
config['extrema_opts/mag_pad_opts/stat_length'] = 3
config['extrema_opts'] = {}

print(config)

# This nested structure is passed as an unpacked dictionary to our sift function.

# Create some random data
x = np.random.randn(1000,)

imf = emd.sift.complete_ensemble_sift(x, **config)

#%%
# Customised sifting with functools.partial
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If you are going to be repeatedly calling a sift function with the same
# arguments many times, you could consider creating a partial function to
# simplify the code. Partial functions are a part of the ``functools`` module
# in python. They act like normal functions but with fixed values for certain
# arguments.
#
# This means we could specify our sift config and use it to create a partial
# function which only needs the data to be passed in as an argument. For
# example:

# Create a mask sift config object and customise some options
config = emd.sift.get_config('mask_sift')
config['max_imfs'] = 5
config['mask_amp_mode'] = 'ratio_sig'
config['envelope_opts/interp_method'] = 'mono_pchip'

# Create a partial function - my_mask_sift is now a function with the arguments
# in config fixed as defaults.
from functools import partial
my_mask_sift = partial(emd.sift.mask_sift, **config)

# my_mask_sift can then be called with the input data as the only argument.
imfs = my_mask_sift(x)

#%%
# We can compare the different options for ``emd.sift.mask_sift`` and
# ``my_mask_sift`` using the python inspect module to print the default
# arguments (or function signature) for each function.

import inspect
print('emd.sift.mask_sift')
print(inspect.signature(emd.sift.mask_sift))
print()
print('my_mask_sift')
print(inspect.signature(my_mask_sift))

#%%
# We can see that the input arguments in the signature of ``my_mask_sift``
# contains all the specified options from the ``config`` so is much longer than
# for ``emd.sift.mask_sift``.

#%%
# Saving and loading sift config files
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We might often want to store, reuse and share sift configurations during a
# project. To help with this, a SiftConfig specification can be stored as a raw
# text file in the YAML format. The config can be saved into a text file and
# loaded back into a SiftConfig object for use in a script. We can also
# directly edit the text file to customise the sift parameters from there if
# preferred. The save and load operations are performed by
# ``emd.sift.SiftConfig.to_yaml_file`` and
# ``emd.sift.SiftConfig.from_yaml_file`` respectively.
#
# Lets look at an example. We're going to store this config in a temporary file
# on your system for this tutorial. This avoids clutter and should work on all
# systems. If you would prefer to use a specific file on your system please
# comment out this section and simply specify ``config_file`` to be a path to
# the file of your choice.

# Create a temporary file OR specify your own file path
import tempfile
config_file = tempfile.NamedTemporaryFile(prefix="ExampleSiftConfig_").name
# Or uncomment the following line and specify your own file
#config_file = '/path/to/my/file'

# Save the config into yaml format
config.to_yaml_file(config_file)

# Open the text file and print its contents
with open(config_file, 'r') as f:
    txt = f.read()
print(txt)

# Load the config back into a SiftConfig object for use in a script
new_config = emd.sift.SiftConfig.from_yaml_file(config_file)
