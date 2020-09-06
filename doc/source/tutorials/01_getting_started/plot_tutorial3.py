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
# The SiftConfig dictionary contains all the arguments for  functions that
# are used in the sift algorithm.
#
# - "sift" contains arguments for the high level sift functions such as ``emd.sift.sift`` or ``emd.sift.ensemble_sift``
# - "imf" contains arguments for ``emd.sift.get_next_imf``
# - "envelope" contains arguments for ``emd.sift.interpolate_envelope``
# - "extrema", "mag_pad" and  "loc_pad" have arguments for extrema detection and padding
import emd

config = emd.sift.get_config('sift')
print(config)

#%%
# These arguments are specific for the each type of sift (particularly at the top "sift" level).

config = emd.sift.get_config('ensemble_sift')
print(config)

#%%
# The SiftConfig dictionary contains arguments and default values for functions
# which are called internally within the different sift implementations. The
# dictionary can be used to viewing and editing the options before they are
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
config['nensembles'] = 24

# This is a sub-arguemnt used by interp_envelope, which is called within
# ensemble_sift.

# Standard
config['envelope_opts']['interp_type'] = 'mono_pchip'
# Shorthard
config['envelope_opts/interp_type'] = 'mono_pchip'

print(config)

#%%
# This nested shorthand can be used to customise the low level extrema padding options

# Standard
config['extrema_opts']['loc_pad_opts']['reflect_type'] = 'even'
# Shorthand
config['extrema_opts/mag_pad_opts/stat_length'] = 3

print(config)

#%%
# This nested structure is passed as an unpacked dictionary to our sift function.

config = emd.sift.get_config('sift')
