#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Helper routines for sifting
"""

import logging
import inspect

from . import utils, sift
import collections

# Housekeeping for logging
logger = logging.getLogger(__name__)


class SiftConfig(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        key = self.__keytransform__(key)
        if isinstance(key, list):
            return self.store[key[0]][key[1]]
        else:
            return self.store[key]

    def __setitem__(self, key, value):
        key = self.__keytransform__(key)
        if isinstance(key, list):
            self.store[key[0]][key[1]] = value
        else:
            self.store[key] = value

    def __delitem__(self, key):
        key = self.__keytransform__(key)
        if isinstance(key, list):
            del self.store[key[0]][key[1]]
        else:
            del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        key = key.split('/')
        if len(key) == 1:
            return key[0]
        else:
            return key

    def to_yaml(self):
        import yaml
        return yaml.dump(self.store)

    def to_opts(self):
        out = self.store['sift']
        out['imf_opts'] = self.store['imf']
        out['imf_opts']['envelope_opts'] = self.store['envelope']
        out['imf_opts']['envelope_opts']['extrema_opts'] = self.store['extrema']
        out['imf_opts']['envelope_opts']['extrema_opts']['mag_pad_opts'] = self.store['mag_pad']
        out['imf_opts']['envelope_opts']['extrema_opts']['loc_pad_opts'] = self.store['loc_pad']
        return out


def sift_config(mode='sift'):

    # Extrema padding opts are hard-coded for the moment, these run through
    # np.pad which has a complex signature
    mag_pad_opts = {'mode': 'median', 'stat_length': 1}
    loc_pad_opts = {'mode': 'reflect', 'reflect_type': 'odd'}

    # Get defaults for extrema detection and padding
    extrema_opts = _get_function_opts(utils.get_padded_extrema, ignore=['X', 'mag_pad_opts',
                                                                        'loc_pad_opts',
                                                                        'combined_upper_lower'])

    # Get defaults for envelope interpolation
    envelope_opts = _get_function_opts(utils.interp_envelope, ignore=['X', 'extrema_opts', 'mode'])

    # Get defaults for computing IMFs
    imf_opts = _get_function_opts(sift.get_next_imf, ignore=['X', 'envelope_opts'])

    # Get defaults for the given sift variant
    sift_types = ['sift', 'ensemble_sift', 'complete_ensemble_sift',
                  'mask_sift', 'mask_sift_adaptive', 'mask_sift_specified']
    if mode in sift_types:
        sift_opts = _get_function_opts(getattr(sift, mode), ignore=['X', 'imf_opts'])
    else:
        raise AttributeError('Sift mode not recognised: please use one of {0}'.format(sift_types))

    sift_opts.update({'max_imfs': 5, 'sift_thresh': 1e-8})

    out = SiftConfig()
    out['sift'] = sift_opts
    out['imf'] = imf_opts
    out['envelope'] = envelope_opts
    out['extrema'] = extrema_opts
    out['mag_pad'] = mag_pad_opts
    out['loc_pad'] = loc_pad_opts

    return out


def _get_function_opts(func, ignore=[]):
    out = {}
    sig = inspect.signature(func)
    for p in sig.parameters:
        if p not in out.keys() and p not in ignore:
            out[p] = sig.parameters[p].default
    return out
