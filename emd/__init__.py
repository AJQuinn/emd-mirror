#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Package for Empirical Mode Decomposition analyses.

Submodules:
    sift - compute Intrinsic Mode Functions from time-series
    spectra - compute frequency transforms and power spectra
    plotting - helper functions for producing figures
    cycles - routines for analysing single cycles
    logger - routines for logging analysis

"""

from . import spectra  # noqa: F401, F403
from . import utils  # noqa: F401, F403
from . import plotting  # noqa: F401, F403
from . import example  # noqa: F401, F403
from . import logger  # noqa: F401, F403
from . import cycles  # noqa: F401, F403
from . import _cycles_support  # noqa: F401, F403
from . import sift  # noqa: F401, F403
from . import support  # noqa: F401, F403

__version__ = support.get_installed_version()
