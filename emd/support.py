#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import os
import pytest
import numpy as np
import pkg_resources  # part of setuptoos

from . import sift

# Housekeeping for logging
import logging
logger = logging.getLogger(__name__)


def get_install_dir():
    """Get directory path of currently installed & imported emd"""
    return os.path.dirname(sift.__file__)


def get_installed_version():
    """Read version of currently installed & imported emd according to
    setup.py. If a user has made local changes this version may not be exactly
    the same as the online package."""
    return pkg_resources.require("emd")[0].version


def run_tests():
    """
    Helper to run tests in python - useful for people without a dev-install to
    run tests perhaps.

    https://docs.pytest.org/en/latest/usage.html#calling-pytest-from-python-code

    """

    inst_dir = get_install_dir()

    if os.path.exists(os.path.join(inst_dir, 'tests')) is False:
        logger.info('Test directory not found in: {0}'.format(inst_dir))
        logger.info('(this is normal for PyPI/pip EMD installs)')
    else:
        logger.info('Running EMD package tests from: {0}'.format(inst_dir))
        out = pytest.main(['-x', inst_dir])

        if out.value != 0:
            logger.warning('EMD package tests FAILED - EMD may not behave as expected')
        else:
            logger.info('EMD package tests passed')


# Ensurance Department


def ensure_equal_dims(to_check, names, func_name, dim=None):
    """
    Check that a set of arrays all have the same dimension. Raise an error with
    details if not.

    Parameters
    ----------
    to_check : list of arrays
        List of arrays to check for equal dimensions
    names : list
        List of variable names for arrays in to_check
    func_name : str
        Name of function calling ensure_equal_dims
    dim : int
        Integer index of specific axes to ensure shape of, default is to compare all dims

    Raises
    ------
    ValueError
        If any of the inputs in to_check have differing shapes

    """

    if dim is None:
        dim = np.arange(to_check[0].ndim)
    else:
        dim = [dim]

    all_dims = [tuple(np.array(x.shape)[dim]) for x in to_check]
    check = [True] + [all_dims[0] == all_dims[ii + 1] for ii in range(len(all_dims[1:]))]

    if np.alltrue(check) == False:  # noqa: E712
        msg = 'Checking {0} inputs - Input dim mismatch'.format(func_name)
        logger.error(msg)
        msg = "Mismatch between inputs: "
        for ii in range(len(to_check)):
            msg += "'{0}': {1}, ".format(names[ii], to_check[ii].shape)
        logger.error(msg)
        raise ValueError(msg)


def ensure_vector(to_check, names, func_name):
    """
    Check that a set of arrays are all vectors with only 1-dimension. Arrays
    with singleton second dimensions will be trimmed and an error will be
    raised for non-singleton 2d or greater than 2d inputs.

    Parameters
    ----------
    to_check : list of arrays
        List of arrays to check for equal dimensions
    names : list
        List of variable names for arrays in to_check
    func_name : str
        Name of function calling ensure_equal_dims

    Returns
    -------
    out
        Copy of arrays in to_check with 1d shape.

    Raises
    ------
    ValueError
        If any input is a 2d or greater array

    """

    out_args = list(to_check)
    for idx, xx in enumerate(to_check):

        if (xx.ndim > 1) and (xx.shape[1] == 1):
            msg = "Checking {0} inputs - trimming singleton from input '{1}'"
            msg = msg.format(func_name, names[idx])
            out_args[idx] = out_args[idx][:, 0]
            logger.warning(msg)
        elif (xx.ndim > 1) and (xx.shape[1] != 1):
            msg = "Checking {0} inputs - Input '{1}' {2} must be a vector or 2d with singleton second dim"
            msg = msg.format(func_name, names[idx], xx.shape)
            logger.error(msg)
            raise ValueError(msg)
        elif xx.ndim > 2:
            msg = "Checking {0} inputs - Shape of input '{1}' {2} must be a vector."
            msg = msg.format(func_name, names[idx], xx.shape)
            logger.error(msg)
            raise ValueError(msg)

    if len(out_args) == 1:
        return out_args[0]
    else:
        return out_args


def ensure_1d_with_singleton(to_check, names, func_name):
    """
    Check that a set of arrays are all vectors with a singleton second
    dimneions. 1d arrays will have a singleton second dimension added and an
    error will be raised for non-singleton 2d or greater than 2d inputs.

    Parameters
    ----------
    to_check : list of arrays
        List of arrays to check for equal dimensions
    names : list
        List of variable names for arrays in to_check
    func_name : str
        Name of function calling ensure_equal_dims

    Returns
    -------
    out
        Copy of arrays in to_check with '1d with singleton' shape.

    Raises
    ------
    ValueError
        If any input is a 2d or greater array

    """

    out_args = list(to_check)
    for idx, xx in enumerate(to_check):

        if (xx.ndim > 2) and np.all(xx.shape[1:] == np.ones_like(xx.shape[1:])):
            # nd input where all trailing are ones
            msg = "Checking {0} inputs - Trimming trailing singletons from input '{1}' (input size {2})"
            logger.debug(msg.format(func_name, names[idx], xx.shape))
            out_args[idx] = np.squeeze(xx)[:, np.newaxis]
        if (xx.ndim > 2) and np.all(xx.shape[1:] == np.ones_like(xx.shape[1:])) == False:  # noqa: E712
            # nd input where some trailing are not one
            msg = "Checking {0} inputs - trailing dims of input '{1}' {2} must be singletons (length=1)"
            logger.error(msg.format(func_name, names[idx], xx.shape))
            raise ValueError(msg)
        elif xx.ndim == 1:
            # Vector input - add a dummy dimension
            msg = "Checking {0} inputs - Adding dummy dimension to input '{1}'"
            logger.debug(msg.format(func_name, names[idx]))
            out_args[idx] = out_args[idx][:, np.newaxis]

    if len(out_args) == 1:
        return out_args[0]
    else:
        return out_args


def ensure_2d(to_check, names, func_name):
    """
    Check that a set of arrays are all arrays with 2 dimensions. 1d arrays will
    have a singleton second dimension added.

    Parameters
    ----------
    to_check : list of arrays
        List of arrays to check for equal dimensions
    names : list
        List of variable names for arrays in to_check
    func_name : str
        Name of function calling ensure_equal_dims

    Returns
    -------
    out
        Copy of arrays in to_check with 2d shape.

    """

    out_args = list(to_check)
    for idx in range(len(to_check)):

        if to_check[idx].ndim == 1:
            msg = "Checking {0} inputs - Adding dummy dimension to input '{1}'"
            logger.debug(msg.format(func_name, names[idx]))
            out_args[idx] = out_args[idx][:, np.newaxis]

    if len(out_args) == 1:
        return out_args[0]
    else:
        return out_args


# Exceptions & Errors

class EMDSiftCovergeError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        logger.exception(self.message)
