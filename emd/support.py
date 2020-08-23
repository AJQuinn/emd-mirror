#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import os
import pytest
import pkg_resources  # part of setuptoos

from . import sift


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
    pytest.main(['-x', get_install_dir()])
