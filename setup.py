#!/usr/bin/python

import pathlib
from setuptools import setup

# Scripts
scripts = []

name = 'emd'
version = '0.3'
release = '0.3.1'

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name=name,

    version=release,

    description='Empirical Mode Decomposition',

    # Author details
    author='Andrew Quinn <andrew.quinn@psych.ox.ac.uk>',
    author_email='andrew.quinn@psych.ox.ac.uk',

    long_description=README,
    long_description_content_type="text/markdown",

    # Choose your license
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='EMD Spectrum Frequency Non-Linear Holospectrum Hilbert-Huang',

    packages=['emd'],

    python_requires='>3.4',

    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'numpydoc',
                      'sphinx_rtd_theme'],

    extras_require={
        'test': ['coverage'],
    },

    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', name),
            'release': ('setup.py', name)}},
)
