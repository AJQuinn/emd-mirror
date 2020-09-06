Installing EMD
=================================

There are several ways to install the EMD toolbox. The best one to use depends
on how you want to use the code.


Stable PyPI version
===================

This version of the code is stable and will be updated relatively slowly. Any updates to PyPI will (hopefully) only contain working changes that have been running without problems on the development versions of the code for a while.

...from pip
"""""""""""

EMD can be install from `PyPI <https://pypi.org/project/emd/>`_ using pip::

    pip install emd

pip will install the latest version of EMD from PyPI alongside any missing dependencies into the current python environment. You can install a specific version by specifying the version number.::

    pip install emd==0.3.0

...from conda
"""""""""""""

If you want to create a conda environment containing EMD, you can use the following yaml config.::

    name: emd
    channels:
    dependencies:
       - pip
       - pip:
         - emd

This environment can be customised to include any other packages that you might be working with. The last two lines can also be added to an existing conda environment configuration file to include emd in that env.

This env can be downloaded `HERE <https://gitlab.com/emd-dev/emd/-/blob/master/envs/emd_conda_env.yml>`_. You can download the config and install the enviromnent by changing directory to the install location and calling these commands.::

    curl https://gitlab.com/emd-dev/emd/-/raw/master/envs/emd_conda_env.yml > emd_conda_env.yml
    conda env create -f emd_conda_env.yml

this will automatically install the required dependancies alongside EMD. The environment can then be activated by calling::

    source activate emd


Development gitlab version
==========================

You can also install the latest development version of EMD on gitlab using a
conda environment. This verion is less stable and likely to change quickly
during active development - however you will get access to new bug-fixes,
features and bugs more quickly.

...from conda
"""""""""""""

A conda environment config file can be specified pointing at the development version of EMD on gitlab.::

    name: emd
    channels:
    dependencies:
       - pip
       - pip:
         - git+https://gitlab.com/emd-dev/emd.git

The env can be downloaded `HERE <https://gitlab.com/emd-dev/emd/-/blob/master/envs/emd-dev_conda_env.yml>`_. You can download the config and install the enviromnent by changing directory to the install location and calling these commands.::

    curl https://gitlab.com/emd-dev/emd/-/raw/master/envs/emd-dev_conda_env.yml > emd-dev_conda_env.yml
    conda env create -f emd-dev_conda_env.yml

this will automatically install the required dependancies alongside EMD. The environment can then be activated by calling::

    source activate emd-dev


...from source (unix)
"""""""""""""""""""""

If you plan to actively contribute to EMD, you will need to install EMD directly from source using git. From the terminal, change into the directory you want to install emd into and run the following command.::


    cd /home/andrew/src
    git clone https://gitlab.com/emd-dev/emd.git
    cd emd
    python setup.py install

You will then be able to use git as normal to switch between development branches of EMD and contribute your own.
