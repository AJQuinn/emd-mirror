Installing EMD
=================================

There are several ways to install the EMD toolbox. The best one to use depends
on how you want to use the code.


Stable PyPI version
*******************

The `stable version of the code <https://pypi.org/project/emd/>`_ is hosted on `PyPI <https://pypi.org>`_ and will be updated relatively slowly. Any updates to PyPI will (hopefully) only contain working changes that have been running without problems on the development versions of the code for a while.

.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>install from pip</h3>

    .. container:: installbody body

        EMD can be install from `PyPI <https://pypi.org/project/emd/>`_ using pip::

            pip install emd

        pip will install the latest version of EMD from PyPI alongside any missing dependencies into the current python environment. You can install a specific version by specifying the version number::

            pip install emd==0.4.0


.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>install in conda environment</h3>

    .. container:: installbody body

        If you want to create a conda environment containing EMD, you can use the following yaml config::

            name: emd
            channels:
            dependencies:
               - pip
               - pip:
                 - emd

        This can be adapted to specify a particular release of EMD by adding the version number to the emd line::

            name: emd
            channels:
            dependencies:
               - pip
               - pip:
                 - emd==0.3.3

        This environment can be customised to include any other packages that you might be working with. The last two lines can also be added to an existing conda environment configuration file to include emd in that env.

        This env can be downloaded `HERE (emd_conda_env.yml) <https://gitlab.com/emd-dev/emd/-/blob/master/envs/emd_conda_env.yml>`_. You can download the config and install the enviromnent by changing directory to the install location and calling these commands::

            curl https://gitlab.com/emd-dev/emd/-/raw/master/envs/emd_conda_env.yml > emd_conda_env.yml
            conda env create -f emd_conda_env.yml

        this will automatically install the required dependancies alongside EMD. The environment can then be activated by calling::

            source activate emd



Development gitlab.com version
******************************

You can also install the `latest development version of EMD
<https://gitlab.com/emd-dev/emd>`_ from gitlab.com using a conda environment.
This version is less stable and likely to change quickly during active
development - however you will get access to new bug-fixes, features and bugs
more quickly.


.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>install in conda environment</h3>

    .. container:: installbody body

        A conda environment config file can be specified pointing at the development version of EMD on gitlab::

            name: emd
            channels:
            dependencies:
               - pip
               - pip:
                 - git+https://gitlab.com/emd-dev/emd.git

        The env can be downloaded `HERE (emd-dev_conda_env.yml) <https://gitlab.com/emd-dev/emd/-/blob/master/envs/emd-dev_conda_env.yml>`_. You can download the config and install the enviromnent by changing directory to the install location and calling these commands::

            curl https://gitlab.com/emd-dev/emd/-/raw/master/envs/emd-dev_conda_env.yml > emd-dev_conda_env.yml
            conda env create -f emd-dev_conda_env.yml

        this will automatically install the required dependancies alongside EMD. The environment can then be activated by calling::

            source activate emd-dev


.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>install development branch in conda environment</h3>

    .. container:: installbody body

        A conda environment config file can be specified pointing at the development version of EMD on gitlab. A specific branch can be indicated by adding the branch name after an @ sign in the line specifying the git repo. Here is an example which installs a branch called 'new_feature'::

            name: emd
            channels:
            dependencies:
               - pip
               - pip:
                 - git+https://gitlab.com/emd-dev/emd.git@new_feature

        We provide `an example env here (emd-dev_conda_env.yml) <https://gitlab.com/emd-dev/emd/-/blob/master/envs/emd-dev_conda_env.yml>`_. You can download the config and add the branch name to the right line. Finally, you can install the enviromnent by changing directory to the install location and calling these commands::

            curl https://gitlab.com/emd-dev/emd/-/raw/master/envs/emd-dev_conda_env.yml > emd-dev_conda_env.yml
            conda env create -f emd-dev_conda_env.yml

        this will automatically install the required dependancies alongside EMD. The environment can then be activated by calling::

            source activate emd-dev

.. container:: toggle body

    .. container:: header body

        .. raw:: html

            <h3 class='installbar'>install from source code</h3>

    .. container:: installbody body

        If you plan to actively contribute to EMD, you will need to install EMD directly from source using git. From the terminal, change into the directory you want to install emd into and run the following command::

            cd /home/andrew/src
            git clone https://gitlab.com/emd-dev/emd.git
            cd emd
            python setup.py install

        You will then be able to use git as normal to switch between development branches of EMD and contribute your own.
