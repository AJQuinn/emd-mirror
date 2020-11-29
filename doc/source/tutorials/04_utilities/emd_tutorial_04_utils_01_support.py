"""
Check install and run tests
===========================
The ``emd.support`` submodule contains a few functions relating to the
installation and testing of EMD on your system. Most users will not need to
use these functions most of the time but they can be helpful for checking
that EMD is properly installed.

"""

#%%
# Firstly, we can use ``emd.support.get_installed_version`` to confirm which
# version of EMD we have installed.

# sphinx_gallery_thumbnail_path = '_static/emd_pytest_thumb.png'

import emd

print(emd.support.get_installed_version())

#%%
# Secondly, we can identify where the EMD files are installed on the computer
# by calling ``emd.support.get_install_dir``. This can be useful if you have a
# couple of EMD versions and want to be sure which is currently in use.

print(emd.support.get_install_dir())

#%%
# Finally, if you have installed EMD from source (rather than via conda or
# pypi) you can run the test suit directly from the support module.

print(emd.support.run_tests())
