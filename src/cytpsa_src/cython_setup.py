# Use this script by issuing the following command (assuming this file is in the
# same directory as "cytpsa.pyx":
# $ python cython_setup.py build_ext --inplace
# This should generate the C file "cytpsa.c", and the binary file "cytpsa.so" on Linux
# or "cytpsa.pyd" on Windows, and the directory named "build". All you need is the binary
# ".so" or ".pyd" file to be located in PYTHONPATH in order to be able to import
# cytpsa. The rest of the auto-generated files/folders can be safely deleted.

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cytpsa.pyx"),
    author="Yoshiteru Hidaka",
    maintainer="Yoshiteru Hidaka",
    maintainer_email="yhidaka@bnl.gov",
)
