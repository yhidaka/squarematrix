program_name = "squarematrix"
version = "0.1.0"

from setuptools import find_packages  # Must come before importing "numpy.distutils"
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from Cython.Build import cythonize

import shlex
from subprocess import Popen, PIPE
from pathlib import Path

try:
    import PyTPSA

    print("* PyTPSA is already installed.")
except ImportError:
    p = Popen(
        shlex.split("pip install git+https://github.com/yhidaka/PyTPSA"),
        stdout=PIPE,
        stderr=PIPE,
        encoding="utf-8",
    )
    out, err = p.communicate()
    print(out)
    if err:
        print("** stderr **")
        print(err)

cython_ext = Extension(f"{program_name}.cytpsa", sources=["src/cytpsa_src/cytpsa.pyx"])
cython_ext.cython_directives = {"language_level": "3"}

fortran_extensions = [
    Extension(
        f"{program_name}.fortran.{fortran_basename}",
        sources=[f"src/fortran_src/{fortran_basename}.f"],
        f2py_options=["--quiet"],
    )
    for fortran_basename in sorted(
        [p.stem for p in Path("src/fortran_src").glob("*.f")]
    )
]


ext_modules = cythonize([cython_ext]) + fortran_extensions

setup(
    name=program_name,
    version=version,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    zip_safe=False,
    description="Square Matrix",
    author="Li Hua Yu, Yoshiteru Hidaka",
    maintainer="Li Hua Yu",
    maintainer_email="lhyu@bnl.gov",
    # install_requires=install_requires,
)
