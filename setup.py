program_name = "squarematrix"
version = "0.1.0"

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import shlex
from subprocess import Popen, PIPE
from pathlib import Path
import shutil

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

recompile_fortran_so = True
# recompile_fortran_so = False

fortran_so_folder = Path(f"src/{program_name}/fortran")
if recompile_fortran_so:
    if fortran_so_folder.exists():
        print("Cleaning up the Fotran SO folder before compilation...")
        shutil.rmtree(fortran_so_folder)
        print("Finished")
fortran_so_folder.mkdir(parents=True, exist_ok=True)

if recompile_fortran_so:
    fortran_basenames = [
        "dot",
        # "findnonzeros",
        "ftm",
        "fts",
        "orderdot",
        "zcolarray",
        "zcol",
        "zcolnew",
        "mysum",
        "lineareq",
    ]
    fortran_so_init_contents = ""
    for base in fortran_basenames:

        fp_base = Path("../../fortran_src").joinpath(base)

        p = Popen(
            shlex.split(f"f2py -c {fp_base}.f -m {base} --quiet"),
            stdout=PIPE,
            stderr=PIPE,
            encoding="utf-8",
            cwd=fortran_so_folder,
        )
        out, err = p.communicate()
        print(out)
        if err:
            print("** stderr **")
            print(err)

        fortran_so_init_contents += f"\nfrom . import {base}"

    with open(fortran_so_folder.joinpath("__init__.py"), "w") as f:
        f.write(fortran_so_init_contents)


extensions = [
    Extension(f"{program_name}.cytpsa", sources=["src/cytpsa_src/cytpsa.pyx"])
]

for e in extensions:
    e.cython_directives = {"language_level": "3"}

setup(
    name=program_name,
    version=version,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={f"{program_name}.fortran": ["*.so"]},
    ext_modules=cythonize(extensions),
    zip_safe=False,
    description="Square Matrix",
    author="Li Hua Yu, Yoshiteru Hidaka",
    maintainer="Li Hua Yu",
    maintainer_email="lhyu@bnl.gov",
    # install_requires=install_requires,
)

if recompile_fortran_so:
    # If you do "$ python setup.py develop", you should NOT delete this folder.
    print("Cleaning up the Fotran SO folder...")
    shutil.rmtree(fortran_so_folder)
    print("Finished.")
