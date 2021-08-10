import numpy as np
from numpy import array


import subprocess


filenames = [
    "dot",
    "findnonzeros",
    "ftm",
    "fts",
    "orderdot",
    "zcolarray",
    "zcol",
    "zcolnew",
    "mysum",
    "lineareq",
]

for fn in filenames:
    ta, xxp = subprocess.getstatusoutput("f2py -c " + fn + ".f -m " + fn)
    print("1.", xxp)
    ta, xxp = subprocess.getstatusoutput("cp " + fn + "*.so ..")
    print("2.", xxp)
