import os

num_threads_to_use = 2
if num_threads_to_use is not None:
    # If this variable is `None`, the script will use up all available threads.
    os.environ["OMP_NUM_THREADS"] = str(num_threads_to_use)
    # Another environment variable "OPENBLAS_NUM_THREADS" works similarly.

import time

import numpy as np
import matplotlib.pyplot as plt

from squarematrix import init_tpsa, scanx, kxmax, showarecord, shortarecord

tt0 = [["example5 start", time.time(), 0]]
print(tt0)

y = 4e-3
ar_ntheta = 12

init_tpsa_input = dict(
    nvar=4,
    ar_ntheta=ar_ntheta,
    ar_cutoff=12,
    norder=5,
    norder_jordan=3,
    use_existing_tpsa=1,  # 0,  #
    oneturntpsa="tpsa",  # "ELEGANT",  #
    deltap=-0.025,
    ltefilename="20140204_bare_1supcell",  # "nsls2sr_supercell_ch77_20150406_1",  # 20140204_bare_1supcell",
    mod_prop_dict_list=[
        {"elem_name": "Qh1G2c30a", "prop_name": "K1", "prop_val": -0.6419573146484081,},
        {
            "elem_name": "sH1g2C30A",
            "prop_name": "K2",
            "prop_val": 19.83291209974166 + 0,
        },
    ],
    tpsacode="yuetpsa",  # "madx",  #
    dmuytol=0.01,  # 0.005, #
)

xyntheta_scan_input = dict(
    ar_iter_number=4,  # 3  #
    ar2_iter_number=12,  # 0  #
    number_of_iter_after_minimum=2,
    applyCauchylimit=True,
    n_theta_cutoff_ratio=(1, 0),  # (2, 1),  # (1, 1)  #
)

init_tpsa_output = init_tpsa(init_tpsa_input=init_tpsa_input)
(ar, minzlist, idxminlist) = scanx(
    xlist=np.arange(-1e-3, -35e-3, -1e-3),
    y=4e-3,
    ntheta=12,
    ar_iter_number=4,
    xyntheta_scan_input=xyntheta_scan_input,
    init_tpsa_input=init_tpsa_input,
    init_tpsa_output=init_tpsa_output,
)

tt1 = [["example5 end", time.time(), time.time() - tt0[0][1]]]
print(tt1)

kxmax(ar, xc=-21e-3)
import pdb

pdb.set_trace()
shortarecord(ar[104])
# showarecord(ar[103])

plt.show()
