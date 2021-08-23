import os

num_threads_to_use = 2
if num_threads_to_use is not None:
    # If this variable is `None`, the script will use up all available threads.
    os.environ["OMP_NUM_THREADS"] = str(num_threads_to_use)
    # Another environment variable "OPENBLAS_NUM_THREADS" works similarly.

import time

import numpy as np
import matplotlib.pyplot as plt

from squarematrix import (
    init_tpsa,
    ar2_xyntheta,
    plot3Dxydconvergence,
    scanmin,
    plot_scanmin,
)

ntheta = 12
y = 4e-3
ar2_iter_number = 12
n_theta_cutoff_ratio = (1, 0)

xyntheta_scan_input = dict(
    ar_iter_number=4,  # 3  #
    ar2_iter_number=ar2_iter_number,
    number_of_iter_after_minimum=2,
    applyCauchylimit=True,
    n_theta_cutoff_ratio=n_theta_cutoff_ratio,
)

init_tpsa_input = {
    "nvar": 4,
    "n_theta": 4,  # 12, #
    "cutoff": 4,  # 12, #
    "norder": 5,
    "norder_jordan": 3,
    "use_existing_tpsa": 1,
    "oneturntpsa": "tpsa",
    "deltap": -0.025,
    "ltefilename": "20140204_bare_1supcell",
    "mod_prop_dict_list": [],
    "tpsacode": "yuetpsa",
    "dmuytol": 0.01,
}
init_tpsa_output = init_tpsa(init_tpsa_input=init_tpsa_input)

tt0 = [["scanx start", time.time(), 0]]
print(tt0)
xyntheta_scan_input["ar2_iter_number"] = ar2_iter_number
xyntheta_scan_input["n_theta_cutoff_ratio"] = n_theta_cutoff_ratio
ar2 = []
for x in np.arange(-1.0e-3, -28.1e-3, -1e-3):
    try:
        ar2 = ar2 + ar2_xyntheta(
            x,
            y,
            ntheta,
            xyntheta_scan_input=xyntheta_scan_input,
            init_tpsa_output=init_tpsa_output,
        )[1]
    except Exception as err:
        print(dir(err))
        print(err.args)
        pass
x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(ar2, plot3d=False)
xset, minzlist, arminidx, idxminlist, diverge = scanmin(x_iter_lndxy, xset, cutoff)
convergeindex = np.where((np.array(idxminlist) - ar2_iter_number >= 0))[0]
# convergeindex = np.where((ar2_iter_number - np.array(idxminlist) - 1) < 1)[0]
xconvergent = xset[convergeindex]
plot_scanmin(xset, minzlist, idxminlist, cutoff)
tt1 = [["scanx end", time.time(), time.time() - tt0[0][1]]]
# ar2_scanx()
print(tt1)

plt.show()
