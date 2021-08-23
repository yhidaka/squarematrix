import os

num_threads_to_use = 2
if num_threads_to_use is not None:
    # If this variable is `None`, the script will use up all available threads.
    os.environ["OMP_NUM_THREADS"] = str(num_threads_to_use)
    # Another environment variable "OPENBLAS_NUM_THREADS" works similarly.

import time

import matplotlib.pyplot as plt

from squarematrix import (
    init_tpsa,
    ar2_xyntheta,
    plot3Dtheta_vs_xydconvergence,
    plot1Dtheta_vs_xydconvergence_from_3Ddata,
    scanmin,
)

x = -7e-3
y = 4e-3
ntheta_lim = 40
ar2_iter_number = 12
n_theta_cutoff_ratio = (1, 0)  # (2, 1),  # (1, 1)  #
scantype = "excluding first iteration"  # "including first iteration"

xyntheta_scan_input = dict(
    ar_iter_number=4,  # 3  #
    ar2_iter_number=ar2_iter_number,  # 0  #
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

xyntheta_scan_input["ar2_iter_number"] = ar2_iter_number
xyntheta_scan_input["n_theta_cutoff_ratio"] = n_theta_cutoff_ratio
tt0 = [["scan_ntheta start", time.time(), 0]]
print(tt0)

ar2 = []
for ntheta in range(4, ntheta_lim):
    try:
        ar2 = ar2 + ar2_xyntheta(
            x,
            y,
            ntheta,
            xyntheta_scan_input=xyntheta_scan_input,
            init_tpsa_output=init_tpsa_output,
            scantype="including first iteration",
        )[1]
    except Exception as err:
        print(dir(err))
        print(err.args)
        pass

x_iter_lndxy, cutoff, n_thetaset = plot3Dtheta_vs_xydconvergence(ar2)
plot1Dtheta_vs_xydconvergence_from_3Ddata(ar2, n_theta=17, plot130=True)
# lt53.plot1Dtheta_vs_xacoeffnewconvergence_from_3Ddata(ar2, n_theta=17, plot131=True)
thetaset, minzlist, arminidx, idxminlist, diverge = scanmin(
    x_iter_lndxy,
    n_thetaset,
    cutoff,
    plotscmain="lndxy vs nth",
    scantype=scantype,
)
plt.tight_layout()
tt1 = [["scan_ntheta end", time.time(), time.time() - tt0[0][1]]]
print(tt1)

plt.show()
