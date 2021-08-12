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
    scanx,
    nuxvzx,
    plot1Dxydconvergence_from_3Ddata,
    plot3Dxydconvergence,
    plotbnm,
    plotferrormin,
)

xyntheta_scan_input = dict(
    ar_iter_number=4,  # 3  #
    ar2_iter_number=12,  # 0  #
    number_of_iter_after_minimum=2,
    applyCauchylimit=True,
    n_theta_cutoff_ratio=(1, 0),  # (2, 1),  # (1, 1)  #
)

init_tpsa_input = {
    "nvar": 4,
    "n_theta": 4,  # 12, #
    "cutoff": 4,  # 12, #
    "norder": 5,
    "norder_jordan": 3,
    "use_existing_tpsa": 0,  # 1,
    "oneturntpsa": "tpsa",
    "deltap": -0.025,
    "ltefilename": "20140204_bare_1supcell",
    "mod_prop_dict_list": [],
    "tpsacode": "yuetpsa",
    "dmuytol": 0.01,
}
init_tpsa_output = init_tpsa(init_tpsa_input=init_tpsa_input)

ar, minzlist, idxminlist = scanx(init_tpsa_output=init_tpsa_output)

tt0 = [["nuxvzx start", time.perf_counter(), 0]]
print(tt0)
idxminlist, convergencerate, nux, nuy, diverge, nuxy, xset = nuxvzx(
    ar, tracking=True, npass=800, xall=[]
)
tt1 = [["nuxvzx end", time.perf_counter(), time.perf_counter() - tt0[0][1]]]
print(tt0)
print(tt1)
plotbnm(ar)
plotferrormin(ar)
plot1Dxydconvergence_from_3Ddata(ar, xc=-22e-3)
plot3Dxydconvergence(ar)
print(tt0)
print(tt1)

nuxerror = [nux[j] - nuxy[0][j] for j in range(len(nux)) if diverge[j, 1] == 0]
nuyerror = [nuy[j] - nuxy[1][j] for j in range(len(nuy)) if diverge[j, 1] == 0]
xset_convergent = [xset[j] for j in range(len(xset)) if diverge[j, 1] == 0]
plt.figure(4)
plt.plot(xset_convergent, nuxerror, ".", xset_convergent, nuyerror, ".")
plt.title("Fig.4 nux, nuy errors relative to tracking for convergent iteration")
plt.xlabel("x (m)")
plt.ylabel("dnux,dnuy")
plt.tight_layout()
plt.show()

print("nuxerror,nuyerror=:\n")
for j in range(len(nuxerror)):
    print(nuxerror[j], nuyerror[j])
