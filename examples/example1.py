import matplotlib.pyplot as plt

from squarematrix import (
    init_tpsa,
    single_point_xyntheta,
    kxmax,
    plot3Dxydconvergence,
    plot1Dxydconvergence_from_3Ddata,
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
    "use_existing_tpsa": 0,  # 1, #
    "oneturntpsa": "tpsa",
    "deltap": -0.025,
    "ltefilename": "20140204_bare_1supcell",
    "mod_prop_dict_list": [],
    "tpsacode": "yuetpsa",
    "dmuytol": 0.01,
}
init_tpsa_output = init_tpsa(init_tpsa_input=init_tpsa_input)

ar, minz, idxmin, iteration_data = single_point_xyntheta(
    x=-15e-3, xyntheta_scan_input=xyntheta_scan_input, init_tpsa_output=init_tpsa_output
)
tmp = [i[0] for i in kxmax(ar, xc=-15e-3)[0]]
artmp = ar[tmp[0] : tmp[-1] + 1]
plot3Dxydconvergence(artmp, plot3d=True)
plot1Dxydconvergence_from_3Ddata(ar, xc=-15e-3)

plt.show()