import numpy as np
import matplotlib.pyplot as plt

from squarematrix import scanxy

if __name__ == "__main__":

    init_tpsa_input = dict(
        nvar=4,
        ntheta=3,
        cutoff=10,
        norder=5,
        norder_jordan=3,
        use_existing_tpsa=0,
        oneturntpsa="tpsa",
        deltap=0.0,
        ltefilename="20140204_bare_1supcell",
        mod_prop_dict_list=[],
        tpsacode="yuetpsa",
        dmuytol=0.01,
    )

    out = scanxy(
        xlist=np.arange(-35e-3, +35e-3, 2e-3),
        ylist=np.arange(1e-3, 15e-3, 2e-3),
        init_tpsa_input=init_tpsa_input,
        return_area=True,
        diff_lvl=-12,
    )
    print(out)

    plt.show()
