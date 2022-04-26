import time
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

from squarematrix import init_tpsa
from squarematrix.lt59 import sv, scanDA, plotxyz

if __name__ == '__main__':
    
    load_from_proc_file = False
    #load_from_proc_file = True
    
    if not load_from_proc_file:
        init_tpsa_input = dict(
            nvar=4,
            ar_ntheta=3,
            ar_cutoff=10,
            norder=5,
            norder_jordan=3,
            use_existing_tpsa=0,  # 0,  #
            oneturntpsa="tpsa",
            deltap=0.0,
            ltefilename=(
                "/GPFS/APC/yhidaka/git_repos/Lattice_optim_scripts_Madx-PTC/"
                #"CD_MAT_LiHua/MCBA_latt_4cell_opt_2abc_4_OCT000"),  
                "CD_MAT_LiHua/MCBA_latt_4cell_opt_2abc_4_OCT050"),  
            # "cb2NSLS2CB65pm_cb0_1cell",# "mcba24pm_v1",  # "nsls2sr_supercell_ch77_20150406_1",  # 20140204_bare_1supcell",
            mod_prop_dict_list=[],
            tpsacode="yuetpsa",
            dmuytol=0.01,
        )
        
        scanDA_parameters = dict(
            nth1=25,
            nth2=30,
            ar2_iter_number=8,
            ar_iter_number=4,
            iteration_step=2e-4,
            step_resolution=1e-4,
            number_of_iter_after_minimum=2,
        )
        searchDA = False
        tt0 = [["scanxy start", time.time(), 0]]
        print(tt0)
        init_tpsa_output = init_tpsa(init_tpsa_input=init_tpsa_input)
        xyminz = []
        
        tt1 = [["scanxy 2", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
        for ymax in np.arange(0e-4, 4.1e-3, 0.5e-4):
            try:
                xset, minzlist, idxminlist, cutoff, ar, scanDAoutput = scanDA(
                    xlist=np.arange(-0e-4, -14.51e-3, -2e-4),
                    ymax=ymax,
                    scan_direction=-1,
                    searchDA=searchDA,
                    scanDA_parameters=scanDA_parameters,
                    init_tpsa_output=init_tpsa_output,
                    init_tpsa_input=init_tpsa_input,
                )
                tmp = [xset.tolist(), (np.ones(len(xset)) * ymax).tolist(), minzlist]
                xyminz = xyminz + list(zip(*tmp))
    
                xset, minzlist, idxminlist, cutoff, ar, scanDAoutput = scanDA(
                    xlist=np.arange(0.0e-4, 14.51e-3, 2e-4),
                    ymax=ymax,
                    scan_direction=1,
                    searchDA=searchDA,
                    scanDA_parameters=scanDA_parameters,
                    init_tpsa_output=init_tpsa_output,
                    init_tpsa_input=init_tpsa_input,
                )
                tmp = [xset.tolist(), (np.ones(len(xset)) * ymax).tolist(), minzlist]
                xyminz = xyminz + list(zip(*tmp))
            except Exception as err:
                print(dir(err))
                print(err.args)
                pass
            tt1 = [["scanxy 3", time.time(), time.time() - tt1[0][1]]]
            print(tt1)
        tt1 = [["scanxy 4", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        tt1 = [["scanxy 5", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
        x, y, z = list(zip(*xyminz))

        with gzip.GzipFile('temp.pgz', 'wb') as f:
            pickle.dump([x, y, z], f)
    else:
        with gzip.GzipFile('temp.pgz', 'rb') as f:
            x, y, z = pickle.load(f)
        
    d = plotxyz(x, y, z, diff_lvl=[-10], markersize=20)
    sv("junk9", [x, y, z])

    plt.show()
