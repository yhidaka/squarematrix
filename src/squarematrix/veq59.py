# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# import tesla
import os
import sys
import time
import pickle
import copy
import subprocess
import pdb
import tempfile

# from importlib import reload

# import commands
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

t0 = time.perf_counter()
timing = [["in veq52, start", time.perf_counter()]]

import PyTPSA

from . import jnfdefinition as jfdf
from . import squarematrixdefinition as sqdf
from . import lte2tpsa2map45 as lte2tpsa
from . import tracy_track, ltemanager, pyele_sdds
from .fortran import zcol, zcolnew

CACHED_TRACY_PASS_FUNCS = None
TRACKING_CODE = "ELEGANT"
# TRACKING_CODE = 'Tracy'

# DEBUG = True
DEBUG = False

if DEBUG:
    print(time.perf_counter() - t0, "in veq52, seconds for import sqdf and jfdf")
t0 = time.perf_counter()


def sv(filename, x):
    ff = open(filename, "wb")
    pickle.dump(x, ff)
    ff.close()


def rl(filename):
    ff = open(filename, "rb")
    xx = pickle.load(ff, encoding="latin1")
    ff.close()
    return xx


#%who	 uarray	 ux0	 ux1,ux2	 ux3	 uy0	 uy1	 uy2	 uy3

global Vm, U, maxchainlenposition, bKi, norder, powerindex, mlen

# checkjnf(ux,Jx,scalex,1,om,0,0,Msx,{'sectionname':'4','Jname':'Jx','uname':'ux','Msname':'Ms','muname':'mux'},1e-8,1e-5)


def checkjnf(u, J, scale, nx, phix0, ny, phiy0, Ms, info, tol1, tol2, powerindex):
    if DEBUG:
        print(
            "\n",
            info["sectionname"],
            ". Check u as left eigenvectors of M: U.M=exp(i*mu)*exp(J).U:",
        )
        print(info["Jname"], "=")
        jfdf.pim(J, len(J), len(J))
    maxchainlenposition, maxchainlen, chain, chainposition = jfdf.findchainposition(J)
    if DEBUG:
        print("position of max length chain=", maxchainlenposition)
        print("max length chain=", maxchainlen)
    tmp1 = np.dot(u, Ms)
    tmp2 = np.exp(1j * (nx * phix0 + ny * phiy0)) * np.dot(jfdf.exp(J, maxchainlen), u)
    tmp3 = tmp1 - tmp2
    if DEBUG:
        print(
            info["sectionname"],
            ". check ",
            info["uname"],
            ".",
            info["Msname"],
            "=exp(i*",
            info["muname"],
            ")*exp(",
            info["Jname"],
            ").",
            info["uname"],
            ",",
            abs(tmp3).max(),
            " relative error:",
            abs(tmp3).max() / abs(tmp1).max(),
        )

    if DEBUG:
        print(info["sectionname"], ". lowest order in ", info["uname"], "[i]:\n")
    for i in range(len(u)):
        tmp = [sum(powerindex[k]) for k, b in enumerate(abs(u[i])) if b > 1e-8]
        if tmp != []:
            if DEBUG:
                print(i, min(tmp))

    def dmt(k, m, tol=tol2):  # Dominating terms of k'th order in uy[m]
        dt = [
            [i, j, abs(u[m][i])]
            for i, j in enumerate(powerindex)
            if (sum(j) == k and abs(u[m][i]) > tol2)
        ]
        return dt

    if DEBUG:
        print(
            "\n",
            info["sectionname"],
            ". dominating terms in ",
            info["uname"],
            "[0], their order, and size:",
        )
    cc = [
        [k, sum(powerindex[k]), b, powerindex[k].tolist()]
        for k, b in enumerate(abs(u[0]))
        if b > tol2
    ]
    cci = np.argsort([i[2] for i in cc])
    ccs = [cc[i] for i in cci]
    if DEBUG:
        print("\n", info["sectionname"], ". 20 lowest order terms:")
        for i in cc[:20]:
            print(i)

    if DEBUG:
        print("\n", info["sectionname"], ". 20 dominating terms:")
        for i in ccs[-20:]:
            print(i)

        print(
            "\n",
            info["sectionname"],
            ". dominating terms of order 1 in ",
            info["uname"],
            "[0] :",
            [[j[0], j[1].tolist(), j[2]] for j in dmt(1, 0)],
        )
        print(
            info["sectionname"],
            ". dominating terms of order 2 in ",
            info["uname"],
            "[0]:",
            [[j[1].tolist(), j[2]] for j in dmt(2, 0)],
        )
        print(
            info["sectionname"],
            ". dominating terms of order 3 in ",
            info["uname"],
            "[0]:",
            [[j[1].tolist(), j[2]] for j in dmt(3, 0)],
        )

        print(
            info["sectionname"],
            ". Showing the lowest of power of x,y in every eigenvector :",
        )
    for k in range(len(u)):
        tmp = [sum(j) for i, j in enumerate(powerindex) if abs(u[k][i]) > tol2]
        if tmp != []:
            if DEBUG:
                print("k=", k, " lowest order=", min(tmp))
    return


# ltefilename="20140204_bare_1supcell"#"nsls2sr_supercell_ch77_20150406_1"#"20140204_bare_1supcell.lte"
"""
def gettpsa(ltefilename,deltap,nv=4,norder=7,uselte2madx=1):
        mfm,powerindex,sequencenumber=lte2tpsa.lte2madx2fort182tpsa(ltefilename,uselte2madx,nv=nv,norder=norder)

        mfmadx=[]

        for var in mfm: # the rows in mfm are the power index of x1,pxp1,y1,yp1, deltap, thus var is one of these
                mfrow=np.zeros(len(powerindex)) #prepare for one row in mfmadx as an array of length of powerindex
                for row in var:
                        inx,coeff,norder1,px,pxp,py,pyp,pdeltap=row #the 8 numbers in each row have coeeficients of the term with powers for x0,xp0,y0,y0, and deltap
                        px,pxp,py,pyp,pdeltap=list(map(int,[px,pxp,py,pyp,pdeltap])) #convert real (read from the file) into integer
                        mfrow[sequencenumber[px,pxp,py,pyp]]+=coeff*deltap**pdeltap #for each set of power of x0,xp0,y0,yp0, add contribution from different power of deltap
                mfmadx.append(mfrow)

        sqmxparameters={'powerindex':powerindex,'sequencenumber':sequencenumber,'nv':nv,'norder':norder}
        return mfmadx,sqmxparameters
"""


def wwinv(ux0, uy0, Zpar):
    x, xp, y, yp = [
        PyTPSA.tpsa(0, k, dtype=complex) for k in range(1, 5)
    ]  # this tpsa is not because of the tpsa.py, but the tpsa class defined in the tpsa.py file.
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], np.array([x, xp, y, yp])) / scalex
    ux0t = PyTPSA.tpsa(
        input_map=ux0, dtype=complex
    )  # uarray is [ux0,ux1,uy0,uy1], u is its tpsa of  zx,zxs,zy,zys
    uy0t = PyTPSA.tpsa(
        input_map=uy0, dtype=complex
    )  # uarray is [ux0,ux1,uy0,uy1], u is its tpsa of  zx,zxs,zy,zys
    wx = ux0t.composite(
        [zx, zxs, zy, zys]
    )  # w[0] corresponds to u[0].Z as tpsa of x,xp,y,y
    wxc = ux0t.conjugate(mode="ComplexPair").composite(
        [zx, zxs, zy, zys]
    )  # w[0] corresponds to u[0].Z as tpsa of x,xp,y,y
    wy = uy0t.composite(
        [zx, zxs, zy, zys]
    )  # w[0] corresponds to u[0].Z as tpsa of x,xp,y,y
    wyc = uy0t.conjugate(mode="ComplexPair").composite(
        [zx, zxs, zy, zys]
    )  # w[0] corresponds to u[0].Z as tpsa of x,xp,y,y
    w = np.array([wx, wxc, wy, wyc])
    winv = np.array(PyTPSA.inverse_map([wx, wxc, wy, wyc]))
    return w, winv


def wwJmatrix(uarray, Zpar3):
    bKi, scalex, norder, powerindex = Zpar3

    sequencenumber = np.zeros((norder + 1, norder + 1, norder + 1, norder + 1), "i")
    for k, ip in enumerate(powerindex):
        sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]] = k

    scalem = 1 / scalex
    mlen = len(powerindex)
    As = np.identity(mlen)
    for i in range(mlen):
        As[i, i] = scalem ** sum(powerindex[i])
    # Dxn is operator matrix to differentiate square matrix:
    # dwdx = np.dot(Dxn[0], wmatrix[0]) is dw0dx0
    # tmp=dwdx-wJmatrix[0,0]=0
    Dxn = np.zeros([4, mlen, mlen]) * (1 + 0.0j)
    for n in range(4):
        for k in range(mlen):
            pw = powerindex[k].copy()
            dpw = pw.copy()
            dpw[n] = dpw[n] - 1

            if dpw[n] > -1:
                sqn = sequencenumber[dpw[0], dpw[1], dpw[2], dpw[3]]
                Dxn[n, sqn, k] = pw[n]
                if k == 1:
                    if DEBUG:
                        print("Dx[n,sqn,k],sqn,k, pw=", Dxn[n, sqn, k], sqn, k, pw)

    wmatrix = jfdf.d3(uarray, As, bKi)
    wJmatrix = np.dot(Dxn, wmatrix.transpose()).transpose([2, 0, 1])
    return wmatrix, wJmatrix


def wwJmatrix_old(uarray, Zpar):
    x, xp, y, yp = [
        PyTPSA.tpsa(0, k, dtype=complex) for k in range(1, 5)
    ]  # this tpsa is not because of the tpsa.py, but the tpsa class defined in the tpsa.py file.
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], np.array([x, xp, y, yp])) / scalex
    utpsa = [
        PyTPSA.tpsa(input_map=ui.tolist(), dtype=complex) for ui in uarray
    ]  # uarray is [ux0,ux1,uy0,uy1], u is its tpsa of  zx,zxs,zy,zys
    w = [
        ui.composite([zx, zxs, zy, zys]) for ui in utpsa
    ]  # w[0] corresponds to u[0].Z as tpsa of x,xp,y,y
    wJ = [[wi.derivative(k, 1) for k in range(1, 5, 1)] for wi in w]
    lenth = len(uarray[0])
    wmatrix = np.array([i.pvl()[1][:lenth] for i in w])
    wJmatrix = np.array([[i.pvl()[1][:lenth] for i in j] for j in wJ])
    return wmatrix, wJmatrix


usecode = dict(tpsacode="tracy", trackingcode="elegant", use_existing_tpsa=1)


def gettpsa(
    ltefilename,
    deltap,
    nv=4,
    norder=7,
    norder_jordan=3,
    usecode=usecode,
    oneturntpsa="ELEGANT",
    mod_prop_dict_list=None,
):
    if oneturntpsa == "ELEGANT" and usecode["tpsacode"] == "madx":
        if usecode["use_existing_tpsa"] == 0:
            mfm, powerindex, sequencenumber = lte2tpsa.lte2madx2fort182tpsa(
                ltefilename, usecode, nv=nv, norder=norder
            )
            mfmadx = []
            for (
                var
            ) in (
                mfm
            ):  # the rows in mfm are the power index of x1,pxp1,y1,yp1, deltap, thus var is one of these
                mfrow = np.zeros(
                    len(powerindex)
                )  # prepare for one row in mfmadx as an array of length of powerindex
                for row in var:
                    (
                        inx,
                        coeff,
                        norder1,
                        px,
                        pxp,
                        py,
                        pyp,
                        pdeltap,
                    ) = row  # the 8 numbers in each row have coeeficients of the term with powers for x0,xp0,y0,y0, and deltap
                    px, pxp, py, pyp, pdeltap = list(
                        map(int, [px, pxp, py, pyp, pdeltap])
                    )  # convert real (read from the file) into integer
                    mfrow[sequencenumber[px, pxp, py, pyp]] += (
                        coeff * deltap**pdeltap
                    )  # for each set of power of x0,xp0,y0,yp0, add contribution from different power of deltap
                mfmadx.append(mfrow)
            sv("mfmadxsaved", [mfmadx, powerindex, sequencenumber])
            oneturntpsa = "no tpsamadx"
        else:
            mfmadx, powerindex, sequencenumber = rl("mfmadxsaved")

    elif oneturntpsa == "ELEGANT" and (
        usecode["tpsacode"] == "yoshitracy" or usecode["tpsacode"] == "yuetpsa"
    ):
        # 3. Generate powerindex
        if DEBUG:
            print("\n3. Generate powerindex and sequencenumber")
        # sequencenumber[i1,i2,i3,i4] gives the sequence number in power index for power of x^i1*xp^i2*y^i3*yp^i4
        sequencenumber = np.zeros((norder + 1, norder + 1, norder + 1, norder + 1), "i")
        powerindex = sqdf.powerindex4(norder)
        powerindex = array(powerindex, "i")
        mlen = len(powerindex)

        for i in range(mlen):
            ip = powerindex[i]
            sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]] = i

        LTE = ltemanager.Lattice(
            LTE_filepath=f"{ltefilename}.lte", used_beamline_name="RING"
        )
        """ it should have form as:
        mod_prop_dict_list = [
            {"elem_name": "Qh1G2c30a", "prop_name": "K1", "prop_val": 1.5},
            {"elem_name": "sH1g2C30A", "prop_name": "K2", "prop_val": 0.0},
        ]
        """
        N_KICKS = dict(CSBEND=20, KQUAD=20, KSEXT=4)
        # N_KICKS = dict(CSBEND=20, KQUAD=20, KSEXT=20)
        # N_KICKS = dict(CSBEND=40, KQUAD=40, KSEXT=20)

        pass_func_list = tracy_track.get_pass_funcs_from_ltemanager(
            LTE, N_KICKS=N_KICKS, mod_prop_dict_list=mod_prop_dict_list
        )

        global CACHED_TRACY_PASS_FUNCS
        CACHED_TRACY_PASS_FUNCS = pass_func_list

        if usecode["tpsacode"] == "yoshitracy":
            tps_module = 0
            if usecode["use_existing_tpsa"] == 0:
                tps_list = tracy_track.tps_cell_pass_trace_space(
                    tps_module, pass_func_list, norder, nv, dp0=deltap
                )
                sv("tracytpsalist", tps_list)
            else:
                tps_list = rl("tracytpsalist")

            mfmadx = []
            for var in tps_list[:nv]:
                mfrow = np.array([var.get_polynom_coeff(i) for i in powerindex])
                mfmadx.append(mfrow)
            oneturntpsa = tps_list

        else:  # usecode['tpsacode'] == 'yuetpsa', Use Hao Yue's PyTPSA
            # import pdb; pdb.set_trace()
            tps_module = 1
            if usecode["use_existing_tpsa"] == 0:
                tps_list_yue = tracy_track.tps_cell_pass_trace_space(
                    tps_module, pass_func_list, norder, nv, dp0=deltap
                )
                # import pdb; pdb.set_trace()
                assert np.all(np.all(tps_list_yue[0].pvl()[0] == powerindex, axis=1))

                mfmadx_yue = []
                for var in tps_list_yue[:nv]:
                    assert var.get_degree() == var.get_max_degree()
                    mfrow = np.array(var.pvl()[1])
                    mfmadx_yue.append(mfrow)

                if False:
                    plt.figure()
                    plt.semilogy(
                        np.abs(np.array(mfmadx) - np.array(mfmadx_yue)).T, ".-"
                    )

                if usecode.get("tmp", True):
                    temp = tempfile.NamedTemporaryFile(
                        suffix="", prefix="tmp_", dir=os.getcwd(), delete=True
                    )
                    mfmadx_fpstr = f"{temp.name}_mfmadx_yue.pkl"
                    oneturntpsa_fpstr = f"{temp.name}_oneturntpsa_yue.tps"
                    wwJmtx_fpstr = f"{temp.name}_wwJmtx.pkl"
                    temp.close()
                else:
                    mfmadx_fpstr = "mfmadx_yuesaved"
                    oneturntpsa_fpstr = "oneturntpsa_yue_saved"
                    wwJmtx_fpstr = "wwJmtx_saved"

                sv(mfmadx_fpstr, [mfmadx_yue, ltefilename, deltap])
                oneturntpsa = tps_list_yue
                PyTPSA.save(oneturntpsa_fpstr, oneturntpsa)
            else:
                mfmadx_fpstr = usecode.get("mfmadx_filepath", "mfmadx_yuesaved")
                oneturntpsa_fpstr = usecode.get(
                    "oneturntpsa_filepath", "oneturntpsa_yue_saved"
                )

                mfmadx_yue, ltefilename_saved, deltap_saved = rl(mfmadx_fpstr)
                if ltefilename_saved != ltefilename or deltap_saved != deltap:
                    print(
                        "\n\n\nltefilename or deltap differ from saved data, use use_existing_tpsa=0"
                    )
                    print(
                        "ltefilename=",
                        ltefilename,
                        "ltefilename_saved=",
                        ltefilename_saved,
                    )
                    print("deltap=", deltap, "deltap_saved=", deltap_saved)
                    raise RuntimeError
                oneturntpsa = PyTPSA.load(oneturntpsa_fpstr)

            mfmadx = mfmadx_yue
    elif oneturntpsa != "ELEGANT" and usecode["tpsacode"] == "yuetpsa":
        mfmadx = rl("mfmadx_yuesaved")
        oneturntpsa = oneturntpsa
        sequencenumber = np.zeros((norder + 1, norder + 1, norder + 1, norder + 1), "i")
        powerindex = sqdf.powerindex4(norder)

    else:
        raise NotImplementedError(f'Invalid "uselte2madx": {"uselte2madx"}')

    sqmxparameters = {
        "powerindex": powerindex,
        "sequencenumber": sequencenumber,
        "nv": nv,
        "norder": norder,
    }
    if (norder == 5 and len(mfmadx[0]) != 126) or (
        norder == 7 and len(mfmadx[0]) != 330
    ):
        print(
            '\n\n\nExit!!\n\nSaved mfmadx doe not match norder!\n\nSet usecode["use_existing_tpsa"]=0, or change norder and try again.'
        )
        sys.exit(0)
    return (
        mfmadx,
        sqmxparameters,
        [mfmadx, norder, powerindex],
        dict(mfmadx=mfmadx_fpstr, oneturntpsa=oneturntpsa_fpstr, wwJmtx=wwJmtx_fpstr),
    )


def getMlist(
    ltefilename,
    deltap,
    mfmadx,
    sqmxparameters,
    norder_jordan,
    mapMatrixMlist,
    usecode="ELEGANT",
    copy_LTE_file=False,
):
    if DEBUG:
        tt0 = [["getM, 1, start", time.time(), 0]]
        print(tt0)
    (
        Ms10,
        phix0,
        phiy0,
        powerindex7,
        norder,
        bK,
        bKi,
        sqrtbetax,
        sqrtbetay,
        msf,
        tbl,
        scalemf,
        deltap,
        xfix,
        xpfix,
        dmuxy,
    ) = lte2tpsa.dpmap(mfmadx, deltap, sqmxparameters)
    # sv("jfM.dat",[Ms10,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf,deltap,xfix,xpfix])
    if DEBUG:
        tt1 = [["getM, 2", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    mapMatrixM = dict(
        zip(
            mapMatrixMlist,
            [
                Ms10,
                phix0,
                phiy0,
                powerindex7,
                norder,
                bK,
                bKi,
                sqrtbetax,
                sqrtbetay,
                msf,
                tbl,
                scalemf,
                deltap,
                xfix,
                xpfix,
                dmuxy,
            ],
        )
    )
    if copy_LTE_file:
        os.system(
            "cp " + ltefilename + ".lte junk.lte"
        )  # choose elegant input lattice lte file junk.lte
    # take jordan vector order as norder_jordan to replace norder
    norder = norder_jordan
    plen = (norder + 1) * (norder + 2) * (norder + 3) * (norder + 4) / 24
    powerindex = sqdf.powerindex4(norder)
    sequencenumber = np.zeros(
        (norder + 1, norder + 1, norder + 1, norder + 1), "i"
    )  # sequence number is given as a function of a power index
    powerindex = array(powerindex, "i")
    mlen = len(powerindex)
    Ms10 = Ms10[:mlen, :mlen]
    bK = bK[:mlen, :mlen]
    bKi = bKi[:mlen, :mlen]

    for i in range(mlen):
        ip = powerindex[i]
        sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]] = i

    mlen = len(Ms10)

    if DEBUG:
        tt1 = [["getM, 3", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    # 3.Check Jordan block phiy0
    uy, uybar, Jy, scaley, Msy, As2y, Asm2y = jfdf.UMsUbarexpJ(
        Ms10, phiy0, 1, powerindex, scalemf, sequencenumber[0, 0, 1, 0], ypowerorder=7
    )
    if DEBUG:
        tt1 = [["getM, 4", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    checkjnf(
        uy,
        Jy,
        scaley,
        0,
        phix0,
        1,
        phiy0,
        Msy,
        {
            "sectionname": "3",
            "Jname": "Jy",
            "uname": "uy",
            "Msname": "Msy",
            "muname": "muy",
        },
        1e-8,
        1e-5,
        powerindex,
    )

    # 4. Check block phix0

    if DEBUG:
        print("\n6. Check ux as left eigenvectors of M: ux.M=exp(i*mux)*exp(Jx).ux:")
    ux, uxbar, Jx, scalex, Msx, As2x, Asm2x = jfdf.UMsUbarexpJ(
        Ms10, phix0, 1, powerindex, scalemf, sequencenumber[1, 0, 0, 0], ypowerorder=7
    )
    checkjnf(
        ux,
        Jx,
        scalex,
        1,
        phix0,
        0,
        phiy0,
        Msx,
        {
            "sectionname": "4",
            "Jname": "Jx",
            "uname": "ux",
            "Msname": "Ms",
            "muname": "mux",
        },
        1e-8,
        1e-5,
        powerindex,
    )

    # u6=jfdf.d3(u6,As6,Asm1) #U.M.Ubar=u.As.M.Asm.ubar=u.Ms.ubar=exp(i*phi0+J), so U=u.As
    # So u41.As41=u42.As42=U4, so u41=u42.As42.Asm41 when we have scaled one block u42 using As42, and
    # we want use scaling As41(=As1) to scale to u41, then for rescaling we use u41=u42.As42.Asm41
    # So u4=u41.As41.Asm1.
    uy = jfdf.d3(
        uy, As2y, Asm2x
    )  # see M scaling in Jordan form reformulation.one, this converts uy by rescaling back to using scalex.
    Ms = Msx
    ux = ux / (ux[0][1])
    uy = uy / (uy[0][3])

    # 5. Construct map matrix between zsbar,zsbars and w,ws: W=V.Zs
    # print("\n5. Construct map matrix V from zsbar plane to w plane")
    # maxchainlenpositionx, maxchainlenx, chainx, chainpositionx=jfdf.findchainposition(Jx)
    # maxchainlenpositiony, maxchainleny, chainy, chainpositiony=jfdf.findchainposition(Jy)
    if DEBUG:
        tt1 = [["getM, 5", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    return (
        mapMatrixM,
        powerindex,
        ux,
        uy,
        norder,
        xfix,
        xpfix,
        scalex,
        bKi,
        deltap,
        Jx,
        Jy,
    )


def wwJwinvf(uarray, Zpar3, use_existing_tpsa=0, saved_filepath=""):
    # Zpar=bKi,scalex,norder,powerindex
    # 5. Construct w,wc tpsa
    """Hao yue's inverse matrix method:
    Btw, I just implement the reverse function of a simple case.  In a 4D case (zx, zx*, zy, zy*), we get the polynomial (w0x, w0x*, w0y, w0y*)=w(zx, zx*, zy, zy*), using the following procedure, we can get the inverse polynomials (zx, zx*, zy, zy*)= w^(-1) (w0x, w0x*, w0y, w0y*), so that w^-1(w)=I.
    The procedure uses the fix point tricks.  First separate the linear part of w out, w=M+N, M is the linear matrix; N is the nonlinear part with zero linear components.
    Then w_n^(-1)=M^(-1) (1- N(w_(n-1)^(-1))).
    """
    norder_winv = 5
    PyTPSA.initialize(4, norder_winv)
    powerindex_winv = sqdf.powerindex4(norder_winv)
    """
    We found for rough calculation of first xyd0 solution, using norder=5 to calculate inverse w is fast and good  enough
    for further more accurate solution by fsolve. So to calculate w5invmtx, we use ux0,uy0 by 3rd order Joardan vector while the inverse tpsa
    is calculated by expansion tpsa to 5'th order
    However, the best Jordan vector should use norder=3.
    And for one turn map calculation, we need more accurate norder=7.
    Hence we use norder =3,5,7 in 3 different places.
    """

    if saved_filepath:
        wwJmtx_fpstr = saved_filepath
    else:
        wwJmtx_fpstr = "wwJmtx_saved"

    if use_existing_tpsa == 0:
        if DEBUG:
            print("now calculating inverse tpsa winv")
            tt0 = [["start", time.time(), 0]]
        # PyTPSA.initialize(4, 3)
        # w, wJ = wwJ(uarray, Zpar3)
        w3mtx, w3Jmtx = wwJmatrix(uarray, Zpar3)

        ux0, uy0 = uarray[0], uarray[2]
        w0, winv = wwinv(ux0, uy0, Zpar3)
        w5mtx = np.array([i.pvl()[1] for i in w0])
        w5invmtx = np.array([i.pvl()[1] for i in winv])
        sv(saved_filepath, [w3mtx, w3Jmtx, w5mtx, w5invmtx])
        wwJwinv = w3mtx, w3Jmtx, w5mtx, w5invmtx, norder_winv, powerindex_winv
    elif use_existing_tpsa == 1:
        w3mtx, w3Jmtx, w5mtx, w5invmtx = rl(wwJmtx_fpstr)
        wwJwinv = w3mtx, w3Jmtx, w5mtx, w5invmtx, norder_winv, powerindex_winv
    return wwJwinv


def vinv_matrix(v1s, v2s, v0norm, winvmtx, norder_winv, powerindex_winv):
    v1 = v1s * v0norm[0]
    v1c = v1.conjugate()
    v2 = v2s * v0norm[1]
    v2c = v2.conjugate()
    v = np.array([v1, v1c, v2, v2c]).transpose()

    Zs7 = zcolnew.zcolarray(v, norder_winv, powerindex_winv).transpose()
    xyd = np.dot(winvmtx, Zs7).transpose().real
    return xyd


# ux0,ux1,ux2=ux[0],ux[1],ux[2]

# 3. Define functions for analysis
print("\n in veq52, 3. Define functions for analysis.")
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import FormatStrFormatter

# 7 define wxp0,wyp0, ux0,uy0,ux1,uy1,ux2,uy2


def Zszaray(xy, Zpar):  # Construct ZsZsp columns from each turn in the tracking result.
    tt0 = [["in Zszaray 1, start", time.time(), 0]]
    print("tt0=", tt0)
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], xy) / scalex
    xxpyyp = [zx, zxs, zy, zys]
    xxpyyp = np.array(xxpyyp).transpose()
    tt1 = [["in Zszaray, 2. ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    Zs = zcolnew.zcolarray(xxpyyp, norder, powerindex)
    tt1 = [["in Zszaray, 3. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    tt1 = [["in Zszaray, 3. ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    return Zs


def Zsz(xs, Zpar):
    x, xp, y, yp = xs
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], array([x, xp, y, yp])) / scalex
    # Zs=sqdf.Zcol(zx,zxs,zy,zys,norder,powerindex) #Zxs is the zsbar,zsbars, column, here zsbar=zbar/scalex
    Zs = zcol.zcol([zx, zxs, zy, zys], norder, powerindex)
    return Zs


def modularToPi(y):
    x = copy.deepcopy(y)
    x = x.real + np.pi
    x = x % (2.0 * np.pi)
    x = x - np.pi
    return x


def modularToNo2(y, N):
    x = copy.deepcopy(y)
    x = x + N / 2
    x = x % (N)
    x = x - N / 2
    return x


def matrixsort(
    x, ndiv=40
):  # calculate the sorted index for a 1600 array before it is converted to 40x40 matrix
    tmp1 = np.argsort(x)

    def indx(n):
        return np.unravel_index(n, (ndiv, ndiv))

    tmp3 = map(indx, tmp1)
    return tmp3


def oneturnmatrixmap(xy0, mfmadx, norder, powerindex7):
    if DEBUG:
        tt0 = [["in oneturnmatrixmap 1, start", time.time(), 0]]
        print("tt0=", tt0)
    Zs0 = zcolnew.zcolarray(xy0, norder, powerindex7)
    # Zs0 = zcolarray.zcolarray(xy0, norder, powerindex7)
    if DEBUG:
        tt1 = [["in oneturnmatrixmap 2", time.time(), time.time() - tt0[0][1]]]
        print("tt1=", tt1)
    xy1 = np.dot(mfmadx, Zs0.transpose()).transpose()
    if DEBUG:
        tt1 = [["in oneturnmatrixmap 3", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)
    return xy1


def vphi(
    xyd0, ab, uvar, v0norm, oneturntpsa, Vmtx, outxv=1
):  # Calculate i*phi as function of the action angle of v to see its DC value and fluctuation
    # import pdb;
    if DEBUG:
        tt0 = [["vphi, 1, start", time.time(), 0]]
        print(tt0)
    if oneturntpsa == "ELEGANT":
        trackcode = "ELEGANT"
    elif oneturntpsa == "Tracy":
        trackcode = "Tracy"
    else:
        trackcode = "tpsa"
    # see /Users/lihuayu/Desktop/nonlineardynamics/henonheiles/sqmxnsls/latticenonlinerresonance.lyx eq.35
    bKi, scalex, norder_jordan, powerindex, xfix, xpfix, deltap = (
        uvar.bKi,
        uvar.scalex,
        uvar.norder_jordan,
        uvar.powerindex,
        uvar.xfix,
        uvar.xpfix,
        uvar.deltap,
    )
    Zpar = bKi, scalex, norder_jordan, powerindex
    # return [x',  px',                y',    py'            ] =
    # return [px,  -x-2xy,             py,    -y -x^2 + y^2]
    # So zx'=x'-1j*px'=px-1j*(-x-2x*y), zy'=y'-1j*py'=py-1j*(-y -x^2 + y^2)
    scan = xyd0.transpose()
    a1, a2 = ab
    fluc = []
    npart = len(scan)
    x0off = 0e-7  # it is found that there is an round off error in elegant that causes residual energy delta non-zero and caused an offset for x0 and xp0
    xp0off = 0e-10  # which is to be removed here for Jordan form calculation. This offsets are found when we reduce the radius in w plane, the circle in the
    y0off = 0e-7  # it is found that there is an round off error in elegant that causes residual energy delta non-zero and caused an offset for x0 and xp0
    yp0off = 0e-10  # which is to be removed here for Jordan form calculation. This offsets are found when we reduce the radius in w plane, the circle in the

    if DEBUG:
        tt1 = [["vphi, 2 end", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
    if trackcode == "ELEGANT":
        with open("beamsddshead", "w") as f:
            f.write(_get_beamsddshead_contents())

        flnm = open("beamsddstail", "w")
        flnm.write("%10i \n" % npart)
        pid = 0
        xxp0 = []
        p_central_mev = 3000
        # import pdb; pdb.set_trace()
        for i in scan:
            # for jk in [1]:
            pid = pid + 1
            # theta1,theta2,x0,xp0,y0,yp0=i
            x0, xp0, y0, yp0 = i
            # zs=Zsz((x0,xp0,y0,yp0),Zpar)
            # wx0n=np.dot(ux0,zs)
            # wy0n=np.dot(uy0,zs)
            # wx1n=np.dot(ux1,zs)
            # wy1n=np.dot(uy1,zs)
            # wx2n=np.dot(ux2,zs)
            # wy2n=np.dot(uy2,zs)
            x = x0 + xfix
            xp = xp0 + xpfix
            y = y0 + y0off
            yp = yp0 + yp0off
            # flnm.write('%10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10i \n'%(x0,xp0,y0,yp0,0,5.870852e+03,0,pid))
            flnm.write(
                "%10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10i \n"
                % (x, xp, y, yp, 0, p_central_mev / 0.511, 0, pid)
            )
            # xxp0.append([theta1,theta2,x0,xp0,y0,yp0])
            xxp0.append([x0, xp0, y0, yp0])

        flnm.close()
        os.system("cat beamsddshead beamsddstail >beam.sdds")

        with open("tracking_dp_ltefilename.ele", "w") as f:
            f.write(_get_tracking_dp_ele_contents())

        # print ('in vph, elegant output into junktmp.')
        os.system(
            "elegant  -macro=npass=2,interval=1,dp="
            + str(deltap)
            + " tracking_dp_ltefilename.ele >junktmp"
        )
        ta, xxp1 = subprocess.getstatusoutput(
            'sddsprocess ring.w1 -filter=par,Pass,1,1 -pipe=out|sddsprintout -pipe=in -col="(x,xp,y,yp)" -noLabel -noTitle '
        )  # read output of elegant for x,xp after one turn
        xxp1 = xxp1.split(
            "\n"
        )  # load the virtual file to convert it into an array of x,xp after 1 turn
        xxp1 = [i.split() for i in xxp1]
        xxp1 = np.array(
            [list(map(eval, i)) for i in xxp1]
        )  # Notice the step x1-xfix, xp1-xpfix is to be carried out later at line 515
        xxp0 = array(xxp0)

    elif trackcode == "Tracy":

        pass_func_list = CACHED_TRACY_PASS_FUNCS

        v0_list = []
        for i in scan:
            x0, xp0, y0, yp0 = i
            x = x0 + xfix
            xp = xp0 + xpfix
            y = y0 + y0off
            yp = yp0 + yp0off
            v0_list.append([x, xp, y, yp, deltap])

            # v = tracy_track.cell_pass_trace_space(
            # pass_func_list, x, xp, y, yp, dp0=deltap)

        v0s = np.array(v0_list).T
        xs, xps, ys, yps, deltaps = v0s
        vs = tracy_track.cell_pass_trace_space(
            pass_func_list, xs, xps, ys, yps, dp0=deltaps
        )
        xxp1 = np.array(vs).T[:, :4]

        # for i in range(xxp1.shape[1]):
        # plt.figure()
        # plt.plot(xxp1[:,i], 'b.-')
        # plt.plot(xxp1_ele[:,i], 'r.-')

        # plt.figure()
        # plt.plot((xxp1[:,i] - xxp1_ele[:,i]) / xxp1_ele[:,i] * 1e2, 'b.-')

        xxp0 = scan
    elif trackcode == "tpsa":
        x0, xp0, y0, yp0 = scan.transpose()
        x, xp, y, yp = (
            x0 + xfix,
            xp0 + xpfix,
            y0 + y0off,
            yp0 + yp0off,
        )
        xxpyyp = np.array([x, xp, y, yp]).transpose()
        mfmadx, norder, powerindex7 = oneturntpsa
        xxp1 = oneturnmatrixmap(xxpyyp, mfmadx, norder, powerindex7)
    else:
        raise NotImplementedError(f"TRACKING_CODE = {TRACKING_CODE}")

    if DEBUG:
        tt1 = [["vphi, 3 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    x1, xp1, y1, yp1 = [xxp1[:, i] for i in range(4)]
    # every row of x0x1 give theta10,theta20,x0,xp0y0,yp0,x1,xp,y1,yp1. x0 is for turn 0, x1 is for turn1, so x0x1 is a map from x0 to x1 as a function on the theta1,theta2 plane
    # theta10,theta20,x0,xp0,y0,yp0,x1,xp1,y1,yp1=array(list(zip(*x0x1)))#x0x1 is z1 as function of z0 on the theta plane.
    x1, xp1 = (
        x1 - xfix,
        xp1 - xpfix,
    )  # change x1 from relative to 0 origin (as obtained from elegant calculation) to relative to xfix as origin
    # x0x1=array([theta10,theta20,x0,xp0,y0,yp0,x1,xp1,y1,yp1]).transpose()#after shifting origin to xfix, recover x0x1.
    # x0x1 = array(
    #    [x0, xp0, y0, yp0, x1, xp1, y1, yp1]
    # ).transpose()  # after shifting origin to xfix, recover x0x1.
    if DEBUG:
        tt1 = [["vphi, 4 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    xy0 = scan.transpose()  # np.array([x0, xp0, y0, yp0])
    xy1 = np.array([x1, xp1, y1, yp1])
    # zs0 = Zszaray(xy0, Zpar)
    # zs1 = Zszaray(xy1, Zpar)
    Zsdx0 = zcolnew.zcolarray(xy0.transpose(), norder_jordan, powerindex)
    Zsdx1 = zcolnew.zcolarray(xy1.transpose(), norder_jordan, powerindex)
    v1mtx, v2mtx = Vmtx
    v10 = np.dot(v1mtx, Zsdx0.transpose())
    v20 = np.dot(v2mtx, Zsdx0.transpose())
    v11 = np.dot(v1mtx, Zsdx1.transpose())
    v21 = np.dot(v2mtx, Zsdx1.transpose())
    """
    v1x0 = np.dot(a1 / v0norm[0], uarray)
    v2x0 = np.dot(a2 / v0norm[1], uarray)
    v10t = np.dot(v1x0, zs0.transpose())
    v20t = np.dot(v2x0, zs0.transpose())
    v11 = np.dot(v1x0, zs1.transpose())
    v21 = np.dot(v2x0, zs1.transpose())
    """
    if DEBUG:
        tt1 = [["vphi, 5, 2zsdx", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    av10, av20 = (
        abs(v10),
        abs(v20),
    )  # turn 0
    av11, av21 = (
        abs(v11),
        abs(v21),
    )  # turn 1
    if DEBUG:
        tt1 = [["vphi, 5.1 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    thetav10, thetav20 = np.log(v10).imag, np.log(v20).imag
    thetav11, thetav21 = np.log(v11).imag, np.log(v21).imag
    """
    pdb.set_trace()
    av10, av20, thetav10, thetav20 = (
        abs(v10),
        abs(v20),
        array(list(map(cmath.phase, v10))),
        array(list(map(cmath.phase, v20))),
    )  # turn 0
    # pdb.set_trace()
    av11, av21, thetav11, thetav21 = (
        abs(v11),
        abs(v21),
        array(list(map(cmath.phase, v11))),
        array(list(map(cmath.phase, v21))),
    )  # turn 1
    """
    if DEBUG:
        tt1 = [["vphi, 6 , logv12", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    th10, th20 = thetav10[0], thetav20[0]
    phi1a = -1j * np.log(av11 / av10) + modularToPi(thetav11 - thetav10)
    phi2a = -1j * np.log(av21 / av20) + modularToPi(thetav21 - thetav20)
    # print("len(x0x1),len(phi1a),len(phi2a)=",len(x0x1),len(phi1a),len(phi2a))
    if DEBUG:
        tt1 = [["vphi, 7 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    phi1a = phi1a.reshape([len(phi1a), 1])
    phi2a = phi2a.reshape([len(phi2a), 1])
    av10, av20, thetav10, thetav20 = (
        av10.reshape([len(phi1a), 1]),
        av20.reshape([len(phi1a), 1]),
        thetav10.reshape([len(phi1a), 1]),
        thetav20.reshape([len(phi1a), 1]),
    )
    v10, v20, v11, v21, thetav11, thetav21 = (
        v10.reshape([len(phi1a), 1]),
        v20.reshape([len(phi1a), 1]),
        v11.reshape([len(phi1a), 1]),
        v21.reshape([len(phi1a), 1]),
        thetav11.reshape([len(phi1a), 1]),
        thetav21.reshape([len(phi1a), 1]),
    )
    if DEBUG:
        tt1 = [["vphi, 8 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    # fluc=np.hstack([x0x1,phi1a,phi2a,av10,av20,thetav10,thetav20,v10,v20,v11,v21])
    fluc = [
        x0,
        xp0,
        y0,
        yp0,
        x1,
        xp1,
        y1,
        yp1,
        np.array(phi1a),
        np.array(phi2a),
        av10,
        av20,
        thetav10,
        thetav20,
        v10,
        v20,
        v11,
        v21,
        thetav11,
        thetav21,
    ]

    """
    fluc = phi1a.reshape(len(phi1a)), phi2a.reshape(len(phi1a)), th10, th20
    if outxv == 1:
        fluc = np.hstack(
            [
                x0x1,
                np.array(phi1a),
                np.array(phi2a),
                av10,
                av20,
                thetav10,
                thetav20,
                v10,
                v20,
                v11,
                v21,
                thetav11,
                thetav21,
            ]
        )
        fluc = list(zip(*fluc))
    """
    if DEBUG:
        tt1 = [["vphi, 9 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        tt1 = [["vphi, 10 end", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
    return fluc  # np.array(fluc)


def veq1(
    x, rhs
):  # Equation for z:  v1(z)==v1,v2(z)==v2 , here v1,v2 are fuctions v1x,v3x defined in section 19.2
    # import pdb; pdb.set_trace()
    (
        acoeff,
        v1r,
        v2r,
        v0norm,
        uarray,
        Zpar,
    ) = rhs  # r in v1r means it is v1 at the RHS of the equation.
    xs = x[0], x[1], x[2], x[3]
    Zs = Zsz(
        xs, Zpar
    )  # x here actually represents z=x,xp,y,yp at one point, Zsz calculate the column from z
    a1, a2 = acoeff
    v1x = np.dot(
        a1 / v0norm[0], uarray
    )  # This is the fft result from section 18 of hn23.
    v2x = np.dot(a2 / v0norm[1], uarray)
    v1 = np.dot(v1x, Zs)  # Notice wx,wy here explicitly are normalized here.
    v2 = np.dot(v2x, Zs)
    v1mv1 = v1 - v1r
    v2mv2 = v2 - v2r
    return (v1mv1.real, v1mv1.imag, v2mv2.real, v2mv2.imag)


# ❇️functions used in early development:
"""
for ik in [0]:
    def wwJ(uarray, Zpar):
        x, xp, y, yp = [
            PyTPSA.tpsa(0, k, dtype=complex) for k in range(1, 5)
        ]  # this tpsa is not because of the tpsa.py, but the tpsa class defined in the tpsa.py file.
        bKi, scalex, norder, powerindex = Zpar
        zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], np.array([x, xp, y, yp])) / scalex
        utpsa = [
            PyTPSA.tpsa(input_map=ui.tolist(), dtype=complex) for ui in uarray
        ]  # uarray is [ux0,ux1,uy0,uy1], u is its tpsa of  zx,zxs,zy,zys
        w = [
            ui.composite([zx, zxs, zy, zys]) for ui in utpsa
        ]  # w[0] corresponds to u[0].Z as tpsa of x,xp,y,y
        wJ = [[wi.derivative(k, 1) for k in range(1, 5, 1)] for wi in w]
        return np.array(w), np.array(wJ)


    def oneturntpsamap(x0, xp0, y0, yp0, oneturntpsa, code="yuetpsa"):
        if code == "yuetpsa":
            x, xp, y, yp = oneturntpsa[:4]
            x1, xp1, y1, yp1 = [i.evaluate([x0, xp0, y0, yp0]) for i in [x, xp, y, yp]]
        elif code == "yoshitracy":
            '''
                    But I found the following works:

                    xt,xpt,yt,ypt=oneturntpsa[:4]
                    var_names = ['x', 'xp', 'y', 'yp']
                    var_values = [1e-3, 0.0, 0.0, 0.0]
                    x = xt.subs(var_names, var_values)
                    x_sub_float=x.get_polynom_coeff([0,0,0,0])
                    '''
            x, xp, y, yp = oneturntpsa[:4]
            tmp = [
                i.subs(["x", "xp", "y", "yp"], [x0, xp0, y0, yp0]) for i in [x, xp, y, yp]
            ]
            x1, xp1, y1, yp1 = [i.get_polynom_coeff([0, 0, 0, 0]) for i in tmp]
        return x1, xp1, y1, yp1


    def veqt(
        X, rhs
    ):  # Equation for z:  v1(z)==v1,v2(z)==v2 , here v1,v2 are fuctions v1x,v3x defined in section 19.2
        # import pdb; pdb.set_trace()
        (
            v1r,
            v2r,
            Vtp,
        ) = rhs  # v12t is tpsa of v12(x,xp,y,yp), r in v1r means it is v1 at the RHS of the equation.
        x, xp, y, yp = X
        v1 = Vtp[0].evaluate(
            [x + 0j, xp, y, yp]
        )  # Notice wx,wy here explicitly are normalized here.
        v2 = Vtp[1].evaluate([x + 0j, xp, y, yp])
        v1mv1 = v1 - v1r
        v2mv2 = v2 - v2r
        return (v1mv1.real, v1mv1.imag, v2mv2.real, v2mv2.imag)


    def veqmatrix(
        Zs, rhs
    ):  # Equation for z:  v1(z)==v1,v2(z)==v2 , here v1,v2 are fuctions v1x,v3x defined in section 19.2
        # import pdb; pdb.set_trace()
        (
            v1r,
            v2r,
            Vmtx,
        ) = rhs  # v12t is tpsa of v12(x,xp,y,yp), r in v1r means it is v1 at the RHS of the equation.
        v1mtx, v2mtx = Vmtx
        v1 = np.dot(v1mtx, Zs)  # Notice wx,wy here explicitly are normalized here.
        v2 = np.dot(v2mtx, Zs)
        v1mv1 = v1 - v1r
        v2mv2 = v2 - v2r
        return (v1mv1.real, v1mv1.imag, v2mv2.real, v2mv2.imag)



    def dVdX(
        X, dVdXt
    ):  # Equation for z:  v1(z)==v1,v2(z)==v2 , here v1,v2 are fuctions v1x,v3x defined in section 19.2
        # import pdb; pdb.set_trace()
        x, xp, y, yp = X
        (
            dv1dXt,
            dv2dXt,
        ) = dVdXt  # dVdXt is tpsa of Jacobian dV/dX(x,xp,y,yp), V=v1r,v1i,v2r,v2i
        dv1dX = np.array([i.evaluate([x + 0j, xp, y, yp]) for i in dVdXt[0]])
        dv2dX = np.array([i.evaluate([x + 0j, xp, y, yp]) for i in dVdXt[1]])
        dVdX = np.array([dv1dX.real, dv1dX.imag, dv2dX.real, dv2dX.imag])
        return dVdX
"""


def dVdXmatrix(
    Zs, dVdXmtx
):  # Equation for z:  v1(z)==v1,v2(z)==v2 , here v1,v2 are fuctions v1x,v3x defined in section 19.2
    # import pdb; pdb.set_trace()
    (
        dv1dXmtx,
        dv2dXmtx,
    ) = dVdXmtx  # dVdXt is tpsa of Jacobian dV/dX(x,xp,y,yp), V=v1r,v1i,v2r,v2i
    dv1dX = np.dot(dv1dXmtx, Zs)
    dv2dX = np.dot(dv2dXmtx, Zs)
    dVdX = np.array([dv1dX.real, dv1dX.imag, dv2dX.real, dv2dX.imag])
    return dVdX


avoidpiby = 1e-6  # avoidpiby is used to rotate the phase of normalized ux to avoid
# exactly at phase pi in determining the initial quadrant in phase space of v10.
# wxp0=np.dot(vph.ux[0],zs0)*np.exp(-1j*avoidpiby), now replaced by v0norm


def _get_tracking_dp_ele_contents():

    contents = """
&run_setup
    lattice = "junk.lte",
        p_central_mev = 3000,
    use_beamline=ring,
        rootname="ring",
!       parameters = %s.param,
    !magnets = %s.mag
    !final = %s.fin,
    ! the second-order is needed here only for the twiss computations.
    ! the tracking is done with kick elements
    losses=%s.los
    default_order = 3,
&end

&alter_elements name=*, type=CSBEN*, item=ISR, value=0 &end
&alter_elements name=*, type=CSBEN*, item=SYNCH_RAD, value=0 &end
&alter_elements name=*, type=CSBEN*, item=N_KICKS, value=64 &end
&alter_elements name=*, type=CSBEN*, item=INTEGRATION_ORDER, value=4 &end
&alter_elements name=*, type=KQUAD*, item=N_KICKS, value=48 &end
!sextupoles kicks
&alter_elements name=*, type=KSEXT*, item=N_KICKS, value=32 &end
&alter_elements name=MA0, item=DY, value=0e-6 &end
&alter_elements name=MA0, item=DP, value=<dp> &end
!&alter_elements name=MA0, item=DX, value=<dx> &end
!&alter_elements name=MA0, item=DXP, value=<dxp> &end
!&alter_elements name=W1, type=WATCH, item=mode, value="coordinate" &end
&alter_elements name=W1, type=WATCH, item=INTERVAL, value=<interval> &end

&run_control
!if n_indices = 1, there will be a scan of one parameter
!        n_steps = 1,
!        n_indices = 0,
        n_passes = <npass>
&end



&sdds_beam
        input = beam.sdds,
        input_type = "elegant",
        prebunched = 0,
&end

&track
&end

&stop &end
EOF
    """

    return contents


def _get_beamsddshead_contents():

    beamsddshead_contents = """SDDS1
!# little-endian
&column name=x, units=m, type=double, &end
&column name=xp, type=double, &end
&column name=y, units=m, type=double, &end
&column name=yp, type=double, &end
&column name=t, units=s, type=double, &end
&column name=p, units=m$be$nc, type=double, &end
&column name=dt, units=s,type=double, &end
&column name=particleID, type=long, &end
&data mode=ascii, &end
!page number 1
"""

    return beamsddshead_contents


def xtracking(xmax, ymax, npass, n_theta, deltap, xfix, xpfix, om1accurate=0):
    # 6. Plot tracking result
    print("\n6. Tracking x,y motion")

    if TRACKING_CODE == "ELEGANT":
        with open("beamsddshead", "w") as f:
            f.write(_get_beamsddshead_contents())

        flnm = open("beamsddstail", "w")
        npart = 1
        flnm.write("%10i \n" % npart)
        # xmax=x00#-1e-3#-1.004e-4#-1.e-3#-1.5e-3#-2.7e-3
        # ymax=y00#0.1e-3#0.26e-3

        ypi = 0
        # npass=819#8193
        xxp0 = []
        p_central_mev = 3000
        # x0off=0e-7  #it is found that there is an round off error in elegant that causes residual energy delta non-zero and caused an offset for x0 and xp0
        # xp0off=0e-10 #which is to be removed here for Jordan form calculation. This offsets are found when we reduce the radius in w plane, the circle in the
        # zbar plane in the following calculation has a center which is shift away from origin.
        x0off, xp0off, y0off, yp0off = (
            0,
            0,
            0,
            0,
        )  # array([  4.06720300e-10,   2.48284400e-11,   1.00000000e-06,0.00000000e+00])# this is xxp[0] obtained after the calculation of xxp
        x0 = xmax - x0off
        xp0 = 0 - xp0off
        y0 = ymax - y0off
        yp0 = ypi - yp0off
        pid = 1
        # write into input file for elegant x0,xp0 for all particles of different theta0
        # flnm.write('%10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10i \n'%(x0,xp0,0,0,0,5.870852e+03,0,pid))
        flnm.write(
            "%10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10.6g %10i \n"
            % (x0, xp0, y0, yp0, 0, p_central_mev / 0.511, 0, pid)
        )
        flnm.close()
        os.system("cat beamsddshead beamsddstail >beam.sdds")

        with open("tracking_dp_ltefilename.ele", "w") as f:
            f.write(_get_tracking_dp_ele_contents())

        os.system(
            "elegant  -macro=npass="
            + str(npass)
            + ",interval=1,dp="
            + str(deltap)
            + " tracking_dp_ltefilename.ele >junktmp"
        )

        import subprocess

        ta, xxp = subprocess.getstatusoutput(
            "sddsprocess ring.w1 -filter=par,Pass,0,"
            + str(npass - 1)
            + ' -pipe=out|sddsprintout -pipe=in -col="(x,xp,y,yp)" -noLabel -noTitle '
        )  # read output of elegant for x,xp after one turn

        # xxp = StringIO(xxp) #retrieve from elegant output file as string, and change the string into a virtual file
        xxp = xxp.split(
            "\n"
        )  # load the virtual file to convert it into an array of x,xp after 1 turn
        xxp = [i.split() for i in xxp]
        xxp = np.array([list(map(eval, i)) for i in xxp])

        ta, tune = subprocess.getstatusoutput(
            'sddsexpand ring.w1 -pipe=out|sddscollapse -pipe=input,output|sddsnaff -pipe=in,out -col=Pass,x,y \
                    -term=frequencies=1 |sddsprintout -pipe=in  -noLabel -notitle -col="(xFrequency,yFrequency)" '
        )
        tune = tune.split()
        nuxt, nuyt = float(tune[-2]), float(tune[-1])

    elif TRACKING_CODE == "Tracy":

        x0off = xp0off = y0off = yp0off = 0.0
        x0 = xmax - x0off
        xp0 = 0.0 - xp0off
        y0 = ymax - y0off
        yp0 = 0.0 - yp0off

        pass_func_list = CACHED_TRACY_PASS_FUNCS
        v_list = [[x0, xp0, y0, yp0, deltap, 0.0]]
        for _ in range(npass):
            v = tracy_track.cell_pass_trace_space(pass_func_list, *v_list[-1])
            v_list.append(v)

        xxp = np.array(v_list)[:-1, :4]

        if False:
            for i in range(xxp.shape[1]):
                plt.figure()
                plt.plot(xxp[:, i], "b.-")
                plt.plot(xxp_ele[:, i], "r.-")

            tbt = pyele_sdds.sdds2dicts("ring.w1")[0]["columns"]
            pass_array = pyele_sdds.sdds2dicts("ring.w1")[0]["params"]["Pass"]
        else:
            tbt = dict(x=xxp[:, 0], y=xxp[:, 2])
            pass_array = np.array(range(xxp.shape[0]))

        temp_tbt_sdds_filepath = "temp.tbt"
        pyele_sdds.dicts2sdds(
            temp_tbt_sdds_filepath,
            columns=dict(Pass=pass_array, x=tbt["x"], y=tbt["y"]),
            outputMode="binary",
            tempdir_path=None,
            suppress_err_msg=True,
        )

        import shlex
        from subprocess import Popen, PIPE

        temp_naff_sdds_filepath = "temp.naff"
        cmd = (
            f"sddsnaff {temp_tbt_sdds_filepath} {temp_naff_sdds_filepath} "
            f"-col=Pass,x,y -term=frequencies=1"
        )
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
        out, err = p.communicate()
        print(out)
        if err:
            print(f"stderr: {err}")
        cmd = (
            f"sddsprintout {temp_naff_sdds_filepath}  -noLabel -notitle "
            f'-col="(xFrequency,yFrequency)"'
        )
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
        out, err = p.communicate()
        # print(out)
        if err:
            print(f"stderr: {err}")
        nuxt, nuyt = [float(_s.strip()) for _s in out.split() if _s.strip()]

        try:
            os.remove(temp_tbt_sdds_filepath)
        except:
            pass
        try:
            os.remove(temp_naff_sdds_filepath)
        except:
            pass

    else:
        raise NotImplementedError(f"TRACKING_CODE = {TRACKING_CODE}")

    xy = xxp.transpose().copy()

    # transform xy as xy relative to fixed point.
    xy[0] = xy[0] - xfix
    xy[1] = xy[1] - xpfix

    print("tune x,y are:", nuxt, nuyt)

    # return np.array([xd,xpd]),omt, np.array([xacoeffnew,xpnew]),omttracking,np.array(xy), om1naff
    return xy, nuxt, nuyt
