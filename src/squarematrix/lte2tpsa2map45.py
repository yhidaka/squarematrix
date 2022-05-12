# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# import squareMatrix_dev as sm
import copy
import os, sys
import time
import pickle

import numpy as np
from numpy import array
import scipy

from . import squarematrixdefinition as sqdf
from . import jnfdefinition as jfdf
from .cytpsa import TPS

dmuytol = 0.005

global Vm, U, maxchainlenposition, bKi, norder, powerindex

# DEBUG = True
DEBUG = False


def modularToPi(y):
    x = copy.deepcopy(y)
    x = x.real + np.pi
    x = x % (2.0 * np.pi)
    x = x - np.pi
    return x


def sv(filename, x):
    ff = open(filename, "wb")
    pickle.dump(x, ff)
    ff.close()


def rl(filename):
    ff = open(filename, "rb")
    xx = pickle.load(ff, encoding="latin1")
    ff.close()
    return xx


def fixpoint(xx, mfmadx, powerindex):
    x, xp = xx
    x1 = sum(
        [
            mfmadx[0][i] * x**nx * xp**nxp
            for i, [nx, nxp, ny, nyp] in enumerate(powerindex)
            if ny + nyp == 0
        ]
    )
    xp1 = sum(
        [
            mfmadx[1][i] * x**nx * xp**nxp
            for i, [nx, nxp, ny, nyp] in enumerate(powerindex)
            if ny + nyp == 0
        ]
    )
    return x1 - x, xp1 - xp


def tpsaOneTurnMap(mfmadx, xfix, xpfix, sqmxparameters, safe_mult=False):
    """"""
    powerindex, sequencenumber, nv, norder = [
        sqmxparameters[i] for i in list(sqmxparameters.keys())
    ]
    var_names = ["x", "xp", "y", "yp"]
    assert len(var_names) == nv
    #
    x = TPS(var_names, norder, safe_mult=safe_mult)
    xp = TPS(var_names, norder, safe_mult=safe_mult)
    y = TPS(var_names, norder, safe_mult=safe_mult)
    yp = TPS(var_names, norder, safe_mult=safe_mult)
    #
    x.set_to_var("x")
    xp.set_to_var("xp")
    y.set_to_var("y")
    yp.set_to_var("yp")
    x0 = x + xfix
    xp0 = xp + xpfix
    y0 = y
    yp0 = yp
    one = x * 0 + 1

    xn = [one, x0]
    xpn = [one, xp0]
    yn = [one, y0]
    ypn = [one, yp0]

    for i in range(2, norder + 1):
        xn.append(xn[-1] * x0)
    for i in range(2, norder + 1):
        xpn.append(xpn[-1] * xp0)
    for i in range(2, norder + 1):
        yn.append(yn[-1] * y0)
    for i in range(2, norder + 1):
        ypn.append(ypn[-1] * yp0)

    u = [xn, xpn, yn, ypn]
    t0 = time.time()
    # Use the table built above, i.e., xn,xnp,.. to construct the square matrix.
    tpsamonomials = []
    for i in range(0, len(powerindex)):
        nz = jfdf.findnonzeros(
            powerindex[i]
        )  # nz is the list of the indexes of the nonzero power terms
        # Now, for this powerindex[i] multiply the power of 4 functions x,xp,y,yp according to their power
        # u[nz[j]] is the relevant term, powerindex[i][nz[j]] is its power
        if len(nz) == 0:
            row = one
        else:
            row = u[nz[0]][powerindex[i][nz[0]]]
            if len(nz) > 1:
                for j in range(1, len(nz)):
                    row = row * u[nz[j]][powerindex[i][nz[j]]]
        tpsamonomials.append(row)
    mftpsa = []
    for row in mfmadx:
        mftpsa.append(np.dot(row, tpsamonomials))
    mftpsa[0] -= xfix
    mftpsa[1] -= xpfix
    return mftpsa


def scalingmf(mf, powerindex):
    # Find a scale s so that s x**m and s is on the same scale if the term in M with maximum absolute value has power m
    # as described in "M scaling" in "Jordan Form Reformulation.one". mf is the first 4 rows of M so the scaling method is the same.
    absM = abs(mf)
    i, j = np.unravel_index(absM.argmax(), absM.shape)
    power = sum(powerindex[j])
    scalex1 = (absM.max()) ** (-(1.0 / (power - 1.0)))
    scalem1 = 1 / scalex1
    mlen = len(powerindex)
    # 	mflen=len(mf)
    if DEBUG:
        print("in scalingmf, scalemf=", scalex1)
    As = np.identity(mlen)
    for i in range(mlen):
        As[i, i] = scalem1 ** sum(powerindex[i])
    Asm = np.identity(mlen)
    for i in range(mlen):
        Asm[i, i] = scalex1 ** sum(powerindex[i])
    mfs = scalem1 * np.dot(mf, Asm)
    return array(mfs), scalex1, As, Asm


def modularToNo2(y, N):
    x = copy.deepcopy(y)
    x = x + N / 2
    x = x % (N)
    x = x - N / 2
    return x


def dpmap(mfmadx, deltap, sqmxparameters):
    if DEBUG:
        tt0 = [["dpmap, 1, start", time.time(), 0]]
        print(tt0)
    powerindex, sequencenumber, nv, norder = [
        sqmxparameters[i] for i in list(sqmxparameters.keys())
    ]
    # 5.1 for a given delatap, from 5 variable tpsa, generate the 4 variable tpsa coefficient table
    if DEBUG:
        print(
            "\n5.1 for a given delatap, generate the 4 variable tpsa coefficient table based on madx output"
        )
    # deltap=0.005#0.002

    if (
        False
    ):  # Commented out by Y. Hidaka on 09/24/2020. This section is now happening in gettpsa() in veq44.py
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
    if DEBUG:
        tt1 = [["dpmap, 2", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    if DEBUG:
        t0.append(
            ["5.1 after given specified deltaP in delta expansion, t=", time.time()]
        )
        print(t0[-1][0], t0[-1][1] - t0[0][1])

    # 5.2 For off-momentum case, with deltap non zero, there should be a point where after one turn the particle remain at the same point,
    #   which is the fixed point, see S.Y's book p.122. Find fixed point xfix,xpfix (dispersion) where for the map x1=f(x0,xp0),xp1=g(x0,xp0) ,
    #   xfix=f(xfix,xpfix),xpfix=g(xfix,xpfix), so for a given deltap, xfix,xpfix is the dispersion.
    if DEBUG:
        print("\n5.2. Find fixed point (dispersion) at the specified deltaP")

    xfix, xpfix = 0, 0
    tmp1 = scipy.optimize.fsolve(
        fixpoint,
        (xfix, xpfix),
        args=(
            mfmadx,
            powerindex,
        ),
        full_output=True,
    )
    xfix, xpfix = tmp1[0]
    if DEBUG:
        tt1 = [["dpmap, 3", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    if DEBUG:
        print("\nxfix,xpfix=", xfix, xpfix)
        print(
            tmp1[-1],
            "ferror=",
            tmp1[1]["fvec"],
            "if not cpnvergent, there is no close orbit.",
        )

    if DEBUG:
        t0.append(["5.2 find xfix, t=", time.time()])
        print(t0[-1][0], t0[-1][1] - t0[0][1])

    # 5.3 Shift origin using TPS variables (not matrix) so that x0=x+xfix,xp0=xp+xpfix
    if DEBUG:
        print(
            "\n5.3 Shift origin using TPS variables (not matrix), to fixed point, so that x0=x+xfix,xp0=xp+xpfix, i.e., x=x0-xfix,xp=xp0-xpfix"
        )
    # After 1 turn map, x1=f(x0,xp0),xp1=g(x0,xp0) here f,g are the tpsa of x0,xp0
    # then for the shifted origin, the new x,xp is x->x1-xfix=f(x+xfix,xp+xpfix)-xfix, xp->xp1-xpfix=g(x+xfix,xp+xpfix)-xpfix
    # Thus x=0,xp=0 is the fixed point, the origin for the shifter variable.
    mftpsa = tpsaOneTurnMap(mfmadx, xfix, xpfix, sqmxparameters, safe_mult=False)
    if DEBUG:
        tt1 = [["dpmap, 4", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    if False:  # Commented out by Y. Hidaka on 09/24/2020. If you want to load
        # TPS from an already-computed TPSA file, do it here.

        # sv('junk',[mftpsa,mfmadx])
        """
        mftpsa,mfmadx=rl('junk')
        tmp1=rl('nsls2sr_supercell_ch77_20150406_1_jf1fastmap.dat')
        mstmp=tmp1[0]
        mf
        """

        tmp = rl("nsls2sr_supercell_ch77_20160525_nslice20_nv4_norder7_delta0.pkl")
        # tmp=rl('nsls2sr_supercell_ch77_20150406_1tpsa_jf1fastmap.pkl')
        mftpsa = [tmp["x"], tmp["px"], tmp["y"], tmp["py"]]  # neglect delta and ct
        """
                tmp1=rl('sqmxnsls2sr_supercell_ch77_20150406_1_deltap0.dat')
                tmp1=rl('nsls2sr_supercell_ch77_20150406_1_jf1fastmap.dat')
                mstmp=tmp1[0]
                """

    if DEBUG:
        t0.append(["5.3 after shifted mftpsa generates, t=", time.time()])
        print(t0[-1][0], t0[-1][1] - t0[0][1])

    # 5.4 Derive first 4 lines of the square matrix mf with shifted origin (dispersion)
    if DEBUG:
        print(
            "\n5.4 Derive first 4 lines of the square matrix mf with shifted origin (dispersion)"
        )

    mf = np.zeros((nv, len(powerindex)))
    for iVar, v in enumerate(mftpsa):
        mf[iVar, :] = np.array([v.get_polynom_coeff(index) for index in powerindex])
    if DEBUG:
        tt1 = [["dpmap, 5", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    # 5.5 Calculate twiss parameters from linear part of the matrix Mx,My
    if DEBUG:
        print("\n5.5 Calculate twiss parameters from linear part of the matrix Mx,My")
        print("linear part of mf matrix showing no coupling even of-momentum:")
        jfdf.prm(mf[:4, 1:5], 4, 4)

    xm, xpm, ym, ypm = mf
    Mx = mf[:2, 1:3]

    tro2x = Mx.trace() / 2
    phix0madx = abs(np.arccos(tro2x)) * np.sign(
        Mx[0, 1]
    )  # the sign of Mx[0,1] determine the sign of phix0madx
    nuxmadx = phix0madx / 2 / np.pi
    betax0madx = Mx[0, 1] / np.sin(
        phix0madx
    )  # only when phix0madx sign is correct we can get betax0>0.
    alphax0madx = (Mx[0, 0] - Mx[1, 1]) / 2.0 / np.sin(phix0madx)
    gammax0madx = (1 + alphax0madx**2) / betax0madx  # -Mx[1,0]/np.sin(phix0madx)
    if DEBUG:
        print(
            "gammax0madx*betax0madx-alphax0madx**2=",
            gammax0madx * betax0madx - alphax0madx**2,
        )
        print(
            "Mx[1,0] ",
            Mx[1, 0],
            " differs from -gammax0madx*np.sin(phix0madx)",
            -gammax0madx * np.sin(phix0madx),
            "\ndue to the error of tpsa:",
        )
        print(
            "modify Mx[1,0] to force them equal so the twiss form is exact for linear part:"
        )
    Mx[1, 0] = -gammax0madx * np.sin(phix0madx)

    if DEBUG:
        print(
            "\ncheck Twiss form Mx, all should be zero:\n",
            np.cos(phix0madx) + alphax0madx * np.sin(phix0madx) - Mx[0, 0],
        )
        print(np.cos(phix0madx) - alphax0madx * np.sin(phix0madx) - Mx[1, 1])
        print(betax0madx * np.sin(phix0madx) - Mx[0, 1])
        print(-gammax0madx * np.sin(phix0madx) - Mx[1, 0])

    My = mf[2:4, 3:5]

    tro2y = My.trace() / 2
    phiy0madx = abs(np.arccos(tro2y)) * np.sign(My[0, 1])
    nuymadx = phiy0madx / 2 / np.pi
    betay0madx = My[0, 1] / np.sin(
        phiy0madx
    )  # only when phix0madx sign is correct we can get betax0>0.
    alphay0madx = (My[0, 0] - My[1, 1]) / 2.0 / np.sin(phiy0madx)
    gammay0madx = (
        1 + alphay0madx**2
    ) / betay0madx  # gammay0madx=-My[1,0]/np.sin(phiy0madx)
    if DEBUG:
        print(
            "gammay0madx*betay0madx-alphay0madx**2=",
            gammay0madx * betay0madx - alphay0madx**2,
        )

    if DEBUG:
        tt1 = [["dpmap, 6", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
    My[1, 0] = -gammay0madx * np.sin(phiy0madx)

    if DEBUG:
        print(
            "\ncheck Twiss form My, all should be zero:\n",
            np.cos(phiy0madx) + alphay0madx * np.sin(phiy0madx) - My[0, 0],
        )
        print(np.cos(phiy0madx) - alphay0madx * np.sin(phiy0madx) - My[1, 1])
        print(betay0madx * np.sin(phiy0madx) - My[0, 1])
        print(-gammay0madx * np.sin(phiy0madx) - My[1, 0])

    sqrtbetax = np.sqrt(betax0madx)
    sqrtbetay = np.sqrt(betay0madx)

    if DEBUG:
        t0.append(["5.5 twiss parameters calculation: t=", time.time()])
        print(t0[-1][0], t0[-1][1] - t0[0][1])

    # 5.6 Construct the BK square matrix using the first 5 rows.
    if DEBUG:
        print("\n5.6 Construct the BK square matrix using the first 5 rows.")
    tol = 1e-12
    bK, bKi = sqdf.BKmatrix(
        betax0madx,
        phix0madx,
        alphax0madx,
        betay0madx,
        phiy0madx,
        alphay0madx,
        0,
        0,
        norder,
        powerindex,
        sequencenumber,
        tol,
    )
    if DEBUG:
        tt1 = [["dpmap, 7", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
        t0.append(["5.6 Construct the BK, t=", time.time()])
        print(t0[-1][0], t0[-1][1] - t0[0][1])

    # 5.7 For resonant block, use tune shift to exact resonance, see "shiftToExactResonance.lyx" in /Users/lihuayu/Dropbox/henonheilesshort/offmnew/theory
    # dmuy=(4*phix0madx+2*phiy0madx-2*np.pi)/2

    nux = phix0madx / 2 / np.pi
    nuy = phiy0madx / 2 / np.pi

    tmp = [
        [
            j,
            abs(modularToNo2((mx - mxs) * nux + (ny - nys) * nuy, 1)),
            modularToNo2((mx - mxs) * nux + (ny - nys) * nuy, 1),
            (mx - mxs) * nux
            + (ny - nys) * nuy
            - modularToNo2((mx - mxs) * nux + (ny - nys) * nuy, 1),
        ]
        for j, (mx, mxs, ny, nys) in enumerate(powerindex)
    ]
    tmp1 = list(zip(*tmp))
    idx0 = jfdf.findzeros(
        tmp1[1], tol
    )  # idx0 are the indixes where mx=mxs,my=mys, so their detune are zero no matter nux,nuy
    idx1 = np.argsort(tmp1[1])  # sort the detune from zero to maximum
    # the first len(idx0) term are zero no matter nux or nuy, so they must be excluded from resonance lines.
    idx = idx1[
        len(idx0)
    ]  # idx is the index in the sorted index after all those indixes which are zeros,i.e., it is the first index which is smallest non-zero
    mx, mxs, my, mys = powerindex[idx]
    k = tmp[idx][3]
    if my - mys == 0:  # if my-mys=0, it is pure x resonance
        dmuy = 0
        dmux = ((mx - mxs) * phix0madx - k * 2 * np.pi) / (mx - mxs)
    else:  # if my-mys!=0, we consider it is y resonance
        dmux = 0
        dmuy = ((mx - mxs) * phix0madx + (my - mys) * phiy0madx - k * 2 * np.pi) / (
            my - mys
        )  # dmuy is the the tune nuy shift from exactly resonance nuy0, i.e. dmuy=2*np.pi*(nuy-nuy0)
    dmuxy = {
        "muxyd": (dmux, dmuy),
        "mx": mx,
        "mxs": mxs,
        "my": my,
        "mys": mys,
        "mux": phix0madx,
        "muy": phiy0madx,
        "k": k,
    }
    if abs(dmuy) > dmuytol:
        dmuy = 0  # if detune is large than tolerance it is consider as off resonance, there no need to shift.
    if abs(dmux) > dmuytol:
        dmux = 0  # if detune is large than tolerance it is consider as off resonance, there no need to shift.
    dmuxy["dmux"] = dmux
    dmuxy["dmuy"] = dmuy

    edmuy = np.identity(4) * (1 + 0j)  # np.exp(1j*dmuy)
    edmuy[0, 0] = np.exp(-1j * dmux)  # zy has additional phase shift dmuy
    edmuy[1, 1] = np.exp(1j * dmux)  # zys has additional phase shift -dmuy
    edmuy[2, 2] = np.exp(-1j * dmuy)  # zy has additional phase shift dmuy
    edmuy[3, 3] = np.exp(1j * dmuy)  # zys has additional phase shift -dmuy

    ln = len(powerindex)
    tmp1 = np.zeros((4, 1)) * (
        1.0 + 0.0j
    )  # prepare to change 4x4 matrix into 330x330 square matrix
    tmp2 = np.zeros((4, ln - 5)) * (1.0 + 0.0j)
    edmuy = np.hstack((tmp1, edmuy, tmp2))
    if DEBUG:
        tt1 = [["dpmap, 8", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    edmuy = sqdf.squarematrix(edmuy, norder, powerindex, sequencenumber, tol)

    # Derive normalized map M=(BK)^(-1).mm.BK, see my notes 'Relation to Normal Form' of Wednesday, March 30, 2011 10:34 PM
    #   here #mfbk is the first 4 rows of M, it is
    if DEBUG:
        print("\n11. Derive normalized map M=(BK)^(-1).mm.BK")

    mforiginal = mf.copy()  # mforiginal corresponds to my original square matrix Mx
    mfbkoriginal = jfdf.d3(
        bKi[1:5, 1:5], mforiginal, bK
    )  # mfbkforiginal corresponds to my original square matrix Mz (the normalized square matrix near resonance)
    mfbk = np.dot(
        mfbkoriginal, edmuy
    )  # mfbk is the square matrix M' exactly on resonance as an useful approximation of Mz

    dphix = np.log(mfbk[0, 1]).imag - np.log(mfbkoriginal[0, 1]).imag
    dphiy = np.log(mfbk[2, 3]).imag - np.log(mfbkoriginal[2, 3]).imag
    dmuxy["dphix"] = dphix
    dmuxy["dphiy"] = dphiy
    phixoriginal = np.log(
        mfbkoriginal[0, 1]
    ).imag  # phix0 is detuned phix0madx so that phix0 is exactly on resonance 4nux+2nuy=1
    phiyoriginal = np.log(
        mfbkoriginal[2, 3]
    ).imag  # phiy0 is detuned phiy0madx so that phiy0 is exactly on resonance 4nux+2nuy=1
    nuxoriginal = np.log(mfbkoriginal[0, 1]).imag / 2 / np.pi
    nuyoriginal = np.log(mfbkoriginal[2, 3]).imag / 2 / np.pi
    phix0 = np.log(
        mfbk[0, 1]
    ).imag  # phix0 is detuned phix0madx so that phix0 is exactly on resonance 4nux+2nuy=1
    phiy0 = np.log(
        mfbk[2, 3]
    ).imag  # phiy0 is detuned phiy0madx so that phiy0 is exactly on resonance 4nux+2nuy=1
    nux0 = phix0 / 2 / np.pi
    nuy0 = phiy0 / 2 / np.pi

    if DEBUG:
        print(
            "nuxoriginal-nux,nuyoriginal-nuy,4*nux+2*nuy-1=",
            nuxoriginal - nux,
            "\t",
            nuyoriginal - nuy,
            "\t",
            4 * nux + 2 * nuy - 1,
        )
        print(
            "nux0-nux,nuy0-nuy,4*nux0+2*nuy0-1=",
            nux0 - nux,
            "\t",
            nuy0 - nuy,
            "\t",
            4 * nux0 + 2 * nuy0 - 1,
        )
        print(
            "dphiy,dmuy=",
            dphiy,
            dmuy,
            "dphix,dmux=",
            dphix,
            dmux,
            " nux0=",
            nux0,
            " nuy0=",
            nuy0,
        )

    # 5.8 Scale the one turn map in z,z* space
    if DEBUG:
        print("\n5.8 Scale the one turn map in z,z* space")
        tt1 = [["dpmap, 9", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    mftmp1 = array(mfbk).copy()
    mfbk, scalemf, As, Asm = scalingmf(mfbk, powerindex)
    # print("mfbk[2,3]-np.exp(1j*phiy0madx)=",mfbk[2,3]-np.exp(1j*phiy0madx))

    # 5.9 Construct scaled square matrix Ms
    if DEBUG:
        print("\n5.9 Construct square matrix Ms")
    # Ms =As.M.Asm is the scaled M
    Ms = sqdf.squarematrix(mfbk, norder, powerindex, sequencenumber, tol)

    if DEBUG:
        t0.append(["5.9 Construct scaled square matrix Ms, t= ", time.time()])
        print(t0[-1][0], t0[-1][1] - t0[0][1])

    if DEBUG:
        print("test deteminantes of linear part of square matrix Ms:")
        print("det(Ms[1:3,1:3])=", np.linalg.det(Ms[1:3, 1:3]))
        print("det(Ms[3:5,3:5])=", np.linalg.det(Ms[3:5, 3:5]))

    if DEBUG:
        print(
            "\n5.10 First, force lower subdiagonal elements to zeros, to be sure Ms is an upper triangular matrix"
        )
    for j in range(len(Ms)):
        for i in range(0, len(Ms)):
            if j < i:
                Ms[i, j] = 0
    tbl, msf = [], []
    if DEBUG:
        tt1 = [["dpmap, 10", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
    return (
        Ms,
        phix0,
        phiy0,
        powerindex,
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
    )


if DEBUG:
    t0 = [["start, t=", time.time()]]

    print(t0[-1][0], t0[-1][1] - t0[0][1])


def lte2madx2fort182tpsa(fn, usecode, nv=4, norder=7):
    # fn='junk'#'20140204_bare_1supcell'#'nsls2sr_supercell_ch77_20150406_1'
    # fn='nsls2sr_supercell_ch77_20150406_1'
    # fn='badfchainloop_inf_loop_20191205_0'#'badfchainloop_inf_loop_20191125_0'#'badinfiniteloop'#'cb2NSLS2CB65pm_cb0_1cell'#'badglare'#'cb2NSLS2CB65pm_cb0_1cell'
    t0.append(["1. start lte2madx2fort182tpsa t=", time.time()])
    print("dt=", t0[-1][0], t0[-1][1] - t0[0][1])

    print("fn=", fn)
    if usecode["tpsacode"] == "madx":
        # 1. From lte generate madx file by lte2madx.py and generate madx output file fort.18'
        print(
            "\n1. From lte generate madx file by lte2madx.py and generate madx output file fort.18"
        )
        # os.system('python lte2madx.py '+fn) #if the format of lte file is similar to that of cb2 lattice, use ltecb2madx_case5.py orltecb2madx_case4.py
        # os.system('python ltecb2madx_case5.py '+fn) #This is specially for cb2 lattice lte file format. case 5 is with exoansion of deltap

        if "cb2" in fn:
            execution = os.system(
                "python ltecb2madx_case5.py  " + fn
            )  # This is specially for cb2 lattice lte file format. case 5 is with exoansion of deltap
        else:
            sedstr = (
                'sed s/"no=7"/"no='
                + str(norder)
                + '"/ madx_case5_tail|cat   >madx_case5_tail1'
            )
            os.system(sedstr)

            execution = os.system(
                "python lte2madx_case5.py  " + fn
            )  # This is specially for cb2 lattice lte file format. case 5 is with exoansion of deltap
        # execution=os.system('python ltensls2sr_supercell_ch77_20150406_1madx_case5.py '+fn)
        if execution != 0:
            print("#1. step failed to transform lte file into madx file. exit.")
            sys.exit(0)

        t0.append(["1. " + fn + ".madx generated,  t=", time.time()])
        print("dt=", t0[-1][0], t0[-1][1] - t0[0][1])

        # 2. Run madx to get output fort.18
        print("\n2. Run madx to get output fort.18")
        print("fn for madx is:", fn)
        execution = os.system(
            "./madx " + fn + ".madx"
        )  # this generate madx file fort.18 to be used here
        if execution != 0:
            print("#1. step failed to 2. Run madx to get output fort.18. exit.")
            sys.exit(0)

        print("1.1 madx execution=", execution)

    t0.append(["2. ./madx, generated fort.18, t=", time.time()])
    print("dt=", t0[-1][0], t0[-1][1] - t0[0][1])

    # 3. Generate powerindex
    print("\n3. Generate powerindex and sequencenumber")
    # sequencenumber[i1,i2,i3,i4] gives the sequence number in power index for power of x^i1*xp^i2*y^i3*yp^i4
    sequencenumber = np.zeros((norder + 1, norder + 1, norder + 1, norder + 1), "i")
    powerindex = sqdf.powerindex4(norder)
    powerindex = array(powerindex, "i")
    mlen = len(powerindex)

    for i in range(mlen):
        ip = powerindex[i]
        sequencenumber[ip[0]][ip[1]][ip[2]][ip[3]] = i

    # 4. Generate 5-variable tpsa table mfm
    print("\n4. Generate 5-variable tpsa table mfm")
    # 4.1 read fort.18 the output from madx
    print("\n4.1 read fort.18 the output from madx")
    fortnm = "fort.18"
    fid = open(fortnm, "r")
    a0 = fid.readlines()
    # 4.2 separate tpsa of x1,xp1,y1,yp1, each of them as polynomila function of x0,xp0,y0,yp0 locate their position as sectionhead
    sectionhead = [[j, i] for j, i in enumerate(a0) if "etall" and "INA" in i]
    sectionhead = [
        i[0] for i in sectionhead
    ]  # example sectionhead=[1, 839, 1677, 2435, 3193, 3201], so from line 1 to 838 are for x1,etc.

    # tbl is the string file in fort.18 divided as 4 sections for coefficients of tpsa polynomial of one turn x1,xp1,y1,yp1
    tbl = (
        []
    )  # Each section of tbl is for x1,xp1,y1,yp1, the one turn coordinates. for example, section 1 has all lines between 1 to 838 for x1.
    for i in range(4):
        tbl.append(a0[sectionhead[i] : sectionhead[i + 1]])

    # 4.3 transform table string into numbers of 8 columns as follows:
    # I           COEFFICIENT               ORDER                      EXPONENTS
    # term number, coefficient of the term,  power order,   x power, xp power,y power yp power, deltap power
    # 1              2                        3            4,       5,       6,      7,         8
    # so mfm is 4 rows of tpsa coefficients
    mfm = []
    for a in tbl:
        mfm.append(
            np.array(
                [
                    list(map(eval, i.split()))
                    for j, i in enumerate(a)
                    if len(i.split()) == 8
                ]
            )
        )

    t0.append(["4.3 ,after extraction from fort.18, t=", time.time()])
    print(t0[-1][0], t0[-1][1] - t0[0][1])

    """
        tmp1=rl('sqmxnsls2sr_supercell_ch77_20150406_1_deltap0.dat')
        mstmp=tmp1[0]
        mfmadx1=[mstmp[1],mstmp[2],mstmp[3],mstmp[4]]
        """

    return mfm, powerindex, sequencenumber


"""
print ('\n5. Generate 4-variable tpsa table for fixed deltap')


def gettpsa(ltefilename,nv=4,norder=7):
        mfm,powerindex,sequencenumber=lte2madx2fort182tpsa(ltefilename,nv=nv,norder=norder)
        sqmxparameters={'powerindex':powerindex,'sequencenumber':sequencenumber,'nv':nv,'norder':norder}
        return mfm,sqmxparameters

fn='junk'#'20140204_bare_1supcell'#'nsls2sr_supercell_ch77_20150406_1'

nv,norder = 4,7
mfm,sqmxparameters=gettpsa(fn,nv=nv,norder=norder)


tol=1e-12
deltap=-0.02
Ms,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf,deltap,xfix,xpfix,dphiy=dpmap(mfm,deltap,sqmxparameters)

sv(fn+"M.dat",[Ms,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf,deltap,xfix,xpfix,dphiy])



deltapstep=0.01
deltaplist=np.arange(-0.02,-0.019,deltapstep)#(0.006,0.007,deltapstep)#(-0.01,0.009,deltapstep)#(-0.01,0.009,0.002)
for deltap in deltaplist:
        #5. Generate 4-variable tpsa table for fixed deltap
        Ms,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf,deltap,xfix,xpfix=dpmap(mfm,deltap)
        #6. Jordan block
        ux,uy,Jx,Jy,scalex=jordanblock(Ms,phix0,phiy0,powerindex,scalemf,sequencenumber)

        Vm=W2Z(ux,uy,Jx,Jy,norder,powerindex,sequencenumber,tol)

        #7. scanxy
        binxydeltap.append([deltap,xfix,scanxy(y_scan_array, x_scan_max, xstep,Vm)])
        t0.append(['7. scanxy, t=',time.time()])
        print ('\n'+t0[-1][0], t0[-1][1]-t0[0][1])





plt.show()
sys.exit(0)
"""
