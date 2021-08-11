# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import sys
import time
import pickle
import copy
import cmath

# import commands
import numpy as np
from numpy import array
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy
import pdb

t0 = time.perf_counter()
timing = [["in iterModule, start", time.perf_counter()]]
# import jnfdefinition
# jfdf = jnfdefinition
# import squarematrixdefinition
# sqdf = squarematrixdefinition

from . import yunaff
from . import (
    veq59 as veqm,
)  # copied from veq52.py. veqnsls2sr_supercell_ch77_20150406_1_deltap0_lt11 as veq#veq20140204_bare_1supcell_deltapm02 as veq
from .fortran import zcolnew, mysum
from .fortran import lineareq as leq

global scalex, tol


tol = 1e-12

print(time.perf_counter() - t0, "in iterModules52, seconds for import sqdf and jfdf")
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


def Zszaray(xy, Zpar):  # Construct ZsZsp columns from each turn in the tracking result.
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], xy) / scalex
    xxpyyp = [zx, zxs, zy, zys]
    xxpyyp = list(zip(*xxpyyp))
    Zs = zcolnew.zcolarray(xxpyyp, norder, powerindex)
    #
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


nsostep = 1


def findso(veq, x0, xp0, y0, yp0, v1s0, v2s0, v1s, v2s, acoeff, v0norm, n):
    v1qr = np.linspace(v1s.real, v1s0.real, num=n)[
        ::-1
    ]  # The [::-1] is added to avoid missing a step if n=1
    v1qi = np.linspace(v1s.imag, v1s0.imag, num=n)[::-1]
    v1q = v1qr + 1j * v1qi
    v2qr = np.linspace(v2s.real, v2s0.real, num=n)[::-1]
    v2qi = np.linspace(v2s.imag, v2s0.imag, num=n)[::-1]
    v2q = v2qr + 1j * v2qi
    xyq = [[[x0, xp0, y0, yp0], []]]
    for i in range(n):
        xt, xpt, yt, ypt = xyq[-1][0]
        tmp1 = scipy.optimize.fsolve(
            veq,
            [xt, xpt, yt, ypt],
            args=[acoeff, v1q[i], v2q[i], v0norm],
            xtol=1e-10,
            factor=0.5,
            full_output=True,
        )  # Find the z value for new v1(z),v2(z)
        xyq.append([tmp1[0], tmp1])
    return xyq[1:]


def bnmphiCauchyCutoffWithIntrokam(
    fluc, n_theta, alphadiv, Cauchylimit, applyCauchylimit, dt,
):  # Calculate period T from given phi(t_k), where t_k is only on approximate period T1 on uniformly divided points
    # as described by Section 3B of "compareIntrokamwithme.lyx".
    # x0,xp0,x1,xp1,phi1,av10,thetav10,v10,v11=np.array(fluc)
    tt0 = [["in Cauchycut       1, start", time.time(), 0]]
    print("tt0=", tt0)
    phi1old, phi2old, th10, th20 = np.array(fluc)
    tt1 = [["in Cauchycut      , 1.2 ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    om1, om2 = mysum.sum(phi1old, phi2old)
    om1, om2 = om1.real / n_theta ** 2, om2.real / n_theta ** 2
    """
    om1 = sum(phi1old.real) / n_theta ** 2
    om2 = sum(phi2old.real) / n_theta ** 2
    """
    tt1 = [["in Cauchycut      , 1.3 ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    phi1old = phi1old.reshape([n_theta, n_theta])
    phi2old = phi2old.reshape([n_theta, n_theta])
    tt1 = [["in Cauchycut      , 2. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    phi1 = (
        phi1old - om1
    )  # phi here is Delta phi of eq.35 of /Users/lihuayu/Desktop/nonlineardynamics/henonheiles/sqmxnsls/latticenonlinerresonance.lyx
    phi2 = phi2old - om2

    th10 = th10.real % (2 * np.pi)  # that is th10 is redefined as mod(th10,2*np.pi)
    th20 = th20.real % (2 * np.pi)
    aphi10 = (
        sum(phi1) / n_theta ** 2
    )  # see Eq.4 of "compareIntrokamwithme.lyx" but this line seems is obsolete (nowhere else uses aphi10)
    aphi20 = (
        sum(phi2) / n_theta ** 2
    )  # see Eq.4 of "compareIntrokamwithme.lyx" /Users/lihuayu/Desktop/nonlineardynamics/henonheiles/latticesqmx/compareIntrokamwithme.lyx
    nux = om1 * dt / 2 / np.pi
    nuy = om2 * dt / 2 / np.pi
    tt1 = [["in Cauchycut      , 3. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)

    aphi1 = (
        np.fft.fft2(phi1) / n_theta ** 2
    )  # aphi[n,m] is the Fourier expansion coefficent of exp(1j*(n*(theta1-th10)+m*(theta2-th20))), ie. its argument is theta1-th10, not theta1!
    aphi2 = (
        np.fft.fft2(phi2) / n_theta ** 2
    )  # This two lines shows the phi1, phi2 here are on unit circle, and aphia,aphi2 are the result of one turn advance with fluctuation.

    tt1 = [["in Cauchycut,      , 4. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)

    if applyCauchylimit:
        a1, b1, a2, b2 = Cauchylimit["Cauchylim"]
        # idxm=min(42,n_theta)
        linefit1 = [-a1 * n + b1 for n in range(alphadiv // 2)]
        linefit2 = [a1 * n + b1 for n in range(-alphadiv // 2, 0)]
        for n in range(-alphadiv // 2, alphadiv // 2):
            for k in range(alphadiv // 2):
                if np.log(abs(aphi1[k, n])) > -a1 * k + b1 + 4:
                    aphi1[k, n] = 0
                if np.log(abs(aphi2[k, n])) > -a1 * k + b1 + 4:
                    aphi2[k, n] = 0
            for k in range(-n_theta // 2, 0):
                if np.log(abs(aphi1[k, n])) > a2 * k + b2 + 4:
                    aphi1[k, n] = 0
                if np.log(abs(aphi2[k, n])) > a2 * k + b2 + 4:
                    aphi2[k, n] = 0
        for k in range(-alphadiv // 2, alphadiv // 2):
            for n in range(alphadiv // 2):
                if np.log(abs(aphi1[k, n])) > -a1 * n + b1 + 4:
                    aphi1[k, n] = 0
                if np.log(abs(aphi2[k, n])) > -a1 * n + b1 + 4:
                    aphi2[k, n] = 0
            for n in range(-n_theta // 2, 0):
                if np.log(abs(aphi1[k, n])) > a2 * n + b2 + 4:
                    aphi1[k, n] = 0
                if np.log(abs(aphi2[k, n])) > a2 * n + b2 + 4:
                    aphi2[k, n] = 0

    #        if np.log(abs(aphi1[k]))> -a1*k+b1 or np.log(abs(aphi1[k]))>a2*k+b2: aphi1[k]=0
    # aphi1tmp=np.fft.fft(phi1[:-1])/n_theta #aphi[n,m] is the Fourier expansion coefficent of exp(1j*(n*theta1+m*theta2))
    # phi1mean=aphi1[0,0]
    # phi2mean=aphi2[0,0]
    tt1 = [["in Cauchycut      , 5. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    """
        2.Use fft of derivatives phi to calculate fft of the first order perturbed trajectory, ie. bnm. see eq.1.14 of "jfn-resonance5.pdf"

        print ("\n16.2.Use fft of derivatives phi to calculate fft of the first order perturbed trajectory, ie. bnm. see eq.1.14 of 'jfn-resonance5.pdf'")
    """
    """
    bnm1, bnm2 = (
        np.zeros([n_theta, n_theta]) * (1 + 0j),
        np.zeros([n_theta, n_theta]) * (1 + 0j),
    )
    """
    tt1 = [["in Cauchycut      , 6. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    cutoff = Cauchylimit["aliasingCutoff"]
    if cutoff > n_theta:
        cutoff = n_theta
    iCut = cutoff // 2
    rows = cols = np.arange(-iCut, iCut)
    inds = np.ix_(rows, cols)
    div_fac = np.exp(1j * (inds[0] * om1 + inds[1] * om2)) - 1
    bnm1cut = aphi1[inds] / div_fac
    bnm2cut = aphi2[inds] / div_fac
    bnm1 = np.zeros_like(aphi1)
    bnm2 = np.zeros_like(aphi2)
    bnm1[inds] = bnm1cut
    bnm2[inds] = bnm2cut
    bnm1[0, 0] = 0j
    bnm2[0, 0] = 0j
    """
    for i in range(-cutoff // 2, cutoff // 2):
        for j in range(
            -cutoff // 2, cutoff // 2
        ):  # Notice that fft output allows negative i,j with bnm[i]=bnm[i+n_theta] if i<0
            if (
                abs(i) + abs(j) != 0
            ):  # exclude DC term i,j=[0,0],there is indeterminant number in bnm[0,0] because it is devided by zero
                bnm1[i, j] = aphi1[i, j] / (
                    np.exp(1j * (i * om1 + j * om2)) - 1
                )  # *np.exp(1j*(nd[i,j]*th10+md[i,j]*th20))
                bnm2[i, j] = aphi2[i, j] / (
                    np.exp(1j * (i * om1 + j * om2)) - 1
                )  # aphi1 is Fourier expansion by (theta-th10)^m
                # bnm1[i,j]=aphi1[i,j]*np.exp(-1j*(i*th10+j*th20))/(np.exp(1j*(i*om1+j*om2))-1) #*np.exp(1j*(nd[i,j]*th10+md[i,j]*th20))
                # bnm2[i,j]=aphi2[i,j]*np.exp(-1j*(i*th10+j*th20))/(np.exp(1j*(i*om1+j*om2))-1) #aphi1 is Fourier expansion by (theta-th10)^m
                # so in order bnl1[m] to represents theta_tilde in in Eq.38 in paper "1DlatticenonlinearResonances.lyx"
                # do not we need this factor np.exp(-1j*i*th10). Based on henonmap study
    """
    tt1 = [["in Cauchycut      , 7. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    bnm1rsh = bnm1.reshape([n_theta ** 2])
    bnm2rsh = bnm2.reshape([n_theta ** 2])
    bnm1rsh, bnm2rsh = mysum.sum(bnm1rsh, bnm2rsh)
    bnm1[0, 0], bnm2[0, 0] = -bnm1rsh, -bnm2rsh

    """
    bnm1[0, 0] = -sum(bnm1.reshape([n_theta ** 2]))  # the
    bnm2[0, 0] = -sum(bnm2.reshape([n_theta ** 2]))  # the
    """
    # it is checked that sum(bnm1)~1e-17, aphi1[0].real~1e-17 as required by Eq.4 of "compareIntrokamwithme.lyx"
    tt1 = [["in Cauchycut      , 8. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    tt1 = [["in Cauchycut      , 8. ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    return aphi1, aphi2, om1, om2, nux, nuy, bnm1, bnm2, th10, th20


def vthetascan_use_x0dlast3(
    acoeff, veq, vscaninput, wwJwinv,
):  # the module "vthetascan previously" is the module "fflucaccurate", see eq.8.1 of "jnf-resonance3.pdf", section 8.
    # For a given initial phi1,phi2, calculate its a,b and initial v10,v20, then scan theta02  to obtain v1,v2 at various rotated actions

    for extract_vscaninput in [1]:
        w0mtx, winvmtx, norder_winv, powerindex_winv = wwJwinv
        (
            acoeff,
            Vmtx,
            dVdXmtx,
            xydlast,
            uarray,
            vna,
            v120i,
            n_theta,
            xi,
            usexp0dlast,
            v0norm,
            bKi,
            scalex,
            norder,
            powerindex,
            Zpar3,
            findsoparamters,
        ) = [vscaninput[i] for i in vscaninput.keys()]
        findsoparamterswwinv = findsoparamters.copy()
        findsoparamterswwinv["w"] = w0mtx
        findsoparamterswwinv["winv"] = winvmtx

    for initial_v120 in [1]:
        v10i, v20i = v120i  # the initial value of v10,v20
        v1s, v2s = v10i, v20i  #
        tmp1 = scipy.optimize.fsolve(
            veq, xi, args=[acoeff, v1s, v2s, v0norm, uarray, Zpar3], full_output=True
        )  # xi is initial x shifted so it is relative to xfix as origin

        # x1,xp1,y1,yp1=tmp1[0]
        # nfev=tmp1[1]['nfev']
        conversionmsg = tmp1[-1]  # [-1][-1]

        if (
            usexp0dlast == 1
        ):  # then use previous solution orbit xyd as initial trial value.
            xyrecordi = dict(
                theta01=0,
                theta02=0,
                vtarget=[v1s, v2s],
                xtrial=xydlast[:, 0],
                conversionmsg=conversionmsg,
            )
            # xydrecord=[[0,0,xyd0last[:,0],[v1s,v2s],tmp1,conversionmsg]]
        else:  # else every new theta value use the xy value from previous xy value from previous theta
            xyrecordi = dict(
                theta01=0,
                theta02=0,
                vtarget=[v1s, v2s],
                xsol=xi,
                conversionmsg=conversionmsg,
            )
            # xydrecord=[[0,0,xi,[v1s,v2s],tmp1,conversionmsg]] #initial shifted x has theta1,theta2= 0,0,
            # tmp=[[xy[:,0],nfev,v1s,tmp1]]
            # theta1last=np.arange(0,2.0*np.pi+2*np.pi/n_theta,2*np.pi/n_theta)

        # scan the phase advances (theta01, theta02) from initial phi1,phi2
        xydrecord = []
        xydrecord.append(xyrecordi)

    # define the target action-angle variable1 v1s,v2s as a perfect circle
    v1s, v2s = np.array(
        [
            [v10i * np.exp(1j * theta01), v20i * np.exp(1j * theta02)]
            for theta01 in np.arange(0, 2.0 * np.pi, 2 * np.pi / n_theta)
            for theta02 in np.arange(0, 2.0 * np.pi, 2 * np.pi / n_theta)
        ]
    ).transpose()

    for first_approximate_solution_xyd0 in [1]:
        # use inverse w, i.e., winvmtx to find first approximate trajectory xyd0 from the target v1s,v2s
        xyd0 = veqm.vinv_matrix(v1s, v2s, v0norm, winvmtx, norder_winv, powerindex_winv)
        bKi, scalex, norder_jordan, powerindex3 = Zpar3
        Zs3 = zcolnew.zcolarray(xyd0, norder_jordan, powerindex3).transpose()
        # from first appximate trajectory xyd0, calculate first appximate action v1s0,v2s0 and its error fvec
        v1mtx, v2mtx = Vmtx
        v1s0 = np.dot(v1mtx, Zs3)  # Notice wx,wy here explicitly are normalized here.
        v2s0 = np.dot(v2mtx, Zs3)
        dv1, dv2 = v1s0 - v1s, v2s0 - v2s
        fvec = np.array([dv1.real, dv1.imag, dv2.real, dv2.imag]).transpose()

        tmp41 = [np.max(i) for i in fvec]
        xy0 = xyd0.transpose()

    outfindso = findsoblock(
        xy0,
        v1s0,
        v2s0,
        v1s,
        v2s,
        Vmtx,
        veqm.dVdXmatrix,  # dVdX,  # veqm.dVdXmatrix
        dVdXmtx,
        Zpar3,
        findsoparamters,
    )

    for extract_solution_xyd0_record in [1]:
        xyd = outfindso["xsol"]
        tmp4 = outfindso["warnflag"]
        tmp41 = outfindso["fval"]
        tmp42 = outfindso["nfev"]
        nfindso = outfindso["nfindso"]
        fvec = outfindso["fvec"]
        vthetascanmsg = {
            "warnflag": tmp4,
            "maxerror": tmp41,  # max(tmp41),
            "fsolveused": "homeMadefsolveBlock",
            "numberdivgence": len(tmp4),
            "nitermean": np.mean(tmp42),
            "nfindsomean": np.mean(nfindso),
            "fvec": fvec,
            "nfev": tmp42,
        }
        xydrecord = {
            "vtarget": [v1s, v2s],
            "xtrial": xy0,
            "vtrial": [v1s0, v2s0],
            "xsol": xyd,
        }
        xyd0 = xyd
        vthetascanmsg["xydrecord"] = xydrecord

    return xyd0, vthetascanmsg


def matrixsort(
    x, ndiv=40
):  # calculate the sorted index for a 1600 1D-array before it is converted to 40x40 matrix
    tmp1 = np.argsort(x)

    def indx(n):
        return np.unravel_index(n, (ndiv, ndiv))

    tmp3 = list(map(indx, tmp1))
    return tmp3


def findphipeaks(bnm1, bnm2, n_theta, nux, nuy, th10, th20, v120i, npeaks=20):
    v10i, v20i = v120i
    bnm2s = bnm2.reshape([n_theta ** 2])
    tmp3 = abs(bnm2s)
    tmp4 = matrixsort(
        tmp3, n_theta
    )  # grid index of sorted phi2 2D fft sequence tmp3 is frm of n_theta**2x1, while tmp4 is of form n_theta**2x2
    tmp5 = list(zip(*tmp4))  # the indexes of top 20 spectral peaks
    tmp6 = np.argsort(tmp3)
    peakvaluesint2 = array(bnm2s)[
        tmp6
    ]  # the value of top 20 peaks in the phi2 2D fft spectrum
    peaksposition0int2 = modularToNo2(array(tmp5[0]), n_theta)
    peaksposition1int2 = modularToNo2(array(tmp5[1]), n_theta)
    # sumpeak2=sum(abs(peakvaluesint2)**2)

    bnm1s = bnm1.reshape([n_theta ** 2])
    tmp3 = abs(bnm1s)
    tmp4 = matrixsort(
        tmp3, n_theta
    )  # grid index of sorted phi2 2D fft sequence tmp3 is frm of n_theta**2x1, while tmp4 is of form n_theta**2x2
    tmp5 = list(zip(*tmp4))  # the indexes of top 20 spectral peaks
    tmp6 = np.argsort(tmp3)
    peakvaluesint1 = array(bnm1s)[
        tmp6
    ]  # the value of top 20 peaks in the phi2 2D fft spectrum
    peaksposition0int1 = modularToNo2(array(tmp5[0]), n_theta)
    peaksposition1int1 = modularToNo2(array(tmp5[1]), n_theta)
    # sumpeak1=sum(abs(peakvaluesint1)**2)

    tmp1 = list(
        zip(
            *[
                (1 + peaksposition0int1) * nux + peaksposition1int1 * nuy,
                peakvaluesint1,
                peaksposition0int1,
                peaksposition1int1,
            ]
        )
    )
    tmp2 = list(
        zip(
            *[
                peaksposition0int2 * nux + (1 + peaksposition1int2) * nuy,
                peakvaluesint2,
                peaksposition0int2,
                peaksposition1int2,
            ]
        )
    )
    fft2Dphi1peaks = array(
        tmp1[-npeaks:]
    )  # Notice: the main peak at 0,0 is excluded in this list it is about ~1.
    fft2Dphi2peaks = array(
        tmp2[-npeaks:]
    )  # Notice: the main peak at 0,0 is excluded in this list it is about ~1.
    print("\n16.2a fft2Dphi1peaks[-5:]=")
    for i in fft2Dphi1peaks[-5:]:
        print(modularToNo2(i[0].real, 1), abs(i[1]), int(i[2]), int(i[3]))
    print("\n16.2b fft2Dphi2peaks[-5:]=")
    for i in fft2Dphi2peaks[-5:]:
        print(modularToNo2(i[0].real, 1), abs(i[1]), int(i[2]), int(i[3]))
    fft2Dphi1peaks = fft2Dphi1peaks.tolist()
    fft2Dphi2peaks = fft2Dphi2peaks.tolist()
    fft2Dphi1peaks = [
        [i[0], i[1] - 1j + bnm1[0, 0], i[2], i[3]] if i[2] == 0 and i[3] == 0 else i
        for i in fft2Dphi1peaks
    ]
    fft2Dphi2peaks = [
        [i[0], i[1] - 1j + bnm2[0, 0], i[2], i[3]] if i[2] == 0 and i[3] == 0 else i
        for i in fft2Dphi2peaks
    ]
    # fft2Dphi1peaks.append([nux, -1j+bnm1[0,0],0,0])
    # fft2Dphi2peaks.append([nuy, -1j+bnm2[0,0],0,0])
    fft2Dphi1peaks = array(fft2Dphi1peaks)
    fft2Dphi2peaks = array(fft2Dphi2peaks)
    return fft2Dphi1peaks, fft2Dphi2peaks


def inversefftnew2(bnm1, bnm2, n_theta, v120i):
    tt0 = [["in inversefft 1, start", time.time(), 0]]
    print("tt0=", tt0)
    v10i, v20i = v120i
    vnm1, vnm2 = bnm1.copy(), bnm2.copy()
    # based on comparison of the sign in the exponents in the definition of fft and eq.1.15
    # the vnm1,vnm2 obtained so far shoulde actually have the sign n,m reversed so
    # it corresponds to change the signs of theta1 and theta2 in indixes of v1n,v2n
    # which are i and j:
    v1nn, v2nn = np.fft.fft2(vnm1), np.fft.fft2(vnm2)
    tt1 = [["in inversefft, 2. ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    """
    # np.flipud(np.fliplr(A))
    v1n, v2n = v1nn.copy(), v2nn.copy()
    for i in range(-n_theta // 2, n_theta // 2):
        for j in range(-n_theta // 2, n_theta // 2):
            v1n[i, j] = v1nn[
                -i, -j
            ]  # Compare eq.(1.15) of "jfn-resonance5.pdf" with definition of fft shows the sign of theta is reversed.
            v2n[i, j] = v2nn[-i, -j]
    # thetav1=np.arange(0,2*np.pi,2*np.pi/n_theta)
    # negtiveindex is to change every index to its reverse sign, except -n_thetae//2 because it is already equal to n_theta//2
    # due to periodicity of v1nn,v2nn, so for n_thetae=40
    # negativeindex=[0,-1-2,-3,...-19,-20,19,18,..0,], since v1nn is mod 20, so v1nn[21]=v1nn[-19]
    negativeindex = (
        np.mod(array(range(-n_theta // 2, -n_theta - n_theta // 2, -1)), n_theta)
        - n_theta // 2
    )
    # the following tbl can be used to test this negativeindex:
    # tbl=[  [i,negativeindex[i]] for i in range(-n_thetae//2,n_thetae//2)]

    v1n = v1nn[negativeindex, :]
    v1n = v1n[:, negativeindex]
    v2n = v2nn[negativeindex, :]
    v2n = v2n[:, negativeindex]
    """

    v1n = v1nn.copy()
    v1n[:, 1:] = np.fliplr(v1nn[:, 1:])
    v1n[1:, :] = np.flipud(v1n[1:, :])
    v2n = v2nn.copy()
    v2n[:, 1:] = np.fliplr(v2nn[:, 1:])
    v2n[1:, :] = np.flipud(v2n[1:, :])
    tt1 = [["in inversefft, 3. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    """
    tmp = []
    for theta01 in np.arange(0, 2.0 * np.pi, 2 * np.pi / n_theta):
        for theta02 in np.arange(0, 2.0 * np.pi, 2 * np.pi / n_theta):
            tmp.append([theta01, theta02])
    thetav1, thetav2 = np.array(tmp).transpose()
    thetav1 = thetav1.reshape([n_theta, n_theta])
    thetav2 = thetav2.reshape([n_theta, n_theta])
    tt1 = [["in inversefft, 4. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    v1n = (
        v10i * np.exp(1j * thetav1) * np.exp(1j * v1n)
    )  # Notice v1n,v2n here are v_1(t),v_2(t) in eq.1.15, they are the first order perturbed v1,v2, v10 is |v10i|, not v10 itself.
    v2n = (
        v20i * np.exp(1j * thetav2) * np.exp(1j * v2n)
    )  # Notice v1n,v2n here are v_1(t),v_2(t) in eq.1.15, they are the first order perturbed v1,v2, v10 is |v10i|, not v10 itself.

    hl = np.arange(0, n_theta) * 2 * np.pi / n_theta
    vl = np.arange(0, n_theta) * 2 * np.pi / n_theta
    hh, vv = np.meshgrid(hl, vl)
    v1n = v10i * np.exp(1j * vv) * np.exp(1j * v1n)
    v2n = v20i * np.exp(1j * hh) * np.exp(1j * v2n)
    """
    hl = np.arange(0, n_theta) * 2 * np.pi / n_theta
    hh, vv = np.meshgrid(hl, hl)
    v1n = v10i * np.exp(1j * (vv + v1n))
    v2n = v20i * np.exp(1j * (hh + v2n))
    tt1 = [["in inversefft, 5. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    tt1 = [["in inversefft, 6. total ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    return v1n, v2n


def scalefun(
    Zs, Vmtx, v1s, v2s, fscale,
):  # Equation for z:  v1(z)==v1,v2(z)==v2 , here v1,v2 are fuctions v1x,v3x defined in section 19.2
    tt0 = [["scalefun, 1, start", time.time(), 0]]
    print(tt0)
    v1mtx, v2mtx = Vmtx
    v1mv1 = np.dot(v1mtx, Zs.transpose()) - v1s
    # Notice wx,wy here explicitly are normalized here.
    tt1 = [["scalefun, 2 end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    v2mv2 = np.dot(v2mtx, Zs.transpose()) - v2s
    tt1 = [["scalefun, 3 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    fun = [
        v1mv1.real * fscale[0],
        v1mv1.imag * fscale[1],
        v2mv2.real * fscale[2],
        v2mv2.imag * fscale[3],
    ]
    tt1 = [["scalefun, 4 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    tt1 = [["scalefun, 4 end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    # Notice scalefun differs from veq(x) only by shifting x to x+s0, and mutiplies veq by fscale: fscale*veq(s0+x)
    return np.array(fun).transpose()


def scaledfprime(
    Zs, dVdXmtx, fscale
):  # Notice scalefun differs from veq(x) only by shifting x to x+s0, and mutiplies veq by fscale: fscale*veq(s0+x)
    (
        dv1dXmtx,
        dv2dXmtx,
    ) = dVdXmtx  # dVdXt is tpsa of Jacobian dV/dX(x,xp,y,yp), V=v1r,v1i,v2r,v2i
    dv1dX = np.dot(dv1dXmtx, Zs.transpose())
    dv2dX = np.dot(dv2dXmtx, Zs.transpose())
    dVdXscaled = np.array(
        [
            dv1dX.real * fscale[0],
            dv1dX.imag * fscale[1],
            dv2dX.real * fscale[2],
            dv2dX.imag * fscale[3],
        ]
    )
    dVdXscaled = dVdXscaled.transpose([2, 0, 1])
    return dVdXscaled


def fsolvebyscaleHomeMadeblock(
    xyt, v1s, v2s, Vmtx, dVdXmatrix, dVdXmtx, findsoparamters, Zpar3
):
    tt0 = [["invdvdx, 1, start", time.time(), 0]]
    print(tt0)
    # see fsolveHomeMade.lyx in /Users/lihuayu/Desktop/nonlineardynamics/henonheiles/offmnewtpsa/theory_vb_iteration
    bKi, scalex, norder_jordan, powerindex3 = Zpar3
    xtol, fscale, xscale, xfactor, factor, fevtol, gu0 = [
        findsoparamters[x]
        for x in "xtol,fscale,xscale,xfactor,factor,fevtol,gu0".split(",")
    ]
    # pdb.set_trace()
    xyt = xyt.transpose()
    """
    u0 = 0.05 / 0.05 * np.array(xscale) * xfactor
    u0 = np.array([u0 for i in range(400)])
    """
    u0 = gu0
    s0 = xyt - u0
    uu = u0
    tt1 = [["invdvdx, 2 end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    xtmp = np.array(s0 + uu)
    Zs = zcolnew.zcolarray(xtmp, norder_jordan, powerindex3)
    tt1 = [["invdvdx, 2.1 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    v = scalefun(Zs, Vmtx, v1s, v2s, fscale)
    tt1 = [["invdvdx, 2.2 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)

    dvdx = scaledfprime(Zs, dVdXmtx, fscale)
    tt1 = [["invdvdx, 3 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)

    uu = u0
    ik = 0
    # dx = np.array([5e-3, 0, 0, 0])
    # dx = np.array([dx for i in range(400)])
    dx = 1e-3
    maxdx = np.amax(abs(dx))
    tt1 = [["invdvdx, 4 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)

    while (
        maxdx > xtol and maxdx > fevtol and ik < 2
    ):  # find solution u such that scalefun(Zs(u))=0
        tt1 = [["invdvdx, 5 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        # Zs = zcolnew.zcolarray(s0 + uu, norder_jordan, powerindex3)

        # v = scalefun(Zs, Vmtx, v1s, v2s, fscale)
        # dvdx = scaledfprime(Zs, dVdXmtx, fscale)
        # tt1 = [["invdvdx, 6 end", time.time(), time.time() - tt0[0][1]]]
        # print(tt1)
        """
        invdvdx = []
        for i in dvdx:
            invdvdx.append(np.linalg.inv(i))

        invdvdx = np.array(invdvdx)
        tt1 = [["invdvdx, 2 end", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

        dx = []

        for m in range(len(v)):
            dx.append(-np.dot(invdvdx[m], v[m]))

        dx = np.array(dx)
        """
        dx, invdvdx = leq.lineareq_block(dvdx, v)
        tt1 = [["invdvdx, 7 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)

        uu = uu + dx
        ik = ik + 1
        maxdx = np.amax(abs(dx))
        tt1 = [["invdvdx, 7.01 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        Zs = zcolnew.zcolarray(s0 + uu, norder_jordan, powerindex3)
        tt1 = [["invdvdx, 7.1 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        v = scalefun(Zs, Vmtx, v1s, v2s, fscale)
        tt1 = [["invdvdx, 7.2 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        dvdx = scaledfprime(Zs, dVdXmtx, fscale)
        tt1 = [["invdvdx, 8 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        """
        print("\n\tin fsolve block v=", np.amax(abs(v)))
        print("\tdvdx=", np.amax(abs(dvdx)))
        print("\tinvdvdx=", np.amax(abs(invdvdx)))
        print("\tDet=", np.amin(abs(np.linalg.det(invdvdx))))
        print("\tdx=", np.amax(abs(dx)))
        """
    # Notice scalefun differs from veq(x) only by shifting x to x+s0, and mutiplies veq by fscale: fscale*veq(s0+x)
    # so Jacobian of scalefun is fsacle*dVdXt
    tt1 = [["invdvdx, 9.1 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    xyd = (uu + s0).transpose()
    xs, xps, ys, yps = xyd
    if ik < 6:
        msg = {"warnflag": "usehomemadefsoveblock"}
    else:
        msg = {"warnflag": "divergent."}
    tt1 = [["invdvdx, 9.2 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    msg["fvec"] = v / fscale  # the error in f is scaled back its true value here.
    tt1 = [["invdvdx, 9.3 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    msg["nfev"] = ik
    tt1 = [["invdvdx, 9.4 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    vscale2 = (v / fscale) ** 2 / 4
    use_fvec_max = True
    if not use_fvec_max:
        tmp1 = mysum.sum(vscale2[:, 0], vscale2[:, 1])
        tmp2 = mysum.sum(vscale2[:, 2], vscale2[:, 3])
        msg["fval"] = np.mean(
            np.sqrt(tmp1 + tmp2)
        )  # np.sqrt(tmp1 + tmp2)  # np.sqrt(sum((v / fscale) ** 2) / 4)
    else:
        msg["fval"] = np.max(
            vscale2[:, 0].tolist()
            + vscale2[:, 1].tolist()
            + vscale2[:, 2].tolist()
            + vscale2[:, 3].tolist()
        )  #
    msg["outfindso"] = "homeMadefsolveBlock"
    tt1 = [["invdvdx, 9.5 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    tt1 = [["invdvdx, 9.6 total", time.time(), time.time() - tt0[0][1]]]
    print(tt1)

    return xyd, Zs, msg


def findsoblock(
    xy0,
    v1s0,
    v2s0,
    v1s,
    v2s,
    Vmtx,
    dVdX,  # veqm.dVdXmatrix
    dVdXmtx,
    Zpar3,
    findsoparamters,
):
    tt0 = [["in findsoblock 1, start", time.time(), 0]]
    print("tt0=", tt0)
    nsosteplim = findsoparamters["nsosteplim"]
    fevtol = findsoparamters["fevtol"]
    nonconverge = True
    nfindso = 1
    while nonconverge and nfindso < nsosteplim:
        dv1 = (v1s - v1s0) / nfindso
        dv2 = (v2s - v2s0) / nfindso
        v1q = np.array([v1s0 + i * dv1 for i in range(1, nfindso + 1)])
        v2q = np.array([v2s0 + i * dv2 for i in range(1, nfindso + 1)])
        xyq = [dict(xsol=xy0)]
        tt1 = [["in findsoblock, 2. ", time.time(), time.time() - tt0[0][1]]]
        print("tt1=", tt1)
        for ik in range(nfindso):
            xyt = xyq[-1]["xsol"]
            v1qik = v1q[ik]
            v2qik = v2q[ik]
            tt1 = [["in findsoblock, 3. ", time.time(), time.time() - tt1[0][1]]]
            print("tt1=", tt1)
            xys, Zsd, msg = fsolvebyscaleHomeMadeblock(
                xyt,
                v1qik,
                v2qik,
                Vmtx,
                veqm.dVdXmatrix,
                dVdXmtx,
                findsoparamters,
                Zpar3,
            )
            # tmp1=scipy.optimize.fsolve(veq,[ xt,xpt,yt,ypt],args=[acoeff,v1q[i],v2q[i],v0norm],xtol=1e-10,factor=0.5,full_output=True)#Find the z value for new v1(z),v2(z)
            tt1 = [["in findsoblock, 4. ", time.time(), time.time() - tt1[0][1]]]
            print("tt1=", tt1)
            msg["xsol"] = xys
            msg["nfindso"] = nfindso
            msg["Zsd"] = Zsd
            xyq.append(msg)
        # print("np.amax(xys)=", np.amax(xys))
        # print("np.amax(abs(xyq[-1]['fval']))=", np.amax(abs(xyq[-1]["fval"])))
        # pdb.set_trace()
        if (
            xyq[-1]["warnflag"] == "usehomemadefsoveblock"
            and np.amax(abs(xyq[-1]["fval"])) < fevtol
        ):
            nonconverge = False
        # print ('nfindso=',nfindso,'\n')
        nfindso = nfindso * 2
        tt1 = [["in findsoblock, 5. ", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)  # tmp=list(zip(*xyq))
        tt1 = [["in findsoblock, 6. ", time.time(), time.time() - tt0[0][1]]]
        print("tt1=", tt1)  # tmp=list(zip(*xyq))
    return xyq[-1]


def inversev1v2(
    veq,
    inversevinput,
    usexyd=1,
    usewinv=0,  # if usefindso=1, then equation used is veqt, i.e.,  equation is calculated by tpsa, and we may choose to use frime or not in fsolve
    usefindsoscale="usefsolvebyscaleHomeMade",
):
    tt0 = [["in inversev1v2, 1, start", time.time(), 0]]
    print(tt0)
    #######################################################
    for extrac_input_data_for_fsolve in [1]:
        (
            acoeff,
            Vmtx,
            dVdXmtx,
            uarray,
            vna,
            v1n,
            v2n,
            n_theta,
            xi,
            xyd0,
            v0norm,
            bKi,
            scalex,
            norder,
            powerindex,
            Zpar,
            findsoparamters,
        ) = [inversevinput[i] for i in inversevinput.keys()]
        # 9.5 of hn49, Calculate the x,xp,y,yp for given first perturbation v1n,v2n
        # print("\n16.4. Calculate the x,xp,y,yp ffor given first perturbation v1n,v2n in 'inversev1v2'")
        a1, a2 = acoeff
        (
            xt,
            xpt,
            yt,
            ypt,
        ) = xi  # Notice that xi is the initial value of xy with origin at xfix,xpfix, not zeros.
        v1s, v2s = v1n[0, 0], v2n[0, 0]
        if usexyd == 1:
            xt, xpt, yp, ypt = xyd0[:, 0]
        xydrecord = [dict(xsol=[xt, xpt, yt, ypt], vtarget=[v1s, v2s])]
        # n=2
        xydlast = xyd0.reshape([4, n_theta, n_theta])
        # import pdb;
        tt1 = [["in inversev1v2, 2 end", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

        bKi, scalex, norder_jordan, powerindex3 = Zpar
        v1s = v1n.reshape([n_theta ** 2])
        v2s = v2n.reshape([n_theta ** 2])
        xy0 = xydlast.reshape([4, n_theta ** 2])
        Zs = zcolnew.zcolarray(xy0.transpose(), norder_jordan, powerindex3).transpose()
        v1mtx, v2mtx = Vmtx
        v1s0 = np.dot(v1mtx, Zs)
        v2s0 = np.dot(v2mtx, Zs)
        v1s = v1n.reshape([n_theta ** 2])
        v2s = v2n.reshape([n_theta ** 2])
    tt1 = [["in inversev1v2, 3 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    #######################################################
    outfindso = findsoblock(
        xy0,
        v1s0,
        v2s0,
        v1s,
        v2s,
        Vmtx,
        veqm.dVdXmatrix,  # dVdX,  # veqm.dVdXmatrix
        dVdXmtx,
        Zpar,
        findsoparamters,
    )
    tt1 = [["in inversev1v2, 4 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    #######################################################
    for extract_output_data in [1]:
        xyd = outfindso["xsol"]
        Zsd = outfindso["Zsd"]
        tmp4 = outfindso["warnflag"]
        tmp41 = outfindso["fval"]
        tmp42 = outfindso["nfev"]
        nfindso = outfindso["nfindso"]
        fvec = outfindso["fvec"]
        tt1 = [["in inversev1v2, 5 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        inversev1v2msg = {
            "warnflag": "useBlockfsolve",
            "maxerror": tmp41,  # max(tmp41),
            "numberdivgence": 0,
            "nitermean": 1,  # np.mean(tmp42),
            "nfindsomean": 1,  # np.mean(nfindso),
            "fvec": fvec,
            "outfindso": outfindso,
        }
        tt1 = [["in inversev1v2, 6 end", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        xydrecord = {
            "vtarget": [v1s, v2s],
            "vtarget": [v1s, v2s],
            "xtrial": xy0,
            "vtrial": [v1s0, v2s0],
        }
        inversev1v2msg["xydrecord"] = xydrecord
    tt1 = [["in inversev1v2, 7 end", time.time(), time.time() - tt1[0][1]]]
    print(tt1)

    tt1 = [["in inversev1v2, 8 end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)

    return xyd, Zsd, inversev1v2msg


def Fmatrix(
    Zsdx, wmtx, n_theta, acoeff, nvar, v0norm, uarray, Vmtx
):  # F here =Wwiglejnm in Eq.16 of "latticenonlinearResonances.lyx, the fft of w_j
    # 19.10 of hn49, Construct w1x0,w1y0,w1x1,w1y1 based on similated trajectory obtained from the spectrum of phifftaccurateplot
    warray = np.dot(wmtx, Zsdx.transpose())
    tt0 = [["in Fmatrix 1, start", time.time(), 0]]
    print("tt0=", tt0)
    a1, a2 = acoeff
    """
    warray = np.dot(uarray, Zsd.transpose())
    """
    tt1 = [["in Fmatrix, 2. ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    warray = np.reshape(warray, [nvar, n_theta, n_theta])
    tt1 = [["in Fmatrix, 3. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    F = array(np.fft.fft2(warray)) / n_theta ** 2
    # see p.6 of "dvovSpectrumDerivation.pdf", we need to divide by n to get Fourier coefficient
    tt1 = [["in Fmatrix, 4. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    tmp1 = np.transpose(F, [1, 2, 0])
    tt1 = [["in Fmatrix, 5. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    fv2a = np.dot(tmp1, a2 / v0norm[1])
    # fv2a is the spectrum of the first order perturbed action, a deviation from the zero order rigid rotation
    fv1a = np.dot(tmp1, a1 / v0norm[0])
    tt1 = [["in Fmatrix, 6. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    W = array(warray)
    tmp2 = np.transpose(W, [1, 2, 0])
    tt1 = [["in Fmatrix, 7. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    """
    v2a = np.dot(tmp2, a2 / v0norm[1])
    # v2a is recalculated v2n from the perturned coordinates xyd, the deviation from the rigid rotation on grid points of theta0 plane.
    v1a = np.dot(tmp2, a1 / v0norm[0])
    """
    v1mtx, v2mtx = Vmtx
    v1a = np.dot(v1mtx, Zsdx.transpose()).reshape([n_theta, n_theta])
    v2a = np.dot(v2mtx, Zsdx.transpose()).reshape([n_theta, n_theta])
    tt1 = [["in Fmatrix, 8. ", time.time(), time.time() - tt1[0][1]]]
    print("tt1=", tt1)
    tt1 = [["in Fmatrix, 9. ", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    return F, v1a, v2a, fv1a, fv2a


def plotpeaktspectrum(
    fv1a,
    fv2a,
    nux,
    nuy,
    v120i,
    n_theta,
    xllim,
    xulim,
    yllim,
    yulim,
    fign=(25, 21, 26, 27),
):
    v10i, v20i = v120i
    plt.figure(fign[2])
    spectrm = [
        [i * nux + j * nuy, abs(fv1a[i, j] / abs(v10i))]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    spectrm = array(list(zip(*spectrm)))
    spectrm[0] = modularToNo2(spectrm[0], 1.0)
    plt.plot(spectrm[0], spectrm[1], "ro", markersize=16, label="fv1a from xyd")
    plt.ylabel("Fourier coeff. of " + r"$v_1$", fontsize=35)
    plt.title("Fig." + str(fign[2]) + " spectrum of v1")
    plt.xlabel("tune " + r"$\nu_1$", fontsize=35, labelpad=11)
    plt.text(0.6, 1, "b", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.legend(loc="best")
    plt.axis([xllim, xulim, yllim, yulim])
    plt.savefig("junk" + str(fign[2]) + ".png")

    plt.figure(fign[3])
    spectrm = [
        [i * nux + j * nuy, abs(fv2a[i, j]) / abs(v20i)]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    spectrm = array(list(zip(*spectrm)))
    spectrm[0] = modularToNo2(spectrm[0], 1.0)
    print("\n16.52 len(spectrm[0])=", len(spectrm[0]))
    plt.plot(spectrm[0], spectrm[1], "ro", markersize=16, label="fv2a from xyd")
    plt.axis([xllim, xulim, yllim, yulim])
    plt.xlabel("tune " + r"$\nu_2$", fontsize=35, labelpad=11)
    plt.ylabel("Fourier coeff. of " + r"$v_2$", fontsize=35)
    plt.text(0.6, 1, "b", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.title("Fig." + str(fign[3]) + " spectrum of v2")

    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[3]) + ".png")
    return


def plotfvn(
    acoeff,
    vn,
    v120i,
    deltap,
    xfix,
    xpfix,
    npass,
    xllim,
    xulim,
    yllim,
    yulim,
    fign=(25, 21, 26, 27),
):
    # v10 is zeroth order actio, v1n is first order action, vn is action with tracking data,
    # fft2Dphi1peaks is peak ov v1n, v1a is duplicated v1n, fv1a is its spectrum.
    a1, a2 = acoeff
    n = len(vn[0])
    ff = np.arange(n) / (1.0 * n)

    plt.figure(fign[2])
    fv1 = np.fft.fft(
        vn[0] / vn[0][0] / n
    )  # see p.5 of "dvovSpectrumDerivation.pdf", fv2/len(vn[1])=f2v/N gives the Fourier expansion coefficient.
    fv1peak = yunaff.naff(vn[0] / vn[0][0])
    fv1nupeak, fv1peak = fv1peak[0][0], fv1peak[1][0]
    fv1p1 = np.insert(fv1, len(fv1), fv1peak)
    ffp1 = np.insert(ff, len(fv1), fv1nupeak)
    ffm = array(
        modularToNo2(ffp1, 1.0)
    )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
    idx = np.argsort(ffm)
    ffm = ffm[idx]
    fv1m = fv1p1[idx]
    plt.plot(ffm, abs(fv1m), "b-", label="fv1n from tracking")
    plt.axis([xllim, xulim, yllim, yulim])

    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[2]) + ".png")

    plt.figure(fign[3])

    fv2 = np.fft.fft(
        vn[1] / vn[1][0] / n
    )  # see p.5 of "dvovSpectrumDerivation.pdf", fv2/len(vn[1])=f2v/N gives the Fourier expansion coefficient.
    fv2peak = yunaff.naff(vn[1] / vn[1][0])
    fv2nupeak, fv2peak = fv2peak[0][0], fv2peak[1][0]
    fv2p2 = np.insert(fv2, len(fv2), fv2peak)
    ffp2 = np.insert(ff, len(fv2), fv2nupeak)
    ffm = array(
        modularToNo2(ffp2, 1.0)
    )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
    idx = np.argsort(ffm)
    ffm = ffm[idx]
    fv2m = fv2p2[idx]
    plt.plot(ffm, abs(fv2m), "b-", label="fv2n from tracking")
    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[3]) + ".png")
    return


def plotfftpeaks(
    fft2Dphi1peaks,
    fft2Dphi2peaks,
    v10,
    v20,
    v1n,
    v2n,
    vn,
    nux,
    nuy,
    n_theta,
    xllim,
    xulim,
    yllim,
    yulim,
    fign=(25, 21, 26, 27),
):
    # v10i,v20i=v120i
    # v10 is zeroth order actio, v1n is first order action, vn is action with tracking data,
    # fft2Dphi1peaks is peak ov v1n, v1a is duplicated v1n, fv1a is its spectrum.
    # a1,a2=acoeff
    plt.figure(fign[0])
    plt.plot(vn[0].real, vn[0].imag, "c.", markersize=5, label="vn[0] tracking")
    plt.xlabel("Re(" + r"$v_1$(n)" + ")", fontsize=30, labelpad=15)
    plt.ylabel("Im(" + r"$v_1$(n)" + ")", fontsize=30, labelpad=-5)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.plot(v10.real, v10.imag, "b.", markersize=10)
    plt.plot(v1n.real, v1n.imag, "r.", markersize=7)
    plt.plot(v10[0].real, v10[0].imag, "b.", markersize=10, label="v10")
    plt.plot(v1n[0].real, v1n[0].imag, "r.", markersize=7, label="v1n")

    plt.legend(loc="best")
    # plt.axis([-1.7,1.7,-1.7,1.7])
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.title(
        "Fig."
        + str(fign[0])
        + ' v1n (red) is the first order perturbation \nof zero order v10 (blue)  calculated by inversefft from \nbnm1 in eq.17 and eq.42 of "latticenonlinearResonances.lyx"'
    )
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1])
    plt.plot(vn[1].real, vn[1].imag, "c.", markersize=5, label="vn[1] traking")
    plt.xlabel("Re(" + r"$v_2$(n)" + ")", fontsize=30, labelpad=15)
    plt.ylabel("Im(" + r"$v_2$(n)" + ")", fontsize=30, labelpad=-5)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.plot(v20[0].real, v20[0].imag, "b.", markersize=10, label="v20")
    plt.plot(v2n[0].real, v2n[0].imag, "r.", markersize=7, label="v2n")
    plt.plot(v20.real, v20.imag, "b.", markersize=10)
    plt.plot(v2n.real, v2n.imag, "r.", markersize=7)
    plt.legend(loc="best")
    # plt.axis([-1.7,1.7,-1.7,1.7])
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    # plt.text(-0.35,0.3,'b',fontsize=30)

    plt.title(
        "Fig."
        + str(fign[1])
        + ' v2n (red) is the first order perturbation \nof zero order v10 (blue)  calculated by inversefft from \nbnm1 in eq.17 and eq.42 of "latticenonlinearResonances.lyx"'
    )
    plt.savefig("junk" + str(fign[1]) + ".png")

    plt.figure(fign[2])
    plt.ylabel("Fourier coeff. of " + r"$v_1$", fontsize=35)
    plt.title("Fig." + str(fign[2]) + " spectrum of v1")
    plt.xlabel("tune " + r"$\nu_1$", fontsize=35, labelpad=11)
    plt.text(0.6, 1, "b", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)
    # plt.tight_layout()
    tmp = fft2Dphi1peaks  # [-5:]
    tmp[:, 0] = modularToNo2(tmp[:, 0].real, 1.0)
    plt.plot(
        tmp[:, 0].real,
        abs(tmp[:, 1]),
        "ro",
        markersize=11,
        label="fft2Dphi1peaks from bnm1",
    )

    n = len(vn[0])
    ff = np.arange(n) / (1.0 * n)
    """
        Notice, see p.6 of dvovSpectrumDerivation.pdf, we need to divide by n to get Fourier coefficient
        """

    # print("\n16.51 vn[1][0], v20i=",vn[1][0], v20i)
    fv1 = np.fft.fft(
        vn[0] / vn[0][0] / n
    )  # see p.5 of "dvovSpectrumDerivation.pdf", fv2/len(vn[1])=f2v/N gives the Fourier expansion coefficient.
    fv1peak = yunaff.naff(vn[0] / vn[0][0])
    fv1nupeak, fv1peak = fv1peak[0], fv1peak[1]
    fv1p1 = np.insert(fv1, len(fv1), fv1peak)
    ffp1 = np.insert(ff, len(fv1), fv1nupeak)
    ffm = array(
        modularToNo2(ffp1, 1.0)
    )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
    idx = np.argsort(ffm)
    ffm = ffm[idx]
    fv1m = fv1p1[idx]
    fv1nupeak = array(modularToNo2(np.array(fv1nupeak), 1.0))

    plt.plot(ffm, abs(fv1m), "b-", label="fv1n from tracking")
    plt.plot(fv1nupeak, abs(fv1peak), "yo", label="fv1 naff peak")
    plt.axis([xllim, xulim, yllim, yulim])

    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[2]) + ".png")

    plt.figure(fign[3])
    # print("\n16.52 len(spectrm[0])=", len(spectrm[0]))

    tmp = fft2Dphi2peaks  # [-5:]
    tmp[:, 0] = modularToNo2(tmp[:, 0].real, 1.0)
    print("\n16.53 len(fft2Dphi2peaks)=", len(fft2Dphi2peaks))
    plt.plot(
        tmp[:, 0].real,
        abs(tmp[:, 1]),
        "ro",
        markersize=11,
        label="fft2Dphi2peaks from bnm2",
    )
    plt.axis([xllim, xulim, yllim, yulim])
    plt.xlabel("tune " + r"$\nu_2$", fontsize=35, labelpad=11)
    plt.ylabel("Fourier coeff. of " + r"$v_2$", fontsize=35)
    plt.text(0.6, 1, "b", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)

    plt.title("Fig." + str(fign[3]) + " spectrum of v2")

    n = len(vn[1])
    ff = np.arange(n) / (1.0 * n)
    fv2 = np.fft.fft(
        vn[1] / vn[1][0] / n
    )  # see p.5 of "dvovSpectrumDerivation.pdf", fv2/len(vn[1])=f2v/N gives the Fourier expansion coefficient.
    fv2peak = yunaff.naff(vn[1] / vn[1][0])
    fv2nupeak, fv2peak = fv2peak[0], fv2peak[1]
    fv2p2 = np.insert(fv2, len(fv2), fv2peak)
    ffp2 = np.insert(ff, len(fv2), fv2nupeak)
    ffm = array(
        modularToNo2(ffp2, 1.0)
    )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
    idx = np.argsort(ffm)
    ffm = ffm[idx]
    fv2m = fv2p2[idx]
    fv2nupeak = array(modularToNo2(np.array(fv2nupeak), 1.0))
    plt.plot(ffm, abs(fv2m), "b-", label="fv2n from tracking")
    plt.plot(fv2nupeak, abs(fv2peak), "yo", label="fv2 naff peak")
    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[3]) + ".png")
    return


def plotvnv0v1(
    acoeff,
    fv1a,
    fv2a,
    v1a,
    v2a,
    fft2Dphi1peaks,
    fft2Dphi2peaks,
    v10,
    v20,
    v1n,
    v2n,
    vn,
    nux,
    nuy,
    v120i,
    numerror,
    Zsd,
    uarray,
    n_theta,
    xllim,
    xulim,
    yllim,
    yulim,
    fign=(25, 21, 26, 27),
    xmax=0,
    ymax=0,
):
    v10i, v20i = v120i
    # v10 is zeroth order actio, v1n is first order action, vn is action with tracking data,
    # fft2Dphi1peaks is peak ov v1n, v1a is duplicated v1n, fv1a is its spectrum.
    a1, a2 = acoeff
    plt.figure(fign[0])
    plt.plot(vn[0].real, vn[0].imag, "c.", markersize=5, label="vn[0] tracking")
    plt.xlabel("Re(" + r"$v_1$(n)" + ")", fontsize=30, labelpad=15)
    plt.ylabel("Im(" + r"$v_1$(n)" + ")", fontsize=30, labelpad=-5)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.plot(v1a.real, v1a.imag, "g.", markersize=15)
    plt.plot(v1n.real, v1n.imag, "r.", markersize=5)
    plt.plot(v10.real, v10.imag, "b.", markersize=2)
    plt.plot(v1a[0].real, v1a[0].imag, "g.", markersize=15, label="v1a")
    plt.plot(v1n[0].real, v1n[0].imag, "r.", markersize=5, label="v1n")
    plt.plot(v10[0].real, v10[0].imag, "b.", markersize=2, label="v10")
    plt.legend(loc="best")
    # plt.axis([-1.7,1.7,-1.7,1.7])
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.title(
        "Fig."
        + str(fign[0])
        + ' v1n (red) is the first order perturbation \nof zero order v10 (blue)  calculated by inversefft from \
                \n bnm1 in eq.17 and eq.42 of "latticenonlinearResonances.lyx" \n xmax='
        + str(xmax)
        + ", ymax="
        + str(ymax)
    )
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1])
    plt.plot(vn[1].real, vn[1].imag, "c.", markersize=5, label="vn[1] traking")
    plt.xlabel("Re(" + r"$v_2$(n)" + ")", fontsize=30, labelpad=15)
    plt.ylabel("Im(" + r"$v_2$(n)" + ")", fontsize=30, labelpad=-5)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.plot(v2a[0].real, v2a[0].imag, "g.", markersize=15, label="v2a")
    plt.plot(v2n[0].real, v2n[0].imag, "r.", markersize=5, label="v2n")
    plt.plot(v20[0].real, v20[0].imag, "b.", markersize=2, label="v20")
    plt.plot(v2a.real, v2a.imag, "g.", markersize=12)
    plt.plot(v2n.real, v2n.imag, "r.", markersize=5)
    plt.plot(v20.real, v20.imag, "b.", markersize=2)
    plt.legend(loc="best")
    # plt.axis([-1.7,1.7,-1.7,1.7])
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    # plt.text(-0.35,0.3,'b',fontsize=30)
    if len(numerror) != 0:
        zerror = array([Zsd[i] for i in numerror]).transpose()
        Werror = np.dot(uarray, zerror)
        v2error = np.dot(a2, Werror)
        plt.plot(v2error.real, v2error.imag, "m.", label="v2 Werror")

    plt.title(
        "Fig."
        + str(fign[1])
        + ' v2n (red) is the first order perturbation \nof zero order v10 (blue)  calculated by inversefft from \
                \n bnm1 in eq.17 and eq.42 of "latticenonlinearResonances.lyx" \n xmax='
        + str(xmax)
        + ", ymax="
        + str(ymax)
    )
    plt.savefig("junk" + str(fign[1]) + ".png")

    plt.figure(fign[2])
    spectrm = [
        [i * nux + j * nuy, abs(fv1a[i, j] / abs(v10i))]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    spectrm = array(list(zip(*spectrm)))
    spectrm[0] = modularToNo2(spectrm[0], 1.0)
    plt.plot(spectrm[0], spectrm[1], "ro", markersize=16, label="fv1a from xyd")
    plt.ylabel("Fourier coeff. of " + r"$v_1$", fontsize=35)
    plt.title("Fig." + str(fign[2]) + " spectrum of v1")
    plt.xlabel("tune " + r"$\nu_1$", fontsize=35, labelpad=11)
    plt.text(0.6, 1, "b", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)
    # plt.tight_layout()
    tmp = fft2Dphi1peaks  # [-5:]
    tmp[:, 0] = modularToNo2(tmp[:, 0].real, 1.0)
    plt.plot(
        tmp[:, 0].real,
        abs(tmp[:, 1]),
        "yo",
        markersize=11,
        label="fft2Dphi1peaks from bnm1",
    )

    n = len(vn[0])
    ff = np.arange(n) / (1.0 * n)
    """
        Notice, see p.6 of dvovSpectrumDerivation.pdf, we need to divide by n to get Fourier coefficient
        """

    print("\n16.51 vn[1][0], v20i=", vn[1][0], v20i)
    fv1 = np.fft.fft(
        vn[0] / vn[0][0] / n
    )  # see p.5 of "dvovSpectrumDerivation.pdf", fv2/len(vn[1])=f2v/N gives the Fourier expansion coefficient.
    fv1nupeak, fv1peak = yunaff.naff(vn[0] / vn[0][0])
    fv1p1 = np.insert(fv1, len(fv1), fv1peak)
    ffp1 = np.insert(ff, len(fv1), fv1nupeak)
    ffm = array(
        modularToNo2(ffp1, 1.0)
    )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
    idx = np.argsort(ffm)
    ffm = ffm[idx]
    fv1m = fv1p1[idx]
    fv1nupeak = array(modularToNo2(np.array(fv1nupeak), 1.0))
    plt.plot(ffm, abs(fv1m), "b-", label="fv1n from tracking")
    plt.axis([xllim, xulim, yllim, yulim])
    plt.plot(fv1nupeak, abs(fv1peak), "mo", label="fv1 naff peak")
    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[2]) + ".png")

    plt.figure(fign[3])
    spectrm = [
        [i * nux + j * nuy, abs(fv2a[i, j]) / abs(v20i)]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    spectrm = array(list(zip(*spectrm)))
    spectrm[0] = modularToNo2(spectrm[0], 1.0)
    print("\n16.52 len(spectrm[0])=", len(spectrm[0]))
    plt.plot(spectrm[0], spectrm[1], "ro", markersize=16, label="fv2a from xyd")
    tmp = fft2Dphi2peaks  # [-5:]
    tmp[:, 0] = modularToNo2(tmp[:, 0].real, 1.0)
    print("\n16.53 len(fft2Dphi2peaks)=", len(fft2Dphi2peaks))
    plt.plot(
        tmp[:, 0].real,
        abs(tmp[:, 1]),
        "yo",
        markersize=11,
        label="fft2Dphi2peaks from bnm2",
    )
    plt.axis([xllim, xulim, yllim, yulim])
    plt.xlabel("tune " + r"$\nu_2$", fontsize=35, labelpad=11)
    plt.ylabel("Fourier coeff. of " + r"$v_2$", fontsize=35)
    plt.text(0.6, 1, "b", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.title("Fig." + str(fign[3]) + " spectrum of v2")

    n = len(vn[1])
    ff = np.arange(n) / (1.0 * n)
    fv2 = np.fft.fft(
        vn[1] / vn[1][0] / n
    )  # see p.5 of "dvovSpectrumDerivation.pdf", fv2/len(vn[1])=f2v/N gives the Fourier expansion coefficient.
    try:
        fv2nupeak, fv2peak = yunaff.naff(vn[1] / vn[1][0])
        fv2p2 = np.insert(fv2, len(fv2), fv2peak)
        ffp2 = np.insert(ff, len(fv2), fv2nupeak)
        ffm = array(
            modularToNo2(ffp2, 1.0)
        )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
        idx = np.argsort(ffm)
        ffm = ffm[idx]
        fv2m = fv2p2[idx]
        fv2nupeak = array(modularToNo2(np.array(fv2nupeak), 1.0))
        plt.plot(ffm, abs(fv2m), "b-", label="fv2n from tracking")
        plt.plot(fv2nupeak, abs(fv2peak), "mo", label="fv2 naff peak")
    except Exception as err:
        print(dir(err))
        print(err.args)
    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[3]) + ".png")
    return


def anew12(F, nux, nuy, nvar, n_theta, Cauchylimit, ntune):
    # see derivation in "weightedIterationFormulas.lyx in offmnew/theory" for weights
    tt0 = [["in anew12 1, start", time.time(), 0]]
    print("tt0=", tt0)
    # cutoff = n_theta
    cutoff = Cauchylimit["aliasingCutoff"]
    print(
        "\n16.6 Calculate optimized linear combination by least \n\
        square method to minimize fluctuation fft amplitude"
    )
    # 24.1 of hn49, Follow p.8 of 'minimizationOfFluctuationOfAction.pdf' in folder 'jfn-resonance5'."
    print(
        "\n16.61 Use F matrix to build constant matrix A,B in equation of Lagrangian multiplier method.nvar=",
        nvar,
    )
    """
        to understand F, let us want find which i gives F[i]=F1[-2,-4]
        we check that tmp1[-2,-4]=52, then we see that F[0,52]=F1[0,-2,-4]
        so F[0,tmp1[m,n]]=F1[0,m,n], lte
        tmp5=[ F1[0,m,n]==F[0,tmp1[m,n]] for m in range(-4,4) for n in range(-4,4)]
        all(tmp5)==True gives True
        """
    F1 = F.copy()
    # F = np.reshape(
    #    F, [nvar, n_theta ** 2]
    # )  # F is already calculated in section 5 above.
    F = F[:nvar]
    # spectrm=[ [i*nux+j*nuy, abs(fv1a[i,j]/abs(v10i))] in vph.plotpeaktspectrum shows
    #  the meaning of fv1a[i,j] is the intensity of tune i*nux+j*nuy and in vph.Fmatrix the relation fv1a and F is clear
    # that F[ik,m,n] is the line intensity of tune m*nux+n*nuy
    # Notice the relation between F[m,n] and b[m,n]: F[m,n]<->bnm1[m-1,n], or F[m+1,n]<->bnm1[m,n]
    # because the term in bnm1 is bnm1[m,n]exp(j(m*om1+n*om2)), then in F, it contribute an extra factor exp(j*om1)
    # For a term F[m,n], it is contributed from bnm1[m-1,n] which is from aphi1[m-1,n]/((exp(j((m-1)*om1+n*om2))-1)
    # The two dominant frequency indexs, we do not sort index because there is a peak at (2,-1) higher than (0,1)
    A = np.zeros([nvar, nvar]) * (
        1 + 0j
    )  # A,B matrix are the F1,F2 matrix in  "weightedIterationFormulas.lyx" (eq.15))
    B = np.zeros([nvar, nvar]) * (1 + 0j)
    # import pdb; pdb.set_trace()
    """
    Fa = np.repeat(F[:, np.newaxis, :], nvar, axis=1)
    Fb = np.repeat(F[np.newaxis, :, :], nvar, axis=0)
    exclude = (Fa * Fb)[:, :, 1, 0]
    A = np.sum(Fa * Fb, axis=(2, 3)) - exclude
    """
    tt1 = [["in anew12 5", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)

    iCut = cutoff // 2
    Fcut = np.concatenate((F[:, :iCut, :], F[:, -iCut:, :]), axis=1)
    Fcut = np.concatenate((Fcut[:, :, :iCut], Fcut[:, :, -iCut:]), axis=2)
    Fa = np.repeat(Fcut[:, np.newaxis, :], nvar, axis=1)
    Fb = np.repeat(Fcut[np.newaxis, :, :], nvar, axis=0)
    Fab = Fa.conj() * Fb
    exclude = Fab[:, :, 1, 0]
    A = np.sum(Fab, axis=(2, 3)) - exclude
    tt1 = [["in anew12 6", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)

    exclude = Fab[:, :, 0, 1]
    B = np.sum(Fab, axis=(2, 3)) - exclude
    tt1 = [["in anew12 7", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)

    """
    alpharangeA = [
        [i, j]
        for i in range(-cutoff // 2, cutoff // 2)
        for j in range(-cutoff // 2, cutoff // 2)
        if [i, j] != [1, 0]
    ]
    for i in range(nvar):
        for j in range(nvar):
            for alpha in alpharangeA:
                A[i, j] = (
                    A[i, j]
                    + np.conjugate(F[i, alpha[0], alpha[1]]) * F[j, alpha[0], alpha[1]]
                )

    B = np.zeros([nvar, nvar]) * (1 + 0j)
    alpharangeB = [
        [i, j]
        for i in range(-cutoff // 2, cutoff // 2)
        for j in range(-cutoff // 2, cutoff // 2)
        if [i, j] != [0, 1]
    ]
    for i in range(nvar):
        for j in range(nvar):
            for alpha in alpharangeB:
                B[i, j] = (
                    B[i, j]
                    + np.conjugate(F[i, alpha[0], alpha[1]]) * F[j, alpha[0], alpha[1]]
                )
    """

    print("\n16.62 the matrixes A and B of p.8:A=")
    # jfdf.pcm(A*1e-4,nvar,nvar)
    print("B=")
    # jfdf.pcm(B*1e-4,nvar,nvar)
    Am = np.linalg.inv(A)
    Bm = np.linalg.inv(B)
    print("\n16.63 the matrixes A and B of p.8:A^-1=")
    # jfdf.pcm(Am,nvar,nvar)
    print("B^-1=")
    # jfdf.pcm(Bm,nvar,nvar)
    """
                The solution of the equation in page 5 of 'minimizationOfFluctuationOfAction.pdf', xxis X, yy is Y
        """
    tt1 = [["in anew12 8", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)

    xx = np.dot(Am, np.conjugate(F[:, 1, 0]))
    xxd = np.dot(F[:, 1, 0], xx)
    a1new = xx / xxd

    yy = np.dot(Bm, np.conjugate(F[:, 0, 1]))
    yyd = np.dot(F[:, 0, 1], yy)
    a2new = yy / yyd
    if ntune == 2:
        mf = array([[F[0, 1, 0], F[0, 0, 1]], [F[1, 1, 0], F[1, 0, 1]]])
        a1new, a2new = np.linalg.inv(mf)
    """
        if nvar==2:
                a1new=np.append(a1new,[0,0])
                a2new=np.append(a2new,[0,0])
        """
    acoeffnew = array([a1new, a2new])
    tt1 = [["in anew12 9", time.time(), time.time() - tt0[0][1]]]
    print("tt1=", tt1)
    return acoeffnew


def plotfvnew(acoeffnew, nux, nuy, F, nvar, n_theta, fign=(26, 27)):
    a1new, a2new = acoeffnew
    Fs = np.reshape(F, [nvar, n_theta ** 2])
    fv1new = np.dot(a1new, Fs)
    fv2new = np.dot(a2new, Fs)
    fv1new = np.reshape(fv1new, [n_theta, n_theta])
    fv2new = np.reshape(fv2new, [n_theta, n_theta])

    plt.figure(fign[0])
    # plt.xlabel('frequency '+r'$\omega$',fontsize=20)
    # plt.ylabel('Fourier expansion',fontsize=20)
    v1spectrm = [
        [i * nux + j * nuy, abs(fv1new[i, j])]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    v1spectrm = array(list(zip(*v1spectrm)))
    v1spectrm[0] = modularToNo2(v1spectrm[0], 1.0)
    plt.plot(v1spectrm[0], v1spectrm[1], "go", markersize=6, label="fv1new")
    # plt.title('Fig.27 the spectrum bnm1 of'+r'$|v_1|$'+' calculated by phifftaccurateplot (green), \n\
    # tracking(blue) fv1,\n  use a1new (green)',fontsize=12)#linear combination fv2a of fft2 of new wx,wy,wx1,wy1(red), use a2new (magenta)',fontsize=12)
    plt.tight_layout()
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1])
    # fig26.set_xlabel('frequency '+r'$\omega$',fontsize=20)
    # fig26.set_ylabel('Fourier expansion',fontsize=20)
    v2spectrm = [
        [i * nux + j * nuy, abs(fv2new[i, j])]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    v2spectrm = array(list(zip(*v2spectrm)))
    v2spectrm[0] = modularToNo2(v2spectrm[0], 1.0)
    plt.plot(v2spectrm[0], v2spectrm[1], "go", markersize=6, label="fv2new")
    plt.tight_layout()
    # plt.title('Fig.26 the spectrum bnm2 of'+r'$|v_2|$'+' calculated by phifftaccurateplot (green), \n\
    # tracking(blue) fv2,\n use a2new (green)',fontsize=12)# linear combination fv2a of fft2 of new wx,wy,wx1,wy1(red), use a2new (green)',fontsize=12)
    plt.savefig("junk" + str(fign[1]) + ".png")
    return


def checka1new(a1new, warray):
    plt.figure(40)
    plt.plot(abs(warray[0]), "ro")
    plt.plot(abs(warray[1]), "go")
    plt.plot(abs(warray[2]), "bo")
    # plt.plot(abs(warray[2]),'bo')
    plt.plot(
        abs(a1new[0] * warray[0] + a1new[1] * warray[1] + a1new[2] * warray[2]),
        "yo",
        markersize=3,
    )
    # plt.plot(abs(a1new[0]*warray[0]+a1new[1]*warray[1]+a1new[2]*warray[2]),'b-')
    v1new = np.dot(a1new, warray)
    plt.plot(abs(v1new), "g-")
    plt.ylabel(
        "fig40 wx0 red, wx1 green,wx2 blue, \na1*wx0+a2*wx1 magenta, a1new.warray b-"
    )
    plt.savefig("junk40.png")
    return v1new


def plotxp0dAspectEqualGrid(x0d, xp0d, xylabel=("x0d", "xp0d"), fign=231):
    plt.figure(fign)
    plt.plot(x0d, xp0d, "b.")
    plt.plot(x0d[1:4], xp0d[1:4], "ro")
    plt.xlabel(xylabel[0])
    plt.ylabel(xylabel[1])
    plt.title("Fig." + str(fign) + ", initial phase space scan")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.savefig("junk" + str(fign) + ".png")
    return


def plot3Dxytrackingxyd(xy0, xy1, xy2, xy3, x0d, xp0d, y0d, yp0d, fign=[1, 2, 4, 5]):
    xy0, xy1 = xy0, xy1
    x0d, xp0d = x0d, xp0d
    st = array([xy0, xy1, xy2, xy3]).real  # (x,xp,y,yp)
    st1 = array([xy2, xy3, xy0, xy1]).real  # (y,yp,x,xp)
    std = array([x0d, xp0d, y0d, yp0d]).real  # (x,xp,y,yp)
    std1 = array([y0d, yp0d, x0d, xp0d]).real  # (y,yp,x,xp)
    label = {
        "xlabel1": "y",
        "ylabel1": "yp",
        "zlabel1": "xp",
        "xlabel2": "x",
        "ylabel2": "xp",
        "zlabel2": "yp",
        "xlabel4": "y",
        "ylabel4": "yp",
        "zlabel4": "x",
        "xlabel5": "x",
        "ylabel5": "xp",
        "zlabel5": "y",
        "fign": fign,
    }

    # sv("junk", [1e0 * st, 1e0 * st1, 1e0 * std, 1e0 * std1, label])
    # exec(compile(open("lt2plot4.py").read(), "lt2plot4.py", "exec"))
    ## execfile('lt2plot.py')
    ## fig.1,st:y,yp vs xp;  fig.2 st1: x,xp-yp;
    ## fig.4,st:y,yp vs. x;  fig.5 st1: x,xp-y;

    xlabel1 = label["xlabel1"]
    ylabel1 = label["ylabel1"]
    xlabel2 = label["xlabel2"]
    ylabel2 = label["ylabel2"]
    xlabel4 = label["xlabel4"]
    ylabel4 = label["ylabel4"]
    xlabel5 = label["xlabel5"]
    ylabel5 = label["ylabel5"]
    zlabel1 = label["zlabel1"]
    zlabel2 = label["zlabel2"]
    zlabel4 = label["zlabel4"]
    zlabel5 = label["zlabel5"]
    fign = label["fign"]

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import FormatStrFormatter

    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    zmax = max(st[1].real)
    zmin = min(st[1].real)
    zmean = np.mean(st[1])
    print(" xp, zmax,zmin=", zmax, zmin)

    ax.scatter(st[2], st[3], st[1], c="r", marker=".")
    ax.scatter(std[2], std[3], std[1], c="b", marker=".")
    ztick = ax.get_zticks(minor=False)
    lbl = [("%0.03e" % a).split("e") for a in ztick]
    lbl2 = list(zip(*[[i[0], str(int(i[1]))] for i in lbl]))
    lbl3 = ["$ \\times 10^{" + str(tmp) + "}$" for tmp in lbl2[1]]
    lbl4 = list(zip(*[lbl2[0], lbl3]))
    lbl5 = [i[0] + i[1] for i in lbl4]
    ax.set_zticklabels(lbl5, rotation=0, ha="center", size="xx-large")
    ax.set_xlabel("\n\n" + xlabel1, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel1, fontsize=20)
    ax.set_zlabel("                    " + zlabel1 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)

    plt.title("Fig." + str(fign[0]))
    plt.savefig("junk1.png")

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")

    zmax = max(st1[1].real)
    zmin = min(st1[1].real)
    zmean = np.mean(st1[1])

    print(" for yp,  zmax,zmin=", zmax, zmin)

    ax.scatter(st1[2], st1[3], st1[1], c="r", marker=".")
    ax.scatter(std1[2], std1[3], std1[1], c="b", marker=".")

    ax.set_xlabel("\n\n" + xlabel2, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel2, fontsize=20)
    ax.set_zlabel("                    " + zlabel2 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)
    # ax.set_zlim3d(zmin*0.,zmax*1.1)
    plt.title("Fig." + str(fign[1]))
    plt.savefig("junk2.png")

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection="3d")

    zmax = max(st[0].real)
    zmin = min(st[0].real)
    zmean = np.mean(st[0])
    print(" for x,  zmax,zmin=", zmax, zmin)
    # ax.set_zlim3d(zmin*0.,zmax*1.1)
    ax.scatter(st[2], st[3], st[0], c="r", marker=".")
    ax.scatter(std[2], std[3], std[0], c="b", marker=".")
    ztick = ax.get_zticks(minor=False)
    lbl = [("%0.3e" % a).split("e") for a in ztick]
    lbl2 = list(zip(*[[i[0], str(int(i[1]))] for i in lbl]))
    lbl3 = ["$ \\times 10^{" + str(tmp) + "}$" for tmp in lbl2[1]]
    lbl4 = list(zip(*[lbl2[0], lbl3]))
    lbl5 = [i[0] + i[1] for i in lbl4]
    ax.set_zticklabels(lbl5, rotation=0, ha="center", size="xx-large")
    # plt.gca().zaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_xlabel("\n\n" + xlabel4, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel4, fontsize=20)
    ax.set_zlabel("                    " + zlabel4 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)

    plt.title("Fig." + str(fign[2]))
    plt.savefig("junk4.png")

    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection="3d")

    zmax = max(st1[0].real)
    zmin = min(st1[0].real)
    zmean = np.mean(st1[0])

    print(" for y,  zmax,zmin=", zmax, zmin)

    ax.scatter(st1[2], st1[3], st1[0], c="r", marker=".")
    ax.scatter(std1[2], std1[3], std1[0], c="b", marker=".")
    ax.set_xlabel("\n\n" + xlabel5, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel5, fontsize=20)
    ax.set_zlabel("                    " + zlabel5 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)
    # ax.set_zlim3d(zmin*0.,zmax*1.1)
    plt.title("Fig." + str(fign[3]))
    plt.savefig("junk5.png")


def plot2Dcrossection(
    x0d,
    xp0d,
    y0d,
    yp0d,
    xy,
    xfix,
    xpfix,
    n_theta,
    xlim=(-2e-2, 2e-2),
    ylim=(-2e-3, 2e-3),
    fign=(72, 73),
    lbl=("x0d,xp0d", "y0d,yp0d"),
):  # plot tracking x,xp and x0d,xp0d from scanning v1,v2 in a cross section in narrow range of yp
    x0n, xp0n, y0n, yp0n = (
        x0d.reshape(n_theta ** 2),
        xp0d.reshape(n_theta ** 2),
        y0d.reshape(n_theta ** 2),
        yp0d.reshape(n_theta ** 2),
    )
    tmp = array([x0n, xp0n, y0n, yp0n]).real.transpose()  # (x,xp,y,yp) in every row
    x0n, xp0n, y0n, yp0n = np.array(
        [[x, xp, y, yp] for x, xp, y, yp in tmp if ylim[0] < yp < ylim[1]]
    ).transpose()
    tmp = xy.transpose()
    xy1 = np.array(
        [[x, xp, y, yp] for x, xp, y, yp in tmp if ylim[0] < yp < ylim[1]]
    ).transpose()
    plt.figure(fign[0])
    plt.plot(xy1[0] + xfix, xy1[1] + xpfix, ".", label="x,xp from tracking")
    plt.plot(x0n + xfix, xp0n + xpfix, ".", label=lbl[0] + " from grid v1,v2 phases ")
    plt.xlabel("x")
    plt.ylabel("xp")
    plt.title(
        "Fig."
        + str(fign[0])
        + " plot tracking x,xp and x0d+xfix,xp0d+xpfix \nfrom scanning v1,v2 in a cross section in narrow range of yp\n , "
        + str(ylim[0])
        + "<yp<"
        + str(ylim[1])
    )
    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[0]) + ".png")

    x0n, xp0n, y0n, yp0n = (
        x0d.reshape(n_theta ** 2),
        xp0d.reshape(n_theta ** 2),
        y0d.reshape(n_theta ** 2),
        yp0d.reshape(n_theta ** 2),
    )
    tmp = array([x0n, xp0n, y0n, yp0n]).real.transpose()  # (x,xp,y,yp) in every row
    x0n, xp0n, y0n, yp0n = np.array(
        [[x, xp, y, yp] for x, xp, y, yp in tmp if xlim[0] < xp < xlim[1]]
    ).transpose()
    tmp = xy.transpose()
    xy1 = np.array(
        [[x, xp, y, yp] for x, xp, y, yp in tmp if xlim[0] < xp < xlim[1]]
    ).transpose()
    plt.figure(fign[1])
    plt.plot(xy1[2], xy1[3], ".", label="y,yp from tracking")
    plt.plot(y0n, yp0n, ".", label=lbl[1] + " from grid v1,v2 phases ")
    plt.xlabel("y")
    plt.ylabel("yp")
    plt.title(
        "Fig."
        + str(fign[1])
        + " plot tracking x,xp and y0d,yp0d \nfrom scanning v1,v2 in a cross section in narrow range of xp\n , "
        + str(xlim[0])
        + "<xp<"
        + str(xlim[1])
    )
    plt.legend(loc="best")
    plt.savefig("junk" + str(fign[1]) + ".png")


def plotv1nv2nThetaVGrid(v1n, v2n, n_theta, fign=(25, 27, 29)):
    thetav1d = list(map(cmath.phase, v1n.reshape([n_theta ** 2])))
    thetav2d = list(map(cmath.phase, v2n.reshape([n_theta ** 2])))
    plt.figure(fign[2])
    plt.plot(thetav1d, thetav2d, ".")
    plt.xlabel("thetav1d")
    plt.ylabel("thetav2d")
    plt.title(
        "Fig."
        + str(fign[2])
        + ' thetav1d,thetav2d of theta1(n), theta2(n) \nin eq.37 of "latticenonlinearResonances.lyx"'
    )
    plt.savefig("junk" + str(fign[2]) + ".png")

    plt.figure(fign[0])
    plt.plot(v1n.real, v1n.imag, "r.", markersize=12)
    plt.xlabel("Re(v1(n))")
    plt.ylabel("Im(v1(n))")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.title(
        "Fig."
        + str(fign[0])
        + 'v1n is the first order perturbation of v10 \n in eq.37 of "latticenonlinearResonances.lyx"'
    )
    plt.savefig("junk" + str(fign[0]) + ".png")
    plt.figure(fign[1])
    plt.plot(v2n.real, v2n.imag, "r.", markersize=12)
    plt.xlabel("Re(v2(n))")
    plt.ylabel("Im(v2(n))")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.title(
        "Fig."
        + str(fign[1])
        + 'v2n is the first order perturbation of v20 \n in eq.37 of "latticenonlinearResonances.lyx"'
    )
    plt.savefig("junk" + str(fign[1]) + ".png")


def plot3d(st, st1, stlabel, st1label, fign=[91, 92, 93, 94]):
    label = {
        "xlabel1": stlabel[2],
        "ylabel1": stlabel[3],
        "zlabel1": stlabel[0],
        "xlabel2": stlabel[2],
        "ylabel2": stlabel[3],
        "zlabel2": stlabel[1],
        "xlabel4": st1label[2],
        "ylabel4": st1label[3],
        "zlabel4": st1label[0],
        "xlabel5": st1label[2],
        "ylabel5": st1label[3],
        "zlabel5": st1label[1],
        "fig1": fign[0],
        "fig2": fign[1],
        "fig3": fign[2],
        "fig4": fign[3],
        "fign": fign,
    }

    # sv("junk", [st, st1, label])
    # exec(compile(open("lt2plot3.py").read(), "lt2plot3.py", "exec"))

    xlabel1 = label["xlabel1"]
    ylabel1 = label["ylabel1"]
    xlabel2 = label["xlabel2"]
    ylabel2 = label["ylabel2"]
    xlabel4 = label["xlabel4"]
    ylabel4 = label["ylabel4"]
    xlabel5 = label["xlabel5"]
    ylabel5 = label["ylabel5"]
    zlabel1 = label["zlabel1"]
    zlabel2 = label["zlabel2"]
    zlabel4 = label["zlabel4"]
    zlabel5 = label["zlabel5"]
    fig1 = label["fig1"]
    fig2 = label["fig2"]
    fig3 = label["fig3"]
    fig4 = label["fig4"]
    fign = label["fign"]

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import FormatStrFormatter

    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)

    fig = plt.figure(fig1)
    ax = fig.add_subplot(111, projection="3d")
    zmax = max(st[0].real)
    zmin = min(st[0].real)
    zmean = np.mean(st[0])
    print(" xp, zmax,zmin=", zmax, zmin)

    ax.scatter(st[2], st[3], st[0], c="r", marker="o")
    ztick = ax.get_zticks(minor=False)
    lbl = [("%0.03e" % a).split("e") for a in ztick]
    lbl2 = list(zip(*[[i[0], str(int(i[1]))] for i in lbl]))
    lbl3 = ["$ \\times 10^{" + str(tmp) + "}$" for tmp in lbl2[1]]
    lbl4 = list(zip(*[lbl2[0], lbl3]))
    lbl5 = [i[0] + i[1] for i in lbl4]
    # ax.set_zticklabels(lbl5,rotation=0,ha='center',size='xx-large')
    ax.set_xlabel("\n\n" + xlabel1, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel1, fontsize=20)
    ax.set_zlabel("                         \n\n\n\n" + zlabel1 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)
    ax.set_zlim3d(zmin * 0.0, zmax * 1.1)

    plt.title("Fig." + str(fign[0]))
    plt.savefig("junk" + str(fig1) + ".png")

    fig = plt.figure(fig2)
    ax = fig.add_subplot(111, projection="3d")

    zmax = max(st[1].real)
    zmin = min(st[1].real)
    zmean = np.mean(st1[1])

    print(" for yp,  zmax,zmin=", zmax, zmin)

    ax.scatter(st[2], st[3], st[1], c="r", marker="o")

    ax.set_xlabel("\n\n" + xlabel2, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel2, fontsize=20)
    ax.set_zlabel("                     \n\n\n\n" + zlabel2 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)
    ax.set_zlim3d(zmin * 0.0, zmax * 1.1)
    plt.title("Fig." + str(fign[1]))
    plt.savefig("junk" + str(fig2) + ".png")

    fig = plt.figure(fig3)
    ax = fig.add_subplot(111, projection="3d")

    zmax = max(st1[0].real)
    zmin = min(st1[0].real)
    zmean = np.mean(st1[0])
    print(" for x,  zmax,zmin=", zmax, zmin)
    ax.set_zlim3d(zmin * 0.0, zmax * 1.1)
    ax.scatter(st1[2], st1[3], st1[0], c="r", marker="o")
    ztick = ax.get_zticks(minor=False)
    lbl = [("%0.3e" % a).split("e") for a in ztick]
    lbl2 = list(zip(*[[i[0], str(int(i[1]))] for i in lbl]))
    lbl3 = ["$ \\times 10^{" + str(tmp) + "}$" for tmp in lbl2[1]]
    lbl4 = list(zip(*[lbl2[0], lbl3]))
    lbl5 = [i[0] + i[1] for i in lbl4]
    # ax.set_zticklabels(lbl5,rotation=0,ha='center',size='xx-large')
    # plt.gca().zaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_xlabel("\n\n" + xlabel4, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel4, fontsize=20)
    ax.set_zlabel("                     \n\n\n\n" + zlabel4 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)

    plt.title("Fig." + str(fign[2]))

    plt.savefig("junk" + str(fig3) + ".png")

    fig = plt.figure(fig4)
    ax = fig.add_subplot(111, projection="3d")

    zmax = max(st1[1].real)
    zmin = min(st1[1].real)
    zmean = np.mean(st1[1])

    print(" for y,  zmax,zmin=", zmax, zmin)

    ax.scatter(st1[2], st1[3], st1[1], c="r", marker="o")
    ax.set_xlabel("\n\n" + xlabel5, fontsize=20)
    ax.set_ylabel("\n\n" + ylabel5, fontsize=20)
    ax.set_zlabel("                     \n\n\n\n" + zlabel5 + "\n\n", fontsize=20)
    # ax.set_xlim3d(-4,4)
    # ax.set_ylim3d(-4,4)
    ax.set_zlim3d(zmin * 0.0, zmax * 1.1)
    plt.title("Fig." + str(fign[3]))

    plt.savefig("junk" + str(fig4) + ".png")


def plotthetavGrid(thetav10, thetav20, thetav11, thetav21, fign=(280, 281)):
    thetav10 = modularToPi(
        thetav10 + 1e-6
    )  # this is to make sure when it is too close to pi, then move to -pi
    thetav20 = modularToPi(thetav20 + 1e-6)
    plt.figure(fign[0])
    plt.plot(thetav10, thetav20, ".")
    plt.xlabel("thetav1")
    plt.ylabel("thetav2")
    plt.title(
        "Fig."
        + str(fign[0])
        + ' thetav10,thetav20 of eq.26,eq.27 \nof "latticenonlinearResonances.lyx" \n(initial phases, zero order action)'
    )
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1])
    plt.plot(thetav11, thetav21, ".")
    plt.xlabel("theta11")
    plt.ylabel("theta21")
    plt.title(
        "Fig."
        + str(fign[1])
        + ' thetav11,thetav21 of eq.26,eq.27 of \n"latticenonlinearResonances.lyx" \n(exact phases after one turn advance from initial phases)'
    )
    plt.savefig("junk" + str(fign[1]) + ".png")


def plotphivstheta(thetav10, thetav20, phi1, phi2, fign=(441, 442)):
    plt.figure(fign[0])
    plt.plot(thetav10, phi1.real, ".", label="Re(phi1)")
    plt.plot(thetav10, phi1.imag, ".", label="Im(phi1)")
    plt.plot(thetav10, phi2.real, ".", label="Re(phi2)")
    plt.plot(thetav10, phi2.imag, ".", label="Im(phi2)")
    plt.xlabel("thetav10")
    plt.legend(loc="best")
    plt.title("Fig." + str(fign[0]) + " thetav10 vs.phi")
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1])
    plt.plot(thetav20, phi1.real, ".", label="Re(phi1)")
    plt.plot(thetav20, phi1.imag, ".", label="Im(phi1)")
    plt.plot(thetav20, phi2.real, ".", label="Re(phi2)")
    plt.plot(thetav20, phi2.imag, ".", label="Im(phi2)")
    plt.xlabel("thetav20")
    plt.legend(loc="best")
    plt.title("Fig." + str(fign[1]) + " thetav20 vs.phi")
    plt.savefig("junk" + str(fign[1]) + ".png")


def plotscatterbnm(bnm1, bnm2, n_theta, cutoff12=(-7.5, -10.5), fign=(46, 47)):
    cutoff = cutoff12[0]
    tmp = np.array(
        [
            [k, n, np.log(abs(bnm1[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(bnm1[n, k]) > cutoff
        ]
    ).transpose()
    plt.figure(fign[0])
    plt.scatter(
        tmp[0],
        tmp[1],
        50,
        tmp[2],
        marker="s",
        cmap=cm.jet,
        vmin=cutoff,
        vmax=max(tmp[2]),
    )
    plt.xlabel("k")
    plt.ylabel("n")
    p = plt.colorbar()
    p.ax.set_ylabel("log|bnm1[k,n]|")
    plt.title("Fig." + str(fign[0]) + " distribution of log|bnm1[k,n]|>" + str(cutoff))
    plt.axis([-21, 21, -21, 21])
    # plt.gca().invert_yaxis()#This make the position of y same as row number of a matrix
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.savefig("junk" + str(fign[0]) + ".png")

    cutoff = cutoff12[1]
    tmp = np.array(
        [
            [k, n, np.log(abs(bnm2[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(bnm1[n, k]) > cutoff
        ]
    ).transpose()
    plt.figure(fign[1])
    plt.scatter(
        tmp[0],
        tmp[1],
        50,
        tmp[2],
        marker="s",
        cmap=cm.jet,
        vmin=cutoff,
        vmax=max(tmp[2]),
    )
    plt.xlabel("k")
    plt.ylabel("n")
    p = plt.colorbar()
    p.ax.set_ylabel("log|bnm2[k,n]|")
    plt.title("Fig." + str(fign[1]) + " distribution of log|bnm2[k,n]|>" + str(cutoff))
    plt.axis([-21, 21, -21, 21])
    # plt.gca().invert_yaxis()#This make the position of y same as row number of a matrix
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.savefig("junk" + str(fign[1]) + ".png")


def plot3DbnmSpectrum(aphi1, aphi2, n_theta, fign=(46, 47)):
    cutoff = -50  # -7.5
    tmp1 = np.array(
        [
            [k, n, np.log(abs(aphi1[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(aphi1[n, k]) > cutoff
        ]
    ).transpose()
    fig = plt.figure(fign[0])
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tmp1[0], tmp1[1], tmp1[2], c="r", marker="o")
    ax.set_xlabel("\n\nk", fontsize=20)
    ax.set_ylabel("\n\nn", fontsize=20)
    ax.set_zlabel("                    " + "ln|aphi1|" + "", fontsize=20)
    plt.title("Fig." + str(fign[0]))
    plt.savefig("junk" + str(fign[0]) + ".png")

    tmp2 = np.array(
        [
            [k, n, np.log(abs(aphi2[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(aphi2[n, k]) > cutoff
        ]
    ).transpose()
    fig1 = plt.figure(fign[1])
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(tmp2[0], tmp2[1], tmp2[2], c="r", marker="o")
    ax1.set_xlabel("\n\nk", fontsize=20)
    ax1.set_ylabel("\n\nn", fontsize=20)
    ax1.set_zlabel("                    " + "ln|aphi2|" + "", fontsize=20)
    plt.title("Fig." + str(fign[1]))
    plt.savefig("junk" + str(fign[1]) + ".png")


def plotAphiCouchyTheorem(
    bnm1, bnm2, n_theta, Cauchylimit, phiname=("phi1", "phi2"), fign=(462, 47)
):
    cutoff = -50
    tmp = np.array(
        [
            [k, n, np.log(abs(bnm1[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(bnm1[n, k]) > cutoff
        ]
    ).transpose()
    tmp1 = tmp.transpose()
    plt.figure(fign[0])
    plt.plot(tmp[1], tmp[2], ".", label="ln|" + phiname[0] + "(k,n)| vs. n")
    tmp = np.array(
        [
            [k, n, np.log(abs(bnm1[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(bnm1[n, k]) > cutoff
        ]
    ).transpose()
    tmp1 = tmp.transpose()
    plt.plot(tmp[0], tmp[2], ".", label="ln|" + phiname[0] + "(k,n)| vs. k")
    plt.savefig("junk" + str(fign[0]) + ".png")
    sv("junk", [bnm2, cutoff, n_theta])
    tmp = np.array(
        [
            [k, n, np.log(abs(bnm2[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(bnm1[n, k]) > cutoff
        ]
    ).transpose()
    tmp1 = tmp.transpose()
    plt.figure(fign[0])
    plt.plot(tmp[1], tmp[2], ".", label="ln|" + phiname[1] + "(k,n)| vs. n")
    tmp = np.array(
        [
            [k, n, np.log(abs(bnm2[k, n]))]
            for n in range(-n_theta // 2, n_theta // 2)
            for k in range(-n_theta // 2, n_theta // 2)
            if abs(bnm1[n, k]) > cutoff
        ]
    ).transpose()
    tmp1 = tmp.transpose()
    plt.plot(tmp[0], tmp[2], ".", label="ln|" + phiname[1] + "(k,n)| vs. k")
    plt.title("Fig." + str(fign[0]) + " Cauchy limit vs. k,n")
    a1, b1, a2, b2 = Cauchylimit["Cauchylim"]
    # idxm=min(42,n_theta)
    linefit1 = [-a1 * n + b1 for n in range(n_theta // 2)]
    linefit2 = [a1 * n + b1 for n in range(-n_theta // 2, 0)]
    plt.plot(linefit1, ".", markersize=12, label="fit y=-a1*n+b1, a1=" + str(a1))
    plt.plot(
        range(-n_theta // 2, 0),
        linefit2,
        ".",
        markersize=12,
        label="fit y=a1*n+b1,b1=" + str(b1),
    )
    plt.legend(loc="lower center")
    plt.savefig("junk" + str(fign[0]) + ".png")


def plotpeakbnm(fft2Dphi1peaks, fft2Dphi2peaks, fign=(461, 471)):
    # 20. plot fft2 spectrum of v1,v2
    # The following result for the case of xy040 has peaks in the last 3 terms
    # i,j=      -1,1       2,-2       1,-1,
    # i,j+1=    -1,2       2,-1       1, 0
    # nu=        2nuy-nux   2nux-nuy   nux
    # nu=        0.05        0.067     0.061
    tune, peakvaluesint2, peaksposition0int2, peaksposition1int2 = list(
        zip(*fft2Dphi2peaks)
    )
    # plt.figure(46,figsize=(3,3))
    fig, ax = plt.subplots(num=fign[0], figsize=(6, 5))
    # plt.scatter(peaksposition0int2[-20:],array(peaksposition1int2[-20:])+1, 100, map(abs,peakvaluesint2[-20:]),  vmin=0.0, vmax=np.max(map(abs,peakvaluesint2[-20:]))*1.2,cmap=cm.jet)
    # map1=ax.scatter(peaksposition0int2[-20:],array(peaksposition1int2[-20:])+1, 100, map(abs,peakvaluesint2[-20:]),\
    #                vmin=0.01, vmax=np.max(map(abs,peakvaluesint2[-20:]))*1.2,cmap=cm.jet,\
    #                norm=matplotlib.colors.LogNorm(0.1,1.2))
    map1 = ax.scatter(
        peaksposition0int2[-20:],
        array(peaksposition1int2[-20:]) + 1,
        100,
        list(map(abs, peakvaluesint2[-20:])),
        vmin=0.015,
        vmax=0.025,
        cmap=cm.jet,
        norm=matplotlib.colors.LogNorm(0.01, 0.03),
    )
    # plt.xlim(-3,4)
    ax.set_xlim(-6, 6)
    # plt.ylim(-3,4)
    ax.set_ylim(-5, 5)
    cbar = fig.colorbar(map1, ticks=np.arange(0.01, 0.03, 0.002))  # , shrink=0.4)
    # plt.colorbar()
    # plt.xlabel('n',fontsize=30)
    ax.set_xlabel("n", fontsize=30)
    ax.set_ylabel("m", fontsize=30)
    # plt.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.text(8, 8, "a", fontsize=30)

    plt.title(
        "Fig."
        + str(fign[0])
        + " plot 2D fft top 20 peaks of \nthe spectrum of integrated phi2"
    )
    plt.savefig("junk" + str(fign[0]) + ".png")

    tune, peakvaluesint1, peaksposition0int1, peaksposition1int1 = list(
        zip(*fft2Dphi1peaks)
    )
    # plt.figure(46,figsize=(3,3))
    fig, ax = plt.subplots(num=fign[1], figsize=(6, 3))
    # plt.scatter(peaksposition0int2[-20:],array(peaksposition1int2[-20:])+1, 100, list(map(abs,peakvaluesint2[-20:])),  vmin=0.0, vmax=np.max(list(map(abs,peakvaluesint2[-20:])))*1.2,cmap=cm.jet)
    """
        map1=ax.scatter(peaksposition0int2[-20:],array(peaksposition1int2[-20:])+1, 100, map(abs,peakvaluesint2[-20:]),\
                        vmin=0.01, vmax=np.max(map(abs,peakvaluesint2[-20:]))*1.2,cmap=cm.jet,\
                        norm=matplotlib.colors.LogNorm(0.1,1.2))
        """
    map1 = ax.scatter(
        array(peaksposition0int1[-20:]) + 1,
        array(peaksposition1int1[-20:]),
        100,
        list(map(abs, peakvaluesint1[-20:])),
        vmin=0.001,
        vmax=0.025,
        cmap=cm.jet,
        norm=matplotlib.colors.LogNorm(1e-3, 0.025),
    )
    # plt.xlim(-3,4)
    ax.set_xlim(-10, 10)
    # plt.ylim(-3,4)
    ax.set_ylim(-5, 5)
    # cbar=fig.colorbar(map1,ticks=np.arange(0.01,0.03,0.002))#, shrink=0.4)
    cbar = fig.colorbar(map1)  # , shrink=0.4)
    # plt.colorbar()
    # plt.xlabel('n',fontsize=30)
    ax.set_xlabel("n", fontsize=30)
    ax.set_ylabel("m", fontsize=30)
    # plt.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.text(8, 8, "a", fontsize=30)

    plt.title(
        "Fig."
        + str(fign[1])
        + " plot 2D fft top 20 peaks of \nthe spectrum of integrated phi1"
    )
    plt.savefig("junk" + str(fign[1]) + ".png")
    return


def plot2Dxytracking(xy, bKi, xfix, xpfix, fign=(311, 312)):
    zbar = np.dot(bKi[1:5, 1:5], xy)

    zx, zxs, zy, zys = zbar
    plt.figure(fign[0])
    plt.plot(zx.real, zx.imag, "g.", zy.real, zy.imag, "r.")
    plt.plot(zx[0].real, zx[0].imag, "bo", zy[0].real, zy[0].imag, "ro")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.title("Fig." + str(fign[0]) + " zx,zxs,zy,zys without xfix,xpfix")
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1])
    plt.plot(xy[0] + xfix, xy[1] + xpfix, "b.")
    plt.plot(xy[2], xy[3], "r.")
    # plt.axes().set_aspect('equal')
    plt.title(
        "Fig."
        + str(fign[1])
        + " x,xp (blue),y,yp (red), obtained from tracking trajectory"
    )
    plt.savefig("junk" + str(fign[1]) + ".png")
    return zx, zxs, zy, zys


def plot2DvnTracking(vn, fign=(24, 23)):
    plt.figure(fign[0], figsize=(5, 5.1))
    plt.plot(vn[1].real, vn[1].imag, "r.", label="v2 from tracking")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    # plt.axis([-1.4,1.4,-1.4,1.4])
    plt.xlabel("Re" + r"$(v_2)$", fontsize=35, labelpad=15)
    plt.ylabel("Im" + r"$(v_2)$", fontsize=35, labelpad=-15)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.title("Fig." + str(fign[0]) + "  vn[1]: v2 based on tracking trajectory")
    plt.savefig("junk" + str(fign[0]) + ".png")

    plt.figure(fign[1], figsize=(5, 5.1))
    plt.plot(vn[0].real, vn[0].imag, "r.", label="v1 from tracking")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    # plt.axis([-1.4,1.4,-1.4,1.4])
    plt.xlabel("Re" + r"$(v_1)$", fontsize=35, labelpad=11)
    plt.ylabel("Im" + r"$(v_1)$", fontsize=35, labelpad=-11)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.title("Fig." + str(fign[0]) + "  vn[0]: v1 based on tracking trajectory")
    plt.savefig("junk" + str(fign[0]) + ".png")


def plot3DzxyvnTracking(xy, bKi, vn, fign=(101, 102, 103, 104)):
    zbar = np.dot(bKi[1:5, 1:5], xy)
    zx, zxs, zy, zys = zbar
    stlabel = "abs(zx);abs(zy);list(map(cmath.phase, zx));list(map(cmath.phase,zy))".split(
        ";"
    )
    st1label = "abs(vn[0]);abs(vn[1]);list(map(cmath.phase, vn[0]));list(map(cmath.phase,vn[1]))".split(
        ";"
    )
    st, st1 = list(map(eval, stlabel)), list(map(eval, st1label))
    plot3d(st, st1, stlabel, st1label, fign=fign)


def plot3DphivzTheta(thetav10, thetav20, phi1, phi2, fign=(91, 92, 93, 94)):
    thetav10 = modularToPi(
        thetav10 + 1e-6
    )  # this is to make sure when it is too close to pi, then move to -pi
    thetav20 = modularToPi(thetav20 + 1e-6)
    stlabel = "phi1.real,phi1.imag,thetav10,thetav20".split(",")
    st1label = "phi2.real,phi2.imag,thetav10,thetav20".split(",")
    st = list(map(eval, stlabel))
    st1 = list(map(eval, st1label))
    plot3d(st, st1, stlabel, st1label, fign=fign)


def plot2DV12xyd0Grid(xyd0, vna, Zpar, fign=(23, 24), label="v10 from xyd0 scan"):
    bKi, scalex, norder, powerindex = Zpar
    tmp = Zszaray(xyd0, Zpar)
    vtmp = np.dot(vna, tmp.transpose())
    plt.figure(fign[0])
    plt.plot(vtmp[0].real, vtmp[0].imag, ".", markersize=9, label="v1 " + label)
    plt.title(
        "Fig."
        + str(fign[0])
        + ' thetav10,thetav20 of eq.26,eq.27 \nof "latticenonlinearResonances.lyx" \n(initial phases, zero order action)'
    )
    plt.legend(loc="best")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.savefig("junk" + str(fign[0]) + ".png")
    plt.figure(fign[1])
    plt.plot(vtmp[1].real, vtmp[1].imag, ".", markersize=9, label="v2 " + label)
    plt.title(
        "Fig."
        + str(fign[1])
        + ' thetav10,thetav20 of eq.26,eq.27 \nof "latticenonlinearResonances.lyx" \n(initial phases, zero order action)'
    )
    plt.legend(loc="best")
    plt.axis("equal")
    # plt.axes().set_aspect("equal")
    plt.savefig("junk" + str(fign[1]) + ".png")
    return vtmp

