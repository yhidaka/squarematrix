# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import copy
import time

import numpy as np
from numpy import array
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from . import jnfdefinition as jfdf

# jfdf = jnfdefinition

from .fortran import ftm

display = ""

# taylor multplication
def tm(f, g, powerindex, norder):
    #!!!taylor multplication: taylor series of the product of function f and g!!!
    # d=[ [0,i[1]] for i in a]
    a = [f, powerindex]
    anonzero = jfdf.findnonzeros(f)
    a = zip(*a)
    b = [g, powerindex]
    bnonzero = jfdf.findnonzeros(g)
    b = zip(*b)
    c = np.zeros(len(powerindex)) * (1.0 + 0j)
    for (
        i
    ) in (
        anonzero
    ):  # only mutiply and add nonzero terms, so choose anonzero and bnonzero
        for j in bnonzero:
            if sum(a[i][1] + b[j][1]) < norder + 1:
                index = a[i][1] + b[j][1]  # add powers.
                ie = index[
                    3
                ]  # Calculate the position of this term in the sequence using the method in "countingTerms.nb"
                iy = (
                    index[2] + ie
                )  # The sequence is arranged as x^(ix-ip)*xp^(ip-iy)*y^(iy-ie)*yp^ie
                ip = index[1] + iy
                ix = index[0] + ip
                sequencenumber = (
                    ie
                    + iy * (iy + 1) / 2
                    + ip * (ip + 1) * (ip + 2) / 6
                    + ix * (ix + 1) * (ix + 2) * (ix + 3) / 24
                )
                c[sequencenumber] = (
                    c[sequencenumber] + a[i][0] * b[j][0]
                )  # multiply coefficients
    return c


def mfunction(m, norder):
    """generate submatrix matrix function mf for map m up to norder
    which is an array with 4 rows of tps representing x,xp,y,yp in terms of x0,xp0,y0,yp0"""
    pv, x, od = m.dump(0)
    pv, xp, od = m.dump(1)
    pv, y, od = m.dump(2)
    pv, yp, od = m.dump(3)
    # pv is the base like:
    # ['0000',
    # '1000',
    # '0100',
    # '0010',
    # '0001',
    # '2000',
    # '1100',
    # '1010',

    # This separates the string of base into individual letters like
    # [['0', '0', '0', '0'],
    # ['1', '0', '0', '0'],
    # ['0', '1', '0', '0'],
    # ['0', '0', '1', '0'],
    # ['0', '0', '0', '1'],
    # ['2', '0', '0', '0'],
    # ['1', '1', '0', '0'],
    # ['1', '0', '1', '0'],
    # ['1', '0', '0', '1']...:

    tmp1 = [list(i) for i in pv]
    # This converts all strings in tmp1 into integer like
    # [[0, 0, 0, 0],
    # [1, 0, 0, 0],
    # [0, 1, 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1],
    # [2, 0, 0, 0],
    # [1, 1, 0, 0],...:
    powerindex = [
        np.array(map(eval, i)) for i in tmp1
    ]  # powetindex(i)+1 is equal to orderp[i+1] in my mathematica program "polynomialMultiplication"
    # taylorCoeff=zip(*[vs,powerindex]) #each taylorCoeff gives a term in taylor exapnsion with its coefficient and its power index
    return [x, xp, y, yp], powerindex


def squarematrix(mf, norder, powerindex, sequencenumber, tol):
    """generate square matrix from the m function mf up to norder"""
    # First define the Taylor expansion of x,xp,y,yp
    ln = len(powerindex)
    x = mf[0]
    xp = mf[1]
    y = mf[2]
    yp = mf[3]
    # To save cpu time, construct a table of [1,x,x^2,x^3], [1,xp,xp^2,xp^3],..., to be used for construction of the map matrix.
    # First start from [1,x], then build [1,x,x^2], then iterate to "norder".
    xn = [np.zeros(ln), x]
    xn[0][0] = 1
    xpn = [np.zeros(ln), xp]
    xpn[0][0] = 1
    yn = [np.zeros(ln), y]
    yn[0][0] = 1
    ypn = [np.zeros(ln), yp]
    ypn[0][0] = 1
    for i in range(2, norder + 1):
        xn.append(ftm.ftm(xn[-1], x, powerindex, sequencenumber, tol))
    for i in range(2, norder + 1):
        xpn.append(ftm.ftm(xpn[-1], xp, powerindex, sequencenumber, tol))
    for i in range(2, norder + 1):
        yn.append(ftm.ftm(yn[-1], y, powerindex, sequencenumber, tol))
    for i in range(2, norder + 1):
        ypn.append(ftm.ftm(ypn[-1], yp, powerindex, sequencenumber, tol))
    u = [xn, xpn, yn, ypn]
    t0 = time.perf_counter()
    # Use the table built above, i.e., xn,xnp,.. to construct the square matrix.
    matrixm = []
    for i in range(0, ln):
        nz = jfdf.findnonzeros(
            powerindex[i]
        )  # nz is the list of the indexes of the nonzero power terms
        # Now, for this powerindex[i] multiply the power of 4 functions x,xp,y,yp according to their power
        # u[nz[j]] is the relevant term, powerindex[i][nz[j]] is its power
        if len(nz) == 0:
            row = np.zeros(ln)
            row[0] = 1.0
        else:
            row = u[nz[0]][powerindex[i][nz[0]]]
            if len(nz) > 1:
                for j in range(1, len(nz)):
                    row = ftm.ftm(
                        row,
                        u[nz[j]][powerindex[i][nz[j]]],
                        powerindex,
                        sequencenumber,
                        tol,
                    )
        matrixm.append(row)
    # print time.perf_counter()-t0 ,"seconds for rows"
    t0 = time.perf_counter()

    # matrixm=array(matrixm)[1:,1:]
    return array(matrixm)


def extracttwiss(mf):
    # Derive twiss function from mf
    a = [i[1:3] for i in mf[:2]]
    cosphi = (a[0][0] + a[1][1]) / 2
    phix0 = np.arccos(cosphi)
    alphax0 = (a[0][0] - a[1][1]) / 2 / np.sin(phix0)
    betax0 = a[0][1] / np.sin(phix0)
    a = [i[3:5] for i in mf[2:4]]
    cosphi = (a[0][0] + a[1][1]) / 2
    phiy0 = np.arccos(cosphi)
    alphay0 = (a[0][0] - a[1][1]) / 2 / np.sin(phiy0)
    betay0 = a[0][1] / np.sin(phiy0)
    return betax0, phix0, alphax0, betay0, phiy0, alphay0


def BKmatrix(
    betax0,
    phix0,
    alphax0,
    betay0,
    phiy0,
    alphay0,
    x0,
    xp0,
    norder,
    powerindex,
    sequencenumber,
    tol,
):

    # build map matrix for B in S.Y.Lee eq.(2.43), see "so" in the section "Generate square matrices" in "xynormal_form_2rd_order-upper-doagonal-Jordan.nb"
    # bf is the tailor function coefficients for B. There are 4 rows for x,xp,y,yp, and each row has 35 columns:(constant, x0,xp0,y0, yp0, x0^2,...)
    # but we first fill only 4 by 5
    ln = len(powerindex)
    bf = array(
        [
            [1.0, 0, 0, 0, 0],
            [x0, np.sqrt(betax0), 0, 0, 0],
            [xp0, -alphax0 / np.sqrt(betax0), 1.0 / np.sqrt(betax0), 0, 0],
            [0, 0, 0, np.sqrt(betay0), 0],
            [0, 0, 0, -alphay0 / np.sqrt(betay0), 1.0 / np.sqrt(betay0)],
        ],
        dtype=complex,
    )

    # build map matrix for K. see my notes "Relation to Normal Form", Wednesday, March 30, 2011 1:59 PM, section "definition"
    Kf = array(
        [
            [1.0, 0, 0, 0, 0],
            [0, 1.0 / 2, 1.0 / 2, 0, 0],
            [0, 1j / 2, -1j / 2, 0, 0],
            [0, 0, 0, 1.0 / 2, 1.0 / 2],
            [0, 0, 0, 1j / 2, -1j / 2],
        ],
        dtype=complex,
    )

    # build the linear part of the square matrix of BK, then fill in the non-linear part with zeros for the first 5 rows(because BK is linear).
    bK = np.dot(bf, Kf)
    bKi = linalg.inv(bK)
    tmp = np.zeros((5, ln - 5)) * (1.0 + 0.0j)
    bK = np.hstack((bK, tmp))
    bKi = np.hstack((bKi, tmp))
    # Construct the BK square matrix using the first 5 rows.
    bK = squarematrix(bK[1:5], norder, powerindex, sequencenumber, tol)
    bKi = squarematrix(bKi[1:5], norder, powerindex, sequencenumber, tol)
    return array(bK), array(bKi)


def Zcol(zx, zxs, zy, zys, norder, powerindex):
    """generate a variable column up to norder"""
    ln = len(powerindex)
    # To save cpu time, construct a table of [1,x,x^2,x^3], [1,xp,xp^2,xp^3],..., to be used for construction of the map matrix.
    # First start from [1,x], then build [1,x,x^2], then iterate to "norder".
    zxn = [1, zx]
    zxsn = [1, zxs]
    zyn = [1, zy]
    zysn = [1, zys]

    for i in range(2, norder + 1):
        zxn.append(zxn[-1] * zx)
    for i in range(2, norder + 1):
        zxsn.append(zxsn[-1] * zxs)
    for i in range(2, norder + 1):
        zyn.append(zyn[-1] * zy)
    for i in range(2, norder + 1):
        zysn.append(zysn[-1] * zys)
    u = [zxn, zxsn, zyn, zysn]
    # t0=time.perf_counter()
    # Use the table built above, i.e., zxn,zxnp,.. to construct the square matrix.
    Z = []
    for i in range(0, ln):
        nz = jfdf.findnonzeros(
            powerindex[i]
        )  # nz is the list of the indexes of the nonzero power terms
        # Now, for this powerindex[i] multiply the power of 4 functions x,xp,y,yp according to their power
        # u[nz[j]] is the relevant term, powerindex[i][nz[j]] is its power
        if len(nz) == 0:
            row = 1.0 + 0j
        else:
            row = u[nz[0]][powerindex[i][nz[0]]]
            if len(nz) > 1:
                for j in range(1, len(nz)):
                    row = row * u[nz[j]][powerindex[i][nz[j]]]

        Z.append(row)
    # print time.perf_counter()-t0 ,"seconds for rows"
    # t0=time.perf_counter()

    # matrixm=array(matrixm)[1:,1:]
    return array(Z)


def Zpcol(zx, zxs, zy, zys, norder, powerindex):
    """generate a variable column up to norder with its first derivatives Z,Zz, Zzs"""
    ln = len(powerindex)
    # To save cpu time, construct a table of [1,x,x^2,x^3], [1,xp,xp^2,xp^3],..., to be used for construction of the map matrix.
    # First start from [1,x], then build [1,x,x^2], then iterate to "norder".
    zxn = [1, zx]
    zxsn = [1, zxs]
    zyn = [1, zy]
    zysn = [1, zys]

    for i in range(2, norder + 1):
        zxn.append(zxn[-1] * zx)
    for i in range(2, norder + 1):
        zxsn.append(zxsn[-1] * zxs)
    for i in range(2, norder + 1):
        zyn.append(zyn[-1] * zy)
    for i in range(2, norder + 1):
        zysn.append(zysn[-1] * zys)
    dzxndz = [i * zxn[i - 1] for i in range(1, len(zxn))]
    dzxndz.insert(0, 0)
    dzxsndzs = [i * zxsn[i - 1] for i in range(1, len(zxsn))]
    dzxsndzs.insert(0, 0)
    u = [zxn, zxsn, zyn, zysn]
    uz = [dzxndz, zxsn, zyn, zysn]
    uzs = [zxn, dzxsndzs, zyn, zysn]
    # t0=time.perf_counter()
    # Use the table built above, i.e., zxn,zxnp,.. to construct the square matrix.
    Z = []
    Zz = []
    Zzs = []
    for i in range(0, ln):
        nz = jfdf.findnonzeros(
            powerindex[i]
        )  # nz is the list of the indexes of the nonzero power terms
        # Now, for this powerindex[i] multiply the power of 4 functions zx,zxs,zy,zys according to their power
        # u[nz[j]] is the relevant term, powerindex[i][nz[j]] is its power
        row = 1.0 + 0j
        rowz = 1.0 + 0j
        rowzs = 1.0 + 0j
        for j in range(4):
            row = row * u[j][powerindex[i][j]]
            rowz = rowz * uz[j][powerindex[i][j]]
            rowzs = rowzs * uzs[j][powerindex[i][j]]

        Z.append(row)
        Zz.append(rowz)
        Zzs.append(rowzs)
    # print time.perf_counter()-t0 ,"seconds for rows"
    # t0=time.perf_counter()
    # matrixm=array(matrixm)[1:,1:]
    return array(Z), array(Zz), array(Zzs)


def contourplot(
    fun,
    xlim=[-0, 5, 0.5, 0.1],
    ylim=[-0.5, 0.5, 0.1],
    levels=np.arange(0, 4, 0.2),
    aspect1="False",
    xlabel="x (m)",
    ylabel="x'",
    ttl="Jordan form abs(b0) contour and tracking",
    cl="r",
    ls="solid",
):
    matplotlib.rcParams["xtick.direction"] = "out"
    matplotlib.rcParams["ytick.direction"] = "out"
    # print "xlim,ylim",xlim,ylim
    x = np.arange(xlim[0], xlim[1], xlim[2])
    y = np.arange(ylim[0], ylim[1], ylim[2])
    X, Y = np.meshgrid(x, y)
    # generate Z function height given the grid specified by X,Y
    # X,Y are the coordinates of the grid points,
    # X and Y have same array structure, rows are evenly spaced in y direction, columns for x
    Z = fun(X, Y)

    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    # plt.figure()
    CS = plt.contour(X, Y, Z, levels, colors=cl, linewidths=4, linestyles=ls)
    # CS = plt.contour(X, Y, Z, levels,colors=('r', 'green', 'blue', (1,1,0), '#afeeee', '0.2'))
    # CS = plt.contour(X, Y, Z, levels,colors='k')
    plt.clabel(CS, inline=1, fontsize=10)
    print("title=", ttl)
    plt.title(ttl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if aspect1 == "True":
        plt.axes("equal")
    # if aspect1=='True': plt.axes().set_aspect('equal', 'datalim')
    plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    # plt.show()
    return


def invutm(V):
    # inverse an upper triangular matrix
    # local{Vm, tmp, y, Y, len},
    Vm = []
    ln = len(V)
    for k in range(ln):
        # (*Solve the k'th column of V^-1*)
        ek = np.zeros(ln)
        ek[k] = 1
        # (*clear variable values for y[j]*)
        # Do[If[! MatchQ[y[j], y[_]], y[j] =.;], {j, 1, len}];
        # Y = Array[y, len];
        # lhs = V.Y - ek;
        y = np.zeros(ln)
        y[k] = ek[k] / V[k, k]
        j = k
        if display == "inver":
            print("k=", k, " j=", j, ", y[", j, "]=", y[j], ", Y=")
            jfdf.pra(y, ln)
        j = k - 1
        while j >= 0:
            # (*Solve for the j'th row of V.Y=ek*)
            if display == "inver":
                print(
                    "k=",
                    k,
                    " j=",
                    j,
                    ", np.dot(V[j,j+1:k],y[j+1:k])=",
                    np.dot(V[j, j + 1 : k + 1], y[j + 1 : k + 1]),
                )
                print(
                    "V[j,j+1]=",
                    V[j, j + 1],
                    " V[j,j+1:k]=",
                    V[j, j + 1 : k + 1],
                    " y[j+1:k]=",
                    y[j + 1 : k + 1],
                )

            y[j] = -np.dot(V[j, j + 1 : k + 1], y[j + 1 : k + 1]) / V[j, j]
            if display == "inver":
                print("k=", k, " j=", j, ", y[", j, "]=", y[j], ", Y=")
                jfdf.pra(y, ln)
            j = j - 1

        # (*Append the column k of V^-1,
        # as a row of the transpose of V^-1*)
        Vm.append(copy.copy(y))

    return np.transpose(array(Vm))


def UlnJ(mn, mu):
    print("Check u as left eigenvectors of m: u.m=J.u")
    u, J = jfdf.leftjordanbasis(mn)
    lhs = np.dot(u, mn)
    rhs = np.dot(J, u)
    tmp = lhs - rhs
    print("abs(u.mn-J.u).max()=", abs(tmp).max())
    print("J=")
    jfdf.pim(J + 1e-9, len(lhs), len(lhs))
    v, lnJ, lnJM, vm = jfdf.vlnJ(J, mu)
    print("lnJM=")
    jfdf.prm(lnJM, len(lnJ), len(lnJ))
    print("lnJ=")
    jfdf.pim(lnJ, len(lnJ), len(lnJ))
    tmp = np.dot(v, np.dot(lnJM, vm)) - J
    print("abs(v.lnJM.vm-J).max()=", abs(tmp).max())
    maxchainlenposition, maxchainlen = jfdf.findchainposition(J)
    print("position of max length chain=", maxchainlenposition)
    print("max length chain=", maxchainlen)
    # U is the eigen basis of the Jordan form of log(M)
    # B=U.Z where b0 at the position of max length Jordan chain is the invariant.
    U = np.dot(v, u)
    return U, maxchainlenposition


# plot3D is not working yet.
def plot3D(fun, xlim=[-0, 5, 0.5, 0.1], ylim=[-0.5, 0.5, 0.1]):

    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(xlim[0], xlim[1], xlim[2])
    y = np.arange(ylim[0], ylim[1], ylim[2])
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=None,linewidth=0, antialiased=False)
    ax.set_zlim(0, 0.4)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return


def powerindex4(n):
    sq = [
        [ix - ip, ip - iy, iy - ie, ie]
        for ix in range(n + 1)
        for ip in range(ix + 1)
        for iy in range(ip + 1)
        for ie in range(iy + 1)
    ]
    return array(sq)


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
    # print "scalemf=", scalex1
    As = np.identity(mlen)
    for i in range(mlen):
        As[i, i] = scalem1 ** sum(powerindex[i])
    Asm = np.identity(mlen)
    for i in range(mlen):
        Asm[i, i] = scalex1 ** sum(powerindex[i])
    mfs = scalem1 * np.dot(mf, Asm)
    return array(mfs), scalex1, As, Asm


def b0z(zbar, zbars, scalex, norder, powerindex, u):
    # given real space x,xp calculate twiss space z=xbar-1j*pbar, then calculate normal form w
    # that is the nonlinear normalized space.
    zsbar = zbar / scalex
    zsbars = zbars / scalex
    Zs = Zcol(
        zsbar, zsbars, 0, 0, norder, powerindex
    )  # Zs is the zsbar,zsbars, column, here zsbar=zbar*scalex
    W = np.dot(
        u[0], Zs
    )  # w is the invariant we denoted as b0 before. Zs is scaled Z, Z is column of zbar, zbars
    return W
