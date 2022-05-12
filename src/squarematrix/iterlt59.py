# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time
import copy
from copy import deepcopy
import pickle
from types import SimpleNamespace
import pdb
from pathlib import Path

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

t0 = [["start", time.time(), 0]]
timing = [["start", time.time(), 0]]

from . import jnfdefinition as jfdf
from . import iterModules59 as vph  # iterModules52 as vph
from . import lte2tpsa2map45 as lte2tpsa
from . import (
    veq59 as veq,
)  # copied from veq52.py. veqnsls2sr_supercell_ch77_20150406_1_deltap0_lt11 as veq#veq20140204_bare_1supcell_deltapm02 as veq
from .fortran import zcol, zcolnew

# DEBUG = True
DEBUG = False

t0.append(["41", time.time(), time.time() - t0[-1][1]])

"""
1.change ltefilename="20140204_bare_1supcell" in  lt40.py
2.check the lte file name
3.Choose uselte2madx=1 in veq.gettpsa(ltefilename,nv=nv,norder=norder,uselte2madx=0) \
        to read lte till fort.18 is generates, otherwise choose 0
4.choose running sequence in iterxacoeffnew(), in particular, acoeff0, xmax, ymax in runbnm in iterxacoeffnew().
5.choose n_theta=16#40,  alphadiv=16#40
6.choose uarray=np.array([ux[0],ux[1],ux[2],uy[0]]) in veq. by setting up veq.uarray here
5.And, change acoeff0=array([[ 1+0j, 0j,0j,0j],[0j, 0j,   0j,1+0j]])
7.change nvar in veq=4 by veq.nvar here
8.make sure MA0, and W1 in tracking_dp_ltefilename.ele used in veq is consistent the MA0 and W1 in the *.lte file (not MA1, or W0 for example)
9.make sure the ring start from ring=(MA0,W1,...)
10. to scan, use: arecord=runm012_usexacoeff0(xlist=np.arange(-35e-3,36e-3,5e-3)+1e-6,ylist=np.arange(0,16e-3,5e-3)+1e-4,n_theta=8)
11 to plot contours of scan, use: frecontour(arecord,-4.4,0,0.1,vlabel='bnm1max')
"""

print(time.time() - timing[-1][1], "seconds for import sqdf and jfdf")
timing.append(["67", time.time(), time.time() - timing[-1][1]])


def sv(filename, x):
    ff = open(filename, "wb")
    pickle.dump(x, ff)
    ff.close()


def rl(filename):
    ff = open(filename, "rb")
    xx = pickle.load(ff, encoding="latin1")
    ff.close()
    return xx


def sva(filename, x):
    ff = open(filename, "a")
    pickle.dump(x, ff)
    ff.close()


def rla(filename):
    ff = open(filename)
    objs = []
    while 1:
        try:
            objs.append(pickle.load(ff))
        except EOFError:
            break
    #
    ff.close()
    return objs


t0.append(["97", time.time(), time.time() - t0[-1][1]])


def x2ari(
    x, *args
):  # start from only 4 complex variables as the coefficients of wy0,wy1 for v1 and v2 respectively
    # sv('junk',[x,args])
    xa = copy.deepcopy(x)
    ln = len(xa)
    aria, nv = args
    # aria=copy.deepcopy(args)#aria is the matrix with row 1,2 represent v1 and v2
    xa = array([xa])
    x1 = array([xa]).reshape(
        [ln // 2, 2]
    )  # change x such that the left and right column are real part and imaginary part of a number
    x2 = x1[:, 0] + 1j * x1[:, 1]
    x2 = x2.reshape(
        [2, ln // 4]
    )  # make x2 a two row matrix, so row 1 and row 2 represent v1 and v2
    aria = np.hstack([x2, np.zeros([2, nv - ln // 4])])
    return aria


def ari2x(ari):
    aria = ari.copy()
    ln = len(ari[0])
    aria = aria.reshape([2 * ln])
    x = array([list(map(np.real, aria)), list(map(np.imag, aria))])
    x = array(list(zip(*x)))
    x = x.reshape(4 * ln)
    return x


def Zszaray(xy, Zpar):  # Construct ZsZsp columns from each turn in the tracking result.
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], xy) / scalex
    xxpyyp = [zx, zxs, zy, zys]
    xxpyyp = list(zip(*xxpyyp))
    Zs = zcolnew.zcolarray(xxpyyp, norder, powerindex)
    # Zs = zcolarray.zcolarray(xxpyyp, norder, powerindex)
    #
    return Zs


def Zsz(xs, Zpar):
    x, xp, y, yp = xs
    bKi, scalex, norder, powerindex = Zpar
    zx, zxs, zy, zys = np.dot(bKi[1:5, 1:5], array([x, xp, y, yp])) / scalex
    # Zs=sqdf.Zcol(zx,zxs,zy,zys,norder,powerindex) #Zxs is the zsbar,zsbars, column, here zsbar=zbar/scalex
    Zs = zcol.zcol([zx, zxs, zy, zys], norder, powerindex)
    return Zs


def latticeparamters(lattice, oneturntpsa="ELEGANT"):
    if DEBUG:
        tt0 = [["latticep, 1, start", time.time(), 0]]
        print(tt0)

    mapMatrixMlist = "Ms10,phix0,phiy0,powerindex,norder,bK,bKi,sqrtbetax,sqrtbetay,msf,tbl,scalemf,deltap,xfix,xpfix,dmuxy".split(
        ","
    )
    mfm, sqmxparameters, oneturntpsa, filepath_d = veq.gettpsa(
        lattice.ltefilename,
        lattice.deltap,
        nv=lattice.nv,
        norder=lattice.norder,
        norder_jordan=lattice.norder_jordan,
        usecode=lattice.usecode,
        oneturntpsa="ELEGANT",
        mod_prop_dict_list=lattice.mod_prop_dict_list,
    )
    if DEBUG:
        tt1 = [["latticep, 2", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
        timing.append(
            [
                " in latticeparamters, 2 after gettpsa",
                time.time(),
                timing[-1][1] - timing[-2][1],
            ]
        )

    if DEBUG:
        tt1 = [["latticep, 3", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    (
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
    ) = veq.getMlist(
        lattice.ltefilename,
        lattice.deltap,
        mfm,
        sqmxparameters,
        lattice.norder_jordan,
        mapMatrixMlist,
        usecode=lattice.usecode,
    )
    if DEBUG:
        tt1 = [["latticep, 4", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
        timing.append(
            [
                " in latticeparamters, 3 after getMlist",
                time.time(),
                timing[-1][1] - timing[-2][1],
            ]
        )
    maxchainlenpositionx, maxchainlenx, chainx, chainpositionx = jfdf.findchainposition(
        Jx
    )
    maxchainlenpositiony, maxchainleny, chainy, chainpositiony = jfdf.findchainposition(
        Jy
    )
    if DEBUG:
        timing.append(
            [
                " in latticeparamters, 4 before mapMatrixM.",
                time.time(),
                timing[-1][1] - timing[-2][1],
            ]
        )
    dmuxy = mapMatrixM["dmuxy"]
    if DEBUG:
        print(
            '\n6. mapMatrixM["dmuxy"]=',
            mapMatrixM["dmuxy"],
            "maxchainlenx=",
            maxchainlenx,
            "maxchainleny=",
            maxchainleny,
            "\n",
        )
    """
    if norder < 5:
        ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3 = (
            ux,
            uy,
            ux[1],
            uy[1],
            ux[2],
            uy[2],
            0,
            0,
        )
    else:
        ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3 = (
            ux[0],
            uy[0],
            ux[1],
            uy[1],
            ux[2],
            uy[2],
            ux[3],
            uy[3],
        )
    """
    for updatelattice in [0]:
        lattice.ux = ux
        lattice.uy = uy
        # lattice.dmuxy = dmuxy
        lattice.deltap = deltap
        lattice.bKi = bKi
        lattice.scalex = scalex
        lattice.norder = norder
        lattice.powerindex = powerindex
        lattice.xfix = xfix
        lattice.xpfix = xpfix

    if DEBUG:
        timing.append(
            [" in latticeparamters, 6 end.", time.time(), timing[-1][1] - timing[-2][1]]
        )
        tt1 = [["latticep, 5", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    return lattice, oneturntpsa, filepath_d


def setup(lattice, oneturntpsa="ELEGANT"):
    timing.append([" in setup, 1", time.time(), timing[-1][1] - timing[-2][1]])

    lattice, oneturntpsa, filepath_d = latticeparamters(lattice, oneturntpsa="ELEGANT")

    if DEBUG:
        timing.append(
            [
                " in setup, 2 after latticeparamters",
                time.time(),
                timing[-1][1] - timing[-2][1],
            ]
        )

    uarray = np.array([lattice.ux[0], lattice.uy[0]])
    lattice.uarray = uarray
    # xacoeff0 = ari2x(lattice.acoeff0)
    return lattice, oneturntpsa, filepath_d


def trackingvn(xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass, lattice):
    tmp = xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass

    xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass = tmp

    xy, nuxt, nuyt = veq.xtracking(
        xmax, ymax - 0e-6, npass, lattice.n_theta, deltap, xfix, xpfix, om1accurate=0
    )  # om1accurate)
    Zse = Zszaray(xy, Zpar)
    vn = np.dot(vna, Zse.transpose())
    return vn, xy, Zse, nuxt, nuyt


t0.append(["222. before bnm", time.time(), time.time() - t0[-1][1]])


def shortarecord(arecordi):
    lattice = arecordi["lattice"]

    print("in shorta,1, xllim=", lattice.xllim)
    # arecordi=arecord[0]
    # arecordi,uarray,acoeff0,nvar=arecord

    ai = SimpleNamespace(**arecordi)
    Zpar = ai.iorecord["inversevinput"]["Zpar"]
    bKi = ai.iorecord["inversevinput"]["bKi"]
    timing.append(["2. plot bnm", time.time()])
    # flucbnm,v120i,acoeff,v0norm,n_theta,xyd0,vna,v12n,xyd,flucful,iorecord,uarray,acoeff0,nvar,xmax,ymax,iklist=[arecordi[i] for i in arecordi.keys()]
    # x0d,xp0d,y0d,yp0d,x1d,xp1d,y1d,yp1d,phi1,phi2,av10,av20,thetav10,thetav20,v10,v20,v11,v21,thetav11,thetav21=[flucful[i] for i in flucful.keys()]

    # flucful=SimpleNamespace(**ai.flucful)
    flucbnm = ai.flucbnm
    bnm1, bnm2, nux, nuy, th10, th20 = [
        ai.v12n[i] for i in "bnm1,bnm2,nux,nuy,th10,th20".split(",")
    ]
    v120i = ai.v120i
    # aphi1,aphi2,bnm1,bnm2,v1n,v2n,vthetascanmsg,F,fv1a,fv2a,v1a,v2a,nux,nuy,acoeffnew,th10,th20,Zsd=[ v12n[i] for i in v12n.keys()]
    vna = ai.vna
    xmax, ymax, xacoeff = [flucbnm[i] for i in ["xmax", "ymax", "xacoeff"]]
    npass = 1024  # lattice.npass
    vn, xy, Zse, nuxt, nuyt = trackingvn(
        xmax,
        ymax,
        vna,
        Zpar,
        lattice.deltap,
        lattice.xfix,
        lattice.xpfix,
        npass,
        lattice,
    )
    vph.plot2Dxytracking(xy, bKi, lattice.xfix, lattice.xpfix, fign=(311, 312))

    xd, xpd, yd, ypd = ai.xyd
    n_theta = ai.n_theta
    if "xyd0" in dir(ai):
        x0d, xp0d, y0d, yp0d = ai.xyd0
    else:
        x0d, xp0d, y0d, yp0d = ai.xydlast
    vph.plot2Dcrossection(
        x0d,
        xp0d,
        y0d,
        yp0d,
        xy,
        lattice.xfix,
        lattice.xpfix,
        n_theta,
        xlim=(-20e-3, 20e-3),
        ylim=(-5e-3, 5e-3),
        fign=(72, 73),
    )
    vph.plot2Dcrossection(
        xd,
        xpd,
        yd,
        ypd,
        xy,
        lattice.xfix,
        lattice.xpfix,
        n_theta,
        xlim=(-20e-3, 20e-3),
        ylim=(-5e-3, 5e-3),
        fign=(721, 731),
        lbl=("xd,xpd", "yd,ypd"),
    )

    fft2Dphi1peaks, fft2Dphi2peaks = vph.findphipeaks(
        bnm1, bnm2, n_theta, nux, nuy, th10, th20, v120i, npeaks=20
    )
    vscannumerror = []

    # if 'fv1a' in list(ai.v12n.keys()):
    if "xacoeffnew" not in list(arecordi.keys()):
        # import pdb; pdb.set_trace()
        print("in shorta,2, xllim=", lattice.xllim)
        acoeff, uarray, v0norm = ai.acoeff, ai.uarray, ai.v0norm
        Zsd = Zszaray(ai.xyd, Zpar)
        F, v1a, v2a, fv1a, fv2a = vph.Fmatrix(
            Zsd, n_theta, acoeff, lattice.nvar, v0norm, uarray
        )
        aphi1, aphi2, v1n, v2n, nux, nuy = [
            ai.v12n[i] for i in "aphi1,aphi2,v1n,v2n,nux,nuy".split(",")
        ]
        v10, v20 = ai.flucful["v10"], ai.flucful["v20"]
        vph.plotvnv0v1(
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
            [],
            Zsd,
            uarray,
            n_theta,
            lattice.xllim,
            lattice.xulim,
            lattice.yllim,
            lattice.yulim,
            fign=(125, 121, 126, 127),
            xmax=xmax,
            ymax=ymax,
        )
        #        n_theta,xllim,xulim,yllim,yulim, fign=(125,121,126,127))
    else:
        acoeff, uarray = ai.acoeff, ai.uarray
        fv1a, fv2a, v1a, v2a, v1n, v2n, nux, nuy, Zsd, F, acoeffnew = [
            ai.v12n[i]
            for i in "fv1a,fv2a,v1a,v2a,v1n,v2n,nux,nuy,Zsd,F,acoeffnew".split(",")
        ]
        nvar = ai.nvar
        v10, v20 = ai.flucful["v10"], ai.flucful["v20"]
        vph.plotvnv0v1(
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
            vscannumerror,
            Zsd,
            uarray,
            n_theta,
            lattice.xllim,
            lattice.xulim,
            lattice.yllim,
            lattice.yulim,
            fign=(125, 121, 126, 127),
            xmax=xmax,
            ymax=ymax,
        )
        vph.plotfvnew(acoeffnew, nux, nuy, F, nvar, n_theta, fign=(126, 127))

    return fft2Dphi1peaks, fft2Dphi2peaks, xmax, ymax, xy, nux, nuy, nuxt, nuyt


def showarecord(arecordi):
    lattice = arecordi["lattice"]

    arecordi = arecordi
    ai = SimpleNamespace(**arecordi)
    Zpar = ai.iorecord["inversevinput"]["Zpar"]
    bKi = ai.iorecord["inversevinput"]["bKi"]
    timing.append(["2. plot bnm", time.time()])
    # flucbnm,v120i,acoeff,v0norm,n_theta,xyd0,vna,v12n,xyd,flucful,iorecord,uarray,acoeff0,nvar,xmax,ymax,iklist=[arecordi[i] for i in arecordi.keys()]
    # x0d,xp0d,y0d,yp0d,x1d,xp1d,y1d,yp1d,phi1,phi2,av10,av20,thetav10,thetav20,v10,v20,v11,v21,thetav11,thetav21=[flucful[i] for i in flucful.keys()]

    # flucful=SimpleNamespace(**ai.flucful)
    flucbnm = ai.flucbnm
    bnm1, bnm2, nux, nuy, th10, th20 = [
        ai.v12n[i] for i in "bnm1,bnm2,nux,nuy,th10,th20".split(",")
    ]
    v120i = ai.v120i
    # aphi1,aphi2,bnm1,bnm2,v1n,v2n,vthetascanmsg,F,fv1a,fv2a,v1a,v2a,nux,nuy,acoeffnew,th10,th20,Zsd=[ v12n[i] for i in v12n.keys()]
    vna = ai.vna
    xmax, ymax, xacoeff = [flucbnm[i] for i in ["xmax", "ymax", "xacoeff"]]
    vn, xy, Zse, nuxt, nuyt = trackingvn(
        xmax,
        ymax,
        vna,
        Zpar,
        lattice.deltap,
        lattice.xfix,
        lattice.xpfix,
        lattice.npass,
        lattice,
    )
    vph.plot2Dxytracking(xy, bKi, lattice.xfix, lattice.xpfix, fign=(311, 312))

    """
        timing.append(["2. plot bnm",time.time()])
        xacoeffnew,flucbnm,v120i,acoeff,v0norm,n_theta,xyd0,vna,v12n,xyd,flucful,iorecord,uarray,acoeff0,nvar,xmax,ymax,iklist=[arecordi[i] for i in arecordi.keys()]
        x0d,xp0d,y0d,yp0d,x1d,xp1d,y1d,yp1d,phi1,phi2,av10,av20,thetav10,thetav20,v10,v20,v11,v21,thetav11,thetav21=[flucful[i] for i in flucful.keys()]
        aphi1,aphi2,bnm1,bnm2,v1n,v2n,vthetascanmsg,F,fv1a,fv2a,v1a,v2a,nux,nuy,acoeffnew,th10,th20,Zsd=[ v12n[i] for i in v12n.keys()]
        xmax, ymax, xacoeffnew, xacoeff=[ flucbnm[i] for i in ['xmax', 'ymax', 'xacoeffnew', 'xacoeff']]
        #numerror,xydrecordvthetascan,
        #print ('xyd[2,7]=',xyd[2,7])
        fluc=phi1,phi2,thetav10[0],thetav20[0]
        #vna,thetav10,thetav20,phi1,phi2,fv1a,fv2a,nux,nuy,v120i,acoeff,acoeffnew,F,numerror,xyd0,xydrecordvthetascan,v1a,v2a,fv1a,fv2a,xyd,thetav11,thetav21=tmp
        """

    acoeff, uarray = ai.acoeff, ai.uarray
    nvar = ai.nvar
    thetav10, thetav20, phi1, phi2, v10, v20, thetav11, thetav21 = [
        ai.flucful[i]
        for i in "thetav10,thetav20,phi1,phi2,v10,v20,thetav11,thetav21".split(",")
    ]

    if "xacoeffnew" in list(arecordi.keys()):
        aphi1, aphi2, fv1a, fv2a, v1a, v2a, v1n, v2n, nux, nuy, Zsd, F, acoeffnew = [
            ai.v12n[i]
            for i in "aphi1,aphi2,fv1a,fv2a,v1a,v2a,v1n,v2n,nux,nuy,Zsd,F,acoeffnew".split(
                ","
            )
        ]
    else:
        aphi1, aphi2, v1n, v2n, nux, nuy = [
            ai.v12n[i] for i in "aphi1,aphi2,v1n,v2n,nux,nuy".split(",")
        ]

    if "xyd0" in dir(ai):
        xydlast = ai.xyd0
    else:
        xydlast = ai.xydlast
    x0d, xp0d, y0d, yp0d = xydlast
    xyd = ai.xyd
    xd, xpd, yd, ypd = xyd

    n_theta = ai.n_theta

    vph.plotphivstheta(thetav10, thetav20, phi1, phi2, fign=(441, 442))

    timing.append(["3. tracking", time.time()])

    vph.plotfvn(
        acoeff,
        vn,
        v120i,
        lattice.deltap,
        lattice.xfix,
        lattice.xpfix,
        lattice.npass,
        lattice.xllim,
        lattice.xulim,
        lattice.yllim,
        lattice.yulim,
        fign=(25, 21, 26, 27),
    )

    timing.append(["4. plot v12n", time.time()])

    vph.plot2Dxytracking(xy, bKi, lattice.xfix, lattice.xpfix, fign=(311, 312))
    vph.plot2DvnTracking(vn, fign=(24, 23))
    vph.plot3DzxyvnTracking(xy, bKi, vn, fign=(101, 102, 103, 104))
    timing.append(["5. plot v12n", time.time()])

    # Zpar=bKi,scalex,norder,powerindex

    tmp55 = vph.plot2DV12xyd0Grid(xydlast, vna, Zpar, fign=(23, 24))
    vph.plotthetavGrid(thetav10, thetav20, thetav11, thetav21, fign=(280, 281))
    timing.append(["6. plot v12n", time.time()])
    vph.plotAphiCouchyTheorem(
        aphi1,
        aphi2,
        n_theta,
        lattice.Cauchylimit,
        phiname=("phi1", "phi2"),
        fign=(462, 47),
    )
    vph.plot3DbnmSpectrum(aphi1, aphi2, n_theta, fign=(146, 147))

    timing.append(["7. plot v12n", time.time()])

    vph.plotv1nv2nThetaVGrid(v1n, v2n, n_theta, fign=(251, 271, 29))
    vph.plot3Dxytrackingxyd(
        xy[0] + lattice.xfix,
        xy[1] + lattice.xpfix,
        xy[2],
        xy[3],
        x0d + lattice.xfix,
        xp0d + lattice.xpfix,
        y0d,
        yp0d,
        fign=[1, 2, 4, 5],
    )
    vph.plot2Dcrossection(
        x0d,
        xp0d,
        y0d,
        yp0d,
        xy,
        lattice.xfix,
        lattice.xpfix,
        n_theta,
        xlim=(-2e-3, 2e-3),
        ylim=(-3e-3, 3e-3),
        fign=(72, 73),
    )
    xd, xpd, yd, ypd = xyd
    vph.plot2Dcrossection(
        xd,
        xpd,
        yd,
        ypd,
        xy,
        lattice.xfix,
        lattice.xpfix,
        n_theta,
        xlim=(-2e-3, 2e-3),
        ylim=(-3e-3, 3e-3),
        fign=(721, 731),
        lbl=("xd,xpd", "yd,ypd"),
    )
    vph.plotxp0dAspectEqualGrid(x0d, xp0d, xylabel=("x0d", "xp0d"), fign=231)
    vph.plotxp0dAspectEqualGrid(y0d, yp0d, xylabel=("y0d", "yp0d"), fign=232)
    timing.append(["8. plot v12n", time.time()])

    vph.plot3DphivzTheta(thetav10, thetav20, phi1, phi2, fign=(91, 92, 93, 94))
    vph.plotscatterbnm(bnm1, bnm2, n_theta, cutoff12=(-7.5, -10.5), fign=(46, 47))

    fft2Dphi1peaks, fft2Dphi2peaks = vph.findphipeaks(
        bnm1, bnm2, n_theta, nux, nuy, th10, th20, v120i, npeaks=20
    )
    vph.plotpeakbnm(
        fft2Dphi1peaks[-5:], fft2Dphi2peaks[-5:], fign=(461, 471)
    )  # fig.461,fig.471

    timing.append(["9. plot v12n", time.time()])

    if "xacoeffnew" not in list(arecordi.keys()):
        # import pdb; pdb.set_trace()
        print("in shorta,2, xllim=", lattice.xllim)
        acoeff, uarray, v0norm = ai.acoeff, ai.uarray, ai.v0norm
        Zsd = Zszaray(ai.xyd, Zpar)
        F, v1a, v2a, fv1a, fv2a = vph.Fmatrix(
            Zsd, n_theta, acoeff, nvar, v0norm, uarray
        )
        aphi1, aphi2, v1n, v2n, nux, nuy = [
            ai.v12n[i] for i in "aphi1,aphi2,v1n,v2n,nux,nuy".split(",")
        ]
        v10, v20 = ai.flucful["v10"], ai.flucful["v20"]
        vph.plotvnv0v1(
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
            [],
            Zsd,
            uarray,
            n_theta,
            lattice.xllim,
            lattice.xulim,
            lattice.yllim,
            lattice.yulim,
            fign=(125, 121, 126, 127),
        )
        # vph.plotfftpeaks(fft2Dphi1peaks,fft2Dphi2peaks,v10,v20,v1n,v2n,vn,nux,nuy,\
        #        n_theta,xllim,xulim,yllim,yulim, fign=(25,21,26,27))
        return (
            vn,
            xy,
            fft2Dphi1peaks,
            fft2Dphi2peaks,
            nux,
            nuy,
            nuxt,
            nuyt,
            bnm1,
            bnm2,
            acoeff,
            v0norm,
            xmax,
            ymax,
        )
    else:
        vscannumerror = []
        vph.plotvnv0v1(
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
            vscannumerror,
            Zsd,
            uarray,
            n_theta,
            lattice.xllim,
            lattice.xulim,
            lattice.yllim,
            lattice.yulim,
            fign=(125, 121, 126, 127),
        )
        vph.plotfvnew(acoeffnew, nux, nuy, F, nvar, n_theta, fign=(126, 127))
        vph.plotpeaktspectrum(
            fv1a,
            fv2a,
            nux,
            nuy,
            v120i,
            n_theta,
            lattice.xllim,
            lattice.xulim,
            lattice.yllim,
            lattice.yulim,
            fign=(25, 21, 26, 27),
        )
        return (
            vn,
            xy,
            fft2Dphi1peaks,
            fft2Dphi2peaks,
            nux,
            nuy,
            nuxt,
            nuyt,
            F,
            bnm1,
            bnm2,
            acoeff,
            lattice.v0norm,
            xmax,
            ymax,
        )


# ❇️outfindsolist,vthetascanrecordlist,...,latticelist=
for ik in []:  # [0]:
    outfindsolist = "warnflag,fval,nfev,niter,xsol,nfindso".split(",")
    vthetascanrecordlist = "theta01,theta02,xsol,vtarget,xtrial,vtrial,numberofpointstaken,outfindso,finalnfindso,conversionmsg".split(
        ","
    )
    vthetascanmsglist = "maxerror,numberdivgence,nfevmean,vthetascanrecord".split(",")
    # flucfullist = "x0d,xp0d,y0d,yp0d,x1d,xp1d,y1d,yp1d,phi1,phi2,av10,av20,thetav10,thetav20,v10,v20,v11,v21,thetav11,thetav21".split(
    # ","
    # )
    # vscaninputlist = "acoeff,Vmtx,dVdXmtx,xydlast,uarray,vna,v120i,n_theta,xi,usexp0dlast,v0norm,bKi,scalex,norder,powerindex,Zpar,findsoparamters".split(
    # ","
    # )
    # inversevinputlist = "acoeff,Vmtx,dVdXmtx,uarray,vna,v1n,v2n,n_theta,xi,xyd0,v0norm,bKi,scalex,norder,powerindex,Zpar,findsoparamters".split(
    # ","
    # )
    # v12nlist = "aphi1,aphi2,bnm1,bnm2,v1n,v2n,vthetascanmsg,F,fv1a,fv2a,v1a,v2a,nux,nuy,acoeffnew,th10,th20,Zsd".split(
    # ","
    # )
    # flucbnmlist = "xmax,ymax,xacoeffnew,xacoeff,np.amax(abs(bnm1)),np.amax(abs(bnm2)),np.std(bnm1),np.std(bnm2)".split(
    # ","
    # )
    # iorecordlist = "vscaninput,vthetascanmsg,inversevinput,inversev1v2msg".split(",")
    # bnmoutlist = "xacoeffnew,flucbnm,v120i,acoeff,v0norm,n_theta,xyd0,vna,v12n,xyd,flucful,iorecord,uarray,acoeff0,nvar,xmax,ymax".split(
    # ","
    # )
    # renewXinputlist = "xacoeffnew,xmax,ymax,xydlast,uarray,acoeff0,nvar,n_theta,uvar,Zpar,findsoparamters,usexp0dlast".split(
    # ","
    # )
    iterxydinputlist = (
        "xacoeffnew,xmax,ymax,xydlast,uarray,acoeff0,nvar,n_theta,uvar,Zpar,findsoparamters,usexp0dlast"
    ).split(",")
    v12niterxydlist = "aphi1,aphi2,bnm1,bnm2,v1n,v2n,nux,nuy,th10,th20".split(",")
    fluciterxydlist = "xmax,ymax,xyd,xacoeff,np.amax(abs(bnm1)),np.amax(abs(bnm2)),np.std(bnm1),np.std(bnm2)".split(
        ","
    )
    iorecorditerxydlist = "inversevinput,inversev1v2msg".split(",")
    iterxydoutlist = "flucbnm,v120i,acoeff,v0norm,n_theta,xydlast,vna,v12n,xyd,flucful,iorecord,uarray,acoeff0,nvar,xmax,ymax".split(
        ","
    )

    latticesetuplist = "ux0,uy0,ux1,uy1,ux2,uy2,ux3,uy3,dmuxy,deltap,bKi,scalex,\
        norder,powerindex,xfix,xpfix".replace(
        " ", ""
    ).split(
        ","
    )

    latticelist = "ux0,uy0,ux1,uy1,ux2,uy2,ux3,uy3,dmuxy,deltap,bKi,scalex,norder,powerindex,\
        xfix,xpfix,ltefilename,usecode,nv,n_theta,alphadiv,ntune,yllim,yulim,xllim,xulim,nvar,acoeff0,npass,\
        uselastxyd,v0norm,a1,b1,a2,b2,norder_jordan,uarray,uvar,Cauchylimit,xacoeff0,acoeff".replace(
        " ", ""
    ).split(
        ","
    )

    lattice = _get_default_lattice_namespace()


def _get_default_lattice_namespace():

    return SimpleNamespace(
        ltefilename="20140204_bare_1supcell",
        usecode=dict(tpsacode="tracy", use_existing_tpsa=1),
        nv=4,
        norder=7,
        deltap=-0.02,
        n_theta=int(16 / 4) * 4,
        # alphadiv = int(16 / 4) * 4,
        ntune=4,
        yllim=0,
        yulim=0.1,
        xllim=-0.51,
        xulim=0.51,
        nvar=2,
        acoeff0=array([[1 + 0j, 0j], [0j, 1 + 0j]]),
        npass=800,
        uselastxyd="never",
        #'no inversev'#'use xydlast for vthetascan trial for ik>0'#'use xydlast after only one vthetascan'#'never',#
        v0norm=(1, 1),
        Cauchylimit={
            "Cauchylim": [20 / 30, 4, 20 / 30, 4],  # "Cauchylim": [a1, b1,a2,b2]
            "aliasingCutoff": 8,
        },
        mod_prop_dict_list=[],
        # mod_prop_dict_list=[
        #    {"elem_name": "Qh1G2c30a", "prop_name": "K1", "prop_val": 1.5},
        #    {"elem_name": "sH1g2C30A", "prop_name": "K2", "prop_val": 0.0},
        # ],
    )


# previous the function bnmnew3 now is called renewX
def renewX(
    renewXinput,
    lattice,
    oneturntpsa="ELEGANT",
    useinversetpsa=False,
    wwJwinv=None,
    usewinv=0,
    applyCauchylimit=False,
):
    if DEBUG:
        tt0 = [["in bnm 1, start", time.time(), 0]]
        print("tt0=", tt0)

    (
        xacoeff,
        xmax,
        ymax,
        xydlast,
        uarray,
        acoeff0,
        nvar,
        n_theta,
        uvar,
        Zpar3,
        findsoparamters,
        usexp0dlast,
    ) = [renewXinput[i] for i in renewXinput.keys()]

    if DEBUG:
        tt1 = [["in bnm,1.1", time.time(), time.time() - tt0[0][1]]]
        print("tt1=", tt1)

    # define action angle variables v1,v2 by matrices##################
    for construct_v12_dv12_wwJ_matrices in [1]:
        xi = xmax - lattice.xfix, 0, ymax, 0
        acoeff = x2ari(xacoeff, acoeff0, nvar)
        vna = np.dot(acoeff, uarray)
        zs0 = Zsz(xi, Zpar3)
        v0norm = np.dot(vna, zs0)
        vna[0] = vna[0] / v0norm[0]  # vna is normalized with v0norm
        vna[1] = vna[1] / v0norm[1]
        v120i = np.dot(vna, zs0)
        if DEBUG:
            tt1 = [["in bnm,1.2", time.time(), time.time() - tt1[0][1]]]
            print("tt1=", tt1)

        # if useinversetpsa:
        if True:
            a1, a2 = acoeff
            wmtx, wJmtx = wwJwinv[:2]
            v1mtx = np.dot(a1 / v0norm[0], wmtx)
            v2mtx = np.dot(a2 / v0norm[1], wmtx)
            Vmtx = v1mtx, v2mtx
            dv1dXmtx = np.dot(
                a1 / v0norm[0], wJmtx.transpose([1, 0, 2])
            )  # This is dv1/dx
            dv2dXmtx = np.dot(
                a2 / v0norm[1], wJmtx.transpose([1, 0, 2])
            )  # This is dv1/dx
            dVdXmtx = [dv1dXmtx, dv2dXmtx]
        else:
            Vmtx, dVdXmtx = None, None
    if DEBUG:
        tt1 = [["in bnm,1.3", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)

    # v_theta scan ####################################################
    for run_vthetascan in [1]:
        # print("in renewX,acoeff=", acoeff, 'eval("acoeff")=', eval("acoeff"))
        vscaninput = dict(
            acoeff=acoeff,
            Vmtx=Vmtx,
            dVdXmtx=dVdXmtx,
            xydlast=xydlast,
            uarray=uarray,
            vna=vna,
            v120i=v120i,
            n_theta=n_theta,
            xi=xi,
            usexp0dlast=usexp0dlast,
            v0norm=v0norm,
            bKi=lattice.bKi,
            scalex=lattice.scalex,
            norder=lattice.norder,
            powerindex=lattice.powerindex,
            Zpar=Zpar3,
            findsoparamters=findsoparamters,
        )
        # decide whether to use vthetascan or not:
        if DEBUG:
            tt1 = [["in bnm,1.4", time.time(), time.time() - tt1[0][1]]]
            print("tt1=", tt1)
        # vthetascan #################################################
        if usexp0dlast == 0:
            # vthetascan #################################################
            if DEBUG:
                tt1 = [["in bnm,1.5", time.time(), time.time() - tt1[0][1]]]
                print("tt1=", tt1)
            xyd0, vthetascanmsg = vph.vthetascan_use_x0dlast3(
                acoeff, veq.veq1, vscaninput, wwJwinv[2:]
            )
        else:
            if DEBUG:
                tt1 = [["in bnm,1.6", time.time(), time.time() - tt1[0][1]]]
                print("tt1=", tt1)
            xyd0 = xydlast
            vthetascanmsg = dict(msg="use xyd as xyd0", numberdivgence=0)

    if DEBUG:
        tt1 = [["in bnm,2 vthetascan", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)

    ## vphi ################################################
    for run_vphi_and_get_fluc in [1]:
        (
            x0d,
            xp0d,
            y0d,
            yp0d,
            x1d,
            xp1d,
            y1d,
            yp1d,
            phi1,
            phi2,
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
        ) = veq.vphi(xyd0, acoeff, uvar, v0norm, oneturntpsa, Vmtx, outxv=1)
        if DEBUG:
            tt1 = [["in bnm, 2.1 vphi", time.time(), time.time() - tt1[0][1]]]
            print("tt1=", tt1)
        flucful = dict(
            x0d=x0d,
            xp0d=xp0d,
            y0d=y0d,
            yp0d=yp0d,
            x1d=x1d,
            xp1d=xp1d,
            y1d=y1d,
            yp1d=yp1d,
            phi1=phi1,
            phi2=phi2,
            av10=av10,
            av20=av20,
            thetav10=thetav10,
            thetav20=thetav20,
            v10=v10,
            v20=v20,
            v11=v11,
            v21=v21,
            thetav11=thetav11,
            thetav21=thetav21,
        )
        fluc = phi1, phi2, thetav10[0], thetav20[0]
    if DEBUG:
        tt1 = [["in bnm, 2.4 ", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)
    # bnmphiCauchyCutoffWithIntrokam #################################################
    (
        aphi1,
        aphi2,
        om1,
        om2,
        nux,
        nuy,
        bnm1,
        bnm2,
        th10,
        th20,
    ) = vph.bnmphiCauchyCutoffWithIntrokam(
        fluc, n_theta, n_theta, lattice.Cauchylimit, applyCauchylimit, dt=1
    )
    if DEBUG:
        tt1 = [["in bnm, 2.5 bnmphiCauchy", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)

    ##################################################

    #  inversefftnew2 #################################################
    for inverse_fft in [1]:
        v1n, v2n = vph.inversefftnew2(
            bnm1, bnm2, n_theta, v120i
        )  # thetav10 is the angle of the rigid rotation v10, as the zeroth order approximation.
        if DEBUG:
            timing.append(
                ["3. inversefftnew2", time.time(), time.time() - timing[-1][1]]
            )
        inversevinput = dict(
            acoeff=acoeff,
            Vmtx=Vmtx,
            dVdXmtx=dVdXmtx,
            uarray=uarray,
            vna=vna,
            v1n=v1n,
            v2n=v2n,
            n_theta=n_theta,
            xi=xi,
            xyd0=xyd0,
            v0norm=v0norm,
            bKi=lattice.bKi,
            scalex=lattice.scalex,
            norder=lattice.norder,
            powerindex=lattice.powerindex,
            Zpar=Zpar3,
            findsoparamters=findsoparamters,
        )
        if DEBUG:
            tt1 = [
                [
                    "in bnm, 3, before inversev1v2,2",
                    time.time(),
                    time.time() - tt1[0][1],
                ]
            ]
            print("tt1=", tt1)
    if DEBUG:
        tt1 = [["in bnm 4 after inversev1v2", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)

    # inverse_v1v2 #################################################
    xyd, Zsdx, inversev1v2msg = vph.inversev1v2(veq.veq1, inversevinput, usexyd=1)
    if np.amax(abs(xyd)) > 1:
        return None

    if DEBUG:
        tt1 = [["in bnm 5 ", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)

    # Fmatrix #################################################
    F, v1a, v2a, fv1a, fv2a = vph.Fmatrix(
        Zsdx, wmtx, n_theta, acoeff, nvar, v0norm, uarray, Vmtx
    )
    if DEBUG:
        tt1 = [["in bnm 6 Fmatrix", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)
    # anew12 #################################################

    try:
        acoeffnew = vph.anew12(
            F, nux, nuy, nvar, n_theta, lattice.Cauchylimit, lattice.ntune
        )
    except Exception as err:
        if DEBUG:
            print(dir(err))
            print("after vph.anew12 in renewX")
        bnmout = None
        return bnmout

    if DEBUG:
        tt1 = [["in bnm 7 anew12", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)

    # prepare_bnmout #################################################
    for prepare_bnmout in [1]:
        xacoeffnew = ari2x(acoeffnew)
        bnm1tmp, bnm2tmp = bnm1.copy(), bnm2.copy()
        if DEBUG:
            tt1 = [["in bnm 8 ", time.time(), time.time() - tt1[0][1]]]
            print("tt1=", tt1)
        bnm1tmp[0, 0], bnm2tmp[0, 0] = 0, 0
        flucbnm = {
            "xmax": xmax,
            "ymax": ymax,
            "xacoeffnew": xacoeffnew,
            "xacoeff": xacoeff,
            "np.amax(abs(bnm1))": np.amax(abs(bnm1tmp)),
            "np.amax(abs(bnm2))": np.amax(abs(bnm2tmp)),
            "np.std(bnm1)": np.std(bnm1tmp),
            "np.std(bnm2)": np.std(bnm2tmp),
        }
        v12n = dict(
            aphi1=aphi1,
            aphi2=aphi2,
            bnm1=bnm1,
            bnm2=bnm2,
            v1n=v1n,
            v2n=v2n,
            vthetascanmsg=vthetascanmsg,
            F=F,
            fv1a=fv1a,
            fv2a=fv2a,
            v1a=v1a,
            v2a=v2a,
            nux=nux,
            nuy=nuy,
            acoeffnew=acoeffnew,
            th10=th10,
            th20=th20,
            Zsd=Zsdx,
        )
        iorecord = dict(
            vscaninput=vscaninput,
            vthetascanmsg=vthetascanmsg,
            inversevinput=inversevinput,
            inversev1v2msg=inversev1v2msg,
        )
        bnmout = dict(
            xacoeffnew=xacoeffnew,
            flucbnm=flucbnm,
            v120i=v120i,
            acoeff=acoeff,
            v0norm=v0norm,
            n_theta=n_theta,
            xyd0=xyd0,
            vna=vna,
            v12n=v12n,
            xyd=xyd,
            flucful=flucful,
            iorecord=iorecord,
            uarray=uarray,
            acoeff0=acoeff0,
            nvar=nvar,
            xmax=xmax,
            ymax=ymax,
        )
        # print ('in bnm:2 xyd[2,7]=',xyd[2,7])
        lattice.v0norm = v0norm
        bnmout["lattice"] = deepcopy(lattice)
    if DEBUG:
        tt1 = [["in bnm, 9 end ", time.time(), time.time() - tt1[0][1]]]
        print("tt1=", tt1)
        tt1 = [["in bnm, 10 end ", time.time(), time.time() - tt0[0][1]]]
        print("tt1=", tt1)
    return bnmout


def errorcount(arecord, ik):
    # import pdb; pdb.set_trace()
    if "v12n" not in arecord[ik].keys():
        return
    inversev1v2errors = arecord[ik]["iorecord"]["inversev1v2msg"]["numberdivgence"]
    if "vthetascanmsg" in arecord[ik]["iorecord"]:
        vthetaserror = arecord[ik]["iorecord"]["vthetascanmsg"]["numberdivgence"]
        return vthetaserror, inversev1v2errors
    else:
        return inversev1v2errors


def iterXs(
    arecord,
    xacoeffnew,
    xydlast,
    iterxydinput,
    xmax=-21e-3,
    ymax=1e-4,
    niter=8,
    oneturntpsa="ELEGANT",
    useinversetpsa=None,
    wwJwinv=None,
    averaging=1,
    number_of_iter_after_minimum=None,
    applyCauchylimit=False,
):
    tt0 = [["iterxacoeffnew, 1. start", time.time(), 0]]
    lattice = arecord[-1]["lattice"]
    renewXinput = deepcopy(iterxydinput)
    del renewXinput["iteration_parameters"]
    deltaxydtable = []
    ##########################################################
    for ik in range(niter):
        # xacoeffnew and xydlast are udated in every iteration

        # iterxydinput=dict(zip(iterxydinputlist,[xacoeffnew,xmax,ymax,xydlast,uarray,acoeff0,nvar,n_theta,uvar,Zpar,findsoparamters,usexp0dlast]))
        if DEBUG:
            tt1 = [["iterxacoeffnew,, 2, ", time.time(), time.time() - tt0[0][1]]]
            print(tt1)
        ##########################################################
        try:
            if DEBUG:
                print("in iterXs, aiter=,ik=", ik, "xmax=", xmax, "ymax=", ymax)
            # renew xacoeffnew and xyd for each iteration
            renewXinput["xacoeffnew"] = xacoeffnew
            renewXinput["xydlast"] = xydlast
            ##########################################################
            bnmout = renewX(
                renewXinput,
                lattice,
                oneturntpsa=oneturntpsa,
                useinversetpsa=useinversetpsa,
                wwJwinv=wwJwinv,
                applyCauchylimit=applyCauchylimit,
            )
            if bnmout == None:
                raise Exception("Sorry, no numbers below zero")
                pdb.set_trace()

            bnmout["ik"] = ik
            lattice.v0norm = bnmout["v0norm"]
            xacoeffnew = bnmout["xacoeffnew"]
            arecord.append(bnmout)
            arecord[-1]["lattice"] = deepcopy(lattice)
            # deepcopy makes the copy immutable from change of original.
            arecord[-1]["deltaxydlist"] = [
                len(arecord),
                xmax,
                ymax,
                deltaxyd(arecord, len(arecord) - 1),
            ]
        except Exception as err:
            if DEBUG:
                print(dir(err))
                print(err.args)
            arecord.append({"ik": ik, "err.args": err.args, "xmax": xmax, "ymax": ymax})
        if DEBUG:
            tt1 = [["iterxacoeffnew,, 3, ", time.time(), time.time() - tt1[0][1]]]
            print(tt1)
        ##########################################################
        arecord, deltaxydtable, iterate = iterate_condition(
            arecord, deltaxydtable, number_of_iter_after_minimum
        )
        if iterate == 1:
            xacoeffnew = arecord[-1]["xacoeffnew"]
            n_theta = arecord[-1]["n_theta"]
            xydlast = averagingxyd(
                arecord[-1]["xyd"], n_theta=n_theta, averaging=averaging
            )
            arecord[-1]["iterxydinput"] = deepcopy(iterxydinput)
        else:
            return arecord
    return arecord


def iterate_condition(arecord, deltaxydtable, number_of_iter_after_minimum):
    if "deltaxydlist" in arecord[-1]:
        deltaxydlist = arecord[-1]["deltaxydlist"]
        deltaxydtable.append(deltaxydlist)
        if (
            deltaxydlist[-1] > 0
            or np.isnan(deltaxydlist[-1])
            or np.log(
                abs(arecord[-1]["iorecord"]["inversev1v2msg"]["outfindso"]["fval"])
            )
            > 0
        ):
            iterate = 0
            return arecord, deltaxydtable, iterate
    if len(deltaxydtable) > 2:
        deltaxyd = list(zip(*deltaxydtable))[-1]
        idx = np.argmin(deltaxyd)
        deltaxydmin = deltaxyd[idx]
        aftermin = deltaxyd[idx:]
        term_larger_later_than_min = [ik for ik in aftermin if ik > deltaxydmin]
        if len(term_larger_later_than_min) > number_of_iter_after_minimum:
            iterate = 0
            return arecord, deltaxydtable, iterate
    if "err.args" in arecord[-1].keys():
        """
        for j in range(ik + 1, niter):
            arecord.append(
                {
                    "ik": ik,
                    "err.args": "blow up in iterXs.",
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )
        """
        iterate = 0
        return arecord, deltaxydtable, iterate
    else:
        iterate = 1
    return arecord, deltaxydtable, iterate


def flucitershow(
    arecord,
    dfmax,
    invmax,
    xmax,
    yulim,
    fig=61,
    rng=(-0.0002e-3, 0.00002e-3, 0.2),
    userng=False,
):
    tmp = [
        [k]
        + [
            j["flucbnm"][i]
            for i in [
                "xmax",
                "ymax",
                "np.amax(abs(bnm1))",
                "np.amax(abs(bnm2))",
                "np.std(bnm1)",
                "np.std(bnm2)",
            ]
        ]
        + [j["ik"]]
        if "flucbnm" in j
        else [k, j["xmax"], np.nan, np.nan, np.nan, np.nan, np.nan, k]
        for k, j in enumerate(arecord)
    ]
    k, xlist1, ylist1, bnm1max, bnm2max, bnm1rms, bnm2rms, iklist = list(zip(*tmp))
    direction = np.sign(rng[1] - rng[0])
    # xset=np.sort(list(set(xlist1)))
    xlist2 = list(zip(*[xlist1, k]))
    xlist = [
        direction * 1e-8 * (xt[1] - len(xlist2) / 2.0 - 10) + xt[0] for xt in xlist2
    ]
    ylist2 = list(zip(*[ylist1, k]))
    ylist = [
        direction * 1e-8 * (xt[1] - len(ylist2) / 2.0 - 10) + xt[0] for xt in ylist2
    ]
    dx = abs(np.max(xlist) - np.min(xlist))
    dy = abs(np.max(ylist) - np.min(ylist))
    # import pdb; pdb.set_trace()
    if dx >= dy - 1e-6:
        xylist = xlist
    elif dy > dx:
        xylist = ylist
    else:
        xylist = xlist
    plt.figure(fig)
    plt.plot(xylist, bnm1rms, "o", markersize=12, label="bnm1rms")
    plt.plot(xylist, bnm2rms, "o", markersize=11, label="bnm2rms")
    plt.plot(xylist, bnm1max, "o", markersize=10, label="bnm1max")
    plt.plot(xylist, bnm2max, "o", markersize=9, label="bnm2max")
    plt.plot(xylist, dfmax, "mo", markersize=6, label="dfmax")
    plt.plot(xylist, invmax, "k.", markersize=5, label="invmax")
    plt.rc("xtick", labelsize=7)
    plt.xlabel("iteration direction is" + str(direction))
    if userng:
        plt.axis([rng[0], rng[1], 0, rng[2]])
    else:
        Dx = np.max(xylist) - np.min(xylist)
        plt.axis([np.min(xylist) - 0.1 * Dx, np.max(xylist) + 0.1 * Dx, 0, rng[2]])
    plt.title("Fig." + str(fig) + "rng=" + str(rng))
    plt.legend(loc="best")
    plt.tight_layout()

    for i in range(len(xylist)):
        plt.text(xylist[i], bnm1max[i] + yulim / 30.0, str(i), color="g", fontsize=8)
    plt.savefig("junk" + str(fig) + ".png")

    for ik in range(len(arecord)):
        if "nvar" in arecord[ik]:
            if dx > dy:
                print(
                    ik, "xmax=", arecord[ik]["xmax"], "nvar=", arecord[ik]["nvar"]
                )  # ,
            elif dy > dx:
                print(
                    ik, "ymax=", arecord[ik]["ymax"], "nvar=", arecord[ik]["nvar"]
                )  # ,
        else:
            print(ik, "xmax=", arecord[ik]["xmax"], "error")

    return xylist, k


# new action variable vb, see /Users/lihuayu/Desktop/nonlineardynamics/henonheiles/hnshort/hnshortlatex.lyx
def vbaction(arecordi):
    v12n = arecordi["v12n"]
    n_theta = arecordi["n_theta"]
    bnm1, bnm2, v1n, v2n = [v12n[i] for i in ["bnm1", "bnm2", "v1n", "v2n"]]
    # the following algorithm comes from /Users/lihuayu/Desktop/nonlineardynamics/henonheiles/henonHeles/hn56new.py line 1907-1919
    u = 0
    v = 0

    for i in range(-n_theta // 2, n_theta // 2):
        for j in range(-n_theta // 2, n_theta // 2):
            u = u - 1j * bnm1[i, j] * v1n**i * v2n**j
            v = v - 1j * bnm2[i, j] * v1n**i * v2n**j

    u = v1n * np.exp(u)
    v = v2n * np.exp(v)
    return (u, v)


def plotvb(arecord, xlist, ylim=0.2):
    if len(arecord[0]["v12n"]["bnm1"]) != arecord[0]["n_theta"]:
        arecord = arecord[1:]  # if so then arecord[0] is the result of
        # chaning n_theta, so v10 cannot be used as fluctuation reference, and hence excluded from the plot.
        xlist = xlist[1:]
    n_theta = arecord[0]["n_theta"]
    v10, v20 = (
        arecord[0]["flucful"]["v10"].reshape([n_theta, n_theta]),
        arecord[0]["flucful"]["v20"].reshape([n_theta, n_theta]),
    )
    vblist = [
        [np.std(vbaction(i)[0] - v10), np.std(vbaction(i)[1] - v20)]
        if "flucful" in i
        else [np.nan, np.nan]
        for i in arecord
    ]
    vblist = np.array(vblist)
    vblist = vblist.transpose()
    plt.figure(128)
    plt.plot(xlist, vblist[0], "o", label="RMS(vb[0])")
    plt.plot(xlist, vblist[1], "o", label="RMS(vb[1])")
    plt.axis([np.min(xlist), np.max(xlist), -0.0, ylim])
    plt.xlabel("x (m)", fontsize=14)
    plt.ylabel("RMS(vbar-v10)", fontsize=14)
    plt.legend(loc="best")
    plt.savefig("junk128.png")


def plotxacoeffnewconvergence(arecord, xc=-20e-3):
    tmp = [[i["xmax"], i["xacoeffnew"]] for i in arecord if "xacoeffnew" in i]
    tmp1 = [
        [tmp[j][0] + j * 1e-9, np.log(np.std(tmp[j][1] - tmp[j - 1][1]))]
        for j in range(1, len(tmp))
    ]
    tmp2 = list(zip(*tmp1))
    minf = np.min([i for i in tmp2[1] if i != -np.inf])
    tmp2 = [i if i[1] != -np.inf else [i[0], minf] for i in tmp1]
    tmp2 = list(zip(*tmp2))
    try:
        plt.figure(129)
        plt.plot(
            tmp2[0],
            tmp2[1],
            "o",
            label="ln(xacoeffnew[n]-xacoeffnew[n-1]) for x center at" + str(xc),
        )
        tmp3 = [i for i in tmp2[0] if abs(i - xc) < 1e-6]
        maxx = np.max(tmp3)
        minx = np.min(tmp3)
        plt.axis([minx, maxx, np.min(tmp2[1]) - 1, np.max(tmp2[1]) + 1])
        plt.xlabel(
            "xmax=-19mm + (number of iteration on both xacoeffnew and xyd times 1e-9) "
        )
        plt.ylabel("ln(xacoeffnew[n]-xacoeffnew[n-1]")
        plt.title(
            "delta xacoeffnew vs. iteration showing convergence,\nat x="
            + str(xc * 1e3)
            + "mm",
            fontsize=15,
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("junk129.png")
    except Exception as err:
        if DEBUG:
            print("error: plot nan. exit", dir(err))
            print(err.args)
    return


def plotxydconvergence(arecord, xc=-20e-3, codeused="tpsa", plot130=True):
    tmp = [
        [k]
        + [
            j["flucbnm"][i]
            for i in [
                "xmax",
                "ymax",
                "np.amax(abs(bnm1))",
                "np.amax(abs(bnm2))",
                "np.std(bnm1)",
                "np.std(bnm2)",
            ]
        ]
        + [j["ik"]]
        if "flucbnm" in j
        else [k, j["xmax"], j["ymax"], np.nan, np.nan, np.nan, np.nan, k]
        for k, j in enumerate(arecord)
        if "lattice" in j.keys()
    ]
    k, xlist1, ylist1, bnm1max, bnm2max, bnm1rms, bnm2rms, iklist = list(zip(*tmp))
    xlist2 = list(zip(*[xlist1, k]))
    xlist = [1e-8 * (xt[1] - len(xlist2) / 2.0 - 10) + xt[0] for xt in xlist2]
    ylist2 = list(zip(*[ylist1, k]))
    ylist = [1e-8 * (xt[1] - len(ylist2) / 2.0 - 10) + xt[0] for xt in ylist2]
    dx = abs(np.max(xlist) - np.min(xlist))
    dy = abs(np.max(ylist) - np.min(ylist))
    if dx >= dy or abs(dx - dy) < 1e-5:
        tmp = [
            [i["xmax"], i["xyd"]] for i in arecord if "err.args" not in i
        ]  # if 'xacoeffnew' in i]
    elif dy > dx:
        tmp = [
            [i["ymax"], i["xyd"]] for i in arecord if "err.args" not in i
        ]  # if 'xacoeffnew' in i]
    tmp1 = [
        [tmp[j][0] + j * 1e-9, np.log(np.std(tmp[j][1] - tmp[j - 1][1]))]
        for j in range(1, len(tmp))
    ]

    tmp12 = np.array([i for j, i in enumerate(tmp1) if i[1] != -np.inf])
    minf = np.min(tmp12[:, 1])
    tmp12 = [[i[0], minf] if i[1] == -np.inf else i for j, i in enumerate(tmp1)]
    tmp2 = np.array(list(zip(*tmp12)))
    tmp3 = np.where(abs(tmp2[0] - xc) < 1e-6)[0]
    iter_number = tmp2[0][tmp3]
    lndxy = tmp2[1][tmp3]
    if plot130:
        plt.figure(130)
        plt.plot(
            iter_number,
            lndxy,
            "o",
            label="x center at x="
            + str(xc * 1e3)
            + "mm,\n cutoff="
            + str(arecord[tmp3[0]]["lattice"].Cauchylimit["aliasingCutoff"])
            + ",n_theta="
            + str(str(arecord[tmp3[0]]["n_theta"]))
            + ",one turn code="
            + codeused,
        )
        plt.axis(
            [
                np.min(tmp2[0][tmp3]) - 2e-9,
                np.max(tmp2[0][tmp3]) + 2e-9,
                np.min(tmp2[1][tmp3]) - 1,
                np.max(tmp2[1][tmp3]) + 1,
            ]
        )
        plt.xlabel("(number of iteration on both xacoeffnew and xyd times 1e-9) ")
        plt.ylabel("ln(xyd[n]-xyd[n-1]")
        plt.title(
            "delta xyd vs. iteration showing convergence\n at x="
            + str(xc * 1e3)
            + "mm",
            fontsize=15,
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("junk130.png")
    return iter_number, lndxy


def plotiteratio(arecord, yulim=0.2):
    print("\nin runiterxyd, error count:")
    for ik in range(len(arecord)):
        print(ik, errorcount(arecord, ik))

    dfmax, invmax = studyiter4(arecord)
    tmp = [[k, i["xmax"], i["ymax"]] for k, i in enumerate(arecord)]
    k, xlist1, ylist1 = list(zip(*tmp))
    xlist2 = list(zip(*[xlist1, k]))
    xlist = [1e-8 * (xt[1] - len(xlist2) / 2.0 - 10) + xt[0] for xt in xlist2]
    ylist2 = list(zip(*[ylist1, k]))
    ylist = [1e-8 * (xt[1] - len(ylist2) / 2.0 - 10) + xt[0] for xt in ylist2]
    dx = abs(np.max(xlist) - np.min(xlist))
    dy = abs(np.max(ylist) - np.min(ylist))
    if dx >= dy - 1e-6:
        xylist = [
            i["xmax"] for i in arecord if "err.args" not in i
        ]  # if 'xacoeffnew' in i]
    elif dy > dx:
        xylist = [
            i["ymax"] for i in arecord if "err.args" not in i
        ]  # if 'xacoeffnew' in i]

    dxlist = (np.max(xylist) - np.min(xylist)) * 0.1
    xymax = xylist[-1]
    xylist, k = flucitershow(
        arecord,
        dfmax,
        invmax,
        -16e-3,
        yulim,
        fig=61,
        rng=(np.min(xylist) - dxlist, np.max(xylist) + dxlist + 1e-7, yulim),
        userng=False,
    )
    print("[k,xylist=")
    for i in list(zip(*[k, xylist])):
        print(i)
    # xc=xmax
    # plt.axis([xc-len(arecord)*1e-7,xc+len(arecord)*1e-7,0,ylim])
    plt.tight_layout()
    plt.savefig("junk61.png")
    try:
        plotvb(arecord, xylist, ylim=yulim)
        plotxydconvergence(arecord, xc=xylist[-1])
        plotxacoeffnewconvergence(arecord, xc=xylist[-1])
    except Exception as err:
        if DEBUG:
            print(dir(err))
            print(err.args)
    plt.show()
    return xlist, k


def deltaxyd(arecord, k):
    tmp1 = (
        [k - 1, arecord[k - 1]["xmax"], arecord[k - 1]["xyd"]]
        if "xacoeffnew" in arecord[k - 1]
        else [k - 1, arecord[k - 1]["xmax"], np.nan]
    )
    tmp2 = (
        [k, arecord[k]["xmax"], arecord[k]["xyd"]]
        if "xacoeffnew" in arecord[k]
        else [k, arecord[k]["xmax"], np.nan]
    )
    deltaxyd = (
        np.nan
        if np.isnan(tmp1[2]).any() or np.isnan(tmp2[2]).any()
        else np.log(np.std(tmp2[2] - tmp1[2]))
    )
    return deltaxyd


def choosepreviousxyd(
    arecord,
):  # generate list of delta xyd, then find those lines where xmax is close but not equal to the last xmax
    # sv('junk1',arecord)
    tmp = [[k, i["xmax"], i["ymax"]] for k, i in enumerate(arecord)]
    k, xlist1, ylist1 = list(zip(*tmp))
    xlist2 = list(zip(*[xlist1, k]))
    xlist = [1e-8 * (xt[1] - len(xlist2) / 2.0 - 10) + xt[0] for xt in xlist2]
    ylist2 = list(zip(*[ylist1, k]))
    ylist = [1e-8 * (xt[1] - len(ylist2) / 2.0 - 10) + xt[0] for xt in ylist2]
    dx = abs(np.max(xlist) - np.min(xlist))
    dy = abs(np.max(ylist) - np.min(ylist))
    if dx > dy:
        xymax = "xmax"
    elif dy > dx:
        xymax = "ymax"
    # , i.e., find previous xmax scan, then find which index gives minimum delta xyd.
    tmp = [
        i["deltaxydlist"] if "deltaxydlist" in i.keys() else [j + 1, i[xymax], np.nan]
        for j, i in enumerate(arecord)
    ]  # collect all deltaxyd data with  its index and xmax
    tmp1 = np.array(list(zip(*tmp)))[1] if dx > dy else np.array(list(zip(*tmp)))[2]
    tmp1 = [
        round(i, 15) for i in tmp1
    ]  # round off the random last digit so iteration is over exactly same xmax
    tmp2 = np.array(list(set(tmp1)))
    lenofsetofall = [len(np.where(tmp1 == i)[0]) for i in tmp2]
    tmp3 = np.array(
        [i["deltaxydlist"] for j, i in enumerate(arecord) if "deltaxydlist" in i.keys()]
    )  # collect all deltaxyd data with  its index and xmax
    tmp4 = np.array(list(zip(*tmp3)))[1] if dx > dy else np.array(list(zip(*tmp3)))[2]
    tmp4 = [round(i, 15) for i in tmp4]
    indexsetofvalidscan = [
        np.where(tmp4 == i)[0] for i in tmp2 if len(np.where(tmp4 == i)[0]) > 0
    ]
    tmp5 = list(zip(*[indexsetofvalidscan, lenofsetofall]))
    indexofvalidset = [
        i[0]
        for i in tmp5
        if ((len(tmp5) > 1 and len(i[0]) == i[1]) or (len(tmp5) == 1))
    ]  # assume most iteration number is larger than 4, if < 4 would be considered abnormal.
    if (
        len(indexofvalidset) == 1
    ):  # if the arecord has only one set of xmax then just use the last arecord as starting point
        arecordindexchosen = -1
    else:  # if there are more than 1 set of xmax, choose the last two sets and choose the record with minimum delta xyd.
        sortedindex = np.argsort([i[-1] for i in indexofvalidset])[-2:]
        indexoflastvalidset = np.hstack(
            [indexofvalidset[sortedindex[0]], indexofvalidset[sortedindex[1]]]
        )
        lastdeltaxydscan = np.array(
            [tmp3[i] for i in indexoflastvalidset if not np.isnan(tmp3[i][3])]
        )
        idx = np.argmin(lastdeltaxydscan[:, 3])
        arecordindexchosen = int(lastdeltaxydscan[idx][0]) - 1
    return arecordindexchosen


def fvec_error_list(ar, xc=-8e-3):
    tmp, xset = kxmax(ar, xc=xc)
    xset = np.sort(list(xset))
    tmp = list(zip(*tmp))
    idx = tmp[0]
    ar1 = [ar[int(i)] for i in idx]  # np.array(ar)[idx]
    ar1 = [i for i in ar1 if "err.args" not in i]
    vthetascanmsg = ar1[0]["iorecord"]["vthetascanmsg"]
    if "usevinv_matrix" in vthetascanmsg["warnflag"]:
        tmp = [max(i) for i in vthetascanmsg["fvec"]]
        tmp2 = [np.std(i) for i in vthetascanmsg["fvec"]]
        tmp1 = vthetascanmsg["nitermean"]
    else:
        tmp = np.amax(abs(vthetascanmsg["fvec"]))
        tmp2 = np.std(vthetascanmsg["fvec"])
        tmp1 = vthetascanmsg["nfev"]
    print(
        "np.max(max(fvec)),np.max(std(fvec)),np.mean(nfev)=",
        np.max(tmp),
        np.max(tmp2),
        np.mean(tmp1),
    )
    ferrortable = []
    for i in range(1, len(ar1) - 1):
        inversev1v2msg = ar1[i]["iorecord"]["inversev1v2msg"]
        if "useBlockfsolve" in inversev1v2msg["warnflag"]:
            tmp = [max(i) for i in inversev1v2msg["fvec"]]
            tmp2 = [np.std(i) for i in inversev1v2msg["fvec"]]
            tmp1 = inversev1v2msg["nitermean"]
            ferrortable.append(np.max(tmp))

            print(
                "np.max(max(fvec)),np.max(std(fvec)),np.mean(nfev)=",
                np.max(tmp),
                np.max(tmp2),
                np.mean(tmp1),
            )

        else:
            tmp0 = [i["outfindso"]["fvec"] for i in inversev1v2msg["xydrecord"]]
            tmp = np.max(list(map(np.max, tmp0)))
            tmp2 = np.std(list(map(np.std, tmp0)))
            tmp1 = inversev1v2msg["nitermean"]
            print(
                "np.max(max(fvec)),np.max(std(fvec)),np.mean(nfev)=",
                np.max(tmp),
                np.max(tmp2),
                np.mean(tmp1),
            )
    ferrorminidx = np.argmin(ferrortable)
    ferrormin = ferrortable[ferrorminidx]
    return ferrormin, ferrorminidx, xset


def plotferrormin(ar):
    scanmin = convergenceplot(ar)
    ferrortable = []  # ferrortable is a list of
    for i in scanmin:
        x, idxconvergent = i[0], i[3]
        if "iorecord" in ar[idxconvergent].keys():
            inversev1v2msg = ar[idxconvergent]["iorecord"]["inversev1v2msg"]
        if "useBlockfsolve" in inversev1v2msg["warnflag"]:
            # tmp = [max(i) for i in inversev1v2msg["fvec"]]
            tmp = abs(inversev1v2msg["outfindso"]["fval"])
            # tmp2 = [np.std(i) for i in inversev1v2msg["fvec"]]
            # tmp1 = inversev1v2msg["nitermean"]
            ferrortable.append([x, np.log(np.max(tmp))])

    plt.figure(132)
    ferrortable = list(zip(*ferrortable))
    plt.plot(ferrortable[0], ferrortable[1], "-")
    plt.plot(ferrortable[0], ferrortable[1], ".")
    plt.xlabel("x (m)")
    plt.ylabel("log(fvec_error minimum)")
    tmp = [i for i in ferrortable[1] if not np.isnan(i)]
    plt.axis([np.min(ferrortable[0]), np.max(ferrortable[0]), np.min(tmp) - 1, 8])
    plt.title("fig.132 log(V_max(err)) vs. x")
    plt.savefig("junk132.png")
    return ferrortable


def plotbnm(ar):
    scanmin = convergenceplot(ar)
    ferrortable = []  # ferrortable is a list of
    maxbmnlogav = []
    for i in scanmin:
        x, idxconvergent = i[0], i[3]
        ari = ar[idxconvergent]
        if "flucbnm" in ari.keys():
            maxbmnlogav.append(
                [
                    ari["xmax"],
                    (
                        np.log(ari["flucbnm"]["np.amax(abs(bnm1))"])
                        + np.log(ari["flucbnm"]["np.amax(abs(bnm2))"])
                    )
                    / 2,
                ]
            )

    maxbmnlogav = list(zip(*maxbmnlogav))
    plt.figure(133)
    plt.plot(maxbmnlogav[0], maxbmnlogav[1], "-")
    plt.plot(maxbmnlogav[0], maxbmnlogav[1], ".")
    plt.xlabel("x (m)")
    plt.ylabel("<log(maxbmnlogav)>")
    plt.axis(
        [np.min(maxbmnlogav[0]), np.max(maxbmnlogav[0]), np.min(maxbmnlogav[1]), -1]
    )
    plt.title("fig.133 <log(max(bmn))> vs. x")
    plt.savefig("junk133.png")
    return maxbmnlogav


t0.append(["3876", time.time(), time.time() - t0[-1][1]])

# 21 interpolation of xyd from n_theta=20 to n_theta=40 xyd2
# scipy.interpolate.interp2d(x, y, z, kind='linear', copy=True, bounds_error=False, fill_value=None)
# xyd is a trajectory with n_theta1, interploation_n_theta(xyd,n_theta1,n_theta2) output xyd2 has n_thetae2
# both xyd and xyd2 has period 2pi
def interploation_xyd(xyd, n_theta1, n_theta2):
    xyd1 = []
    xyd2 = []
    x1 = np.linspace(0, n_theta1 - 1, n_theta1)
    y1 = np.linspace(0, n_theta1 - 1, n_theta1)
    x2 = np.linspace(0, n_theta1 - 1, n_theta2)
    y2 = np.linspace(0, n_theta1 - 1, n_theta2)
    X, Y = np.meshgrid(x1, y1)
    for xydi in xyd:
        Z = xydi.reshape([n_theta1, n_theta1])
        f = interp2d(x1, y1, Z, kind="cubic")
        Z2 = f(x2, y2)
        xyd1.append(Z)
        xyd2.append(Z2)

    X2, Y2 = np.meshgrid(x2, y2)
    """
    for i in range(4):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].pcolormesh(X, Y, xyd1[i])
        ax[1].pcolormesh(X2, Y2, xyd2[i])
        plt.savefig("junk" + str(i) + ".png")

    plt.show()
    """
    return np.array(xyd2).reshape([4, n_theta2**2])


# tmp=interploation_n_theta(xyd,20,40)


def iterpolaten_theta(arecordi, n_theta2, cutoff2):
    arecord2 = arecordi.copy()
    xyd = arecord2["xyd"]
    n_theta1 = arecord2["n_theta"]
    xyd2 = interploation_xyd(xyd, n_theta1, n_theta2)
    arecord2["xyd"] = xyd2
    arecord2["n_theta"] = n_theta2
    arecord2["lattice"].n_theta = n_theta2
    arecord2["iterxydinput"]["n_theta"] = n_theta2
    arecord2["iterxydinput"]["iteration_parameters"] = arecordi["iterxydinput"][
        "iteration_parameters"
    ]
    arecord2["lattice"].Cauchylimit["aliasingCutoff"] = cutoff2
    return arecord2


def studyiter4(arecord):
    dfmax = []
    niters = []
    k = 0
    for i in range(len(arecord)):
        k = k + 1
        print("k=", k)
        if "iorecord" in arecord[i]:
            if (
                "vthetascanmsg" in arecord[i]["iorecord"]
                and "msg" not in arecord[i]["iorecord"]["vthetascanmsg"]
            ):
                vscan = arecord[i]["iorecord"]["vthetascanmsg"]["vthetascanrecord"]
                numberdivgence = arecord[i]["iorecord"]["vthetascanmsg"][
                    "numberdivgence"
                ]
                ferror = [i["outfindso"]["fval"] for i in vscan]
                fvecerror = [np.max(i["outfindso"]["fvec"]) for i in vscan]
                niter = [
                    i["outfindso"]["niter"]
                    if "niter" in list(i["outfindso"].keys())
                    else i["outfindso"]["nfev"]
                    for i in vscan
                ]
                dfmax.append(np.amax(ferror))
                niters.append(np.mean(niter))
                print(
                    i,
                    "\t",
                    np.amax(ferror),
                    "\t",
                    "\t",
                    "\t",
                    np.mean(niter),
                    "\t",
                    numberdivgence,
                )
            elif (
                "vthetascanmsg" in arecord[i]["iorecord"]
                and "msg" in arecord[i]["iorecord"]["vthetascanmsg"]
            ):
                # if arecord[i]['iorecord']['vthetascanmsg']['msg']!='use xyd as xyd0':
                dfmax.append(np.nan)
                niters.append(np.nan)
            else:
                dfmax.append(np.nan)
                niters.append(np.nan)
        else:
            dfmax.append(np.nan)
            niters.append(np.nan)
        print("k=", k, "len(dfmax)=", len(dfmax))

    if len(dfmax) > 0:
        dfmaxnotnan = [i for i in dfmax if not np.isnan(i)]
        nitersnotnan = [i for i in niters if not np.isnan(i)]
        print("mean(maxerror)=", np.mean(dfmaxnotnan))
        print("mean(niter)=", np.mean(nitersnotnan))
        print("max(maxerror)=", np.max(dfmaxnotnan))
        print("max(niter)=", np.max(nitersnotnan), "\n\n")

    invmax = []
    invniters = []
    for i in range(len(arecord)):
        if "iorecord" in arecord[i]:
            if "inversev1v2msg" in arecord[i]["iorecord"]:
                inv = arecord[i]["iorecord"]["inversev1v2msg"]["xydrecord"]
                numberdivgence = arecord[i]["iorecord"]["inversev1v2msg"][
                    "numberdivgence"
                ]
                inverror = [i["outfindso"]["fval"] for i in inv]
                invfvecerror = [np.max(i["outfindso"]["fvec"]) for i in inv]
                invniter = [
                    i["outfindso"]["niter"]
                    if "niter" in list(i["outfindso"].keys())
                    else i["outfindso"]["nfev"]
                    for i in inv
                ]
                invmax.append(np.amax(inverror))
                invniters.append(np.mean(invniter))
                print(
                    i,
                    "\t",
                    np.amax(inverror),
                    "\t",
                    "\t",
                    "\t",
                    np.mean(invniter),
                    "\t",
                    numberdivgence,
                )
            else:
                invmax.append(0)
                niters.append(0)
        else:
            invmax.append(0)
            niters.append(0)

    if len(invmax) > 0:
        print("mean(invmaxerror)=", np.mean(invmax))
        print("mean(invniter)=", np.mean(invniters))
        print("max(invmaxerror)=", np.max(invmax))
        print("max(invniter)=", np.max(invniters))
        # time=[ i[2]  for i in timing[2:] if "2" in i[0]]
    # print ('time invthetascan=',np.max(time))
    ##time=[ i[2]  for i in timing[2:] if "4" in i[0]]
    # print ('maxtime inversev=',np.max(time))
    if len(arecord) > 4 and "iorecord" in arecord[-1]:
        k = -1
        if "vthetascanmsg" in arecord[k]["iorecord"]:
            if "vthetascanrecord" in arecord[i]["iorecord"]["vthetascanmsg"]:
                vscan = arecord[k]["iorecord"]["vthetascanmsg"]["vthetascanrecord"]
                ferror = [i["outfindso"]["fvec"] for i in vscan]
                plt.figure(1)
                plt.title("Fig.1 the errors fvec of vthetascan in iteration " + str(k))
                plt.plot(ferror, ".")
                ferrora = list(
                    zip(*[list(map(abs, i["outfindso"]["fvec"])) for i in vscan])
                )
                sortedferror = [np.sort(i) for i in ferrora]
                plt.figure(2)
                for j, i in enumerate(sortedferror):
                    plt.plot(i, ".", label="ferror[" + str(j) + "]")
                plt.title(
                    "Fig.2 the sorted errors fvec of vthetascan in iteration " + str(k)
                )
                plt.savefig("junk2.png")
        k = -1
        if "inversev1v2msg" in arecord[k]["iorecord"]:
            inv = arecord[k]["iorecord"]["inversev1v2msg"]["xydrecord"]
            inverror = [i["outfindso"]["fvec"] for i in inv]
            plt.figure(3)
            plt.plot(inverror, ".")
            plt.title("Fig.3 the errors fvec of inversev1v2 in iteration " + str(k))
            plt.savefig("junk3.png")
            inverrora = list(
                zip(*[list(map(abs, i["outfindso"]["fvec"])) for i in inv])
            )
            sortedinverror = [np.sort(i) for i in inverrora]
            argsortedinverror = [np.argsort(i) for i in inverrora]
            plt.figure(4)
            for j, i in enumerate(sortedinverror):
                plt.plot(i, ".", label="inverror[" + str(j) + "]")
            plt.title(
                "Fig.4 the sorted errors fvec of inversev1v2 in iteration " + str(k)
            )
            plt.savefig("junk4.png")
    return dfmax, invmax


def init_tpsa(
    xmax=-18e-3,
    ymax=1e-4,
    ylim=0.2,
    trackcode="tpsa",
    npass=800,
    init_tpsa_input=None,
    lattice=None,
    keep_saved_files=True,
):
    if DEBUG:
        tt0 = [["in init, 1, start", time.time(), 0]]
        print(tt0)

    if lattice is None:
        lattice = _get_default_lattice_namespace()

    (
        nvar,
        n_theta,
        cutoff,
        norder,
        norder_jordan,
        use_existing_tpsa,
        oneturntpsa,
        deltap,
        ltefilename,
        mod_prop_dict_list,
        tpsacode,
        dmuytol,
    ) = [init_tpsa_input[i] for i in init_tpsa_input.keys()]
    ########################################################

    for prepare_lattice in [1]:
        lattice.ltefilename = (
            ltefilename  ##"20140204_bare_1supcell"#"nsls2sr_supercell_ch77_20150406_1"#
        )
        # lattice.'usecode']=dict(tpsacode='yoshitracy',use_existing_tpsa=0)
        # lattice.'usecode']=dict(tpsacode="yuetpsa",use_existing_tpsa=0)
        lattice.usecode = dict(tpsacode=tpsacode, use_existing_tpsa=use_existing_tpsa)
        # lattice.'usecode']=dict(tpsacode='madx',use_existing_tpsa=0)
        lattice.uselastxyd = "use xydlast for vthetascan trial for ik>0"  #'never',#'use xydlast after only one vthetascan'#'no inversev'#
        lattice.deltap = deltap
        lattice.n_theta = n_theta
        # lattice.alphadiv = lattice.n_theta
        lattice.nvar = nvar
        lattice.norder = norder
        lattice.norder_jordan = norder_jordan
        lattice.yulim = ylim
        lattice.npass = npass
        lattice.mod_prop_dict_list = mod_prop_dict_list
        xscale = (100, 1.3, 50, 0.13)
        xfactor = 33
        gu0 = 0.05 / 0.05 * np.array(xscale) * xfactor
        gu0 = np.array([i * np.ones(n_theta**2) for i in gu0]).transpose()
        gu0 = np.array(gu0)
        findsoparamters = {
            "xtol": 1e-8,  # 1e-10,
            "fscale": (2.7, 2.7, 0.36, 0.13),
            "xscale": (100, 1.3, 50, 0.13),
            "xfactor": 33,
            "factor": 0.5,
            "nsosteplim": 2,
            "fevtol": 8e-4,  # 1e-2,
            "gu0": gu0,
        }

    ########################################################
    if nvar == 4:
        acoeff0 = array([[1 + 0j, 0j, 0j, 0j], [0j, 0j, 1 + 0j, 0j]])
    elif nvar == 6:
        acoeff0 = array([[1 + 0j, 0j, 0j, 0j, 0j, 0j], [0j, 0j, 0j, 1 + 0j, 0j, 0j]])
    else:
        raise ValueError("`nvar` must be 4 or 6.")
    lattice.acoeff0 = acoeff0
    veq.TRACKING_CODE = "ELEGANT"  #'Tracy'
    lte2tpsa.dmuytol = dmuytol  # criterion to decide resonance condition
    if DEBUG:
        tt1 = [["in init 2 ", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    # setup oneturntpsa and lattice #######################################################
    for setup_lattice in [1]:
        # generate lattice and depends on whether using 'ELEGANT' or 'tpsa' as tracking code, and generate oneturntpsa if oneturntpsa='tpsa'
        # if oneturntpsa!='tpsa', then oneturntpsa already exists and must be given the input to replace it as oneturntpsa=oneturntpsasaved
        if trackcode == "ELEGANT":
            lattice.usecode = dict(tpsacode="yuetpsa", use_existing_tpsa=1)
            lattice, oneturntpsa, filepath_d = setup(lattice)
        elif oneturntpsa == "tpsa":
            lattice.usecode = dict(
                tpsacode=tpsacode, use_existing_tpsa=use_existing_tpsa
            )
            lattice, oneturntpsa, filepath_d = setup(lattice)
        elif oneturntpsa != "tpsa":
            lattice.usecode = dict(tpsacode="yuetpsa", use_existing_tpsa=1)
            lattice, junk, filepath_d = setup(lattice)

        lattice.Cauchylimit["aliasingCutoff"] = cutoff
    if DEBUG:
        tt1 = [["in init 3 ", time.time(), time.time() - tt1[0][1]]]
        print(tt1)

    # extract parameters to parepare for renewXinput #######################################################
    for parepare_renewXinput in [1]:

        # choose uarray for renewXinput #######################################################
        if nvar == 3:
            uarray = np.array([lattice.ux[0], lattice.ux[1], lattice.uy[0]])
        if nvar == 4:
            uarray = np.array(
                [lattice.ux[0], lattice.ux[1], lattice.uy[0], lattice.uy[1]]
            )
        if nvar == 6:
            uarray = np.array(
                [
                    lattice.ux[0],
                    lattice.ux[1],
                    lattice.ux[2],
                    lattice.uy[0],
                    lattice.uy[1],
                    lattice.uy[2],
                ]
            )

        xacoeff0 = ari2x(acoeff0)
        xacoeffnew = xacoeff0

        # uvarlist = "bKi,scalex,norder_jordan,powerindex,xfix,xpfix,deltap".split(",")

        # uvar = [eval("lattice." + i) for i in uvarlist]
        uvar = SimpleNamespace(
            bKi=lattice.bKi,
            scalex=lattice.scalex,
            norder_jordan=lattice.norder_jordan,
            powerindex=lattice.powerindex,
            xfix=lattice.xfix,
            xpfix=lattice.xpfix,
            deltap=lattice.deltap,
        )

        # lattice.uvar = uvar  # uarray in lattice and uvar noth need to be updated

        Zpar3 = lattice.bKi, lattice.scalex, lattice.norder_jordan, lattice.powerindex

        usexp0dlast = 0
        xydlast = np.zeros([4, n_theta**2])

    renewXinput = dict(
        xacoeffnew=xacoeff0,
        xmax=xmax,
        ymax=ymax,
        xydlast=xydlast,
        uarray=uarray,
        acoeff0=acoeff0,
        nvar=lattice.nvar,
        n_theta=n_theta,
        uvar=uvar,
        Zpar=Zpar3,
        findsoparamters=findsoparamters,
        usexp0dlast=usexp0dlast,
    )
    if DEBUG:
        tt1 = [["in init 4 ", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    wwJwinv = veq.wwJwinvf(
        uarray,
        Zpar3,
        use_existing_tpsa=use_existing_tpsa,
        saved_filepath=filepath_d["wwJmtx"],
    )
    # w,wc=wtps(uarray,Zpar)
    if DEBUG:
        tt1 = [["in init 5 end", time.time(), time.time() - tt0[0][1]]]
        print(tt1)
    # wwJwinv: wwJ is Jacobian matrix of dw/dx

    if not keep_saved_files:
        for fp in filepath_d.values():
            try:
                Path(fp).unlink()
            except:
                pass

    return renewXinput, lattice, oneturntpsa, wwJwinv


def change_n_theta(
    init_tpsa_output, n_theta2, xyntheta_scan_input=None  # xyntheta_scan_input
):
    renewXinput, lattice, oneturntpsa, wwJwinv = init_tpsa_output
    (
        ar_iter_number,
        ar2_iter_number,
        number_of_iter_after_minimum,
        applyCauchylimit,
        n_theta_cutoff_ratio,
    ) = [xyntheta_scan_input[i] for i in xyntheta_scan_input.keys()]
    cutoff2 = n_theta2 // n_theta_cutoff_ratio[0] - n_theta_cutoff_ratio[1]

    findsoparamters = renewXinput["findsoparamters"]
    xscale = findsoparamters["xscale"]
    xfactor = findsoparamters["xfactor"]
    gu0 = 0.05 / 0.05 * np.array(xscale) * xfactor
    gu0 = np.array([i * np.ones(n_theta2**2) for i in gu0]).transpose()
    gu0 = np.array(gu0)
    findsoparamters["gu0"] = gu0
    renewXinput["findsoparamters"] = findsoparamters
    renewXinput["n_theta"] = n_theta2

    lattice.n_theta = n_theta2
    # lattice.alphadiv = n_theta2
    lattice.Cauchylimit["aliasingCutoff"] = cutoff2
    init_tpsa_output = renewXinput, lattice, oneturntpsa, wwJwinv
    return init_tpsa_output


# previous iteration_on_xyntheta now is xyntheta_from_X
def xyntheta_from_X0(
    xmax=-17e-3,
    ymax=4e-3,
    ntheta=12,
    init_tpsa_output=None,
    xyntheta_scan_input=None,
    useinversetpsa=None,
    use_existing_tpsa=None,
    averaging=1,
    scantype="including first iteration",
):
    try:
        # This is "start_from_xacoeff0_iterate_on_xacoeffnew_and_xyd" together with xyminz, i.e., 3d data for x,y,z, where z is the lowest error point of iteration
        # iteration starts from the linear combination xacoeff0 by renewXfromX0, then followed by iterXs where the xacoeffnew and xyd are iterated
        # according to iteration_parameters in the imput.
        if DEBUG:
            tt0 = [["snew0, 1. start", time.time(), 0]]
            print(tt0)

        init_tpsa_output = change_n_theta(
            init_tpsa_output, ntheta, xyntheta_scan_input=xyntheta_scan_input
        )
        renewXinput, lattice, oneturntpsa, wwJwinv = init_tpsa_output
        ar_iter_number = xyntheta_scan_input["ar_iter_number"]
        number_of_iter_after_minimum = xyntheta_scan_input[
            "number_of_iter_after_minimum"
        ]
        iteration_parameters = dict(
            iter_number=ar_iter_number,
            number_of_iter_after_minimum=number_of_iter_after_minimum,
        )

        try:
            arecordi = renewXfromX0(
                xmax,
                ymax,
                renewXinput=renewXinput,
                lattice=lattice,
                oneturntpsa=oneturntpsa,
                use_existing_tpsa=use_existing_tpsa,
                wwJwinv=wwJwinv,
                useinversetpsa=useinversetpsa,  # False,  #
                usewinv=1,
            )
            if arecordi == None:
                return None
        except Exception as err:
            if DEBUG:
                print(dir(err))
                print(err.args)
            return None
        arecordi["lattice"].uselastxyd = "always"

        if DEBUG:
            tt1 = [["snew0, 2, ", time.time(), time.time() - tt0[0][1]]]
            print(tt1)
        n_theta = arecordi["n_theta"]
        xyd = averagingxyd(arecordi["xyd"], n_theta=n_theta, averaging=averaging)

        for update_iterxydinput in [1]:
            iterxydinput = renewXinput.copy()
            iterxydinput["xmax"] = xmax
            iterxydinput["ymax"] = ymax
            iterxydinput["xacoeffnew"] = arecordi["xacoeffnew"]
            iterxydinput["xydlast"] = xyd
            iterxydinput["usexp0dlast"] = 1
            iterxydinput["iteration_parameters"] = iteration_parameters

        arecordi["iterxydinput"] = iterxydinput
        if DEBUG:
            tt1 = [["snew0, 3, ", time.time(), time.time() - tt1[0][1]]]
            print(tt1)
        ar1 = iterXs(
            [arecordi],
            arecordi["xacoeffnew"],
            xyd,
            iterxydinput,
            xmax=xmax,
            ymax=ymax,
            niter=iteration_parameters["iter_number"],
            oneturntpsa=oneturntpsa,
            useinversetpsa=useinversetpsa,
            wwJwinv=wwJwinv,
            averaging=averaging,
            number_of_iter_after_minimum=iteration_parameters[
                "number_of_iter_after_minimum"
            ],
        )

        if DEBUG:
            tt1 = [["snew0, 4, ", time.time(), time.time() - tt1[0][1]]]
            print(tt1)
            tt1 = [["snew0, 5, ", time.time(), time.time() - tt0[0][1]]]

        x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(ar1, plot3d=False)
        x, y, z, mainidx, iteration_parameters = x_iter_lndxy
        z1 = [i for i in z if not np.isnan(i)]
        if scantype == "excluding first iteration":
            z1 = [i for i in z if not np.isnan(i)][1:]
        elif scantype == "including first iteration":
            z1 = [i for i in z if not np.isnan(i)][0:]

        if len(z1) > 0:
            minz = np.min(z1)
            idxmin = np.where(z1 == minz)[0][-1]  # choose the last minz position
            # we need to add 2 to idxmin because the scantype="including first iteration"
        else:
            minz = np.nan
            idxmin = np.where(np.isnan(z))[0][0]

        iteration_data = {
            "arecord": ar1,
            "minz": minz,
            "x_iter_lndxy": x_iter_lndxy,
            "idxmin": idxmin,
        }

    except Exception as err:
        if DEBUG:
            print(dir(err))
            print(err.args)
        return

    return iteration_data


# previous runxnew0 now is renewXfromX0
def renewXfromX0(
    xmax,
    ymax,
    renewXinput="None",
    lattice=None,
    uarray=None,
    oneturntpsa="tpsa",
    use_existing_tpsa=0,
    wwJwinv=None,
    useinversetpsa=None,
    usewinv=1,
):
    renewXinput["xmax"] = xmax
    renewXinput["ymax"] = ymax
    if useinversetpsa:
        arecordi = renewX(
            renewXinput,
            lattice,
            oneturntpsa=oneturntpsa,
            useinversetpsa=useinversetpsa,
            wwJwinv=wwJwinv,
            usewinv=usewinv,
        )
    else:
        arecordi = renewX(
            renewXinput,
            lattice,
            oneturntpsa=oneturntpsa,
            useinversetpsa=useinversetpsa,
        )
    arecordi["ik"] = 0

    arecordi["deltaxydlist"] = [
        len(arecordi),
        xmax,
        ymax,
        np.nan,
    ]

    return arecordi


def checkxydconvergence(arecord, xc=-20e-3):
    tmp = [
        [
            k,
            j["deltaxydlist"][1],
            j["deltaxydlist"][-1],
        ]  # j["deltaxydlist"] =k,xmax,ymax, deltaxyd
        for k, j in enumerate(arecord)
        if not np.isnan(j["deltaxydlist"][-1]) and abs(j["xmax"] - xc) < 1e-10
    ]
    """
    tmp = [
        [k]
        + [
            j["flucbnm"][i]
            for i in [
                "xmax",
                "ymax",
                "np.amax(abs(bnm1))",
                "np.amax(abs(bnm2))",
                "np.std(bnm1)",
                "np.std(bnm2)",
            ]
        ]
        + [j["ik"]]
        if "flucbnm" in j
        else [k, j["xmax"], j["ymax"], np.nan, np.nan, np.nan, np.nan, k]
        for k, j in enumerate(arecord)
    ]
    k, xlist1, ylist1, bnm1max, bnm2max, bnm1rms, bnm2rms, iklist = list(zip(*tmp))
    xlist2 = list(zip(*[xlist1, k]))
    xlist = [1e-8 * (xt[1] - len(xlist2) / 2.0 - 10) + xt[0] for xt in xlist2]
    ylist2 = list(zip(*[ylist1, k]))
    ylist = [1e-8 * (xt[1] - len(ylist2) / 2.0 - 10) + xt[0] for xt in ylist2]
    dx = abs(np.max(xlist) - np.min(xlist))
    dy = abs(np.max(ylist) - np.min(ylist))
    if dx >= dy or abs(dx - dy) < 1e-5:
        tmp = [
            [i["xmax"], i["xyd"]] for i in arecord if "err.args" not in i
        ]  # if 'xacoeffnew' in i]
    elif dy > dx:
        tmp = [
            [i["ymax"], i["xyd"]] for i in arecord if "err.args" not in i
        ]  # if 'xacoeffnew' in i]
    """
    if len(tmp) == 0:
        return None
    tmp1 = [i[-1] for i in tmp if i[1] != -np.inf]
    minf = np.min(tmp1)
    tmp12 = [
        [i[1], minf] if i[2] == -np.inf else [i[1], i[2]] for j, i in enumerate(tmp)
    ]
    tmp2 = np.array(list(zip(*tmp12)))
    tmp3 = np.where(abs(tmp2[0] - xc) < 1e-6)[0]
    return np.min(tmp2[1]), np.argmin(tmp2[1][tmp3])


def convergenceplot(arecord):
    kxmx = [
        [j, i["xmax"], i["ymax"]] for j, i in enumerate(arecord) if "acoeff" in i.keys()
    ]
    xset = np.sort(list(set([i[1] for i in kxmx])))
    scanmin1 = []
    for xcc in xset:
        ar1 = [arecord[i[0]] for i in kxmx if abs(xcc - i[1]) < 1e-9]
        ar1index = [j for j, i in enumerate(kxmx) if abs(xcc - i[1]) < 1e-9]
        tmp = checkxydconvergence(ar1, xc=xcc)
        if tmp != None:
            scanmin1.append([xcc, tmp[0], tmp[1], ar1index[tmp[1] + 1]])
    scanmin = list(zip(*scanmin1))
    plt.figure(131)
    plt.plot(scanmin[0], scanmin[1], ".")
    plt.plot(scanmin[0], scanmin[1], "-")
    plt.plot(scanmin[0], scanmin[2], "-", label="index of minimum")
    plt.xlabel("x (m)")
    plt.ylabel("ln(xyd[n]-xyd[n-1]")
    plt.legend(loc="best")
    plt.title("Fig.131 minimum in iteration at x")
    axes = plt.gca()
    axes.set_ylim([np.min(scanmin[1]), 10])
    plt.savefig("junk131.png")
    return scanmin1


def change_n_theta2_in_findsoparamters(arecordi, n_theta2):
    findsoparamters = arecordi["iterxydinput"]["findsoparamters"]
    xscale = findsoparamters["xscale"]
    xfactor = findsoparamters["xfactor"]
    gu0 = 0.05 / 0.05 * np.array(xscale) * xfactor
    gu0 = np.array([i * np.ones(n_theta2**2) for i in gu0]).transpose()
    gu0 = np.array(gu0)
    findsoparamters["gu0"] = gu0
    return findsoparamters


# use the arecord resulted from scan_around_DA()
def plot3Dtheta_vs_xydconvergence(arecord, codeused="tpsa", plot_3d=True):
    tmp = [
        [k]
        + j["deltaxydlist"]
        + [j["lattice"].Cauchylimit["aliasingCutoff"]]
        + [j["n_theta"]]
        for k, j in enumerate(arecord)
        if "deltaxydlist" in j.keys() and "err.args" not in j
    ]
    k, iter_number, xlist1, ylist1, lndeltaxyd, cutoff, n_theta = list(zip(*tmp))
    # n_theta2 = arecord[k[-2]]["iterxydinput"]["n_theta"]
    norder = arecord[k[-1]]["lattice"].norder
    tmp12 = np.array([i for j, i in enumerate(tmp) if not np.isnan(i[4])])
    minf = np.min([i for i in tmp12[:, 4] if (not np.isnan(i)) and i != -np.inf])
    tmp12 = [
        [i[0], i[1], i[2], i[3], minf, i[5], i[6]] if i[4] == -np.inf else i
        for j, i in enumerate(tmp12)
    ]
    tmp2 = np.array(list(zip(*tmp12)))

    n_thetaset = np.sort(list(set(n_theta)))

    x = []
    y = []
    z = []
    mainidx = []
    for n_theta in n_thetaset:
        tmp3 = np.where(abs(tmp2[6] - n_theta) < 1e-6)[0]  # tp2[2] is xmax
        mainidx.append(
            list(map(int, tmp2[0][tmp3].tolist()))
        )  # the first term is the seed from previous run
        x = x + (list(range(len(tmp3))))
        y = y + (np.ones(len(tmp3)) * n_theta).tolist()
        z = z + tmp2[4][tmp3].tolist()

    if plot_3d:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c="r", marker="o")

        ax.set_xlabel("\n\niteration", fontsize=20)
        ax.set_ylabel("\n\nn_theta", fontsize=20)
        ax.set_zlabel(
            "                         \n\n\n\nln(|xyd(n)-xyd[n+1]|)\n\n", fontsize=20
        )
        # ax.set_xlim3d(-4,4)
        # ax.set_ylim3d(-4,4)
        ax.set_zlim3d(-25, 0)
        plt.title(
            "Fig.1 lndeltaxyd vs. n_theta and iteration\n number with cutoff=n_theta, x="
            + str(xlist1[0])
            + ",norder="
            + str(norder)
        )
        plt.savefig("junk1.png")

    ik = 0
    while ik >= 0:
        if "iterxydinput" in arecord[ik]:
            iteration_parameters = arecord[ik]["iterxydinput"]["iteration_parameters"]
            ik = -1
        else:
            ik = ik + 1
    # iteration_parameters = arecord[1]["iterxydinput"]["iteration_parameters"]
    x_iter_lndxy = x, y, z, mainidx, iteration_parameters
    # here x,y,z are list of iteration number for every arecord[i]
    # y is a list of n_theta for every arecord[i]
    # z is a list of ln(delta_xyd) for every arecord[i]
    return x_iter_lndxy, cutoff, n_thetaset


# use result from plot3Dtheta_vs_xydconvergence
def plot1Dtheta_vs_xydconvergence_from_3Ddata(arecord, n_theta=12, plot130=False):
    x_iter_lndxy, cutoff, n_thetaset = plot3Dtheta_vs_xydconvergence(
        arecord, plot_3d=False
    )
    x, y, z, mainidx, iteration_parameters = x_iter_lndxy
    idx = [j for j, i in enumerate(y) if abs(i - n_theta) < 1e-9]
    z1 = np.array(z)[idx]
    if plot130:
        plt.figure(130)
        if len(z1) != 0:
            plt.plot(z1, "o", label="n_theta=" + str(n_theta))
        plt.legend(loc="best")
        plt.xlabel("iteration number")
        plt.ylabel("ln(|xyd[n]-xyd[n+1]|)")
        plt.title("Fig.130 convergence rate vs. iteration number")
        plt.savefig("junk130.png")
    return idx, z1


# use result from plot3Dtheta_vs_xydconvergence
def plot1Dtheta_vs_xacoeffnewconvergence_from_3Ddata(
    arecord, n_theta=12, plot131=False
):
    x_iter_lndxy, cutoff, n_thetaset = plot3Dtheta_vs_xydconvergence(
        arecord, plot_3d=False
    )
    x, y, z, mainidx, iteration_parameters = x_iter_lndxy
    idx_theta = np.where(n_thetaset == n_theta)[0][0]
    xacoeffnewlist = [arecord[i]["xacoeffnew"] for i in mainidx[idx_theta]]
    lndxacoeffnew = [
        np.log(np.std(xacoeffnewlist[i + 1] - xacoeffnewlist[i]))
        for i in list(range(len(xacoeffnewlist) - 1))
    ]
    if plot131:
        plt.figure(131)
        if len(lndxacoeffnew) != 0:
            plt.plot(lndxacoeffnew, "o", label="n_theta=" + str(n_theta))
        plt.legend(loc="best")
        plt.xlabel("iteration number")
        plt.ylabel("ln(|xyd[n]-xyd[n+1]|)")
        plt.title("Fig.131 ln(dxacoeffnew) vs. iteration number")
        plt.savefig("junk131.png")
    return idx_theta, lndxacoeffnew


# for ik in [0]:
# previous test_change_n_theta_accumulate_by_tpsa2 now is xyntheta_from_X_change_ntheta
def xyntheta_from_X_change_ntheta(
    xmax=-22e-3,
    ymax=4e-3,
    arecordi=None,
    n_theta2=40,
    cutoff2=20,
    oneturntpsa="tpsa",
    iteration_parameters=dict(iter_number=12, number_of_iter_after_minimum=4),
    plotconvergence=True,
    wwJwinv=0,
    scan_variable="n_theta",  # "xy", #
    applyCauchylimit=False,
):  # ELEGANT" #'oneturntpsasaved):
    tt0 = [["change n_theta start", time.time(), 0]]
    print(tt0)

    lattice = arecordi["lattice"]
    if wwJwinv == 0:
        bKi, scalex, powerindex, uarray = (
            lattice.bKi,
            lattice.scalex,
            lattice.powerindex,
            lattice.uarray,
        )
        norder_jordan = lattice.norder_jordan
        Zpar3 = bKi, scalex, norder_jordan, powerindex
        wwJwinv = veq.wwJwinvf(uarray, Zpar3, use_existing_tpsa=1)

    tt1 = [["change n_theta 2", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    ##########################################################
    findsoparamters = change_n_theta2_in_findsoparamters(arecordi, n_theta2)

    arecordi["iterxydinput"]["findsoparamters"] = findsoparamters
    arecordi["iterxydinput"]["iteration_parameters"] = iteration_parameters

    tt1 = [["change n_theta 3", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    arecord2 = iterpolaten_theta(arecordi, n_theta2=n_theta2, cutoff2=cutoff2)
    arecord2["lattice"].uselastxyd = "always"
    n_theta = arecord2["n_theta"]
    xyd = averagingxyd(arecord2["xyd"], n_theta=n_theta, averaging=1)

    iterxydinput = arecord2["iterxydinput"]
    for update_iterxydinput in [1]:
        iterxydinput["xmax"] = xmax
        iterxydinput["ymax"] = ymax
        iterxydinput["xacoeffnew"] = arecord2["xacoeffnew"]
        iterxydinput["xydlast"] = xyd
        iterxydinput["usexp0dlast"] = 1
        iterxydinput["iteration_parameters"] = iteration_parameters

    arecord2["iterxydinput"] = iterxydinput
    tt1 = [["change n_theta 4", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    ##########################################################
    arecord = iterXs(
        [arecord2],
        arecord2["xacoeffnew"],
        xyd,
        iterxydinput,
        xmax=xmax,
        ymax=ymax,
        niter=iteration_parameters["iter_number"],
        oneturntpsa=oneturntpsa,
        useinversetpsa=None,
        wwJwinv=wwJwinv,
        averaging=1,
        number_of_iter_after_minimum=iteration_parameters[
            "number_of_iter_after_minimum"
        ],
        applyCauchylimit=applyCauchylimit,
    )
    tt1 = [["change n_theta 5", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    try:
        if plotconvergence:
            if oneturntpsa != "ELEGANT":
                iter_number, lndxy = plot1Dtheta_vs_xydconvergence_from_3Ddata(
                    arecord, n_theta=n_theta2, plot130=False
                )
            else:
                iter_number, lndxy = plotxydconvergence(
                    arecord, xc=xmax, codeused="ELEGANT"
                )
    except Exception as err:
        if DEBUG:
            print(dir(err))
            print(err.args)
        lndxy = np.nan
    tt1 = [["change n_theta 6", time.time(), time.time() - tt1[0][1]]]
    tt1 = [["change n_theta 7", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    plt.show()
    # import gc
    # gc.collect()
    return arecord, lndxy


def averagingxyd(xyd, n_theta=20, averaging=1):
    # import pdb; pdb.set_trace()
    # xyd=arecord[67]['xyd']
    if averaging == 0:
        return xyd
    xydr = xyd.reshape([4, n_theta, n_theta])
    xydn = xydr.copy()
    for ik in range(n_theta):
        xydn[0][ik] = np.mean(xydr[0][ik])
        xydn[1][ik] = np.mean(xydr[1][ik])

    for ik in range(n_theta):
        xydn[2][:, ik] = np.mean(xydr[2][:, ik])
        xydn[3][:, ik] = np.mean(xydr[3][:, ik])
    """
        plt.figure(61)
        plt.plot(xyd[0],xyd[1],'.')
        plt.plot(xydr[0][1],xydr[1][1],'g.')
        plt.plot(xydn[0],xydn[1],'r.')
        #plt.axes().set_aspect('equal')
        plt.figure(62)
        plt.plot(xyd[2],xyd[3],'.')
        plt.plot(xydr[2][:,0],xydr[3][:,0],'g.')
        plt.plot(xydn[2],xydn[3],'r.')
        """
    return xydn.reshape([4, n_theta**2])


"""
ar,ar4=test_difference_of_one_turn_tpsa_with_ELEGANT()
ar5=xyntheta_from_X_change_ntheta(xmax=-22e-3,arecordi=ar4[-1],n_theta2=30, cutoff2=15)
for ik in range(len(ar)):
        print (ik, errorcount(ar,ik))

dfmax, invmax=studyiter4(ar)
"""


def db_main_lines(ar3, i):
    db = ar3[i]["v12n"]["bnm1"] - ar3[i + 1]["v12n"]["bnm1"]
    n_theta = ar3[i]["lattice"].n_theta
    nux = ar3[i]["v12n"]["nux"]
    nuy = ar3[i]["v12n"]["nuy"]
    dbs = db.reshape([n_theta**2])
    tmp3 = abs(dbs)
    tmp4 = vph.matrixsort(
        tmp3, n_theta
    )  # grid index of sorted phi2 2D fft sequence tmp3 is frm of n_theta**2x1, while tmp4 is of form n_theta**2x2
    tmp5 = list(zip(*tmp4))  # the indexes of top 20 spectral peaks
    tmp6 = np.argsort(tmp3)
    peakvaluesint1 = array(dbs)[
        tmp6
    ]  # the value of top 20 peaks in the phi2 2D fft spectrum
    peaksposition0int1 = vph.modularToNo2(array(tmp5[0]), n_theta)
    peaksposition1int1 = vph.modularToNo2(array(tmp5[1]), n_theta)

    tmp1 = list(
        zip(
            *[
                (peaksposition0int1) * nux + peaksposition1int1 * nuy,
                peakvaluesint1,
                peaksposition0int1,
                peaksposition1int1,
            ]
        )
    )
    for i in tmp1[-6:]:
        print(
            "{:20} {:40} {:20} {:20} ".format(
                i[0].real, abs(i[1]), int(i[2]), int(i[3])
            )
        )
        # print(i[0].real, "\t", abs(i[1]), "\t", int(i[2]), "\t", int(i[3]))


def plot3Dxydconvergence(arecord, plot3d=True):
    codeused = "tpsa"
    for ar in arecord:
        if "lattice" not in ar:
            pass
        else:
            cutoff = ar["lattice"].Cauchylimit["aliasingCutoff"]
            break
    tmp = [
        [k] + f["deltaxydlist"]
        for k, f in enumerate(arecord)
        if "deltaxydlist" in f.keys() and "err.args" not in f
        # and "iterxydinput" in f.keys()
    ]

    # "deltaxydlist" in f.keys() and "err.args" not in f and 'iterxydinput' in f.keys() are first condition for convergence
    k, iter_number, xlist1, ylist1, lndeltaxyd = list(zip(*tmp))

    # tmp12 = np.array([i for j, i in enumerate(tmp) if not np.isnan(i[4])])
    tmp12 = np.array(tmp)
    minf = np.min([i for i in tmp12[:, -1] if (not np.isnan(i)) and i != -np.inf])
    tmp12 = [
        [i[0], i[1], i[2], i[3], minf] if i[4] == -np.inf else i
        for j, i in enumerate(tmp12)
    ]
    tmp2 = np.array(list(zip(*tmp12)))

    xset = np.sort(list(set(xlist1)))

    x = []
    y = []
    z = []
    mainidx = []
    for xc in xset:
        tmp3 = np.where(abs(tmp2[2] - xc) < 1e-6)[0]  # tp2[2] is xmax
        mainidx.append(
            list(map(int, tmp2[0][tmp3].tolist()))
        )  # the first term is the seed from previous run
        x = x + (list(range(len(tmp3))))
        y = y + (np.ones(len(tmp3)) * xc).tolist()
        z = z + tmp2[4][tmp3].tolist()

    if plot3d:
        fig = plt.figure(151)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c="r", marker="o")

        ax.set_xlabel("\n\niteration", fontsize=20)
        ax.set_ylabel("\n\nx (m)", fontsize=20)
        ax.set_zlabel(
            "                         \n\n\n\nln(|xyd(n)-xyd[n+1]|)\n\n", fontsize=20
        )
        # ax.set_xlim3d(-4,4)
        # ax.set_ylim3d(-4,4)
        ax.set_zlim3d(minf, 0)
        plt.title("Fig.1 lndeltaxyd vs. x and iteration number\n cutoff=" + str(cutoff))
        plt.savefig("junk1.png")
    ik = 0
    while ik >= 0:
        if "iterxydinput" in arecord[ik]:
            iteration_parameters = arecord[ik]["iterxydinput"]["iteration_parameters"]
            ik = -1
        else:
            ik = ik + 1
    x_iter_lndxy = x, y, z, mainidx, iteration_parameters
    # here x,y,z are list of iteration number for every arecord[i]
    # y is a list of xmax for every arecord[i]
    # z is a list of ln(delta_xyd) for every arecord[i]
    return x_iter_lndxy, xset, cutoff


# use result from plot3Dxydconvergence
def plot1Dxydconvergence_from_3Ddata(y, z, cutoff, xc=-21.1e-3):
    idx = [j for j, i in enumerate(y) if abs(i - xc) < 1e-9]
    z1 = np.array(z)[idx]
    plt.figure(130)
    if len(z1) != 0:
        plt.plot(z1, "o", label="x=" + str(xc))
    plt.legend(loc="best")
    plt.xlabel("iteration number")
    plt.ylabel("ln(|xyd[n]-xyd[n+1]|)")
    plt.title("Fig.130 convergence rate vs. iteration number,cutoff=" + str(cutoff))
    plt.savefig("junk130.png")
    return


def plot_scanmin(xset, minzlist, idxminlist, cutoff, plotscmain="lndxy vs x", label=""):
    plt.figure(141)
    plt.plot(xset, minzlist, "-o", label=label)  # , label=r"$\ln(|X_n-X_{n-1}|)$")
    plt.plot(xset, idxminlist, "-")  # , label="index where iteration stops ")
    plt.legend(loc="best")
    if plotscmain == "lndxy vs x":
        xlabel = "x (m)"
    elif plotscmain == "lndxy vs nth":
        xlabel = r"$n_{\theta}$"
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(r"$\ln(|X_n-X_{n-1}|)$", fontsize=20)
    # plt.ylabel("ln(|xyd[n]-xyd[n+1]|)")
    if type(cutoff) == int:
        plt.title(
            "Fig.141 max convergence rate vs. " + xlabel + ", cutoff=" + str(cutoff)
        )
    else:
        plt.title("Fig.141 max convergence rate vs. " + xlabel)
    axes = plt.gca()
    tmp = [i for i in minzlist if not np.isnan(i) and not np.isinf(i)]
    axes.set_ylim([np.min(tmp), np.max(idxminlist) + 1])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("junk141.png")
    return


def scanmin(x_iter_lndxy, xset, cutoff, plotscmain="no plot", label="", scantype="x"):
    x, y, z, mainidx, iteration_parameters = x_iter_lndxy
    iter_number = iteration_parameters["iter_number"]
    x1 = []
    minzlist = []
    idxminlist = []
    arminidx = []
    flatmainidx = [i for j in mainidx for i in j]
    # remove list in list adn empty list in mainidx
    diverge = []
    xset1 = np.array([i for i in xset if i in y])

    for xc in xset1:
        xidx = [j for j, i in enumerate(y) if abs(i - xc) < 1e-9 and not np.isnan(z[j])]
        # xidx is the indixes for all j for ar[j]['xmax']=xc and ar[j]["deltaxydlist"][-1] is not nan
        # the index in xidx refers to x,y,z, which is not the index for ar
        aridx = np.array(flatmainidx)[xidx]  # [1:]
        # aridx is the index in arecord[aridx]
        # the point before first xidx is from previous run of ar, or from just change n_theta,
        # so is already removed in lt54.plot3Dxydconvergence
        # here we take xidx[1:] agin because there is no iteration at 0, so should be excluded, but we keep it.
        if scantype == "ntheta including first iteration":
            z1 = np.array(z)[xidx][0:]
        elif scantype == "x":
            z1 = np.array(z)[xidx][1:]
        elif scantype == "ntheta excluding first iteration":
            ztmp = [i for i in np.array(z)[xidx] if not np.isnan(i)]
            z1 = np.array(ztmp)[1:]

        # [0:] the first point is excluded because there is no iteration at 0, so should be excluded, but we keep it.
        # z1 is lndxy for selected xc
        if len(z1) > 0:
            minz = np.min(z1)
            idxmin = np.where(z1 == minz)[0][-1]  # choose the last minz position
            # the index refers to z1 or aridx, which is not the index in arecord
            min_index = aridx[idxmin]  # min_index refers to arecord[min_idex]
            arminidx.append(min_index)
            # accumulate the min index (refers to index of ar) and minz for xc
            minzlist.append(minz)
            idxminlist.append(idxmin)  # + 1)
            # the item in idxminlist is aindex referes to the index of ze, not ar.
            # determine whether iteration diverges.
            if len(z1) < iter_number - 1:  # - 1:
                # if the iteration terminated before reaching iter_number, it means divergence
                diverge.append([xc, 1])
            elif (z1[-3] - z1[-1]) / 2 < 0.25:
                # if last two steps slope less than 0.5/step, takes as divergent.
                diverge.append([xc, 1])
            else:
                diverge.append([xc, 0])
            # idxmin+1 is then sequence number of all ar[i] where xmax=xc
        else:
            minzlist.append(np.nan)
            idxminlist.append(0)
            diverge.append([xc, 1])
            if len(xidx) > 0:
                arminidx.append(xidx[0])
            else:
                arminidx.append(0)

    if plotscmain != "no plot":
        plot_scanmin(
            xset1, minzlist, idxminlist, cutoff, plotscmain=plotscmain, label=label
        )  # "lndxy vs x", or "lndxy vs nth"
    return xset1, minzlist, arminidx, idxminlist, diverge


def survival_turn_average(
    arecord, i=185, npass=6400, dxrange=np.arange(-20e-6, -10e-6, 1e-7)
):
    lattice = arecord[1]["lattice"]
    xfix = lattice.xfix
    xpfix = lattice.xpfix
    deltap = lattice.deltap
    Zpar = arecord[1]["iorecord"]["inversevinput"]["Zpar"]
    vna = arecord[i]["vna"]
    ymax = arecord[i]["ymax"]
    xmax0 = arecord[i]["xmax"]

    nuxy = []
    nloss = []
    nturn = []
    for dx in dxrange:
        xmax = xmax0 + dx
        vn, xy, Zse, nuxt, nuyt = trackingvn(
            xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass, lattice
        )
        nuxy.append([nuxt, nuyt])
        nturn.append([xmax, len(xy[0])])
        if len(xy[0]) < npass:
            nloss.append([i, nuxt, nuyt, xmax])
    nuxy = list(zip(*nuxy))
    nturn = np.array(list(zip(*nturn)))
    nloss = list(zip(*nloss))
    plt.figure(131)
    plt.plot(
        nturn[0],
        nturn[1],
        "ro",
        label="number of turn when tracking ends, npass=" + str(npass),
    )
    plt.legend(loc="best")
    plt.savefig("junk131.png")
    return nturn, nloss


def turnnumberaccumulate(ar3, dxrange=np.arange(520e-6, 540e-6, 1e-7), npass=409600):
    # add number of survival turns data in dxrange to exssiting data in tmp2, this is to be used by cut and paste
    nturn, nloss = survival_turn_average(ar3, dxrange=dxrange, npass=npass)
    tmp2 = nturn.tolist()  # tmp2 + nturn.tolist()
    tmp3 = np.array(tmp2)
    tmp4 = tmp3.reshape([33, 2])
    plt.figure(133)
    plt.plot(sum(tmp4[:, 0]), sum(tmp4[:, 1]), ".")
    # tmp4 is a list of (xlist1,nturnlist1,xlist2,nturnlist2,xlist3,nturnlist3,...xlistm,nturnlistm)
    return tmp4


def turn_number_averaging(
    tmp4, dn=30
):  # this function needs to be used as cut and paste, there is bug.
    tmp2 = rl("junknturn")  # rl("nturn_record")
    tmp3 = np.array(tmp2)
    ln = int(len(tmp3) / 2)
    tmp4 = tmp3.reshape([ln, 2])
    plt.figure(133)
    flatx = [i for subi in tmp4[:, 0] for i in [subi]]
    flatnturn = [i for subi in tmp4[:, 1] for i in [subi]]
    plt.plot(flatx, flatnturn, ".")
    sx = flatx  # sum of list is to join the list , not to add them!
    sturn = flatnturn
    idx = np.argsort(sx)
    sx1 = np.array(sx)[idx]
    sturn1 = np.array(sturn)[idx]
    sturn2 = []
    for i in range(dn, len(sx1) - dn):
        sturn2.append([sx1[i], np.mean(sturn1[i - dn : i + dn])])

    sturn2 = list(zip(*sturn2))
    plt.figure(134)
    plt.plot(
        sturn2[0], sturn2[1], ".", label="number of turns average over dn=" + str(dn)
    )
    plt.legend(loc="best")
    plt.xlabel("x (m)")
    plt.ylabel("number of turns of survival averaged")
    plt.savefig("junk134.png")
