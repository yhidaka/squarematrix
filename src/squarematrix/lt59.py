# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time
import pickle
import pdb
import gc
from types import SimpleNamespace

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

import PyTPSA

t0 = [["start", time.time(), 0]]
timing = [["start", time.time(), 0]]
from . import squarematrixdefinition as sqdf

# from .fortran import zcolnew
# import scipy
# from scipy import loadtxt
# from scipy.interpolate import interp2d
# from scipy import optimize

# import zcol
from . import yunaff
from . import iterModules59 as vph  # iterModules52 as vph

# import lte2tpsa2map45 as lte2tpsa
# from cytpsa import TPS
from . import (
    veq59 as veq,
)  # copied from veq52.py. veqnsls2sr_supercell_ch77_20150406_1_deltap0_lt11 as veq#veq20140204_bare_1supcell_deltapm02 as veq
from . import iterlt59 as lt53
from .iterlt59 import xyntheta_from_X0
from .iterlt59 import change_n_theta

# from iterlt59 import lattice

t0.append(["41", time.time(), time.time() - t0[-1][1]])

## reload(lte2tpsa)
# reload(vph)
# reload(veq)
# reload(lt53)
## reload(jfdf)

"""
1.change ltefilename="20140204_bare_1supcell" in  lt40.py
2.check the lte file name
3.Choose uselte2madx=1 in veq.gettpsa(ltefilename,nv=nv,norder=norder,uselte2madx=0) \
        to read lte till fort.18 is generates, otherwise choose 0
4.choose running sequence in iterxacoeffnew(), in particular, acoeff0, xmax, ymax in runbnm in iterxacoeffnew().
5.choose n_theta=16#40,  alphadiv=16#40
6.choose uarray=np.array([ux0,ux1,ux2,uy0]) in veq. by setting up veq.uarray here
5.And, change acoeff0=array([[ 1+0j, 0j,0j,0j],[0j, 0j,   0j,1+0j]])
7.change nvar in veq=4 by veq.nvar here
8.make sure MA0, and W1 in tracking_dp_ltefilename.ele used in veq is consistent the MA0 and W1 in the *.lte file (not MA1, or W0 for example)
9.make sure the ring start from ring=(MA0,W1,...)
10. to scan, use: arecord=runm012_usexacoeff0(xlist=np.arange(-35e-3,36e-3,5e-3)+1e-6,ylist=np.arange(0,16e-3,5e-3)+1e-4,n_theta=8)
11 to plot contours of scan, use: frecontour(arecord,-4.4,0,0.1,vlabel='bnm1max')
"""

# plt.ion()

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
    ff = open(filename, "ab")
    pickle.dump(x, ff)
    ff.close()


def rla(filename):
    ff = open(filename, "rb")
    objs = []
    while 1:
        try:
            objs.append(pickle.load(ff, encoding="latin1"))
        except EOFError:
            break
    #
    ff.close()
    return objs


def cl():
    gc.collect()
    gc.collect()
    gc.collect()
    plt.close("all")


t0.append(["97", time.time(), time.time() - t0[-1][1]])

# 3. Define functions for analysis
# print("\n3. Define functions for analysis.")
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import FormatStrFormatter


def nuxvzx(
    arecord,
    tracking=False,
    npass=800,
    plotnuxvzx=True,
    xall=np.arange(-18e-3, -19.02e-3, -0.05e-3),
    diverge_condition=3,
):  # just give k vs. xmax, ymax
    x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(arecord, plot3d=False)
    xset, convergencerate, arminidx, idxminlist, diverge = scanmin(
        x_iter_lndxy, xset, cutoff, diverge_condition=diverge_condition,
    )
    idx = [j for j, i in enumerate(convergencerate) if not np.isnan(i)]
    convergencerate = np.array(convergencerate)[idx]
    arminidx = np.array(arminidx)[idx]
    diverge = np.array(diverge)[idx]
    xset = np.array(xset)[idx]
    nux = [arecord[i]["v12n"]["nux"] for i in arminidx]
    nuy = [arecord[i]["v12n"]["nuy"] for i in arminidx]
    y = arecord[1]["ymax"]
    xnan = [i for i in xall if len(np.where(abs(xset - i) < 1e-10)[0]) == 0]
    # find points in xall where there is no match in xset
    if plotnuxvzx:
        plt.figure(143)
        plt.scatter(
            nux,
            nuy,
            10,
            convergencerate,
            marker="s",
            cmap=cm.jet,
            vmin=-20,  # min(convergencerate),
            vmax=0,  # max(convergencerate),
        )
        didx = np.where(np.array(diverge).transpose()[1] == 1)
        plt.plot(
            array(nux)[didx],
            array(nuy)[didx],
            "rx",
            markersize=5,
            label="nuy non-convergent point",
        )
        plt.xlabel("nux", fontsize=15)
        plt.ylabel("nuy", fontsize=15)
        p = plt.colorbar(fraction=0.04, pad=0.01)
        p.ax.set_ylabel("convergence rate", fontsize=15)
        plt.legend(loc="best", prop={"size": 6})
        plt.title(
            "Fig.1 nux,nuy color coded by convergence rate\nfor various x, with y="
            + str(y)
            + ", cutoff="
            + str(cutoff),
            fontsize=15,
        )
        # plt.gca().invert_yaxis()#This make the position of y same as row number of a matrix
        # plt.axes().set_aspect("equal")
        plt.savefig("junk1.png")
        plt.tight_layout()

        plt.figure(2)
        plt.plot(xset, nux, "o", label="nux")
        plt.plot(
            array(xset)[didx], array(nux)[didx], "rx", label="nuy non-convergent point"
        )
        plt.xlabel("x (m)", fontsize=15)
        plt.ylabel("nux", fontsize=15)
        plt.legend(loc="best", prop={"size": 6})
        plt.title(
            "Fig.2 nux, vz. x, with y=" + str(y) + ", cutoff=" + str(cutoff),
            fontsize=15,
        )
        plt.tight_layout()
        plt.savefig("junk2.png")

        plt.figure(3)
        plt.plot(xset, nuy, "o", label="nuy")
        plt.plot(
            array(xset)[didx], array(nuy)[didx], "rx", label="nuy non-convergent point"
        )
        plt.xlabel("x (m)", fontsize=15)
        plt.ylabel("nuy", fontsize=15)
        plt.legend(loc="best", prop={"size": 6})
        plt.title(
            "Fig.3 nuy vz. x, with y=" + str(y) + ", cutoff=" + str(cutoff), fontsize=15
        )
        plt.tight_layout()
        plt.savefig("junk3.png")
        if tracking == True:
            lattice = arecord[1]["lattice"]
            xfix = lattice.xfix
            xpfix = lattice.xpfix
            deltap = lattice.deltap
            Zpar = arecord[1]["iorecord"]["inversevinput"]["Zpar"]

            nuxy = []
            nloss = []
            nturn = []
            for i in arminidx:
                vna = arecord[i]["vna"]
                ymax = arecord[i]["ymax"]
                xmax = arecord[i]["xmax"]
                vn, xy, Zse, nuxt, nuyt = lt53.trackingvn(
                    xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass, lattice
                )
                nuxy.append([nuxt, nuyt])
                nturn.append([xmax, len(xy[0])])
                if len(xy[0]) < npass:
                    nloss.append([i, nuxt, nuyt, xmax])

            n_theta = arecord[arminidx[0]]["n_theta"]
            ymax = arecord[arminidx[0]]["ymax"]
            nuxynan = []
            nlossnan = []
            nturnnan = []
            for i, xmax in enumerate(xnan):
                print("2: i,xmax", i, xmax)
                xy, nuxt, nuyt = veq.xtracking(
                    xmax,
                    ymax - 0e-6,
                    npass,
                    n_theta,
                    deltap,
                    xfix,
                    xpfix,
                    om1accurate=0,
                )
                nuxynan.append([nuxt, nuyt])
                nturnnan.append([xmax, len(xy[0])])
                if len(xy[0]) < npass:
                    nlossnan.append([i, nuxt, nuyt, xmax])

            nuxy = list(zip(*nuxy))
            nturn = np.array(list(zip(*nturn)))
            nloss = list(zip(*nloss))

            nuxynan = list(zip(*nuxynan))
            nturnnan = np.array(list(zip(*nturnnan)))
            nlossnan = list(zip(*nlossnan))

            plt.figure(131)
            plt.plot(
                nturn[0],
                nturn[1],
                "ro",
                label="number of turn when tracking ends, npass=" + str(npass),
            )
            if len(xnan) > 0:
                plt.plot(
                    nturnnan[0],
                    nturnnan[1],
                    "ro",
                    label="number of turn when tracking ends, npass=" + str(npass),
                )
            plt.legend(loc="best")
            plt.savefig("junk131.png")
            plt.figure(143)
            plt.plot(nuxy[0], nuxy[1], "y.", markersize=2, label="tracking")
            if len(xnan) > 0:
                plt.plot(nuxynan[0], nuxynan[1], "y.")
            if len(nloss) > 0:
                plt.plot(
                    nloss[1],
                    nloss[2],
                    "kx",
                    markersize=6,
                    label="tracking lost particle, npass=" + str(npass),
                )
            plt.legend(loc="best", prop={"size": 10})
            plt.savefig("junk1.png")

            plt.figure(2)
            plt.plot(xset, nuxy[0], ".", label="nux tracking")
            if len(xnan) > 0:
                plt.plot(xnan, nuxynan[0], "y.")
            if len(nloss) > 0:
                plt.plot(
                    nloss[3],
                    nloss[1],
                    "yx",
                    label="tracking lost particle, npass=" + str(npass),
                )
            if len(nlossnan) > 0:
                plt.plot(
                    nlossnan[3],
                    nlossnan[1],
                    "yx",
                    label="tracking lost particle, npass=" + str(npass),
                )
            plt.legend(loc="best", prop={"size": 6})
            plt.savefig("junk2.png")
            plt.figure(3)
            plt.plot(xset, nuxy[1], ".", label="nuy tracking")
            if len(xnan) > 0:
                plt.plot(xnan, nuxynan[1], "y.")
            if len(nloss) > 0:
                plt.plot(
                    nloss[3],
                    nloss[2],
                    "yx",
                    label="tracking lost particle, npass=" + str(npass),
                )
            if len(nlossnan) > 0:
                plt.plot(
                    nlossnan[3], nlossnan[2], "yx",
                )
            plt.legend(loc="best", prop={"size": 6})
            plt.savefig("junk3.png")
        else:
            nuxy = None
        plt.show()
    return idxminlist, convergencerate, nux, nuy, diverge, nuxy, xset


# this uses the result of scan_around_DA() to find the maximum growing rate at index i (e.g., i=185) in the bnm spectrum
def db_main_lines(ar3, i):
    db = ar3[i]["v12n"]["bnm1"] - ar3[i + 1]["v12n"]["bnm1"]
    n_theta = ar3[i]["lattice"].n_theta
    nux = ar3[i]["v12n"]["nux"]
    nuy = ar3[i]["v12n"]["nuy"]
    dbs = db.reshape([n_theta ** 2])
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


def kxmax(arecord, xc=-21e-3):  # just give k vs. xmax, ymax
    # for ik in range(len(arecord)):
    #    print(ik, errorcount(arecord, ik))

    tmp = [
        [j, i["xmax"], i["ymax"]] for j, i in enumerate(arecord) if "acoeff" in i.keys()
    ]
    # print(tmp)
    tmp1 = list(zip(*tmp))
    xset = set(tmp1[1])
    tmp2 = np.where(abs(np.array(tmp1[1]) - xc) < 1e-6)[0]
    tmp3 = [tmp[i] for i in tmp2]
    return tmp3, xset


def kntheta(arecord, ntheta=16):  # just give k vs. xmax, ymax
    # for ik in range(len(arecord)):
    #    print(ik, errorcount(arecord, ik))

    tmp = [[j, i["n_theta"]] for j, i in enumerate(arecord) if "n_theta" in i.keys()]
    # print(tmp)
    # print(tmp)
    tmp1 = list(zip(*tmp))
    nthetaset = set(tmp1[1])
    tmp2 = np.where(abs(np.array(tmp1[1]) - ntheta) < 1e-6)[0]
    tmp3 = [tmp[i] for i in tmp2]
    return tmp3, nthetaset


# use the arecord resulted from scan_around_DA()
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

    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.ticker import FormatStrFormatter

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
def plot1Dxydconvergence_from_3Ddata(arecord, xc=-21.0e-3, label=True):
    x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(arecord, plot3d=False)
    x, y, z, mainidx, iteration_parameters = x_iter_lndxy
    idx = [j for j, i in enumerate(y) if abs(i - xc) < 1e-9]
    z1 = np.array(z)[idx]
    plt.figure(140)
    if len(z1) != 0:
        plt.plot(
            z1, "o", label="x=" + str(xc) + "\n one tunr map:tpsa",
        )
    if label:
        plt.legend(loc="best", prop={"size": 16})
    plt.xlabel("iteration number", fontsize=20)
    plt.ylabel(r"$\ln(|X_n-X_{n-1}|)$", fontsize=20)
    plt.title(
        ""
    )  # Fig.140 convergence rate vs. iteration number,cutoff=" + str(cutoff))
    plt.savefig("junk130.png")
    return


# use result from plot3Dxydconvergence
def scanmin(
    x_iter_lndxy,
    xset,
    cutoff,
    plotscmain="no plot",
    label="",
    scantype="x",
    diverge_condition=3,
):
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
        xidx = [j for j, i in enumerate(y) if abs(i - xc) < 1e-9]
        # xidx is the indixes for all j for ar[j]['xmax']=xc and ar[j]["deltaxydlist"][-1] is not nan
        # the index in xidx refers to x,y,z, which is not the index for ar
        aridx = np.array(flatmainidx)[xidx]  # [1:]
        # aridx is the index in arecord[aridx]
        # the point before first xidx is from previous run of ar, or from just change n_theta,
        # so is already removed in lt54.plot3Dxydconvergence
        # here we take xidx[1:] agin because there is no iteration at 0, so should be excluded, but we keep it.
        if scantype == "including first iteration":
            z1 = np.array(z)[xidx][0:]
        elif scantype == "x" or "excluding first iteration":
            z1 = np.array(z)[xidx][1:]

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
            idxminlist.append(
                idxmin + 1
            )  # + 1) when using scantype="including first iteration"
            # the item in idxminlist is aindex referes to the index of ze, not ar.
            # determine whether iteration diverges.
            if (
                idxmin < diverge_condition
            ):  # previously len(z1) <  iter_number was too stringent
                # if the iteration terminated before reaching diverge_condition, it means divergence
                diverge.append([xc, 1])
            # elif (z1[-3] - z1[-1]) / 2 < 0.25:# we do not use whether last two step increases as divergence condition from 7/11/2021
            # if last two steps slope less than 0.5/step, takes as divergent.
            # diverge.append([xc, 1])
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


def plot_scanmin(xset, minzlist, idxminlist, cutoff, plotscmain="lndxy vs x", label=""):
    plt.figure(141)
    plt.plot(xset, minzlist, "-o", label=label)  # , label=r"$\ln(|X_n-X_{n-1}|)$")
    plt.plot(
        xset, idxminlist, "-", label=label
    )  # , label="index where iteration stops ")
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


def survival_turn(arecord, npass=65536):
    i = 185
    lattice = arecord[1]["lattice"]
    xfix = lattice.xfix
    xpfix = lattice.xpfix
    deltap = lattice.deltap
    Zpar = arecord[1]["iorecord"]["inversevinput"]["Zpar"]
    vna = arecord[i]["vna"]
    ymax = arecord[i]["ymax"]
    nturn_record_data = []
    for xc in np.arange(-19e-3, -21.0e-3, -0.1e-3):
        xrange = xc + np.arange(-1e-6, 1e-6, 1e-7)
        nturn = []
        for xmax in xrange:
            print("xmax=", xmax)
            vn, xy, Zse, nuxt, nuyt = lt53.trackingvn(
                xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass, lattice
            )
            nturn.append([xmax, len(xy[0]), npass])
        nturn = list(zip(*nturn))
        nturn_record_data.append(nturn)
        sv("junknturn", nturn_record_data)
    plt.figure(131)
    for nturn in nturn_record_data:
        plt.plot(
            nturn[0], nturn[1], "o",
        )
    plt.legend(loc="best")
    plt.title("Fig.131 npass=" + str(nturn[2][-1]))
    plt.ylabel("number  of survival turns vs. x")
    plt.savefig("junk131.png")
    plt.figure(132)
    for nturn in nturn_record_data:
        plt.plot(
            np.mean(nturn[0]), np.mean(nturn[1]), "o",
        )
    plt.title("Fig.132 npass=" + str(nturn[2][-1]))
    plt.ylabel("number  of mean turn vs. x")
    plt.legend(loc="best")
    plt.savefig("junk132.png")

    return nturn_record_data


"""
tmp9 = rl("nturn_record")
tmp21 = tmp9 + list(zip(*nturn.tolist()))
tmp4 =np.array(tmp21)
plt.figure(133)
flatx = [i for subi in tmp4[:, 0] for i in [subi]]
flatnturn = [i for subi in tmp4[:, 1] for i in [subi]]
plt.plot(flatx, flatnturn, ".")

sv('nturn_record',tmp21)
"""

"""
# use the arecord resulted from scan_around_DA()
def plot3Dtheta_vs_xydconvergence(arecord, codeused="tpsa"):
    tmp = [
        [k]
        + j["deltaxydlist"]
        + [j["lattice"]["Cauchylimit"]["aliasingCutoff"]]
        + [j["n_theta"]]
        for k, j in enumerate(arecord)
        if "deltaxydlist" in j.keys() and "err.args" not in j
    ]
    k, iter_number, xlist1, ylist1, lndeltaxyd, cutoff, n_theta = list(zip(*tmp))
    # n_theta2 = arecord[k[-2]]["iterxydinput"]["n_theta"]
    norder = arecord[k[-1]]["lattice"]["norder"]
    tmp12 = np.array([i for j, i in enumerate(tmp) if not np.isnan(i[4])])
    minf = np.min(tmp12[:, -2])
    tmp12 = [
        [i[0], i[1], i[2], i[3], minf, i[5], i[6]] if i[4] == -np.inf else i
        for j, i in enumerate(tmp12)
    ]
    tmp2 = np.array(list(zip(*tmp12)))

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

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
        + str(xlist1[1])
        + ",norder="
        + str(norder)
    )
    plt.savefig("junk1.png")
    iteration_parameters = arecord[1]["iterxydinput"]["iteration_parameters"]
    x_iter_lndxy = x, y, z, mainidx, iteration_parameters
    # here x,y,z are list of iteration number for every arecord[i]
    # y is a list of n_theta for every arecord[i]
    # z is a list of ln(delta_xyd) for every arecord[i]
    return x_iter_lndxy, cutoff, n_thetaset


# use result from plot3Dtheta_vs_xydconvergence
def plot1Dtheta_vs_xydconvergence_from_3Ddata(arecord, n_theta=12):
    x_iter_lndxy, cutoff, n_thetaset = plot3Dtheta_vs_xydconvergence(arecord)
    x, y, z, mainidx, iteration_parameters = x_iter_lndxy
    idx = [j for j, i in enumerate(y) if abs(i - n_theta) < 1e-9]
    z1 = np.array(z)[idx]
    plt.figure(130)
    if len(z1) != 0:
        plt.plot(z1, "o", label="n_theta=" + str(n_theta))
    plt.legend(loc="best")
    plt.xlabel("iteration number")
    plt.ylabel("ln(|xyd[n]-xyd[n+1]|)")
    plt.title("Fig.130 convergence rate vs. iteration number")
    plt.savefig("junk130.png")
    return
"""


def determine_starting_index_for_scantheta(
    xmax, xconvergent, convergeindex, minzlist, arminidx
):

    # see choosepreviousxydnew about the following 4 lines to get the two lists: convergeindex and xconvergent
    # and how find the index idx as the starting index in ar for ar2 iteration:
    tmp = abs(xmax - xconvergent)
    tmp1 = np.argsort(tmp)
    tmp2 = convergeindex[tmp1[:3]]
    tmp3 = np.array(minzlist)[tmp2]
    tmp4 = np.argmin(tmp3)
    tmp5 = tmp2[tmp4]
    idx = arminidx[tmp5]
    return idx


def choosepreviousxydnew(ar, xmax=-24e-3, ar_iter_number=4):
    x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(ar, plot3d=False)
    xset, minzlist, arminidx, idxminlist, diverge = scanmin(x_iter_lndxy, xset, cutoff)
    convergeindex = np.where((np.array(idxminlist) - ar_iter_number >= 0))[0]
    # convergeindex = np.where(abs(np.array(idxminlist) - (ar_iter_number - 1)) < 1e-4)[0]
    xconvergent = xset[convergeindex]

    # see choosepreviousxydnew about the following 4 lines to get the two lists: convergeindex and xconvergent
    # and how find the index idx as the starting index in ar for ar2 iteration:
    tmp = abs(xmax - xconvergent)
    tmp1 = np.argsort(tmp)
    tmp2 = convergeindex[tmp1[:3]]
    tmp3 = np.array(minzlist)[tmp2]
    tmp4 = np.argmin(tmp3)
    tmp5 = tmp2[tmp4]
    idx = arminidx[tmp5]
    return idx


def choosepreviousxydnew_old(ar, xmax=-21e-3, ar_iter_number=4):
    # find the index in arecord such that arecord[idx]['xmax'] is either equal to xmax, or next close to xmax,
    # and for all arecord[i] such that the arecord[i]['xmax']==xmax, form a convergent iteration sequence.
    # ar_iter_number=4 means the arecord scan was obtained with maximum iteration number is 4.
    x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(ar, codeused="tpsa", plot3d=False)
    xse, minzlist, arminidx, idxminlist, diverge = scanmin(
        x_iter_lndxy, xset, cutoff, plotscmain="no plot"
    )

    convergeindex = np.where((np.array(idxminlist) - ar_iter_number >= 0))[0]
    # convergeindex = np.where(abs(np.array(idxminlist) - (ar_iter_number - 1)) < 1e-4)[0]
    # convergeindex lists all the indixes in xset where all ar for that x within xset gives a convergent iteration,
    # so len(convergeindex)=<len(xset)

    xconvergent = xset[convergeindex]
    # xconvergent gives the list of xmax, such if all those ar[i] with ar[i]['xmax']==xmax give a sequence of convergent iteration.
    # so len(convergeindex)=len(xconvergent)=<len(xset)
    xidx = np.where(abs(xset - xmax) < 1e-4)[0][0]
    # xidx is the single index in the list xset such that xset[xidx]==xmax)

    tmp = abs(xmax - xconvergent)
    # a list of length=len(xconvergent), each give how far the x in the convergent x list is from the given xmax
    tmp1 = np.argsort(tmp)
    # sort of the indixes in xconvergent (same as the indixes of convergeindex) so that the first few are where it is close to the specified xmax
    tmp2 = convergeindex[tmp1[:3]]
    # choose only the first 3 x's clost to specified xmax, get their index in xset.
    tmp3 = np.array(minzlist)[tmp2]
    # find the minimum lndxy in the 3 terms in minzlist (len(minzlist)==len(xset))
    tmp4 = np.argmin(tmp3)
    # tmp4 is the index of the minimum in tmp3, so the index is the index of tmp2
    tmp5 = tmp2[tmp4]
    # tmp5 gives the index in convergeindex, which gives the minimum lndxy, so tmp5 is not the index in xset yet.
    tmp6 = arminidx[tmp5]
    # tmp6 gives the index is xset which is chosen to give the mimimum of lndxy and its x is close to the specified xmax
    return tmp6


def determine_next_iterate_step_xmax(
    xmax, iterate, x_conv_div_list, iteration_step, step_resolution, scan_direction
):
    if len(x_conv_div_list) == 1:
        # Only first scan of xmax use this to determine direction
        if x_conv_div_list[0][1] == 1:
            xmax = xmax - scan_direction * iteration_step
        elif x_conv_div_list[0][1] == 0:
            xmax = xmax + scan_direction * iteration_step
        iterate = 1
    elif len(x_conv_div_list) > 1:
        tmp = list(zip(*x_conv_div_list))
        # tmp[1] is a list of 1(diverge), or 0(converge), tmp[0] is the list of xmax scanned already
        if 0 in tmp[1][-2:] and 1 in tmp[1][-2:]:
            # if 0 and 1 are in tmp[1] list, then reduce step by half
            iteration_step = iteration_step / 2
        if iteration_step < step_resolution - 1e-4:
            # if step is lower than resolution, stop iteration
            iterate = 0
        if tmp[1][-1] == 1:
            # if last point diverges, next step go opposit to scan direction
            xmax = xmax - scan_direction * iteration_step
        elif tmp[1][-1] == 0:
            # if last point converges, next step go scan direction
            xmax = xmax + scan_direction * iteration_step
        if (
            xmax in tmp[0]
            or abs(xmax) < 1e-3
            or np.sign(xmax) == -np.sign(scan_direction)
        ):  # if the next step already done, stop iteration
            iterate = 0

    return iterate, xmax, iteration_step


def determine_DA(x_conv_div_list):
    x_conv_div_list = list(zip(*x_conv_div_list))
    idx = np.argsort(np.array(x_conv_div_list[0]))
    conv_div_list = np.array(x_conv_div_list[1])[idx]
    x_list = np.array(x_conv_div_list[0])[idx]
    last_convergence = np.where(conv_div_list == 0)[0]
    if len(last_convergence) == 0:
        x_last_convergence = 0.0
        x_diverge = 0.0
    else:
        last_convergence = last_convergence[0]
        x_last_convergence = x_list[last_convergence]
        x_diverge = x_list[last_convergence - 1]
    return x_diverge, x_last_convergence, x_list, conv_div_list


def plotDA(
    x_list,
    ymax,
    x_conv_div_list,
    x_diverge,
    x_last_convergence,
    nth1,
    nth2,
    cutoff,
    norder,
    iteration_step0,
    step_resolution,
    conv_div_list,
):
    plt.figure(141)

    plt.axvline(
        x=x_diverge,
        linewidth=4.0,
        label="point iteration diverges \nnth1=" + str(nth1) + ", nth2=" + str(nth2),
        color="red",
    )
    plt.axvline(
        x=x_last_convergence, label="point iteration still converges", color="cyan",
    )
    plt.legend(loc="lower left", handlelength=1, handleheight=1, prop={"size": 9})
    if type(cutoff) == int:
        plt.title(
            "Fig.141 max convergence rate vs. x, cutoff="
            + str(cutoff)
            + ",\n norder="
            + str(norder)
            + ", iteration_step0="
            + str(iteration_step0)
            + ", \nstep_resolution="
            + str(step_resolution)
            + ", ymax="
            + str(ymax)
        )
    else:
        plt.title("Fig.141 max convergence rate vs. n_theta")
    plt.savefig("junk141.png")

    plt.figure(145)
    x_conv_div_list = list(zip(*x_conv_div_list))
    plt.plot(
        x_list, conv_div_list, "-", label="nth1,nth2=" + str(nth1) + "," + str(nth2),
    )
    plt.plot(x_conv_div_list[0], x_conv_div_list[1], "o")
    plt.xlabel("x (m)")
    plt.ylabel("\niteration divergence 1, convergence 0")
    plt.title(
        "Fig.145 scanDA(nth1,nth2), norder="
        + str(norder)
        + "\n iteration_step0="
        + str(iteration_step0)
        + ", step_resolution="
        + str(step_resolution)
    )
    plt.savefig("junk145.png")
    plt.show()


def plotxyz(
    xset,
    yset,
    minzlist,
    dlow=-30,
    dhi=0,
    plotfilename="",
    xlabel="x (m)",
    ylabel="y (m)",
    spectrallabel="minimum of "
    + r"$ \ln(|X_n-X_{n-1}|_{rms}))$"
    + "\nfor all iteration number n ",
    markersize=50,
    figsize=(12, 4),
):
    # plot x-xp distribution with spectrum of energy as colorbar(linewidth thin to have thin circle, pickradius gives dot size)
    # plt.subplot(111)
    # fig = plt.figure(146)
    fig = plt.figure(num=146, figsize=figsize)
    # dlow, dhi = np.min(minzlist), np.max(minzlist)
    plt.scatter(
        xset,
        yset,
        c=minzlist,
        s=markersize,
        marker="s",
        vmin=dlow,
        vmax=dhi,
        cmap=cm.jet,
        linewidths=0.1,
        pickradius=0.5,
    )
    p = plt.colorbar()
    p.ax.set_ylabel(spectrallabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title("Fig.146 Minimum of convergence per iteration")
    plt.title("")
    plt.grid(True)
    plt.axis([np.min(xset), np.max(xset), np.min(yset), np.max(yset)])
    plt.savefig("junk146.png")
    # plt.show()
    # plt.savefig(plotfilename)
    # plt.close()
    return fig


# ar, ar2=example6(
# xyspectrum(ar2[125])
# gc.collect()  # clean up memory used and not to be used now in previous run
# scanxy()
# plt.show()


def single_point_xyntheta(
    x=-31e-3,
    y=4e-3,
    ntheta=12,
    ar_iter_number=4,
    init_tpsa_output=None,  # init_tpsa_output,
    xyntheta_scan_input=None,  # xyntheta_scan_input,
):  # this function just tell how to use iteration_on_xyntheta, one should directly use iteration_on_xyntheta
    xyntheta_scan_input["ar_iter_number"] = ar_iter_number
    iteration_data = xyntheta_from_X0(
        xmax=x,
        ymax=y,
        ntheta=ntheta,
        init_tpsa_output=init_tpsa_output,
        xyntheta_scan_input=xyntheta_scan_input,
        useinversetpsa=True,
        use_existing_tpsa=1,
        scantype="excluding first iteration",  # "including first iteration",  #
    )
    ar = iteration_data["arecord"]  # this line slow down by about 30%
    minz = iteration_data["minz"]
    idxmin = iteration_data["idxmin"]

    # ar, minz, idxmin, iteration_data=single_point_xyntheta(x=-15e-3)
    # tmp=[ i[0] for i in kxmax(ar,xc=-15e-3)[0] ]
    # artmp=ar[tmp[0]:tmp[-1]+1]
    # plot3Dxydconvergence(artmp,plot3d=True)

    return ar, minz, idxmin, iteration_data


def example1():
    ar, minz, idxmin, iteration_data = single_point_xyntheta(x=-15e-3)
    tmp = [i[0] for i in kxmax(ar, xc=-15e-3)[0]]
    artmp = ar[tmp[0] : tmp[-1] + 1]
    plot3Dxydconvergence(artmp, plot3d=True)
    plot1Dxydconvergence_from_3Ddata(ar, xc=-15e-3)


def example2(ar):
    lt53.kxmax(ar, xc=-21e-3)
    # lt53.shortarecord(ar[104])
    lt53.showarecord(ar[103])


def example3(
    ar, xall=[],  # np.arange(-17.47e-3, -17.51e-3, -0.001e-3),
):
    tt0 = [["nuxvzx start", time.time(), 0]]
    print(tt0)
    idxminlist, convergencerate, nux, nuy, diverge = nuxvzx(
        ar, tracking=True, npass=800, xall=xall,
    )  # False)  # True)  #
    tt1 = [["nuxvzx end", time.time(), time.time() - tt0[0][1]]]
    print(tt0)
    print(tt1)
    lt53.plotbnm(ar)
    lt53.plotferrormin(ar)
    plot1Dxydconvergence_from_3Ddata(ar, xc=-22e-3)
    plot3Dxydconvergence(ar)
    print(tt0)
    print(tt1)
    return (
        idxminlist,
        convergencerate,
        nux,
        nuy,
        diverge,
    )
    gc.collect()
    gc.collect()
    gc.collect()
    """
        xset,
        nturn,
        nuxy,
        nloss,
        xnan,
        nturnnan,
        nuxynan,
        nlossnan,
    )
    """


def scanx(
    xlist=np.arange(-1e-3, -35e-3, -1e-3),
    y=4e-3,
    ntheta=12,
    ar_iter_number=4,
    xyntheta_scan_input=None,
    init_tpsa_input=None,
    init_tpsa_output=None,
):
    tt0 = [["scanxy start", time.time(), 0]]
    print(tt0)

    for input_and_initialize in [0]:
        if xyntheta_scan_input == None:
            xyntheta_scan_input = dict(
                ar_iter_number=ar_iter_number,  # 3  #
                ar2_iter_number=None,  # 0  #
                number_of_iter_after_minimum=2,
                applyCauchylimit=True,
                n_theta_cutoff_ratio=(1, 0),  # (2, 1),  # (1, 1)  #
            )
        if init_tpsa_output is None:
            if init_tpsa_input is None:
                init_tpsa_input = {
                    "nvar": 4,
                    "n_theta": ntheta,  # 4,  #
                    "cutoff": ntheta,  # 12, #
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
            init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)

        tt1 = [["scanxy 2", time.time(), time.time() - tt0[0][1]]]
        print(tt1)

    xyntheta_scan_input["ar_iter_number"] = ar_iter_number
    # ar = []
    minzlist = []
    idxminlist = []

    tt1 = [["scanxy 4", time.time(), time.time() - tt1[0][1]]]
    print(tt1, y)
    ar = []
    xset = []
    minzlist = []
    idxminlist = []
    for x in xlist:
        iteration_data = xyntheta_from_X0(
            xmax=x,
            ymax=y,
            ntheta=ntheta,
            init_tpsa_output=init_tpsa_output,
            xyntheta_scan_input=xyntheta_scan_input,
            useinversetpsa=True,
            use_existing_tpsa=1,
            scantype="excluding first iteration",  # "including first iteration",  #
        )
        if iteration_data != None:
            ar = ar + iteration_data["arecord"]  # this line slow down by about 30%
            minzlist.append(iteration_data["minz"])
            idxminlist.append(iteration_data["idxmin"])
            xset.append(x)

    tt1 = [["scanx end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    n_theta_cutoff_ratio = xyntheta_scan_input["n_theta_cutoff_ratio"]
    cutoff2 = (ntheta // n_theta_cutoff_ratio[0] - n_theta_cutoff_ratio[1],)
    label = "n_theta=" + str(ntheta)
    plot_scanmin(xset, minzlist, idxminlist, cutoff2, label=label)
    gc.collect()
    gc.collect()
    gc.collect()
    # ar, minzlist, idxminlist=scanx()
    return ar, minzlist, idxminlist


def scanxy(
    xlist=np.arange(-32e-3, 35.1e-3, 1e-3),  # (-29e-3, -27.1e-3, 1e-3),  #
    ylist=np.arange(1e-3, 15.1e-3, 1e-3),  # (1.0e-3, 1.1e-3, 1e-3),  #
    ntheta=12,
    mod_prop_dict_list=[],
    savearecord=False,
    xyntheta_scan_input=None,
    init_tpsa_input=None,
    init_tpsa_output=None,
):
    tt0 = [["scanxy start", time.time(), 0]]
    print(tt0)

    for initialize_input in [0]:
        if xyntheta_scan_input == None:
            xyntheta_scan_input = dict(
                ar_iter_number=3,  # 4,  #
                ar2_iter_number=11,  # 12,  # 0,  #
                number_of_iter_after_minimum=2,
                applyCauchylimit=True,
                n_theta_cutoff_ratio=(1, 0),  # (2, 1),  # (1, 1)  #
            )

        if init_tpsa_output is None:
            if init_tpsa_input is None:
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
                    "mod_prop_dict_list": mod_prop_dict_list,
                    "tpsacode": "yuetpsa",
                    "dmuytol": 0.01,
                }
            init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)

    tt1 = [["scanxy 2", time.time(), time.time() - tt0[0][1]]]
    print(tt1)

    init_tpsa_output = change_n_theta(
        init_tpsa_output, ntheta, xyntheta_scan_input=xyntheta_scan_input
    )

    tt1 = [["scanxy 3", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    ar = []
    xyminz = []
    for y in ylist:
        try:
            tt1 = [["scanxy 4", time.time(), time.time() - tt1[0][1]]]
            print(tt1, y)
            for x in xlist:
                iteration_data = xyntheta_from_X0(
                    xmax=x,
                    ymax=y,
                    ntheta=ntheta,
                    init_tpsa_output=init_tpsa_output,
                    xyntheta_scan_input=xyntheta_scan_input,
                    useinversetpsa=True,
                    use_existing_tpsa=1,
                    scantype="excluding first iteration",  # "including first iteration",  #
                )
                if iteration_data == None:
                    pass
                else:
                    xyminz.append([x, y, iteration_data["minz"]])
                    if savearecord:  # this line slow down by about 30%
                        ar = ar + iteration_data["arecord"]

            tt1 = [["scanxy1 5 ", time.time(), time.time() - tt1[0][1]]]
            print(tt1, y)
        except Exception as err:
            print(dir(err))
            print(err.args)
            pass

    tt1 = [["scanxy end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    x, y, z = list(zip(*xyminz))
    fig = plotxyz(x, y, z, markersize=130)
    sv("junk9", [x, y, z])
    # scanxy()
    if savearecord:
        return ar
    else:
        return


def scan_n_theta(
    x=-7e-3,
    y=4e-3,
    ntheta_lim=20,
    ar_iter_number=4,
    xyntheta_scan_input=None,
    init_tpsa_input=None,
    init_tpsa_output=None,
):
    tt0 = [["scan_ntheta start", time.time(), 0]]
    print(tt0)

    for initialize_input in [0]:
        if xyntheta_scan_input == None:
            xyntheta_scan_input = dict(
                ar_iter_number=3,  # 4,  #
                ar2_iter_number=None,  # 12,  # 0,  #
                number_of_iter_after_minimum=2,
                applyCauchylimit=True,
                n_theta_cutoff_ratio=(1, 0),  # (2, 1),  # (1, 1)  #
            )

        if init_tpsa_output is None:
            if init_tpsa_input is None:
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
            init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)

    tt1 = [["scanxy 2", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    xyntheta_scan_input["ar_iter_number"] = ar_iter_number

    ar = []
    thetaset = []
    minzlist = []
    idxminlist = []
    for ntheta in range(4, ntheta_lim):
        iteration_data = xyntheta_from_X0(
            xmax=x,
            ymax=y,
            ntheta=ntheta,
            init_tpsa_output=init_tpsa_output,
            xyntheta_scan_input=xyntheta_scan_input,
            useinversetpsa=True,
            use_existing_tpsa=1,
            scantype="excluding first iteration",  # "including first iteration",  #
        )
        if iteration_data != None:
            ar = ar + iteration_data["arecord"]  # this line slow down by about 30%
            thetaset.append(ntheta)
            minzlist.append(iteration_data["minz"])
            idxminlist.append(iteration_data["idxmin"])

    x_iter_lndxy, cutoff, n_thetaset = lt53.plot3Dtheta_vs_xydconvergence(ar)
    lt53.plot1Dtheta_vs_xydconvergence_from_3Ddata(ar, n_theta=79, plot130=True)

    n_theta_cutoff_ratio = xyntheta_scan_input["n_theta_cutoff_ratio"]
    cutoff = (ntheta // n_theta_cutoff_ratio[0] - n_theta_cutoff_ratio[1],)
    label = "x,y=" + str(x) + ", " + str(y)
    plot_scanmin(
        thetaset, minzlist, idxminlist, cutoff, plotscmain="lndxy vs nth", label=label
    )  # "lndxy vs x", or "lndxy vs nth"
    tt1 = [["scan_ntheta end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    # ar = scan_n_theta(x=-22e-3, ntheta_lim=20)
    # kntheta(ar,ntheta=7)
    # lt53.plot1Dtheta_vs_xydconvergence_from_3Ddata(ar,n_theta=7,plot130=True)
    # xyspectrum(ar[51])
    gc.collect()
    gc.collect()
    return ar


def xyspectrum(ari, npass=800, fign=(128, 129), tracking=True):
    # example: after ar,ar2=example6(), found ntheta=16 is convergent, so
    # use
    # kntheta(ar2,ntheta=16)
    # to find ar2[112] is the convergent result
    # so use xyspectrum(ar2[112]) to plot the spectrum of zx and zy.
    xyd = ari["xyd"]
    Zpar = ari["iorecord"]["inversevinput"]["Zpar"]
    bKi, scalex, norder, powerindex = Zpar
    zxyd = np.dot(bKi[1:5, 1:5], xyd) / scalex
    n_theta = ari["n_theta"]
    zxyd = zxyd.reshape(4, n_theta, n_theta)
    zxd, zyd = zxyd[0], zxyd[2]
    fxd = np.fft.fft2(zxd) / n_theta ** 2
    fyd = np.fft.fft2(zyd) / n_theta ** 2
    nux = ari["v12n"]["nux"]
    nuy = ari["v12n"]["nuy"]
    ###################################################
    plt.figure(fign[0])
    spectrm = [
        [i * nux + j * nuy, abs(fxd[i, j] / abs(fxd[1, 0]))]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    spectrm = array(list(zip(*spectrm)))
    spectrm[0] = vph.modularToNo2(spectrm[0], 1.0)
    plt.plot(spectrm[0], spectrm[1], "ro", markersize=5, label="fxd from xyd")
    plt.ylabel("Fourier coeff. of " + r"$z_x$", fontsize=35)
    plt.title("Fig.128 spectrum of zx")
    plt.xlabel("tune " + r"$\nu_x$", fontsize=35, labelpad=11)
    plt.tick_params(axis="both", which="major", labelsize=18)

    plt.figure(fign[1])
    spectrm = [
        [i * nux + j * nuy, abs(fyd[i, j] / abs(fyd[0, 1]))]
        for i in range(-n_theta // 2, n_theta // 2)
        for j in range(-n_theta // 2, n_theta // 2)
    ]
    spectrm = array(list(zip(*spectrm)))
    spectrm[0] = vph.modularToNo2(spectrm[0], 1.0)
    plt.plot(spectrm[0], spectrm[1], "ro", markersize=5, label="fyd from xyd")
    plt.ylabel("Fourier coeff. of " + r"$z_y$", fontsize=35)
    plt.title("Fig." + str(fign[1]) + " spectrum of zy")
    plt.xlabel("tune " + r"$\nu_y$", fontsize=35, labelpad=11)
    plt.tick_params(axis="both", which="major", labelsize=18)

    ###################################################
    if tracking:
        vna = ari["vna"]
        ymax = ari["ymax"]
        xmax = ari["xmax"]
        lattice = ari["lattice"]
        xfix = lattice.xfix
        xpfix = lattice.xpfix
        deltap = lattice.deltap
        vn, xy, Zse, nuxt, nuyt = lt53.trackingvn(
            xmax, ymax, vna, Zpar, deltap, xfix, xpfix, npass, lattice
        )
        n = len(xy[0])
        ff = np.arange(n) / (1.0 * n)
        zxy = np.dot(bKi[1:5, 1:5], xy) / scalex
        zx, zy = zxy[0], zxy[2]
        fx = np.fft.fft(zx) / n
        fy = np.fft.fft(zy) / n

        plt.figure(fign[0])
        fxnupeak, fxpeak = yunaff.naff(zx)
        fxp1 = np.insert(fx, len(fx), fxpeak)
        ffp1 = np.insert(ff, len(fx), fxnupeak)
        ffm = array(
            vph.modularToNo2(ffp1, 1.0)
        )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
        idx = np.argsort(ffm)
        ffm = ffm[idx]
        fxm = fxp1[idx]
        fxnupeak = array(vph.modularToNo2(np.array(fxnupeak), 1.0))
        plt.plot(ffm, abs(fxm) / np.max(abs(fxm)), "b-", label="fx from tracking")

        plt.axis([-0.5, 0.5, 0, 0.1])
        plt.plot(
            fxnupeak,
            abs(fxpeak) / np.max(abs(fxpeak)),
            "yo",
            markersize=3,
            label="fx naff peak",
        )
        plt.legend(loc="best")
        plt.savefig("junk" + str(fign[0]) + ".png")
        ###################################################

        plt.figure(fign[1])
        fynupeak, fypeak = yunaff.naff(zy)
        fyp1 = np.insert(fy, len(fy), fypeak)
        ffp1 = np.insert(ff, len(fy), fynupeak)
        ffm = array(
            vph.modularToNo2(ffp1, 1.0)
        )  # This is make the spectrum range from -0.5 to 0.5 instead of 0 to 1.
        idx = np.argsort(ffm)
        ffm = ffm[idx]
        fym = fyp1[idx]
        fynupeak = array(vph.modularToNo2(np.array(fynupeak), 1.0))
        plt.plot(ffm, abs(fym) / np.max(abs(fym)), "b-", label="fy from tracking")

        plt.axis([-0.5, 0.5, 0, 0.1])
        plt.plot(
            fynupeak,
            abs(fypeak) / np.max(abs(fypeak)),
            "yo",
            markersize=3,
            label="fy naff peak",
        )
        plt.legend(loc="best")
        plt.savefig("junk" + str(fign[1]) + ".png")


################################# less impotatfunctions


def example4(y=4e-3, ar_ntheta=12):
    tt0 = [["example4 start", time.time(), 0]]
    print(tt0)

    init_tpsa_input = dict(
        nvar=4,
        ar_ntheta=ar_ntheta,
        ar_cutoff=12,
        norder=5,
        norder_jordan=3,
        use_existing_tpsa=1,  # 0,  #
        oneturntpsa="tpsa",  # "ELEGANT",  #
        deltap=-0.025,
        ltefilename="20140204_bare_1supcell",  # "nsls2sr_supercell_ch77_20150406_1",  # 20140204_bare_1supcell",
        mod_prop_dict_list=[
            {
                "elem_name": "Qh1G2c30a",
                "prop_name": "K1",
                "prop_val": -0.6419573146484081,
            },
            {
                "elem_name": "sH1g2C30A",
                "prop_name": "K2",
                "prop_val": 19.83291209974166 + 0,
            },
        ],
        tpsacode="yuetpsa",  # "madx",  #
        dmuytol=0.01,  # 0.005, #
    )
    scanDA_parameters = dict(
        nth1=25,  # 50,  # 60,  # 40,  #
        nth2=30,  # 55,  # 70,  # 45,  #
        ar2_iter_number=8,
        ar_iter_number=4,
        iteration_step=2e-3,
        step_resolution=1e-3,
        number_of_iter_after_minimum=2,
    )

    init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)
    (xset, minzlist, idxminlist, cutoff, ar, scanDAoutput) = scanDA(
        xlist=np.arange(-1.0e-3, -28.1e-3, -1e-3),
        ymax=y,
        scan_direction=-1,
        searchDA=True,  # False,
        init_tpsa_output=init_tpsa_output,
        scanDA_parameters=scanDA_parameters,
        init_tpsa_input=init_tpsa_input,
    )
    tt1 = [["example4 end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    return ar


# First scanusing xacoeff0 from -1mm to -28mm, then search for DA using nth1,nth2 only at divergence border.
def scanDA(
    xlist=np.arange(-1.0e-3, -28.1e-3, -1e-3),
    ymax=4e-3,
    scan_direction=-1,
    searchDA=False,
    init_tpsa_output=None,
    scanDA_parameters=None,
    init_tpsa_input=None,
    mod_prop_dict_list=[],
):
    tt0 = [["scanDA start", time.time(), 0]]
    print(tt0)

    xylist = [[x, ymax] for x in xlist]
    (
        nvar,
        ar_ntheta,
        ar_cutoff,
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

    (
        nth1,
        nth2,
        ar2_iter_number,
        ar_iter_number,
        iteration_step,
        step_resolution,
        number_of_iter_after_minimum,
    ) = [scanDA_parameters[i] for i in scanDA_parameters.keys()]

    renewXinput, lattice, oneturntpsa, wwJwinv = init_tpsa_output
    tt1 = [["scanDA 1.0", time.time(), time.time() - tt0[0][1]]]
    print(tt1)

    ar, minzlist, idxminlist = scanx(
        xlist=xlist,
        y=ymax,
        ntheta=ar_ntheta,
        ar_iter_number=4,
        init_tpsa_input=init_tpsa_input,
        init_tpsa_output=init_tpsa_output,
    )

    tt1 = [["scanDA 1.1", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    # ar, oneturntpsa = rl("junk8")
    # ar is result of scan starting from xacoeff0 to provide initial xyd for n_theta scan to find DA
    # see choosepreviousxydnew about the following 4 lines to get the two lists: convergeindex and xconvergent
    x_iter_lndxy, xset, cutoff = lt53.plot3Dxydconvergence(ar, plot3d=False)
    xset, minzlist, arminidx, idxminlist, diverge = lt53.scanmin(
        x_iter_lndxy, xset, cutoff
    )

    convergeindex = np.where((np.array(idxminlist) - (ar_iter_number - 2) >= 0))[0]
    # convergeindex = np.where(abs(np.array(idxminlist) - (ar_iter_number - 1)) < 1e-4)[0]
    xconvergent = xset[convergeindex]
    tt1 = [["scanDA 2.0", time.time(), time.time() - tt1[0][1]]]
    print(tt1)
    if searchDA:
        if len(xconvergent) != 0:
            tt1 = [["scanDA 2", time.time(), time.time() - tt1[0][1]]]
            print(tt1)

            scantheta_input_list = (
                nth1,
                nth2,
                ar2_iter_number,
                norder,
                use_existing_tpsa,
                oneturntpsa,
                wwJwinv,
                ar,
                convergeindex,
                xconvergent,
                minzlist,
                arminidx,
                number_of_iter_after_minimum,
            )

            idx_of_last_converge_in_convergeindex = np.argmax(abs(xconvergent))
            idx_of_last_converge_in_xset = convergeindex[
                idx_of_last_converge_in_convergeindex
            ]
            xmax_start = xset[idx_of_last_converge_in_xset]

            iterate = 1
            maxlist = []
            x_conv_div_list = []
            xmax = xmax_start
            iteration_step0 = iteration_step
            ar2 = []

            # iterate for specified nth to test convergence as a way to search DA
            while iterate == 1:
                tt1 = [["scanDA 3", time.time(), time.time() - tt1[0][1]]]
                print(tt1)
                starting_index = determine_starting_index_for_scantheta(
                    xmax, xconvergent, convergeindex, minzlist, arminidx
                )
                artmp, lndxylist = scan_nth12_around_DA(
                    xmax, starting_index, scantheta_input_list
                )

                tt1 = [["scanDA 4", time.time(), time.time() - tt1[0][1]]]
                print(tt1)
                growth = []
                for i in lndxylist:
                    # print("i=", i)
                    if type(i) == float or np.isnan(i).any():
                        # i.e., i is number 0, or there is nan in i.
                        growth.append(
                            1
                        )  # this means diverge occured in the iteration process.
                    elif len(i) < ar2_iter_number:
                        # this means the iteration terminated before reaching the specified number, the it diverges
                        growth.append(1)
                    elif i[-1] > i[-3]:
                        # this means even thought the iteration reaches the specified number, if the end is higher than the second last point, it is a  sgrowth
                        growth.append(1)
                    else:
                        growth.append(0)
                        # if the last 3 conditions are not satisfied, iteration converges
                growth = growth[0] * growth[1]
                # only both nth1,nth2 diverge, it the particle is considered lost
                x_conv_div_list.append([xmax, growth])
                maxlist.append([xmax, lndxylist])
                # pdb.set_trace()
                # next: decide whether to continue to search for DA, and if so which direction to go
                iterate, xmax, iteration_step = determine_next_iterate_step_xmax(
                    xmax,
                    iterate,
                    x_conv_div_list,
                    iteration_step,
                    step_resolution,
                    scan_direction,
                )
                ar2 = ar2 + artmp

            tt1 = [["scanDA 5", time.time(), time.time() - tt0[0][1]]]
            print(tt1)
        elif len(xconvergent) == 0:
            x_conv_div_list = [[0, 1]]
            iteration_step0 = iteration_step
            maxlist = [[0, 1]]
            ar2 = []

        x_diverge, x_last_convergence, x_list, conv_div_list = determine_DA(
            x_conv_div_list
        )
        plot_scanmin(xset, minzlist, idxminlist, cutoff)

        plotDA(
            x_list,
            ymax,
            x_conv_div_list,
            x_diverge,
            x_last_convergence,
            nth1,
            nth2,
            cutoff,
            norder,
            iteration_step0,
            step_resolution,
            conv_div_list,
        )
        tt1 = [["scanDA 6", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        scanDAoutput = (
            x_conv_div_list,
            maxlist,
            ar2,
            convergeindex,
            xconvergent,
            xset,
        )
    else:
        scanDAoutput = None
    tt1 = [["scanDA 7", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    return xset, minzlist, idxminlist, cutoff, ar, scanDAoutput


# scan_nth12_around_DA scan for n_theta=50 and 60 only. To be used in scanDA
def scan_nth12_around_DA(xmax, idx, scantheta_input_list):
    (
        nth1,
        nth2,
        ar2_iter_number,
        norder,
        use_existing_tpsa,
        oneturntpsa,
        wwJwinv,
        ar,
        convergeindex,
        xconvergent,
        minzlist,
        arminidx,
        number_of_iter_after_minimum,
    ) = scantheta_input_list
    tt0 = [["scan3 start", time.time(), 0]]
    print(tt0)

    ar2 = []
    lndxylist = []
    # idx = lt53.kxmax(ar, xc=xmax)[0][-1][0]
    tt1 = [["scan3 2", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    for n_theta2 in [nth1, nth2]:
        tt1 = [["scan3 3", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
        artmp, lndxy = lt53.xyntheta_from_X_change_ntheta(
            xmax=xmax,
            ymax=ar[idx]["ymax"],
            arecordi=ar[idx],
            n_theta2=n_theta2,  # 40,
            cutoff2=n_theta2,  # n_theta2 // 2,
            oneturntpsa=oneturntpsa,
            iteration_parameters=dict(
                iter_number=ar2_iter_number,
                number_of_iter_after_minimum=number_of_iter_after_minimum,
            ),
            plotconvergence=True,
            wwJwinv=wwJwinv,
            scan_variable="n_theta",
        )
        ar2 = ar2 + artmp
        lndxylist = lndxylist + [lndxy]
        tt1 = [["scan3 4", time.time(), time.time() - tt1[0][1]]]
        print(tt1)
    # axes = plt.gca()
    # axes.set_ylim([-12, 0])
    tt1 = [["scan3 5", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    return ar2, lndxylist


################################# start a2 scan function
def find_minimum_converge(
    x=-7e-3,
    y=10e-3,
    ntheta_lim=10,
    init_tpsa_output=None,  # init_tpsa_output,
    xyntheta_scan_input=None,  # xyntheta_scan_input,
):
    tt0 = [["scan_ntheta start", time.time(), 0]]
    print(tt0)
    ar = []
    for ntheta in range(4, ntheta_lim):
        iteration_data = xyntheta_from_X0(
            xmax=x,
            ymax=y,
            ntheta=ntheta,
            init_tpsa_output=init_tpsa_output,
            xyntheta_scan_input=xyntheta_scan_input,
            useinversetpsa=True,
            use_existing_tpsa=1,
            scantype="excluding first iteration",  # "including first iteration",  #
        )
        if iteration_data != None:
            ar = ar + iteration_data["arecord"]  # this line slow down by about 30%

    x_iter_lndxy, cutoff, n_thetaset = lt53.plot3Dtheta_vs_xydconvergence(
        ar, plot_3d=False
    )
    thetaset, minzlist, arminidx, idxminlist, diverge = scanmin(
        x_iter_lndxy,
        n_thetaset,
        cutoff,
        plotscmain="no plot",
        label="x,y=" + str(x) + ", " + str(y),
    )
    minzidx = np.argmin(minzlist)
    ntheta = thetaset[minzidx]
    idxmin = arminidx[minzidx]
    return ar, ntheta, idxmin


def find_first_converge(
    x=-2e-3, y=14e-3, ntheta_lim=10, init_tpsa_output=None,  # init_tpsa_output
):
    # ar,ntheta=find_first_converge(x=-4e-3, y=14e-3, ntheta_lim=10)
    # lt53.plot1Dtheta_vs_xydconvergence_from_3Ddata(ar, n_theta=ntheta, plot130=True)
    tt0 = [["scan_ntheta start", time.time(), 0]]
    print(tt0)
    for ntheta in range(4, ntheta_lim):
        try:

            iteration_data = xyntheta_from_X0(
                xmax=x,
                ymax=y,
                ntheta=ntheta,
                init_tpsa_output=init_tpsa_output,
                xyntheta_scan_input=xyntheta_scan_input,
                useinversetpsa=True,
                use_existing_tpsa=1,
                scantype="excluding first iteration",  # "including first iteration",  #
            )
            if iteration_data != None:
                ar = iteration_data["arecord"]  # this line slow down by about 30%

            x_iter_lndxy, cutoff, n_thetaset = lt53.plot3Dtheta_vs_xydconvergence(
                ar, plot_3d=False
            )
            thetaset, minzlist, arminidx, idxminlist, diverge = scanmin(
                x_iter_lndxy,
                n_thetaset,
                cutoff,
                plotscmain="no plot",
                label="x,y=" + str(x) + ", " + str(y),
            )
            if idxminlist[0] > 0:
                return ar, ntheta, idxminlist[0]
        except Exception as err:
            print(dir(err))
            print(err.args)
            pass

    return ar, ntheta, idxminlist[0]


def ar2_xyntheta(
    x=-7e-3,
    y=4e-3,
    n_theta2=12,
    xyntheta_scan_input=None,  # xyntheta_scan_input,
    init_tpsa_output=None,  # init_tpsa_output,
    scantype="including first iteration",
):
    renewXinput, lattice, oneturntpsa, wwJwinv = init_tpsa_output
    (
        ar_iter_number,
        ar2_iter_number,
        number_of_iter_after_minimum,
        applyCauchylimit,
        n_theta_cutoff_ratio,
    ) = [xyntheta_scan_input[i] for i in xyntheta_scan_input.keys()]

    # ar, ntheta, idxmin = find_first_converge(x=x, y=y, ntheta_lim=10)
    ar, ntheta, idxmin = find_minimum_converge(
        x=x,
        y=y,
        ntheta_lim=10,
        init_tpsa_output=init_tpsa_output,
        xyntheta_scan_input=xyntheta_scan_input,
    )

    if ar2_iter_number == 0:
        ar2 = ar
    else:
        # ar is result of scan starting from xacoeff0 to provide initial xyd for n_theta scan to find DA
        # see choosepreviousxydnew about the following 4 lines to get the two lists: convergeindex and xconvergent
        x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(ar, plot3d=False)
        """
        xset, minzlist, arminidx, idxminlist, diverge = scanmin(
            x_iter_lndxy, xset, cutoff, plotscmain="no plot"
        )
        convergeindex = np.where(
            abs(np.array(idxminlist) - (ar_iter_number - 1)) < 1e-4
        )[0]
        xconvergent = xset[convergeindex]
        starting_index = determine_starting_index_for_scantheta(
            xmax, xconvergent, convergeindex, minzlist, arminidx
        )
        """
        starting_index = idxmin
        ar2 = lt53.xyntheta_from_X_change_ntheta(
            xmax=x,
            ymax=y,
            arecordi=ar[starting_index],
            n_theta2=n_theta2,  # 40,
            cutoff2=n_theta2 // n_theta_cutoff_ratio[0] - n_theta_cutoff_ratio[1],
            oneturntpsa=oneturntpsa,
            iteration_parameters=dict(
                iter_number=ar2_iter_number,
                number_of_iter_after_minimum=number_of_iter_after_minimum,
            ),
            plotconvergence=True,
            wwJwinv=wwJwinv,
            scan_variable="n_theta",
            applyCauchylimit=applyCauchylimit,
        )[0]
    # ar,ar2=ar2_xyntheta(x=-2e-3,y=14e-3,n_theta2=6)
    # lt53.plot1Dtheta_vs_xydconvergence_from_3Ddata(ar2, n_theta=6,plot130=True)
    return ar, ar2


def ar2_scanx(ntheta=12, y=4e-3, ar2_iter_number=12, n_theta_cutoff_ratio=(1, 0)):
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
        "use_existing_tpsa": 1,
        "oneturntpsa": "tpsa",
        "deltap": -0.025,
        "ltefilename": "20140204_bare_1supcell",
        "mod_prop_dict_list": [],
        "tpsacode": "yuetpsa",
        "dmuytol": 0.01,
    }
    init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)

    tt0 = [["scanx start", time.time(), 0]]
    print(tt0)
    xyntheta_scan_input["ar2_iter_number"] = ar2_iter_number
    xyntheta_scan_input["n_theta_cutoff_ratio"] = n_theta_cutoff_ratio
    ar2 = []
    for x in np.arange(-1.0e-3, -28.1e-3, -1e-3):
        try:
            ar2 = (
                ar2
                + ar2_xyntheta(
                    x,
                    y,
                    ntheta,
                    xyntheta_scan_input=xyntheta_scan_input,
                    init_tpsa_output=init_tpsa_output,
                )[1]
            )
        except Exception as err:
            print(dir(err))
            print(err.args)
            pass
    x_iter_lndxy, xset, cutoff = plot3Dxydconvergence(ar2, plot3d=False)
    xset, minzlist, arminidx, idxminlist, diverge = scanmin(x_iter_lndxy, xset, cutoff)
    convergeindex = np.where((np.array(idxminlist) - ar2_iter_number >= 0))[0]
    # convergeindex = np.where((ar2_iter_number - np.array(idxminlist) - 1) < 1)[0]
    xconvergent = xset[convergeindex]
    plot_scanmin(xset, minzlist, idxminlist, cutoff)
    tt1 = [["scanx end", time.time(), time.time() - tt0[0][1]]]
    # ar2_scanx()
    print(tt1)
    return ar2


def ar2_scan_n_theta(
    x=-7e-3,
    y=4e-3,
    ntheta_lim=40,
    ar2_iter_number=12,
    n_theta_cutoff_ratio=(1, 0),
    scantype="excluding first iteration",  # "including first iteration", #
):
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
        "use_existing_tpsa": 1,
        "oneturntpsa": "tpsa",
        "deltap": -0.025,
        "ltefilename": "20140204_bare_1supcell",
        "mod_prop_dict_list": [],
        "tpsacode": "yuetpsa",
        "dmuytol": 0.01,
    }
    init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)

    xyntheta_scan_input["ar2_iter_number"] = ar2_iter_number
    xyntheta_scan_input["n_theta_cutoff_ratio"] = n_theta_cutoff_ratio
    tt0 = [["scan_ntheta start", time.time(), 0]]
    print(tt0)

    ar2 = []
    for ntheta in range(4, ntheta_lim):
        try:
            ar2 = (
                ar2
                + ar2_xyntheta(
                    x,
                    y,
                    ntheta,
                    xyntheta_scan_input=xyntheta_scan_input,
                    init_tpsa_output=init_tpsa_output,
                    scantype="including first iteration",
                )[1]
            )
        except Exception as err:
            print(dir(err))
            print(err.args)
            pass

    x_iter_lndxy, cutoff, n_thetaset = lt53.plot3Dtheta_vs_xydconvergence(ar2)
    lt53.plot1Dtheta_vs_xydconvergence_from_3Ddata(ar2, n_theta=17, plot130=True)
    # lt53.plot1Dtheta_vs_xacoeffnewconvergence_from_3Ddata(ar2, n_theta=17, plot131=True)
    thetaset, minzlist, arminidx, idxminlist, diverge = scanmin(
        x_iter_lndxy, n_thetaset, cutoff, plotscmain="lndxy vs nth", scantype=scantype,
    )
    plt.tight_layout()
    tt1 = [["scan_ntheta end", time.time(), time.time() - tt0[0][1]]]
    print(tt1)
    # ar2=ar2_scan_n_theta(x=-7e-3,ntheta_lim=20)
    # lt53.plot1Dtheta_vs_xydconvergence_from_3Ddata(ar2,n_theta=17,plot130=True)
    # xyspectrum(ar2[51])
    return ar2


if __name__ == "__main__":

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
    init_tpsa_output = lt53.init_tpsa(init_tpsa_input=init_tpsa_input)

    if True:
        # Same as examples/example1.py
        example1()

        plt.show()

    if False:
        # Same as examples/example2.py
        ar, minzlist, idxminlist = scanx()
        example3(ar)

    if False:
        # Same as examples/example3.py
        scanxy()

    if False:
        # Same as examples/example4.py
        ar = scan_n_theta()
        xyspectrum(ar[51])

    if False:
        # differs from examples/example5.py by scanDA in exampl4 is replaced by scanx in example5.
        ar = example4()
        example2(ar)

    if False:
        # Same as examples/example6.py
        ar2 = ar2_scanx()
    if False:
        # Same as examples/example7.py
        ar2 = ar2_scan_n_theta()
