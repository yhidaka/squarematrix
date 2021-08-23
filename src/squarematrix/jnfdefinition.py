# Find left Jordan zero eigen basis u for upper-triangular matrix m: u.m=J.u
# see mathematica code in "JordanNormalFormAndTriangularMatrix.nb"

import sys
import copy
import pickle

import numpy as np
from numpy import array
from numpy import linalg

from .fortran import orderdot

global tol
# mn=np.loadtxt("mathematica-JNF-example.dat")
# mn[7,9]=5.33333333333333333333
tol = 1e-9  # 1e-9
tolrank = 1e-9  # 1e-7
display = ""  # "semileftjordanbasis"


eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal = 0, [], 0, [], [], [], [], [], []


def sv(filename, x):
    ff = open(filename, "wb")
    pickle.dump(x, ff)
    ff.close()


def rl(filename):
    ff = open(filename, "rb")
    xx = pickle.load(ff, encoding="latin1")
    ff.close()
    return xx


def initialize():
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal
    diagonal = [mp[i][i] for i in range(0, len(mp))]
    zeros = findzeros(diagonal, tol)
    if display == "initialize":
        print(
            "in initialize 1: zeros=",
            zeros,
            " len(diagonal)=",
            len(diagonal),
            " tol=",
            tol,
        )
    # print diagonal
    tmp = np.zeros(len(mp), dtype=int)
    for j in range(0, len(zeros)):
        tmp[zeros[j]] = j
    # eigenposition is a list of diagonal element of mt, paired with a sequence of zero position while non-zero position is paired with zeros
    # eigenposition =zip(*[diagonal, tmp])
    eigenposition = np.array([diagonal, tmp]).transpose()
    lz = len(zeros)
    tm = np.zeros((lz, lz)) * 0.0j
    if display == "initialize":
        print("in initialize 2: zeros=", zeros)
    return zeros, eigenposition, diagonal, tm


def initializeX():
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal
    # Set row zeros[[eik]] = 1 and all rows number larger than this as 0
    x = np.zeros(len(mp)) * 0.0j
    x[zeros[eik]] = 1.0 + 0.0j
    return x


# def findzeros():
# 	global eik,mp,j,x,tm,eigenposition,zeros,xi,diagonal
#        zeros=[ i for i in range(0,len(diagonal)) if abs(diagonal[i])<tol]
#        warning=[ i for i in range(0,len(diagonal)) if (abs(diagonal[i])>tol and abs(diagonal[i])<10*tol)]
#        return zeros,warning
def findzeros(tmp6, tol):
    zerorows = []
    for i in range(0, len(tmp6)):
        if abs(tmp6[i]) < tol:
            zerorows.append(i)
    return zerorows


def solvingNonzeroRows():
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal
    # For first eigenvector, or the last sequential rows of non-zero rows in the matrix, the RHS of the equation is 0.
    if display == "solvingNonzeroRows":
        print(
            "At p0, in non-zero Solving non-zero row, j=",
            j,
            " eigenposition=",
            eigenposition[j],
        )
    if display == "solvingNonzeroRows":
        print(
            "in solvingNonzeroRows, At p0, eik= ",
            eik,
            " j=",
            j,
            " zeros[eik-1]=",
            zeros[eik - 1],
        )
    if eik == 0.0 or (eik > 0 and j > zeros[eik - 1]):
        # Solve the j'th row of equation mp.x=J.X, i.e., (mp.x)[j]==0 for x[j]
        # i.e., solve for x[j] in equation m[j][k]x[k]=0
        if display == "solvingNonzeroRows":
            print(
                "At p1, in solvingNonzeroRows, eik= ",
                eik,
                " j=",
                j,
                " zeros[eik-1]=",
                zeros[eik - 1],
            )
        if display == "solvingNonzeroRows":
            print(
                "At p1, np.dot(mp[j][j+1:],x[j+1:])=",
                np.dot(mp[j][j + 1 :], x[j + 1 :]),
                " mp[j][j]=",
                mp[j][j],
            )
        # print "mp[j]=", mp[j]
        if display == "solvingNonzeroRows":
            print("x=", x)
        x[j] = -np.dot(mp[j][j + 1 :], x[j + 1 :]) / mp[j][j]
        uu = -np.dot(mp[j][j + 1 :], x[j + 1 :]) / mp[j][j]
        yy = x[j]
        if display == "solvingNonzeroRows":
            print("At p1, Found x[", j, "]=", x[j], "check uu.yy.", uu, yy)
        if display == "solvingNonzeroRows":
            print("At p1, Found x[", j, "]=", x[j])
    elif j < zeros[eik - 1]:
        # For other non-zero rows, the RHS of the equation is determined by the known eigenvectors with lower indexes
        if display == "solvingNonzeroRows":
            print(
                "At p2, in non-zero",
                " Y[[j]]=",
                np.dot(mp[j][j + 1 :], x[j + 1 :]),
                " eik=",
                eik,
                " j=",
                j,
                " x[j]=",
                x[j],
            )
        if display == "solvingNonzeroRows":
            print(
                "At p2, Sum[y[eik,m-1]Xi[[m-1,j]],{m,eik,2,-1}]=",
                sum([tm[eik, m - 1] * xi[m - 1][j] for m in np.arange(eik, 0, -1)]),
            )
        x[j] = (
            sum([tm[eik, m - 1] * xi[m - 1][j] for m in np.arange(eik, 0, -1)])
            - np.dot(mp[j][j + 1 :], x[j + 1 :])
        ) / mp[j][j]
        if display == "solvingNonzeroRows":
            print("At p2, Found x[", j, "]=", x[j])
        if display == "solvingNonzeroRows":
            print(
                "At p2,  Y=",
                np.dot(mp[j][j:], x[j:]),
                ",sum( [tm[eik,m-1]*xi[m-1][j] for m in np.arange(eik,0,-1) ])",
                sum([tm[eik, m - 1] * xi[m - 1][j] for m in np.arange(eik, 0, -1)]),
                " X=",
                x,
            )
    return


def solvingZeroRows():
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal
    # For first eigenvector, or the last sequential rows of non-zero rows in the matrix, the RHS of the equation is 0.
    if display == "solvingZeroRows":
        print("p0 in solvingZeroRows, j=", j, " eigenposition=", eigenposition[j])
    if display == "solvingZeroRows":
        print(
            "in solvingZeroRows, eik= ", eik, " j=", j, " zeros[eik-1]=", zeros[eik - 1]
        )
    if display == "solvingZeroRows":
        print(
            "np.dot(mp[j][j+1:],x[j+1:])=",
            np.dot(mp[j][j + 1 :], x[j + 1 :]),
            " mp[j][j]=",
            mp[j][j],
        )
    # print "mp[j]=", mp[j]
    if display == "solvingZeroRows1":
        print("x=    ", x)
    if display == "solvingZeroRows":
        print("zeros=    ", zeros)
    # ep is the index of eigenvalue at position j
    ep = int(np.real(eigenposition[j][1]))
    vi = [i for i in range(0, j + 1) if abs(mp[j][i]) > tol]
    if display == "solvingZeroRows":
        print("vi=", vi, " mp[j]=", [mp[j][i] for i in range(0, j + 1)])
    if vi == []:
        if display == "solvingZeroRows":
            print("p1", "Y[[j]]=", " eik=", eik, " j=", j, " vi=", vi)
        if display == "solvingZeroRows":
            print(tm[eik, 0], xi[0][j])
        if display == "solvingZeroRows":
            print(
                " \n y[",
                eik,
                ",",
                ep,
                "]=",
                tm[eik, ep],
                ", Y[[",
                j,
                "]]= ",
                np.dot(mp[j][j + 1 :], x[j + 1 :]),
                " -Sum[y[eik,m-1]Xi[[m-1,j]],{m,eik,2,-1}]=",
                sum([tm[eik, m - 1] * xi[m - 1][j] for m in np.arange(eik, 0, -1)]),
            )
        tm[eik, ep] = np.dot(mp[j][j + 1 :], x[j + 1 :]) - sum(
            [tm[eik, m - 1] * xi[m - 1][j] for m in np.arange(eik, 0, -1)]
        )
        if display == "solvingZeroRows":
            print(
                " \n after asign y[eik,ep], y[",
                eik,
                ",",
                ep,
                "]=",
                tm[eik, ep],
                ", Y[",
                j,
                "]= ",
                np.dot(mp[j][j + 1 :], x[j + 1 :]),
                ", Sum[y[eik,m-1]Xi[[m-1,j]],{m,eik,2,-1}]=",
                sum([tm[eik, m - 1] * xi[m - 1][j] for m in np.arange(eik, 0, -1)]),
            )
    return


def eigenv():
    # This code is described in "Generalized Eigenvectors" (Monday, October 29, 2012
    # ), which is in the note "Jordan Form Reformulation", in folder "nonlineaDyn\Relation to driving terms\Jordan Form Reformulation"
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal

    if display == "eigenv":
        print("in eigen, len(mp)=", len(mp))
    zeros, eigenposition, diagonal, tm = initialize()
    if display == "eigenv":
        print("in eigenv(): zeros=", zeros)
    if display == "eigenv":
        print("in eigen, pos2", len(diagonal))
    xi = []

    eik = 0

    while eik < len(zeros):
        x = initializeX()
        j = zeros[eik] - 1
        if display == "eigenv":
            print("\n  \n   Solve: eik= ", eik, " \n \nY = ", " X= ", x)

        if display == "eigenv":
            print("xi=", xi, ", j=", j)

        while j >= 0:
            # Solve equations for each row of the eigenvector # eik, start from the last row at position eik - 1 to 1
            if display == "eigenv":
                print(
                    "\n \n p0 Solve j=",
                    j,
                    " eigenposition[[j]]= ",
                    eigenposition[j],
                    "Y=",
                )
            if abs(eigenposition[j][0]) < tol:
                solvingZeroRows()
            else:
                solvingNonzeroRows()
                if display == "eigenv":
                    print("in solvingNonzeroRows,now j=", j)
            if display == "eigenv":
                print("now j=", j)
            if display == "eigenv":
                print("now eik=", eik, " x=", x)
            j = j - 1

        xi.append(x)
        eik = eik + 1

    return xi, array(tm)


def eigensubspace(mn):
    # this subroutine corresponds to part of "jordan[m_]" in "JordanNormalFormAndTriangularMatrix.nb"
    # except the part of jordan normal form is not included, which is to be carried out in
    # a separate small subroutine.
    # The meaning of xi and tm: Xi.mn=tij.Xj
    # eigensubspace find the eigenspace xi of eigenvalue 0 for matrix mn
    # Each row of xi is a left eigenvector of mn so that Xi.mn is still a linear combination of Xi
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal
    """See "find-eigenvecor.nb" in folder "Relation to driving terms". """
    # Solve for X and check that X indeed form the basis for the eigenspace: mp.X is invariant within this space
    mt = np.transpose(mn)
    S1 = np.identity(len(mn))[::-1] * (1 + 0j)
    # See "Reverse Order Matrix" in "Jordan Form Reformulation"
    # mp=S1.mt.S1 so that mp is still an upper-triangular matrix, even though mt is a lower triangular matrix.(*Change eigenvector so that it is the eigenvectors of the reversed \
    # order of mp:mt=S1.mp.S1,i.e.,the result should be the eigenbasis for \
    # mt.As in "Reverse Order Matrix",which is in "Generalized \
    # Eigenvectors",which is in "Jordan Form Reformulation" of \
    # Sunday,December 02,2012 onenotes.So,after this transformation,we have \
    # mt.X=t.X *)
    mp = np.dot(np.dot(S1, mt), S1)
    xi, tm = eigenv()
    xi1 = np.dot(xi, S1)
    return tm, xi1


def righteigensubspace(mn):
    # this subroutine corresponds to part of "jordan[m_]" in "JordanNormalFormAndTriangularMatrix.nb"
    # except the part of jordan normal form is not included, which is to be carried out in
    # a separate small subroutine.
    # The meaning of xi and tm: mn.Xi=tij.Xj
    # eigensubspace find the right eigenspace xi of eigenvalue 0 for matrix mn
    # Each column of xi is a right eigenvector of mn so that mn.Xi is still a linear combination of Xi
    global eik, mp, j, x, tm, eigenposition, zeros, xi, diagonal
    mp = mn
    """See "find-eigenvecor.nb" in folder "Relation to driving terms". """
    # Solve for X and check that X indeed form the basis for the eigenspace: mp.X is invariant within this space
    # Eigenvectors",which is in "Jordan Form Reformulation" of \
    # Sunday,December 02,2012 onenotes.So,after this transformation,we have \
    # mp.X=t.X *)
    xi, tm = eigenv()
    return tm, np.transpose(xi)


def pca(a, m):
    # print real array
    for j in range(m):
        print("  %8.2f   " % a[j].real, end=" ")
    print("")
    for j in range(m):
        print("    %8.2fj" % a[j].imag, end=" ")
    print("\n")


def pra(a, m):
    # print complex array
    for j in range(m):
        print("  %8.2f   " % a[j].real, end=" ")
    print("\n")


def prp(a, m):
    # print complex array
    for j in range(m):
        print("  %10.1f %6s  " % (a[0][j].real, a[1][j]), end=" ")
        if (j - 4) % 5 == 0:
            print("")
    print("")


def prm(a, m, n):
    # print real matrix
    for i in range(m):
        print("")
        for j in range(n):
            print("  %8.2f   " % a[i, j].real, end=" ")
    print("\n")


def pim(a, m, n):
    # print interger matrix
    for i in range(m):
        print("")
        for j in range(n):
            print("%3d" % a[i, j].real, end=" ")
    print("\n")


def pim(a, m, n):
    # print real matrix
    for i in range(m):
        print("")
        for j in range(n):
            print("  %d   " % a[i, j].real, end=" ")
    print("\n")


def pcm(a, m, n):
    # print complex matrix
    for i in range(m):
        print("")
        for j in range(n):
            print("  %8.4f   " % a[i, j].real, end=" ")
        print("")
        for j in range(n):
            print("    %8.4fj" % a[i, j].imag, end=" ")
        print("\n")
    return


# Subroutines called by JNF

# Hermition conjugate and matrix form


def ht(m):
    return np.transpose(np.conj(m))


# findzeros and findnonzeros
# (*Subroutine to find which element is zero in a list*)
# def findzeros(tmp6,tol):
# 	zerorows = []
# 	for i in range(0,len(tmp6)):
# 		if abs(tmp6[i])< tol: zerorows.append(i)
# 	return zerorows


def findnonzeros(tmp6):
    zerorows = []
    for i in range(0, len(tmp6)):
        if abs(tmp6[i]) > tol:
            zerorows.append(i)
    return zerorows


def rk(m):
    # local[{u, v, w, len, eigen},
    # use SVD to determine rank is more stable,
    #   it may need to define tolerance for findzeros*)
    u, w, v = linalg.svd(m)
    eigen = w * (1.0 + 0j)
    # output is rank of m
    return len(w) - len(findzeros(eigen, tolrank))


def rkchop(m):
    # local[{chp, rank, rkm}
    # (*To find rank for high order power *)
    m[abs(m) < tol] = 0  # this line is added 11/19/2019
    chp = [m]
    rank = rk(chp[-1])
    rkm = [rank]
    ln = len(m)
    while rank > 0:
        mp = np.dot(chp[-1], m)  # mp is the power of m
        maxmp = np.amax(abs(mp))
        if display == "rkchop":
            print("len(chp)=", len(chp), "maxmp=", maxmp)
        if maxmp < tol:
            # print "last mp=", prm(mp*1e6,10,10)
            return array(rkm)
        mp = (
            mp / maxmp
        )  # when mp is too small, we need renormalize mp#removed 11/19/2019
        if display == "rkchop":
            prm(mp * 1e6, 10, 10)
        mp[abs(mp) < tol] = 0  # chop every term less than tol
        chp.append(mp)
        rank = rk(chp[-1])
        if rank < rkm[-1]:
            rkm.append(rank)
        else:
            rank = -1
    # output is the rank of m, m.m, m.m.m, ...till the rank is 0
    if display == "rkchop":
        sv("junk1", chp)
    return array(rkm)


# Apply SVD to null spaces using subsvd
# SVD and subsvd


def svd(s):
    # local Module[{u, v, w, vh},
    # (*SVD for the matrix s,
    # with singular value changed to increasing order using t matrix
    # Notice, numpy svd output v is actually conjugate transpose of v, i.e., vh, so use vh here
    u, w, vh = linalg.svd(s)
    t1 = (np.identity(len(u[0])))[::-1, :] * (1 + 0j)
    u = np.dot(u, t1)
    v = ht(vh)
    t2 = (np.identity(len(vh)))[::-1, :] * (1 + 0j)
    vh = np.dot(t2, vh)
    v = np.dot(v, t2)
    w = np.dot(t1, np.dot(np.diag(w), t2))
    # output is u,w,vh with s=u.w.vh
    return u, w, vh


def subsvd(s, ml):
    # local{u, w, v, subs, vp, mp},
    # Making a square submatrix SVD
    # (*Take submatrix of s, the lower right corner of s, start from index ml+1 *)
    mp = ml
    ln = len(s)
    subs = s[mp:ln, mp:ln]
    # (*svd for the submatrix,
    # with singular value changed to increasing order using t matrix*)
    u, w, vh = svd(subs)
    # (*Change the transformation matrix back to full space dimention*)
    v = ht(vh)
    vp = np.identity(ln) * (1.0 + 0.0j)
    vp[mp:ln, mp:ln] = v
    v = vp
    vh = ht(v)
    # output is vh, vh.s.v, v
    return vh, np.dot(vh, np.dot(s, v)), v


def couplingelmentposition(tm):  # tm is the eigenspace matrix in Xi.M=tm.Xi
    # See See  Onenotes : nonlineaDyn\Relation to driving terms\Jordan Form Reformulation\Jordan Decomposition : Positive diagonal elements in QR used in triangularization
    # Also see JordanNormalFormAndTriangularMatrix.nb: "Nullblock sizes and coupling elment position"
    EigenspaceDimension = len(tm)
    # NumberofChains=(norder+1)/2, but we do not have norder passed into this module so we have to use tm to calculate it
    NumberofChains = np.roots([1, 1, -EigenspaceDimension * 2])
    NumberofChains = max(NumberofChains)
    # Check that dimension of the space is consistent with number of chains
    if abs(EigenspaceDimension - NumberofChains * (NumberofChains + 1) / 2) > 1e-4:
        sys.exit(0)
    nc = array(range(int(NumberofChains + 1e-4), -1, -1))
    ml = EigenspaceDimension - nc * (nc + 1) / 2
    BlockSize = [[ml[i], ml[i + 1] - 1] for i in range(int(NumberofChains + 1e-4))]
    T = [[BlockSize[i], BlockSize[i + 1]] for i in range(len(BlockSize) - 1)]
    CouplingElementsPosition = np.zeros([EigenspaceDimension], dtype=int)
    for i in range(len(T)):
        for j in range(int(T[i][1][1] - T[i][1][0]) + 1):
            CouplingElementsPosition[int(T[i][1][0]) + j] = int(T[i][0][0]) + j
    return ml, CouplingElementsPosition


def nullblocksvd(
    s,
):  # see JordanNormalFormAndTriangularMatrix.nb in /Users/lihuayu/Desktop/nonlineardynamics/nonlineaDyn/Relation to driving terms/Jordan Form Reformulation
    # import pdb; pdb.set_trace()
    # local {v, v0, s0, v1, s1, len, ml},
    ln = len(s)
    ml = ln - rkchop(s)
    if display == "nullblocksvd":
        print("in nullblocksvd, ml=", ml)
    v0h, s0, v0 = subsvd(s, 0)
    s0[:, : ml[0]] = 0  # added 11/19/2019
    for i in range(0, len(ml) - 1):
        v1h, s1, v1 = subsvd(s0, ml[i])
        s1[ml[i] :, ml[i] : ml[i + 1]] = 0  # added 11/19/2019
        v0 = np.dot(v0, v1)
        v0h = np.dot(v1h, v0h)
        s0 = s1
    # s1[ml[-1]:,ml[-1]]=0#added 11/19/2019#removed 12/25/2020
    # output is v0,s0 with s1=v0h.s.v0
    if display == "nullblocksvd":
        sv("junk1", [ml, s, s0])
    return v0h, s0, v0


# QRDecomposition
def qrd(A):
    # local {r, q, subq, qc, u, v, w},
    lrow = len(A)
    lcol = len(A[0])
    q, r = linalg.qr(A)
    subq = q
    q = np.zeros([lrow, lrow]) * (1.0 + 0j)
    q[:, 0:lcol] = subq
    qc = ht(q)
    u, w, vh = svd(qc)
    v = ht(vh)
    # Notice that svd here give singular values in increasing order
    # so we take the first columns of v as the null space of qc to fill into q
    q[:, lcol:lrow] = v[:, : (lrow - lcol)]
    qc = ht(q)
    # output is q,r with A=q.r
    return q, np.dot(qc, A)


def ik(i, ml):
    # (*ik gives the start and end of one null space block *)
    if i == 0:
        interval = [0, ml[0]]
    else:
        interval = [ml[i - 1], ml[i]]
    return interval


def triangularsubblocks(s1):
    # local{T, ml, len, U, Uh, k, j, msub, q, r},
    T = s1
    ln = len(s1)
    ml = ln - rkchop(T)  # ml is the null space dimension
    ml = np.append(
        ml, ln
    )  # When the last block of nullspace has dimension greater than 1, this step is critical so that QR decomposition for the last block is treated properly.
    U = np.identity(ln) * (1 + 0j)
    # (*Construct U matrix block by block*)
    if len(ml) == 1:
        Uh = ht(U)

    for i in (range(0, len(ml) - 1))[::-1]:
        k = ik(i, ml)  # (*ik gives the start and end of one null space block *)
        j = ik(i + 1, ml)
        msub = T[k[0] : k[1], j[0] : j[1]]
        q, r = qrd(msub)
        U[k[0] : k[1], k[0] : k[1]] = ht(q)
        Uh = ht(U)
        T = np.dot(U, np.dot(s1, Uh))
    return U, T, Uh


# Gauss elimination of all upperdiagonal element except the lowest one for each column
def gausselim(T):
    # local {B, m, u, um, v, vm, couplingelm, j, len},
    B = copy.copy(T)
    ln = len(T)
    m = np.identity(ln) * (1 + 0j)
    u = copy.copy(m)  # use copy to avoid both u and m point to same address.
    um = copy.copy(m)
    v = copy.copy(m)
    vm = copy.copy(m)
    j = ln
    couplingelm = ln
    couplingelmposition = np.zeros([ln], dtype=int)
    while couplingelm > 1:
        j = j - 1
        couplingelm = findnonzeros(B[:, j])[-1]
        couplingelmposition[j] = couplingelm
        v = copy.copy(m)
        vm = copy.copy(m)
        v[:, couplingelm] = -B[:, j] / B[couplingelm, j]
        vm[:, couplingelm] = B[:, j] / B[couplingelm, j]
        v[couplingelm, couplingelm] = 1.0 + 0.0j
        u = np.dot(v, u)
        um = np.dot(um, vm)
        B = np.dot(v, np.dot(B, vm))

    if display == "gausselim":
        sv("junk1", B)
    if display == "gausselim":
        print("in gausselim, coupling element row position =", couplingelmposition)
    ml, couplingelmpositionTheory = couplingelmentposition(T)
    if display == "gausselim":
        print(
            "max(|couplingelmposition-theoty|)=",
            max(abs(couplingelmposition - couplingelmpositionTheory)),
        )
    if display == "gausselim":
        print("couplingelmpositiontheoty=", couplingelmpositionTheory)
    couplingelement = [
        T[couplingelmposition[i]][i]
        for i in range(len(couplingelmposition))
        if couplingelmposition[i] > 0
    ]
    if display == "gausselim":
        print("couplingelement=", abs(array(couplingelement)))
    return u, np.dot(u, np.dot(T, um)), um


def gausselimacc(T):
    u1, J1, u1m = gausselim(T)
    u2, J, u2m = gausselim(
        J1
    )  # iterate gauss elimination once to make elimination of each upper column much more accuraate
    u = np.dot(u2, u1)
    um = np.dot(u1m, u2m)
    # Notice u.T.um only approximately =J, there near zero elements that correspond to exact zero in J
    # To be accurate, we have u2.(u1.s.u1m).u2m=J, but not (u2.u1).s.(u1m.u2m)!!
    # So we output also u2,u1,u1m,u2m
    return u, J, um, u2, u1, u1m, u2m


# Normalize a jordan form not normalized yet

# Find chains from a near jordan form not normalized yet
def findchains(J):
    # local {chain, chains, chainlinks, chainremain, j, len},
    ln = len(J)
    j = np.zeros((ln, ln), dtype=int)
    for i in range(0, ln):
        for k in range(0, ln):
            if abs(J[i, k]) > tol:
                j[i, k] = 1

    chainlinks = np.dot(np.transpose(j), range(1, ln + 1))
    chainremain = range(1, ln + 1)
    chains = []
    while chainremain != []:
        it = chainremain[-1]
        chain = [it]
        while it > 0:
            it = chainlinks[it - 1]
            if it > 0:
                chain.append(it)

        chains.append(chain[::-1])
        chainremain = [
            i for i in range(1, ln + 1) if i not in [j for subc in chains for j in subc]
        ]
    return chains[::-1]


def semiJ(J):
    # For a given set of eigenvector basis J, find the matrix to transform it into semi-Jordan form
    # see "JordanNormalFormAndTriangularMatrix" for the principle of this module
    # local{X, m, j, i, z, zh, len, ch},
    ln = len(J)
    if display == "semiJ":
        print("ln=", ln)
    if display == "semiJ":
        sv("junk", J)
    if display == "semiJ":
        prm(J, 10, 10)
    ch = findchains(J)
    lst = list(map(len, ch))
    lst = np.argsort(lst)  # sort the chain according to its length
    lst = lst[::-1]  # lengest chain is the first chain now
    ch = [ch[i] for i in lst]

    X = np.identity(ln) * (
        1.0 + 0.0j
    )  # X serves as a set of initial transformation basis
    fch = [
        i - 1 for subc in ch for i in subc
    ]  # fch is the index sequence of the chain arrange according to Jordan chain
    z = [X[fch[i]] for i in range(len(J))]
    z = np.transpose(z)  # z is the new basis relative to the initial basis X
    zm = linalg.inv(z)
    return zm, np.dot(zm, np.dot(J, z)), z


def jnfinsubspace(s):
    # local{v1, s1, v1h, U, T, Uh, u, um, V, Vh, J, z, zh, Jn},
    if display == "jnfinsubspace":
        print("in jnfinsubspace, s=")
        if len(s) > 9:
            prm(s, 10, 10)
        else:
            prm(s, len(s), len(s))

    v1h, s1, v1 = nullblocksvd(
        s
    )  # build columns of null space of powers of mn by iteration
    if display == "jnfinsubspace":
        print("in jnfinsubspace, s1=")
        if len(s1) > 9:
            prm(s1, 10, 10)
        else:
            prm(s1, len(s1), len(s1))
        print("max|v1|=", abs(v1).max(), "max|v1h|", abs(v1h).max())
    U, T, Uh = triangularsubblocks(
        s1
    )  # using QR decomposition to make avery super diagonal blocks upper triagular
    if display == "jnfinsubspace":
        print("in jnfinsubspace, T=")
        if len(T) > 9:
            prm(T, 10, 10)
        else:
            prm(T, len(T), len(T))
        print("max|U|=", abs(U).max(), "max|Uh|", abs(Uh).max())
    u, pJ, um, u2, u1, u1m, u2m = gausselimacc(
        T
    )  # use iterated gauss elimination to obtain accurate non-normalized Jordan form
    if display == "jnfinsubspace":
        print("in jnfinsubspace, pJ=")
        if len(pJ) > 9:
            prm(pJ, 10, 10)
        else:
            prm(pJ, len(pJ), len(pJ))
    # Notice u.T.um only approximately =J, there near zero elements that correspond to exact zero in J
    # To be accurate, we have u2.(u1.s.u1m).u2m=J, but not (u2.u1).s.(u1m.u2m)!!
    V = np.dot(u, np.dot(U, v1h))  # accumulate all 3 transform matrix above into one
    Vm = np.dot(v1, np.dot(Uh, um))
    # sv('junk',[s,s1,T,pJ])
    zm, sJ, z = semiJ(
        pJ
    )  # Find chains from the list of general eigenvectors in pJ and sequence them in order to form semi-Jordan form where one vector transform into next after acted upon by the matrix mn from right only different by one facctor.
    V = np.dot(zm, V)
    Vm = np.dot(Vm, z)
    if display == "jnfinsubspace":
        print("in jnfinsubspace, sJ=")
        if len(sJ) > 9:
            prm(sJ, 10, 10)
        else:
            prm(sJ, len(sJ), len(sJ))
    # Notice V.s.Vm only approximately =Jn, there near zero elements in V.s.Vm that correspond to exact zero in Jn
    # We have V=zm.u.U.v1h and Vm=v1.Uh.um.z. But to be accurate, as explained in gausselimacc
    # we need to break u and um into u=u2.u1, and um=u1m.u2m, so
    # V=zm.u2.u1.U.v1h=V2.V1, with V2=zm.u2 and V1=u1.U.v1h
    # Vm=v1.Uh.u1m.u2m.z=V1m.V2m, with V1m=v1.Uh.u1m and V2m=u2m.z
    V2 = np.dot(zm, u2)
    V1 = d3(u1, U, v1h)
    V1m = d3(v1, Uh, u1m)
    V2m = np.dot(u2m, z)
    # So accurately we have Jn=V2.(V1.s.V1m).V2m but Jn=V.s.Vm is not accurate!
    return V, sJ, Vm, V2, V1, V1m, V2m


def d3(u, T, um):
    return np.dot(u, np.dot(T, um))


def semileftjordanbasis(mn):
    tm, xi = eigensubspace(
        mn
    )  # Find a set of generalized eigenvectors to form an eigenspace for eigenvalue 0.
    if display == "semileftjordanbasis":
        prm(mn, 10, 10)
    if display == "semileftjordanbasis":
        print("tm=", tm)
    if display == "semileftjordanbasis":
        print("tol=", tol)
    if np.amax(abs(tm)) < tol:
        return (
            xi,
            0 * tm,
        )  # if so then J=tm=0, u=xi , i.e., all xi are proper eigenvectors
    S, J, Sm, S2, S1, S1m, S2m = jnfinsubspace(
        tm
    )  # From one eigensubspace build the semi-Jorda basis for semi-Jordan form J
    u = np.dot(S2, np.dot(S1, xi))  # u is the left semi-Jordan baisis of mn
    # maxchainlenposition, maxchainlen, chain,lc=findchainposition(J) #chain is the coefficients list of superdiagonal of J, lc is the list of chain position.

    # Notice that due to machine precision,u=S2.S1.xi is much more accurate than u=S.xi=(S2.S1).xi
    return u, J


def vJlnJM(J, mu):
    # JlnJM=v.lnJM.vm, where lnJM=ln(1+exp(-j*mu)Jn), with un.mn=Jn.un and Jn is in standard Jordan form, mn=M-exp(i*mu)
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)
    # Construct ln(J) where J is the standard Jordan form of the matrix M
    lnJM = np.zeros(len(J))
    u = -np.identity(len(J)) * (1 + 0j)
    n = 1
    while n < maxchainlen:
        u = np.dot(u, (-J) * np.exp(-1j * mu))
        lnJM = lnJM + u / n
        n = n + 1
    v, JlnJM = semileftjordanbasis(lnJM)
    v, JlnJM, normalizer, A = orthonormalize(
        v, JlnJM
    )  # normalize all vectors in the basis
    v, JlnJM, normalizer, A = crossblockorthonormalize(v, JlnJM)
    vm = np.linalg.inv(v)
    # Notice lnJM=u.log(M)(u^-1)-i*mu=u.ln(exp(-i*mu)M).um, lnJM is the log of the Jordan form of M,
    # so lnJM is not in Jordan form, while JlnJM is in Jordan form
    # and v.lnJM.vm=JlnJM. vn=normalizer.v is the left basis for standardized Jordan form of lnJM
    # i.e., vn.lnJM.vnm is in Jordan form
    return v, JlnJM, lnJM, vm, normalizer


def exp(J, norder):
    # J here is in Jordan form with diagonal all zeros
    u = np.identity(len(J)) * (1 + 0j)
    ex = u.copy()
    n = 1
    while n < norder + 1:
        u = np.dot(u, J) / n
        ex = ex + u
        n = n + 1
    return ex


def findchainposition(J):
    # Find chains in J and give a list of their position
    chain = np.diag(J, 1)
    chain = chain.tolist()
    chain.insert(0, 0)  # chain is the superdiagonal of J
    tol = 1e-4
    lc = findzeros(chain, tolrank)  # zeros determine the structure of eigenspace
    lc2 = [lc[i + 1] - lc[i] for i in range(len(lc) - 1)]
    lc2.append(len(chain) - lc[-1])
    maxchainlen = max(lc2)
    maxchainlenposition = lc[lc2.index(maxchainlen)]
    lc.append(len(chain))
    chainposition = []  # this gives the nonzero position in chain
    for i in range(len(lc) - 1):
        chainposition.append(range(lc[i], lc[i + 1]))
    return maxchainlenposition, maxchainlen, chain, chainposition


def normalize(u, J):
    # module to normalize left eigen basis, renormalize the chain and build the matrix to unnormalize into standrd Jordan form
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)
    # 1. normalize the left eigenvectors in u to become un
    # Since u.m=J.u, so un.m=A.u.m=A.J.Am.A.u=Jn.u, where un=A.u normalizes u, Am=A^(-1)
    norm = [np.sqrt(np.dot(vec, ht(vec))).real for vec in u]
    A = np.identity(len(u)) * (1.0 + 0j)
    for i in range(len(u)):
        A[i, i] = 1 / norm[i]
    Am = np.linalg.inv(A)
    un = np.dot(A, u)
    Jn = np.dot(np.dot(A, J), Am)
    for i in range(1, len(chain)):
        chain[i] = Jn[i - 1, i]

    un = un.tolist()
    # un.mn=Jn.un, where the vectors in un are normalized.
    # 2.Find the normalization constants to transform normalized semi-Jordan basis un into standard Jordan basis unn
    normalfactor = copy.copy(chain)

    for link in chainposition:
        for j in range(
            1, len(link) - 1
        ):  # for every multiplier, the norm of all vectors down the chain from it must be increased by the multiplier in order to change to standard Jordan form.
            for k in range(j + 1, len(link)):
                normalfactor[link[k]] = normalfactor[link[k]] * chain[link[j]]
    for link in chainposition:
        normalfactor[link[0]] = 1.0

    # 3. Generate the matrix "nomalizer" for transformation from un to unn
    normalizer = np.identity(len(J)) * 1.0 + 0j
    for i in range(len(normalizer)):
        normalizer[i, i] = normalfactor[i]

    # Output un is nomalized left semi-Jordan basis, Jn is its semi-Jordan form
    # normalizer is the normalization matrix such that unn=normalizer.un is
    # standard Jordan form: unn.mn=Jnn.unn, where Jnn is in standard Jordan form.
    # un=A.u, Jn=A.J.Am
    return un, Jn, normalizer, chain, A, Am


def checkJordan(u, J, mn):
    lhs = np.dot(u, mn)
    rhs = np.dot(J, u)
    tmp = lhs - rhs
    print("abs(u.mn-J.u).max()=", abs(tmp).max(), " with J=")
    # 	prm(J, len(J), len(J))
    prm(J, 5, 10)
    return abs(tmp).max()


def checkcorrelation(unn, display="False"):
    mcc = np.tile(0j, [len(unn), len(unn)])
    for i in range(len(unn)):
        for j in range(len(unn)):
            mcc[i, j] = np.dot(ht(unn[i]), unn[j])

    if display == True:
        prm(mcc * 1e2, len(mcc), len(mcc))
    return mcc


def orthonormalize(us, Js):
    # us.mn=Js.us where Js is insemi-Jordan form and us is not normalized, and not orthogonalized
    # This module transform us to a normalized left basis with all vectors orthogonal to the first vector in every chain
    u, J, normalizer, chain, A1, A1m = normalize(
        us, Js
    )  # normalized semi-Jordan form J and normalized left basis u. So u=A.us, J=A1.Js.A1m
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)
    t = np.identity(len(J)) * (1 + 0.0j)
    for (
        link
    ) in (
        chainposition
    ):  # for each jordan block, follow proceedure in "Gauge transform of JNF". Section: "How to make u[1] orthogonal to other eigenvectors."
        if len(link) == 1:
            continue  # if len(link)=0, this is an single element eigenstate, so there is no other elements to be orthogonalized.
        subchain = [
            chain[i] for i in link
        ]  # the list a1,a2,a3,.. in semi-Jordan form J

        alp = [
            1 + 0.0j
        ]  # alp=[a1,a1 a2, a1 a2 a3, a1 a2 a3 a4, a1 a2 a3 a4 a5]. a1, a2,.. are chain ratio.
        i = 1
        while i < len(subchain):
            alp.append(alp[-1] * subchain[i])
            i = i + 1
        alp = alp[
            1:
        ]  # drop first one alp=[a1,a1 a2, a1 a2 a3, a1 a2 a3 a4, a1 a2 a3 a4 a5]. a1, a2,.. are chain ratio.

        # correlation matrix mc_ij=<ui|uj>  i>0,j>0 see my notes in "Gauge transform of JNF". Section: "How to make u[1] orthogonal to other eigenvectors."
        # solve equation to find the coefficients beta for gauge transformation matrix tao
        mc = checkcorrelation(u[link[1] : link[-1] + 1])  # matrix <e_i|e_j> for i,j>0

        ene1 = np.empty(len(link) - 1, dtype=complex)  # column <e_i|e_0>
        for i in range(len(link) - 1):
            ene1[i] = np.dot(ht(u[link[i + 1]]), u[link[0]])

        abeta = np.linalg.solve(mc, -ene1)  # solve mc.X=-ene1
        beta = (
            abeta / alp
        )  # beta=X/alp is the solution for coefficients of gauge transform

        # Construct for every block part of the transformation matrix to maintain Jordan form invariant when subtracting other chains from the main chain to make other chains orthogonal to the main chain
        # See rule2 in "JordanNormalFormAndTriangularMatrix", section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant  "
        abeta = np.insert(
            abeta, 0, 0
        )  # this makes it easy to use indexes to identify which raio is used later
        for i in range(
            1, len(link)
        ):  # assing value to superdiagonal i, start from row=0 and column=i
            ax = abeta[i]
            t[link[0], link[i]] = ax
            # print "link[0]=",link[0]," link[i]=",link[i]," ax=",ax
            j = 1
            while i + j < len(
                link
            ):  # assigning value to super diagonal i, while j is the index along the superdiagonal
                # print "i=",i," i+j=",i+j,"link[i+j]=",link[i+j],"link[j]=",link[j]
                ax = ax * chain[link[i + j]] / chain[link[j]]
                # print "j=",j," ax=", ax, "chain[link[i+j]]=",chain[link[i+j]], "chain[link[j]]",chain[link[j]]
                # print "link[j]=",link[j]," link[i+j]=",link[i+j]," ax=",ax
                t[link[j], link[i + j]] = ax
                j = j + 1

    unn = np.dot(t, u)  # So unn=t.u=t.A1.us

    # normalize unn to uo so all vectors are nomalized with u[0] orthogonal to others.
    uo, Jo, normalizer, chain, A2, A2m = normalize(
        unn, J
    )  # So uo=A2.unn=A2.t.A1.us, Jo=A2.J.A2m=A2.A1.Js.A1m.A2m and normalizer.uo is standard Jordan basis
    A = d3(A2, t, A1)
    return uo, Jo, normalizer, A  # uo=A.us


def crossblockorthonormalize(u, J):
    # orthogonalize all vectors in other chain blocks to the u[0] of the longest chain, input u should be already orthonormalized within each block to its first eigenvector.
    # See  "JordanNormalFormAndTriangularMatrix", section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant "
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)
    # Construct the matrix equation for the transformation coefficients abeta
    v = copy.copy(u)
    v.pop(maxchainlenposition)  # remove the row from v at this position
    mc = checkcorrelation(v)
    ene1 = np.empty(len(v)) * (1.0 + 0.0j)  # column <e_i|e_0>
    for i in range(len(v)):
        ene1[i] = np.dot(ht(v[i]), u[0])

    abeta = np.linalg.solve(mc, -ene1)  # solve mc.X=-ene1
    abeta = np.insert(
        abeta, 0, 0
    )  # this makes it easy to use indexes to identify which raio is used later
    t = np.identity(len(J)) * (1 + 0.0j)

    link0 = chainposition[maxchainlenposition]

    # construct the cross block upper triangular matrix for substracting vector in other chains from the main vector in the main chain
    for k in range(
        1, len(chainposition)
    ):  # for all chain except the main chain construct the upper-triangular part of the transformation matrix
        link = chainposition[k]

        # See rule1 in "JordanNormalFormAndTriangularMatrix", section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant "
        for i in range(
            len(link)
        ):  # assing value to superdiagonal i, start from row=0 and column=i
            ax = abeta[link[i]]
            t[link0[0], link[i]] = ax
            j = 1
            while i + j < len(link):  # j is the index along the superdiagonal
                ax = ax * chain[link[i + j]] / chain[link0[j]]
                t[link0[j], link[i + j]] = ax
                j = j + 1

    # Construct the main block part of the transformation matrix to maintain Jordan form invariant when subtracting other chains from the main chain to make other chains orthogonal to the main chain
    # See rule2 in "JordanNormalFormAndTriangularMatrix", section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant  "
    for i in range(
        1, len(link0)
    ):  # assing value to superdiagonal i, start from row=0 and column=i
        ax = abeta[link0[i]]
        t[link0[0], link0[i]] = ax
        j = 1
        while i + j < len(
            link0
        ):  # assigning value to super diagonal i, while j is the index along the superdiagonal
            ax = ax * chain[link0[i + j]] / chain[link0[j]]
            t[link0[j], link0[i + j]] = ax
            j = j + 1

    un = np.dot(t, u)  # un=t.u
    # checkcorrelation(un,True)
    u, J, normalizer, chain, A1, A1m = normalize(
        un, J
    )  # So u=A1.un=A1.t.u, J=A1.J.A1m, normalizer.u gives standard Jordan basis, so noramlizer.J.normalizer^(-1) is the standard Jordan form.
    A = np.dot(A1, t)  # So output u=A.u for input u.
    return u, J, normalizer, A


def standardizeJNF(u, J, normalizer):
    # use normalizer to change semi-Jordan basis u into standard Jordan basis un, and change semi-Jordan form to standard Jordan form Jn
    un = np.dot(normalizer, u)
    Jn = J.copy()
    for i in range(1, len(J)):
        if abs(J[i - 1, i]) > tol:
            Jn[i - 1, i] = 1
    return un, Jn


def UMexpJold(M, mu):
    # mn=M-exp(i*mu) is the uppertriangular matrix of one turn map M, mu is its tune of interest.
    mn = M - np.identity(len(M)) * np.exp(1j * mu)
    # This module calculate U such that:
    # U.M=exp(i*mu)*exp(J).U, and
    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M
    # A derivation of U.M=exp[j mu]*exp(J).U is given in "test_jnfdefinition2.py"
    # print "Check us as left eigenvectors of m in semi Jordan form: us.mn=Js.u"
    us, Js = semileftjordanbasis(mn)
    us, Js, unormalizer, A = orthonormalize(us, Js)
    us, Js, unormalizer, A = crossblockorthonormalize(us, Js)
    # checkJordan(us,Js,mn)
    # checkcorrelation(us)

    # print "Check standard Jordan form of m: un.mn=Jn.un"
    un, Jn = standardizeJNF(
        us, Js, unormalizer
    )  # un, Jn is in the standard Jordan form, un=unormalizer.u
    # checkJordan(un,Jn,mn)
    # pim(Jn,len(Jn),len(Jn))

    v, JlnJM, lnJM, vm, vnormalizer = vJlnJM(Jn, mu)
    # mn=M-exp(i*mu), and un.mn.unm=Jn
    # Let lnJM=ln(1+exp(-j*mu)Jn), so 1+exp(-j*mu)Jn=exp(lnJM)
    # lnJM=ln(1+exp(-j*mu)un.mn.unm)=un.ln(1+exp(-j*mu)mn).unm=un.ln(exp(-j*mu)(exp(j*mu)+mn)).unm=un.ln(exp(-j*mu)M).unm
    # So lnJM=un.ln(exp(-j*mu)M).unm (notice: calculate unm is not as easy as invert un becasue un is not square matrix, it is a submatrix of Un.)
    # JlnJM=semi-Jordan form of lnJM: JlnJM=v.lnJM.vm =v.un.ln(exp(-j*mu)M).unm.vm,  JlnJM is in semi-Jordan form
    u1 = np.dot(
        v, un
    )  # u1=v.un is the left basis of ln(exp(-j*mu)M):   u1.ln(exp(-j*mu)M)=JlnJM.u1   (because JlnJM=v.un.ln(exp(-j*mu)M).unm.vm, so JlnJM.v.un=v.un.ln(exp(-j*mu)M)   )
    u2, J2, normalizer, A1 = orthonormalize(
        u1, JlnJM
    )  # Hence u2=A1.u1 is still the left basis of ln(exp(-j*mu)M):   u2.ln(exp(-j*mu)M)=J2.u2, and all the vectors in u2 are normalized
    # checkcorrelation(u2) #u2 have all vectors nomralized and each block has its first vector orthogonal to all others
    u3, J3, u3normalizer, A2 = crossblockorthonormalize(
        u2, J2
    )  # Hence u3=A2.u2 is still the left basis of ln(exp(-j*mu)M):   u3.ln(exp(-j*mu)M)=J3.u3, and all the vectors in u are normalized and all are orthogonal to u[0] even between Jordan blocks

    # To check u3.ln(exp(-j*mu)M)=J3.u3 using u.m=J.u, we need to calculate ln(exp(-j*mu)M), which is a infinite series (may even be divergent), unlike a Jordan form such as ln(1+exp(-j mu)J), which truncates at the power of n.
    # To avoid this, we need to track the transfrom from u to un (un=unormalizer.u) to u1 (u1=v.un), to u2 (u2=t2.u1),
    # Then to u3 (u3=A2.u2). We also need to track the transfrom from J to lnJM to JlnJM, to  J2, to J3.
    # Thus starting from us.mn=Js.mn, we next have u1.ln(exp(-j*mu)M)=JlnJM.u1, then we have u2.ln(exp(-j*mu)M)=A1.u1.ln(exp(-j*mu)M)=A1.JlnJM.A1m.A1.u1=A1.JlnJM.A1m.u2=J2.u2
    # So u3=A2.u2=A2.A1.u1, and u3.ln(exp(-j*mu)M)=A2.u2.ln(exp(-j*mu)M)=A2.A1.u1.ln(exp(-j*mu)M)=A2.J2.u2=A2.J2.A2m.A2.u2=J3.u3. That is, u1.ln(exp(-j*mu)M)=A1m.A2m.J3.A2.A1. A1m.A2m.u3=JlnJM.u1
    # Hence we should check that JlnJM=A1m.A2m.J3.A2.A1, and u3=A2.A1.u1
    # This would confirm u3.ln(exp(-j*mu)M)=J3.u3 using the transformation from u1 to u3.

    U, J = standardizeJNF(
        u3, J3, u3normalizer
    )  # U=u3normalizer.u3=u3normalizer.A2.A1.u1=u3normalizer.A2.A1.v.un=u3normalizer.A2.A1.v.unormalizer.u
    # checkcorrelation(U,True)

    # Hence it is clear that the left eigenbasis is U
    # the vectors in u3 are normalized, u[0] is orthogonalized to other vectors. U are not normalized, but they transform M and lnJM into standard Jordan form.
    # U is the eigen basis of the Jordan form of log(exp(-i*mu)M)
    # B=U.Z where b0 at the position of max length Jordan chain is the invariant.

    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)

    vn = np.dot(d3(u3normalizer, A2, A1), v)
    # vnm=np.linalg.inv(vn)
    # Since un.mn=Jn.un,
    # we have U.M=vn.un.M=vn.un.(exp(j*mu)+mn)=vn.(exp(j*mu)+Jn).un=vn.(exp(j*mu)+Jn).vnm.vn.un=(  exp(j*mu)+(vn.Jn.vnm)  ).U
    # lnJM=ln(1+exp(-j*mu)Jn), so 1+exp(-j*mu)Jn=exp(lnJM)
    # So U.M=(  exp(j*mu)+(vn.Jn.vnm)  ).U=exp(j*mu)*vn.( 1+ exp(-j*mu)*Jn  ).vnm.U=exp(j*mu)*vn.exp(lnJM).vnm.U=
    # =exp(j*mu)*exp(vn.lnJM.vnm).U=exp(j*mu)*exp(J4).U
    # we have U.M=exp(j*mu)*exp(J).U

    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M such that U.M=exp(i*mu)exp(J).U,
    # while un is the standarized left basis of mn=M-exp(i*mu) such that un.mn=Jn.un. Both J and Jn are instandard Jordan form.
    # U=vn.un, mu is the tune
    return U, J, vn, un


# def orderdot(u1,u2,order,powerindex):
# this subroutine was used before 1/30/2015, see why thsi is replaced by the current orderdot in jf26.py section 7.
# 	v1=array([ u1[i] for i in range(len(u1)) if sum(powerindex[i])==order])
# 	v2=array([ u2[i] for i in range(len(u2)) if sum(powerindex[i])==order])
# 	u1u2=np.dot(v1,v2)
# 	return u1u2


def orderdotnew(u1, u2, order, powerindex, ypowerorder):
    # see jf26.py section 7 for why we need to modify orderdot to this by adding [:2] to powerindex[i] to correct an old error of using the tatal power including y power
    v1 = array(
        [
            u1[i]
            for i in range(len(u1))
            if sum(powerindex[i]) == order and sum(powerindex[i][2:4]) <= ypowerorder
        ]
    )
    v2 = array(
        [
            u2[i]
            for i in range(len(u2))
            if sum(powerindex[i]) == order and sum(powerindex[i][2:4]) <= ypowerorder
        ]
    )
    u1u2 = np.dot(v1, v2)
    return u1u2


def findorder(i, chainposition):
    # The order is determined as given in section 7 of discussion on minimization for any specific order see "Jordan Form Reformulation/Gauge-invariance-in-Jordan-space.one"
    chainindex, indexinchain = next(
        ([k, sublist] for k, sublist in enumerate(chainposition) if i in sublist), -1
    )  # Find which chain the row i is in and find the position in this chain
    indexinchain = indexinchain.index(i)
    order = 2 * (chainindex + indexinchain) + 1
    return order


def findordernew(i, chainposition, u, powerindex, ypowerorder, u0idx):
    # Directly find the lowest order terms in zx power in u[i], thus applied in gauge transform of Jordan form u[i] can be used to minimize terms of this order in u[0]
    # print "in findordernew, u0idx=",u0idx, " abs(u[0][u0idx])=",abs(u[0][u0idx])
    a = [
        sum(powerindex[k])
        for k, b in enumerate(abs(u[i]))
        if b / abs(u[0][u0idx]) > 1e-7 and sum(powerindex[k][2:4]) <= ypowerorder
    ]
    # print a,ypowerorder
    if a == []:
        order = np.amax(powerindex)
    else:
        order = min(a)
    return order


def correlationatorder(u, chainposition, powerindex, index, lowestpower, ypowerorder):
    # see jf26.py section 7 for why we need to replace findorder by findordernew here
    mcc = np.tile(0j, [len(index), len(index)])
    # if ypowerorder==0, then use orderdot with x power counted only, else y powers are also take into account
    for i in range(len(index)):
        order = lowestpower[index[i]]  # order is determined by the row number i in u[i]
        for j in range(len(index)):
            # mcc[i,j]=orderdotnew(ht(u[index[i]]),u[index[j]], order,powerindex,ypowerorder)
            mcc[i, j] = orderdot.orderdot(
                ht(u[index[i]]), u[index[j]], order, powerindex, ypowerorder
            )

    # if display==True: prm(mcc*1e2,len(mcc),len(mcc))
    return mcc


def correlationatordernew(u, chainposition, powerindex, ypowerorder, u0idx):
    # see jf26.py section 7 for why we need to replace findorder by findordernew here
    mcc = np.tile(0j, [len(u) - 1, len(u) - 1])
    # if ypowerorder==0, then use orderdot with x power counted only, else y powers are also take into account
    for i in range(1, len(u)):
        order = findordernew(
            i, chainposition, u, powerindex, ypowerorder, u0idx
        )  # order is determined by the row number i in u[i]
        for j in range(1, len(u)):
            # mcc[i-1,j-1]=orderdotnew(ht(u[i]),u[j], order,powerindex,ypowerorder)
            mcc[i - 1, j - 1] = orderdot.orderdot(
                ht(u[i]), u[j], order, powerindex, ypowerorder
            )

    # if display==True: prm(mcc*1e2,len(mcc),len(mcc))
    return mcc


def correlationatorderall(u, chainposition, powerindex, ypowerorder, u0idx):
    # see jf26.py section 7 for why we need to replace findorder by findordernew here
    mcc = np.tile(0j, [len(u), len(u)])
    # if ypowerorder==0, then use orderdot with x power counted only, else y powers are also take into account
    if ypowerorder == 0:
        for i in range(len(u)):
            if display == "correlationatorderall":
                "print i=", i
            order = findordernew(
                i, chainposition, u, powerindex, u0idx
            )  # order is determined by the row number i in u[i]
            if display == "correlationatorderall":
                "print i,order=", i, order
            for j in range(len(u)):
                # mcc[i,j]=orderdotnew(ht(u[i]),u[j], order,powerindex)
                mcc[i, j] = orderdot.orderdot(ht(u[i]), u[j], order, powerindex)
    else:
        for i in range(len(u)):
            order = findordernew(i, chainposition, u, powerindex, u0idx)
            for j in range(len(u)):
                mcc[i, j] = orderdot.orderdot(ht(u[i]), u[j], order, powerindex)

    # if display==True: prm(mcc*1e2,len(mcc),len(mcc))
    return mcc


def minimizeu0byorder(u, J, chainposition, powerindex, u0idx, ypowerorder):
    # This module is modified from the module "minimizeu0byordernew" following the study in "henon-Heiles7orderCompareGustavsonE0.0833.nb"
    # The purpose of modification in 3/28/2017 is to take into account resonance case where there are two vector with lowest power equal to 1.
    # Minimize the high order terms in u[0] using gauge transform by other chains's lowest order terms
    # Given u and J such that u.M=J.u, M can be replaced by ln(exp(-j*mu)M), as it it used most often in UMexpJ.
    # see jf26.py section 7 for why we need to replace correlationatorder by correlationatordernew here
    # orthogonalize all vectors in other chain blocks to the u[0] of the longest chain.
    # See  "Gauge transform of JNF-onthonormalized.nb"
    # in "nonlineaDyn/Relation to driving terms/Jordan Form Reformulation",
    # section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant "
    # and the pdf file in the folder "jfn-resoance3".
    # For discussion on minimization for any specific order see "Jordan Form Reformulation/Gauge-invariance-in-Jordan-space.one"
    # and "Gauge transform of JNF minimize high power terms Joined Blocks.nb"
    if display == "minimizeu0byordernew":
        print("in minimizeu0byordernew, J=", prm(J, len(J), len(J)))
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)
    lowestpower = [
        findordernew(i, chainposition, u, powerindex, ypowerorder, u0idx)
        for i in range(len(u))
    ]
    index1 = [i for i in range(len(u)) if lowestpower[i] == 1]
    index = [i for i in range(len(u)) if lowestpower[i] != 1]
    # Construct the matrix equation for the transformation coefficients abeta
    # if ypowerorder==0, then use orderdot with x power counted only, else y powers are also take into account
    mc = correlationatorder(
        u, chainposition, powerindex, index, lowestpower, ypowerorder
    )  # the matrix <e_i|e_j>_{power of i}
    ene0 = np.empty(len(index)) * (1.0 + 0.0j)  # column <e_i|e_0>
    for i in range(len(index)):
        order = lowestpower[
            index[i]
        ]  # For each i select the power order for the minimization
        # ene0[i]=orderdotnew(ht(u[index[i]]),u[0], order, powerindex, ypowerorder)
        ene0[i] = orderdot.orderdot(
            ht(u[index[i]]), u[0], order, powerindex, ypowerorder
        )

    abeta = np.linalg.solve(mc, -ene0)  # solve mc.X=-ene1
    for i in index1:
        abeta = np.insert(
            abeta, i, 0
        )  # this makes it easy to use indexes to identify which ratio is used later

    t = np.identity(len(J)) * (1 + 0.0j)

    link0 = chainposition[maxchainlenposition]

    # construct the cross block upper triangular matrix for substracting vector in other chains from the main vector in the main chain
    for k in range(
        0, len(chainposition)
    ):  # for all chain except the main chain construct the upper-triangular part of the transformation matrix
        link = chainposition[k]

        # See rule1 in "Gauge transform of JNF minimize high power terms"
        for i in range(
            len(link)
        ):  # assing value to superdiagonal with index i (i.e., distance from diagonal), start from row=0 and column=i
            ax = abeta[
                link[i]
            ]  # The solution abeta has alpha1, alpha2, .. for one chain, gamma1, gamma2,.. for another chain, etc.
            t[link0[0], link[i]] += ax
            j = 1
            while i + j < len(link):  # j is the index along the superdiagonal
                ax = (
                    ax * chain[link[i + j]] / chain[link0[j]]
                )  # ax=alphan*xm, x2=a1/b1,x3=x2*a2/b2,.., y2=a2/b1, y3=y2*a2/b2,..
                t[link0[j], link[i + j]] += ax
                j = j + 1

    un = np.dot(t, u)  # un=t.u gives u a gauge transform and keep J invariant
    return un, t


def minimizeu0byordernew(u, J, chainposition, powerindex, u0idx, ypowerorder):
    # Minimize the high order terms in u[0] using gauge transform by other chains's lowest order terms
    # Given u and J such that u.M=J.u, M can be replaced by ln(exp(-j*mu)M), as it it used most often in UMexpJ.
    # see jf26.py section 7 for why we need to replace correlationatorder by correlationatordernew here
    # orthogonalize all vectors in other chain blocks to the u[0] of the longest chain.
    # See  "Gauge transform of JNF-onthonormalized .nb"
    # in "nonlineaDyn/Relation to driving terms/Jordan Form Reformulation",
    # section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant "
    # For discussion on minimization for any specific order see "Jordan Form Reformulation/Gauge-invariance-in-Jordan-space.one"
    # and "Gauge transform of JNF minimize high power terms Joined Blocks.nb"
    if display == "minimizeu0byordernew":
        print("in minimizeu0byordernew, J=", prm(J, len(J), len(J)))
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(J)
    # Construct the matrix equation for the transformation coefficients abeta
    # if ypowerorder==0, then use orderdot with x power counted only, else y powers are also take into account
    mc = correlationatordernew(u, chainposition, powerindex, ypowerorder, u0idx)
    ene0 = np.empty(len(mc)) * (1.0 + 0.0j)  # column <e_i|e_0>
    for i in range(1, len(u)):
        order = findordernew(
            i, chainposition, u, powerindex, ypowerorder, u0idx
        )  # For each i select the power order for the minimization
        # if ypowerorder==0, then use orderdot with x power counted only, else y powers are also take into account
        ene0[i - 1] = orderdotnew(ht(u[i]), u[0], order, powerindex, ypowerorder)

    abeta = np.linalg.solve(mc, -ene0)  # solve mc.X=-ene1
    abeta = np.insert(
        abeta, 0, 0
    )  # this makes it easy to use indexes to identify which ratio is used later
    t = np.identity(len(J)) * (1 + 0.0j)

    link0 = chainposition[maxchainlenposition]

    # construct the cross block upper triangular matrix for substracting vector in other chains from the main vector in the main chain
    for k in range(
        1, len(chainposition)
    ):  # for all chain except the main chain construct the upper-triangular part of the transformation matrix
        link = chainposition[k]

        # See rule1 in "Gauge transform of JNF minimize high power terms"
        for i in range(
            len(link)
        ):  # assing value to superdiagonal with index i (i.e., distance from diagonal), start from row=0 and column=i
            ax = abeta[
                link[i]
            ]  # The solution abeta has alpha1, alpha2, .. for one chain, gamma1, gamma2,.. for another chain, etc.
            t[link0[0], link[i]] = ax
            j = 1
            while i + j < len(link):  # j is the index along the superdiagonal
                ax = (
                    ax * chain[link[i + j]] / chain[link0[j]]
                )  # ax=alphan*xm, x2=a1/b1,x3=x2*a2/b2,.., y2=a2/b1, y3=y2*a2/b2,..
                t[link0[j], link[i + j]] = ax
                j = j + 1

    # Construct the main block part of the transformation matrix to maintain Jordan form invariant when subtracting other chains from the main chain to make other chains orthogonal to the main chain
    # See rule2 in "JordanNormalFormAndTriangularMatrix", section "How to make all vectors orthogonal to one main vectors while mainaining Jordan form invariant  "
    for i in range(
        1, len(link0)
    ):  # assing value to superdiagonal i, start from row=0 and column=i
        ax = abeta[
            link0[i]
        ]  # beta1, beta2, .. are in the solution abeta of the part in the chain "link0"
        t[link0[0], link0[i]] = ax
        j = 1
        while i + j < len(
            link0
        ):  # assigning value to super diagonal i, while j is the index along the superdiagonal
            ax = (
                ax * chain[link0[i + j]] / chain[link0[j]]
            )  # p2=b2/b1, p3=p2*b3/b2,.., q2=b3/b1, q3=q2*b4/b2,..
            t[link0[j], link0[i + j]] = ax
            j = j + 1

    un = np.dot(t, u)  # un=t.u gives u a gauge transform and keep J invariant
    return un, t


def UMexpJ(M, mu, powerindex, u0idx, ypowerorder):
    # This module calculate U such that:
    # U.M=exp(i*mu)*exp(J).U, and
    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M
    # A derivation of U.M=exp[j mu]*exp(J).U is given in "testjnfdefinition3.py"
    # mn=M-exp(i*mu) is the uppertriangular matrix of one turn map M, mu is its tune of interest.
    mn = M - np.identity(len(M)) * np.exp(1j * mu)
    us, Js = semileftjordanbasis(mn)
    us, Js, usnormalizer, chain, A1, A1m = normalize(us, Js)
    # print "check us.mn=Js.us:"
    # checkJordan(us,Js,mn)

    # print "Check standard Jordan form of m: un.mn=Jn.un"
    un, Jn = standardizeJNF(
        us, Js, usnormalizer
    )  # un, Jn is in the standard Jordan form, un=usnormalizer.u
    # checkJordan(un,Jn,mn)

    v, JlnJM, lnJM, vm, vnormalizer = vJlnJM(Jn, mu)
    # mn=M-exp(i*mu), and un.mn.unm=Jn
    # Let lnJM=ln(1+exp(-j*mu)Jn), so 1+exp(-j*mu)Jn=exp(lnJM)
    # lnJM=ln(1+exp(-j*mu)un.mn.unm)=un.ln(1+exp(-j*mu)mn).unm=un.ln(exp(-j*mu)(exp(j*mu)+mn)).unm=un.ln(exp(-j*mu)M).unm
    # So lnJM=un.ln(exp(-j*mu)M).unm (notice: calculate unm is not as easy as invert un becasue un is not square matrix, it is a submatrix of Un.)
    # JlnJM=semi-Jordan form of lnJM: JlnJM=v.lnJM.vm =v.un.ln(exp(-j*mu)M).unm.vm,  JlnJM is in semi-Jordan form
    u1 = np.dot(
        v, un
    )  # u1=v.un is the left basis of ln(exp(-j*mu)M):   u1.ln(exp(-j*mu)M)=JlnJM.u1   (because JlnJM=v.un.ln(exp(-j*mu)M).unm.vm, so JlnJM.v.un=v.un.ln(exp(-j*mu)M)   )
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(JlnJM)
    u2, t2 = minimizeu0byorder(
        u1, JlnJM, chainposition, powerindex, u0idx, ypowerorder
    )  # Hence u2=t2.u1 is still the left basis of ln(exp(-j*mu)M):   u2.ln(exp(-j*mu)M)=JlnJM.u2, and the high order terms in u2[0] are minimized
    # Compare with u1[0], u2[0] has its high order terms minimized by subtracing the short chains from the longest chain, even though u1 and u2 both are left eigenvectors of ln(exp(-j*mu)M)

    # To check u2.ln(exp(-j*mu)M)=JlnJM.u2 using u.m=J.u, we need to calculate ln(exp(-j*mu)M), which is a infinite series (may even be divergent), unlike a Jordan form such as ln(1+exp(-j mu)J), which truncates at the power of n.
    # To avoid this, we need to track the transfrom from u to un (un=unormalizer.u) to u1 (u1=v.un), to u2 (u2=t2.u1),
    # We also need to track the transfrom from J to lnJM to JlnJM.
    # Thus starting from us.mn=Js.mn, we next have u1.ln(exp(-j*mu)M)=JlnJM.u1, then we have u2.ln(exp(-j*mu)M)=t2.u1.ln(exp(-j*mu)M)=t2.JlnJM.t2m.t2.u1=t2.JlnJM.t2m.u2=JlnJM.u2
    # Hence we should check that JlnJM=t2.JlnJM.t2m, and u2=t2.u1
    # This would confirm u2.ln(exp(-j*mu)M)=JlnJM.u2 using the transformation from u1 to u2.

    u3, J3, u3normalizer, chain, A2, A2m = normalize(
        u2, JlnJM
    )  # according to module "normalize", every row of u3 is normalized, J3 is semiJordan form, u3=A2.u2, J3=A2.JlnJM.A2m
    U, J = standardizeJNF(
        u3, J3, u3normalizer
    )  # U=u3normalizer.u3=u3normalizer.A2.t2.u1=u3normalizer.A2.t2.v.un, "standardizeJNF" chang semiJordan form J3 into standard J

    # A=d3(u3normalizer,A2,t2)
    # Am=np.linalg.inv(A)
    vn = np.dot(d3(u3normalizer, A2, t2), v)
    # vnm=np.linalg.inv(vn)
    # tmp=U-np.dot(vn,un)
    # print "check U=vn.un:", abs(tmp).max()
    # tmp=J-d3(A, JlnJM,Am)
    # print "check J=A.JlnJM.Am:", abs(tmp).max()
    # tmp=J-d3(vn, lnJM,vnm)
    # print "check J=vn.lnJM.vnm:", abs(tmp).max()

    # Hence it is clear that the left eigenbasis is U
    # the vectors in u3 are normalized, u3[0] is has higher order terms minimized.
    # as it is shown in "Gauge-invariance-in-Jordan-space" in Onenote,
    # this means ni_<u3[0]|u3[i]>_ni=0, i.e., for a product limited to ni'th order.
    # U is not normalized, but they transform M and lnJM into standard Jordan form.
    # U is the eigen basis of the Jordan form of log(exp(-i*mu)M)

    # B=U.Z is the most important vector, where b0 at the position of max length Jordan chain is the invariant.

    # Since un.mn=Jn.un,
    # we have U.M=vn.un.M=vn.us.(exp(j*mu)+mn)=vn.(exp(j*mu)+Jn).un=vn.(exp(j*mu)+Jn).vnm.vn.un=(  exp(j*mu)+(vn.Jn.vnm)  ).U
    # Since lnJM=ln(1+exp(-j*mu)Jn), so 1+exp(-j*mu)Jn=exp(lnJM)
    # So U.M=(  exp(j*mu)+(vn.Jn.vnm)  ).U=exp(j*mu)*vn.( 1+ exp(-j*mu)*Jn  ).vnm.U=exp(j*mu)*vn.exp(lnJM).vnm.U=
    # =exp(j*mu)*exp(vn.lnJM.vnm).U=exp(j*mu)*exp(J).U
    # we have U.M=exp(j*mu)*exp(J).U

    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M such that U.M=exp(i*mu)exp(J).U,
    # while un is the standarized left basis of mn=M-exp(i*mu) such that un.mn=Jn.un. Both J and Jn are instandard Jordan form.
    # U=vn.un, mu is the tune
    return U, J, vn, un


def UMexpJnew(M, mu, powerindex, nres, u0idx, ypowerorder):
    # This module calculate U such that:
    # U.M=exp(i*mu)*exp(J).U, and
    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M
    # A derivation of U.M=exp[j mu]*exp(J).U is given in "testjnfdefinition3.py"
    # mn=M-exp(i*mu) is the uppertriangular matrix of one turn map M, mu is its tune of interest.
    mn = M - np.identity(len(M)) * np.exp(1j * nres * mu)
    if display == "UMexpJnew":
        prm(mn, 10, 10)
    us, Js = semileftjordanbasis(mn)
    us, Js, usnormalizer, chain, A1, A1m = normalize(us, Js)
    # print "check us.mn=Js.us:"
    # checkJordan(us,Js,mn)

    # print "Check standard Jordan form of m: un.mn=Jn.un"
    un, Jn = standardizeJNF(
        us, Js, usnormalizer
    )  # un, Jn is in the standard Jordan form, un=usnormalizer.u
    # checkJordan(un,Jn,mn)
    if display == "UMexpJnew":
        print("in UMexpJnew, Jn=", Jn)
    if display == "UMexpJnew":
        print("in UMexpJnew, len(Jn)=", len(Jn))
    if (
        len(Jn) == 1
    ):  # when len(Jn)==1, the eigenspace is just the eigenvector, and lnJM is ill defined, so we directly give the simple Jordan form solution which is [0]
        u3, J3, u3normalizer, chain, A2, A2m = normalize(
            un, Jn
        )  # according to module "normalize", every row of u3 is normalized, J3 is semiJordan form, u3=A2.u2, J3=A2.JlnJM.A2m
        if display == "UMexpJnew":
            print("in UMexpJnew,2, len(Jn)=", len(Jn))
        U, J = standardizeJNF(
            u3, J3, u3normalizer
        )  # U=u3normalizer.u3=u3normalizer.A2.t2.u1=u3normalizer.A2.t2.v.un, "standardizeJNF" chang semiJordan form J3 into standard J
        vn = np.identity(len(un))
        return U, J, vn, un

    if np.amax(abs(Jn)) < tol:
        return un, Jn, np.identity(len(un)) * (1 + 0j), un

    v, JlnJM, lnJM, vm, vnormalizer = vJlnJM(Jn, nres * mu)
    # mn=M-exp(i*mu), and un.mn.unm=Jn
    # Let lnJM=ln(1+exp(-j*mu)Jn), so 1+exp(-j*mu)Jn=exp(lnJM)
    # lnJM=ln(1+exp(-j*mu)un.mn.unm)=un.ln(1+exp(-j*mu)mn).unm=un.ln(exp(-j*mu)(exp(j*mu)+mn)).unm=un.ln(exp(-j*mu)M).unm
    # So lnJM=un.ln(exp(-j*mu)M).unm (notice: calculate unm is not as easy as invert un becasue un is not square matrix, it is a submatrix of Un.)
    # JlnJM=semi-Jordan form of lnJM: JlnJM=v.lnJM.vm =v.un.ln(exp(-j*mu)M).unm.vm,  JlnJM is in semi-Jordan form
    u1 = np.dot(
        v, un
    )  # u1=v.un is the left basis of ln(exp(-j*mu)M):   u1.ln(exp(-j*mu)M)=JlnJM.u1   (because JlnJM=v.un.ln(exp(-j*mu)M).unm.vm, so JlnJM.v.un=v.un.ln(exp(-j*mu)M)   )
    maxchainlenposition, maxchainlen, chain, chainposition = findchainposition(JlnJM)
    u2, t2 = minimizeu0byordernew(
        u1, JlnJM, chainposition, powerindex, u0idx, ypowerorder
    )  # Hence u2=t2.u1 is still the left basis of ln(exp(-j*mu)M):   u2.ln(exp(-j*mu)M)=JlnJM.u2, and the high order terms in u2[0] are minimized

    # To check u2.ln(exp(-j*mu)M)=JlnJM.u2 using u.m=J.u, we need to calculate ln(exp(-j*mu)M), which is a infinite series (may even be divergent), unlike a Jordan form such as ln(1+exp(-j mu)J), which truncates at the power of n.
    # To avoid this, we need to track the transfrom from u to un (un=unormalizer.u) to u1 (u1=v.un), to u2 (u2=t2.u1),
    # We also need to track the transfrom from J to lnJM to JlnJM.
    # Thus starting from us.mn=Js.mn, we next have u1.ln(exp(-j*mu)M)=JlnJM.u1, then we have u2.ln(exp(-j*mu)M)=t2.u1.ln(exp(-j*mu)M)=t2.JlnJM.t2m.t2.u1=t2.JlnJM.t2m.u2=JlnJM.u2
    # Hence we should check that JlnJM=t2.JlnJM.t2m, and u2=t2.u1
    # This would confirm u2.ln(exp(-j*mu)M)=JlnJM.u2 using the transformation from u1 to u2.

    u3, J3, u3normalizer, chain, A2, A2m = normalize(
        u2, JlnJM
    )  # according to module "normalize", every row of u3 is normalized, J3 is semiJordan form, u3=A2.u2, J3=A2.JlnJM.A2m
    U, J = standardizeJNF(
        u3, J3, u3normalizer
    )  # U=u3normalizer.u3=u3normalizer.A2.t2.u1=u3normalizer.A2.t2.v.un, "standardizeJNF" chang semiJordan form J3 into standard J

    # A=d3(u3normalizer,A2,t2)
    # Am=np.linalg.inv(A)
    vn = np.dot(d3(u3normalizer, A2, t2), v)
    # vnm=np.linalg.inv(vn)
    # tmp=U-np.dot(vn,un)
    # print "check U=vn.un:", abs(tmp).max()
    # tmp=J-d3(A, JlnJM,Am)
    # print "check J=A.JlnJM.Am:", abs(tmp).max()
    # tmp=J-d3(vn, lnJM,vnm)
    # print "check J=vn.lnJM.vnm:", abs(tmp).max()

    # Hence it is clear that the left eigenbasis is U
    # the vectors in u3 are normalized, u3[0] is has higher order terms minimized.
    # as it is shown in "Gauge-invariance-in-Jordan-space" in Onenote,
    # this means ni_<u3[0]|u3[i]>_ni=0, i.e., for a product limited to ni'th order.
    # U is not normalized, but they transform M and lnJM into standard Jordan form.
    # U is the eigen basis of the Jordan form of log(exp(-i*mu)M)

    # B=U.Z is the most important vector, where b0 at the position of max length Jordan chain is the invariant.

    # Since un.mn=Jn.un,
    # we have U.M=vn.un.M=vn.us.(exp(j*mu)+mn)=vn.(exp(j*mu)+Jn).un=vn.(exp(j*mu)+Jn).vnm.vn.un=(  exp(j*mu)+(vn.Jn.vnm)  ).U
    # Since lnJM=ln(1+exp(-j*mu)Jn), so 1+exp(-j*mu)Jn=exp(lnJM)
    # So U.M=(  exp(j*mu)+(vn.Jn.vnm)  ).U=exp(j*mu)*vn.( 1+ exp(-j*mu)*Jn  ).vnm.U=exp(j*mu)*vn.exp(lnJM).vnm.U=
    # =exp(j*mu)*exp(vn.lnJM.vnm).U=exp(j*mu)*exp(J).U
    # we have U.M=exp(j*mu)*exp(J).U

    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M such that U.M=exp(i*mu)exp(J).U,
    # while un is the standarized left basis of mn=M-exp(i*mu) such that un.mn=Jn.un. Both J and Jn are instandard Jordan form.
    # U=vn.un, mu is the tune
    return U, J, vn, un


def UMJ(M, mu, powerindex):
    # This module calculate U such that:
    # U.M=J.U, and
    # W=U.Z, where w0 is the main invariant, J is the Jordan form of the matrix M
    # mn=M-i*mu is the uppertriangular matrix of derivative map \dot{Z}=M.Z, mu is its tune of interest.
    mn = M - np.identity(len(M)) * mu * 1j
    if display == "UMJ":
        prm(mn, 10, 10)
    us, Js = semileftjordanbasis(mn)
    us, Js, usnormalizer, chain, A1, A1m = normalize(us, Js)
    # print "check us.mn=Js.us:"
    # checkJordan(us,Js,mn)

    # print "Check standard Jordan form of m: un.mn=Jn.un"
    un, Jn = standardizeJNF(
        us, Js, usnormalizer
    )  # un, Jn is in the standard Jordan form, un=usnormalizer.u
    # checkJordan(un,Jn,mn)
    return Jn, un, mn


def scaling(M, powerindex):
    # Find A such that ms=A.M.Am has maximum reduced to same order as first order
    absM = abs(M)
    i, j = np.unravel_index(absM.argmax(), absM.shape)
    power = sum(powerindex[j])
    scalex = (absM.max()) ** (-(1.0 / (power - 1.0)))
    scalem = 1 / scalex
    mlen = len(powerindex)
    if display == "scaling":
        print("scalex=", scalex, " scalem=", scalem)
    As = np.identity(mlen)
    for i in range(mlen):
        As[i, i] = scalem ** sum(powerindex[i])
    Asm = np.identity(mlen)
    for i in range(mlen):
        Asm[i, i] = scalex ** sum(powerindex[i])
    ms = d3(As, M, Asm)
    return ms, As, Asm


def scalingnew(Ms1, powerindex, scalemf):
    # Find A2 such that ms=A2.Ms1.Am2 has maximum reduced to same order as first order
    # Only in final scalex we take into account the first scaling "scalemf" generated in "jf1map.py" before the final construction of square matrix
    # for the reduction of high order power terms errors
    # The difference from scaling is that now scalex=scalex2*scalemf is the final scaling.
    # Notice that the output are As2, not As,  because ms=d3(As2,Ms1,Asm2), not ms=d3(As,Ms1,Asm)
    # The square matrix M lost its precision, so only Ms1 and Ms are useful. So only use As2 to connect Ms1 with ms
    absM = abs(Ms1)
    i, j = np.unravel_index(absM.argmax(), absM.shape)
    power = sum(powerindex[j])
    scalex2 = (absM.max()) ** (-(1.0 / (power - 1.0)))
    scalem2 = 1 / scalex2
    mlen = len(powerindex)
    if display == "scalingnew":
        print("scalex2=", scalex2, " scalem1=", scalem2)
    As2 = np.identity(mlen)
    for i in range(mlen):
        As2[i, i] = scalem2 ** sum(powerindex[i])
    Asm2 = np.identity(mlen)
    for i in range(mlen):
        Asm2[i, i] = scalex2 ** sum(powerindex[i])
    ms = d3(
        As2, Ms1, Asm2
    )  # Notice!! ms does not take the first scaling into account, so ms cannot operate on Z column made from zbar, not even zbars which is scaled twice from zbar to zbar/scalefm to zbar/scalex
    scalex = (
        scalex2 * scalemf
    )  # scalex now takes the first scaling "scalemf" into account, see line 164 of jf24.py
    return ms, As2, Asm2, scalex


def UMexpJScaling(M, mu, powerindex, u0idx, ypowerorder):
    # This module calculate u such that:
    # u.Ms=exp(i*mu)*exp(J).u, and scaling matrix so that ms=As.M.Asm has much smaller singular value spectrum range than M, and JFN of ms is easy to manage and stable.
    # U=u.As, however, As is very simple scaling factor powered according to order.
    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M
    Ms, As, Asm = scaling(M, powerindex)
    u, J, vn, un = UMexpJ(Ms, mu, powerindex, u0idx, ypowerorder)
    # B=U.Z=(u.As).Z=u.(As.Z), where b0 is the main invariant, J is the Jordan form of the matrix M such that U.M=exp(i*mu)exp(J).U,
    # while un is the standarized left basis of mn=Ms-exp(i*mu) such that un.mn=Jn.un. Both J and Jn are instandard Jordan form.
    # u=vn.un, mu is the tune
    return u, J, vn, un, Ms, As


def UMexpJScalingnew(M, mu, powerindex, nres, u0idx, ypowerorder):
    # This module calculate u such that:
    # u.Ms=exp(i*mu)*exp(J).u, and scaling matrix so that ms=As.M.Asm has much smaller singular value spectrum range than M, and JFN of ms is easy to manage and stable.
    # U=u.As, however, As is very simple scaling factor powered according to order.
    # B=U.Z, where b0 is the main invariant, J is the Jordan form of the matrix M
    Ms, As, Asm = scaling(M, powerindex)
    u, J, vn, un = UMexpJnew(Ms, mu, powerindex, nres, u0idx, ypowerorder)
    # B=U.Z=(u.As).Z=u.(As.Z), where b0 is the main invariant, J is the Jordan form of the matrix M such that U.M=exp(i*mu)exp(J).U,
    # while un is the standarized left basis of mn=Ms-exp(i*mu) such that un.mn=Jn.un. Both J and Jn are instandard Jordan form.
    # u=vn.un, mu is the tune
    return u, J, vn, un, Ms, As


def UMsUbarexpJ(Ms1, phix0, nres, powerindex, scalemf, u0idx, ypowerorder):
    # usage:
    # jf44.py:uy,uybar,Jy,scalexy,Ms,As2,Asm2=
    # jfdf.UMsUbarexpJ(Ms10,phiy0,1,powerindex,scalemf,sequencenumber[0,0,1,0],ypowerorder=5)
    # Hence u0idx=sequencenumber[0,0,1,0] here is the position of the dominating lowest order term in u, when we #consider muy, we use the term [0,0,1,0]
    Ms, As2, Asm2, scalex = scalingnew(
        Ms1, powerindex, scalemf
    )  # We no longer use Ms2 as in jf24.py, scalex now include both initial scaling scalemf, and the scaling generated by second scaling
    u, J, vn, un = UMexpJnew(
        Ms, phix0, powerindex, nres, u0idx, ypowerorder
    )  # Temporarily we still use all the power including y
    mn = Ms - np.identity(len(Ms)) * np.exp(1j * nres * phix0)
    tm1, g = righteigensubspace(
        mn
    )  # the calculation of ubar is based on derivation in "Resonance Blocks.one" in the folder /nonlineardyn/Relation to driving term/Jordan Form Reformulation
    A = np.dot(u, g)
    Am = np.linalg.inv(A)
    ubar = np.dot(
        g, Am
    )  # ubar here is the E=g.(f.g)^-1=G(UG)^-1 in  "Resonance-Blocks-Calculation-of-reciprocal-basis.pdf"
    # Ms is construct from Ms1 by scaling zbar=zsbar1/scalex2=zbar/scalemf/scalex2=zbar/scalex

    # 8-24-2018, modified to take into account Jordan form of indentity, for which every vector is an eigenvector.
    if np.amax(abs(J)) < tol:
        tmp = map(
            np.argmax, abs(u)
        )  # When J=0, all rows in u are eigenvectors, we need to order them according to u with lower power first
        tmp1 = [
            sum(powerindex[i]) for i in tmp
        ]  # tmp1 has the index of the maximum term in every row of u
        tmp2 = np.argsort(
            tmp1
        )  # tmp2 sort u according to the lowest power of the maximum term arranged first
        u = u[tmp2]
        return u, ubar, J, scalex, Ms, As2, Asm2

    # The module developed in jf26.py for UMsUbarexpJ has an error when we use nres=-4 as the following line
    # dmt=[ i for i in range(len(u[0])) if (powerindex[i]==[abs(nres),0,0,0]).all() ][0]#dominating term [5,0,0,0]
    # So it is modified to find the dominating term in u[0]
    tmp = [
        [k, abs(b)]
        for k, b in enumerate(abs(u[0]))
        if sum(powerindex[k][2:4]) <= ypowerorder
    ]  # takes only x powers for its index and value in u[0]
    tmp1 = [
        i[-1] for i in tmp
    ]  # keep only absolute values of u[0] for those terms with only x powers
    tmp2 = np.argmax(
        tmp1
    )  # find the position of the maximum in the list of x power only terms of u[0]
    dmt = tmp[tmp2][0]  # find the index in u[0] for the maximum term
    # Hence the normalization of this term is:
    unormalizer = u[0][dmt]
    u = (
        u / unormalizer
    )  # this normalizes u matrix so to first order w is same as zsbar=zbar/scalex
    ubar = ubar * unormalizer  # while for resonance block it is also normalized
    # The output gives left and right eigenvectors so that
    # u.Ms=exp(i*mu)*exp(J).u, and Ms2.ubar=ubar.exp(i*mu)*exp(J)
    # and u.ubar=1, with zsbar=zbar/scalex, w=u.Zsbar

    # See my notes of 5/2/2013 in section "higher order" in "Courant-Snyder variables and U transform and relation to Twiss Transform"
    # U is the eigen basis of the Jordan form of M in form of exp(J), U=u.As, Ms=As.M.Asm
    # B=U.Z where b0 at the position of max length Jordan chain is the invariant.
    # B=u.Zs, where Zs=As.Z is scaled Z with zsbar=zbar/s while u has small singular value range.
    # We can use w=b0=u(zsbar,zsbars) to construct V
    # Notice!!! that u1=u.As2, Zs=As2.Zs1=As.Z, but we we do not calculate As anyway.
    return u, ubar, J, scalex, Ms, As2, Asm2
