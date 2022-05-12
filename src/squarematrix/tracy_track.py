import sys
from types import SimpleNamespace
from enum import IntEnum
from functools import partial

import numpy as np

from .cytpsa import TPS
from .cytpsa import sqrt as tps_sqrt

try:
    import PyTPSA
    from PyTPSA import tpsa
except:
    print("\n** Hao Yue's TPSA module failed to be imported.\n")

USE_ELEGANT_EDGE_FOCUS = False
# USE_ELEGANT_EDGE_FOCUS = True

# Defines global variables for Tracy code
# See src/tracy/physlib.cc
globval = SimpleNamespace()
globval.H_exact = False  # Small Ring Hamiltonian
globval.quad_fringe = False  # quadrupole fringe fields on/off
globval.EPU = False  # Elliptically Polarizing Undulator
globval.Cavity_on = False  # Cavity on/off
globval.radiation = False  # radiation on/off
globval.IBS = False  # diffusion on/off
globval.emittance = False  # emittance  on/off
globval.pathlength = False  # Path lengthening computation
globval.CODimax = 40  # maximum number of iterations for COD algo

# #############################################
# Definitions under "pytracy/src/tracy/tracy.h"
# #############################################
HOMmax = 21
#
class ThickType(IntEnum):
    thick = 0
    thin = 1


#
class PartsKind(IntEnum):
    drift = 0
    Wigl = 1
    Mpole = 2
    Cavity = 3
    marker = 4
    undef = 5
    Insertion = 6
    FieldMap = 7
    Spreader = 8
    Recombiner = 9
    Solenoid = 10


#
class MpoleType(IntEnum):
    All = 0
    Dip = 1
    Quad = 2
    Sext = 3
    Oct = 4
    Dec = 5
    Dodec = 6


#
class Plane(IntEnum):
    Horizontal = 1
    Vertical = 2


#
class Method(IntEnum):
    Meth_Linear = 0
    Meth_First = 1
    Meth_Second = 2
    Meth_Fourth = 4
    Meth_genfun = 5


# #############################################
# Definitions under "pytracy/src/tracy/field.h"
# #############################################
class SpatialIndex(IntEnum):
    X_, Y_, Z_ = list(range(3))


#
class PhaseSpaceIndex(IntEnum):
    x_, px_, y_, py_, delta_, ct_ = list(range(6))


#
x_, px_, y_, py_, delta_, ct_ = (
    PhaseSpaceIndex.x_,
    PhaseSpaceIndex.px_,
    PhaseSpaceIndex.y_,
    PhaseSpaceIndex.py_,
    PhaseSpaceIndex.delta_,
    PhaseSpaceIndex.ct_,
)
X_, Y_, Z_ = (SpatialIndex.X_, SpatialIndex.Y_, SpatialIndex.Z_)

# See "src/pytracy.sip"
SymplecticIntegCoeffs = SimpleNamespace()
SymplecticIntegCoeffs.c_1 = 1.0 / (2 * (2.0 - 2 ** (1 / 3)))
SymplecticIntegCoeffs.c_2 = 0.5 - SymplecticIntegCoeffs.c_1
SymplecticIntegCoeffs.d_1 = 2.0 * SymplecticIntegCoeffs.c_1
SymplecticIntegCoeffs.d_2 = 1.0 - 2.0 * SymplecticIntegCoeffs.d_1


def sqr(x):
    """"""

    return x * x


def cell_pass(pass_func_list, x0, px0, y0, py0, dp0=0.0, ct0=0.0):
    """"""

    globval.pathlength = True

    # mcount = 0
    # ncells = len(pass_func_list)
    v = [x0, px0, y0, py0, dp0, ct0]
    for i, func in enumerate(pass_func_list):
        # print(f'#{i+1:d} / {ncells:d}')
        # if func.func is Mpole_Pass:
        # print('Mpole_Pass')
        # mcount += 1
        func(v)

    return v


def cell_pass_trace_space(pass_func_list, x0, xp0, y0, yp0, dp0=0.0, ct0=0.0):
    """"""

    px0 = (1 + dp0) * xp0
    py0 = (1 + dp0) * yp0

    v = cell_pass(pass_func_list, x0, px0, y0, py0, dp0=dp0, ct0=ct0)

    x1, px1, y1, py1, dp1, ct1 = v

    xp1 = px1 / (1 + dp0)
    yp1 = py1 / (1 + dp0)

    return [x1, xp1, y1, yp1, dp1, ct1]


def _tps_cell_pass(
    is_phase_space, tps_module, pass_func_list, norder, nv, dp0=0.0, safe_mult=False
):
    """"""

    global tps_sqrt

    if tps_module == 0:  # Use Y. Hidaka's "cytpsa"

        from cytpsa import sqrt as tps_sqrt

        if is_phase_space:

            if nv == 4:
                var_names = ["x", "px", "y", "py"]
            elif nv == 5:
                var_names = ["x", "px", "y", "py", "dp"]
            else:
                raise NotImplementedError('"nv" must be 4 or 5.')
            assert len(var_names) == nv
            #
            x = TPS(var_names, norder, safe_mult=safe_mult)
            px = TPS(var_names, norder, safe_mult=safe_mult)
            y = TPS(var_names, norder, safe_mult=safe_mult)
            py = TPS(var_names, norder, safe_mult=safe_mult)
            dp = TPS(var_names, norder, safe_mult=safe_mult)
            ct = TPS(var_names, norder, safe_mult=safe_mult)
            #
            x.set_to_var("x")
            px.set_to_var("px")
            y.set_to_var("y")
            py.set_to_var("py")
            if nv == 5:
                dp.set_to_var("dp")
            else:
                if dp0 != 0.0:
                    dp += dp0
            ct.set_to_zero()

        else:  # pass in trace space (x, x', y, y')

            if nv == 4:
                var_names = ["x", "xp", "y", "yp"]
            elif nv == 5:
                var_names = ["x", "xp", "y", "yp", "dp"]
            else:
                raise NotImplementedError('"nv" must be 4 or 5.')
            assert len(var_names) == nv
            #
            x = TPS(var_names, norder, safe_mult=safe_mult)
            xp = TPS(var_names, norder, safe_mult=safe_mult)
            y = TPS(var_names, norder, safe_mult=safe_mult)
            yp = TPS(var_names, norder, safe_mult=safe_mult)
            dp = TPS(var_names, norder, safe_mult=safe_mult)
            ct = TPS(var_names, norder, safe_mult=safe_mult)
            #
            x.set_to_var("x")
            xp.set_to_var("xp")
            y.set_to_var("y")
            yp.set_to_var("yp")
            if nv == 5:
                dp.set_to_var("dp")
            else:
                if dp0 != 0.0:
                    dp += dp0
            ct.set_to_zero()

            px = (1 + dp) * xp
            py = (1 + dp) * yp

    else:  # Use H. Yue's "PyTPSA"

        from PyTPSA import sqrt as tps_sqrt

        tpsa.initialize(nv, norder)

        if is_phase_space:

            x = tpsa(value=0.0, variable=1)
            px = tpsa(value=0.0, variable=2)
            y = tpsa(value=0.0, variable=3)
            py = tpsa(value=0.0, variable=4)
            if nv == 5:
                dp = tpsa(value=0.0, variable=5)
            else:
                dp = tpsa(value=dp0, variable=0)
            ct = tpsa(value=0.0, variable=0)

        else:  # pass in trace space (x, x', y, y')

            x = tpsa(value=0.0, variable=1)
            xp = tpsa(value=0.0, variable=2)
            y = tpsa(value=0.0, variable=3)
            yp = tpsa(value=0.0, variable=4)
            if nv == 5:
                dp = tpsa(value=0.0, variable=5)
            else:
                dp = tpsa(value=dp0, variable=0)
            ct = tpsa(value=0.0, variable=0)

            px = (1 + dp) * xp
            py = (1 + dp) * yp

    globval.pathlength = True

    mcount = 0
    ncells = len(pass_func_list)
    v = [x, px, y, py, dp, ct]
    for i, func in enumerate(pass_func_list):
        # print(f"#{i+1:d} / {ncells:d}")
        if func.func is Mpole_Pass:
            # print('Mpole_Pass')
            mcount += 1
        func(v)

    return v


def tps_cell_pass(tps_moduel, pass_func_list, norder, nv, dp0=0.0, safe_mult=False):
    """"""

    is_phase_space = True

    return _tps_cell_pass(
        is_phase_space,
        tps_moduel,
        pass_func_list,
        norder,
        nv,
        dp0=dp0,
        safe_mult=safe_mult,
    )


def tps_cell_pass_trace_space(
    tps_moduel, pass_func_list, norder, nv, dp0=0.0, safe_mult=False
):
    """"""

    is_phase_space = False

    v = _tps_cell_pass(
        is_phase_space,
        tps_moduel,
        pass_func_list,
        norder,
        nv,
        dp0=dp0,
        safe_mult=safe_mult,
    )

    x1, px1, y1, py1, dp1, ct1 = v

    xp1 = px1 / (1 + dp0)
    yp1 = py1 / (1 + dp0)

    return [x1, xp1, y1, yp1, dp1, ct1]


def Mpole_Pass(cell, x):
    """"""

    elemp = cell.Elem
    M = cell.Elem.M

    Quad = MpoleType.Quad

    GtoL(x, cell.dS, cell.dT, M.Pc0, M.Pc1, M.Ps1)

    # fringe fields
    if globval.quad_fringe and M.PB[Quad + HOMmax] != 0.0:
        quad_fringe(M.PB[Quad + HOMmax], x)
    if not globval.H_exact:
        if M.Pirho != 0.0:
            EdgeFocus(M.Pirho, M.PTx1, M.Pgap, x)
    else:
        p_rot(M.PTx1, x)
        bend_fringe(M.Pirho, x)

    if M.Pthick == ThickType.thick:
        if not globval.H_exact:
            # polar coordinates
            h_ref = M.Pirho
            dL = elemp.PL / M.PN
        else:
            # Cartesian coordinates
            h_ref = 0.0
            if M.Pirho == 0.0:
                dL = elemp.PL / M.PN
            else:
                dL = 2.0 / M.Pirho * np.sin(elemp.PL * M.Pirho / 2.0) / M.PN

    if M.Pthick == ThickType.thick:
        dL1 = SymplecticIntegCoeffs.c_1 * dL
        dL2 = SymplecticIntegCoeffs.c_2 * dL
        dkL1 = SymplecticIntegCoeffs.d_1 * dL
        dkL2 = SymplecticIntegCoeffs.d_2 * dL

        for seg in range(1, M.PN + 1):
            Drift(dL1, x)
            thin_kick(M.Porder, M.PB, dkL1, M.Pirho, h_ref, x)
            Drift(dL2, x)
            thin_kick(M.Porder, M.PB, dkL2, M.Pirho, h_ref, x)

            Drift(dL2, x)
            thin_kick(M.Porder, M.PB, dkL1, M.Pirho, h_ref, x)
            Drift(dL1, x)
    else:
        thin_kick(M.Porder, M.PB, 1.0, 0.0, 0.0, x)

    # fringe fields
    if not globval.H_exact:
        if M.Pirho != 0.0:
            EdgeFocus(M.Pirho, M.PTx2, M.Pgap, x)
    else:
        bend_fringe(-M.Pirho, x)
        p_rot(M.PTx2, x)
    if globval.quad_fringe and (M.PB[Quad + HOMmax] != 0.0):
        quad_fringe(-M.PB[Quad + HOMmax], x)

    # Local -> Global
    LtoG(x, cell.dS, cell.dT, M.Pc0, M.Pc1, M.Ps1)


def GtoL(X, S, R, c0, c1, s1):
    """ """

    # Simplified rotated p_rot
    X[px_] += c1
    X[py_] += s1
    # Translate
    X[x_] -= S[X_]
    X[y_] -= S[Y_]
    # Rotate
    x1 = [v.copy() if isinstance(v, (TPS, tpsa)) else v for v in X]
    X[x_] = R[X_] * x1[x_] + R[Y_] * x1[y_]
    X[px_] = R[X_] * x1[px_] + R[Y_] * x1[py_]
    X[y_] = -R[Y_] * x1[x_] + R[X_] * x1[y_]
    X[py_] = -R[Y_] * x1[px_] + R[X_] * x1[py_]
    # Simplified p_rot
    X[px_] -= c0


def LtoG(X, S, R, c0, c1, s1):
    """"""

    # Simplified p_rot
    X[px_] -= c0
    # Rotate
    x1 = [v.copy() if isinstance(v, (TPS, tpsa)) else v for v in X]
    X[x_] = R[X_] * x1[x_] - R[Y_] * x1[y_]
    X[px_] = R[X_] * x1[px_] - R[Y_] * x1[py_]
    X[y_] = R[Y_] * x1[x_] + R[X_] * x1[y_]
    X[py_] = R[Y_] * x1[px_] + R[X_] * x1[py_]
    # Translate
    X[x_] += S[X_]
    X[y_] += S[Y_]
    # p_rot rotated
    X[px_] += c1
    X[py_] += s1


def Drift(L, x):
    """
    Based on t2elem.cc in PyTracy
    """

    if not globval.H_exact:
        u = L / (1.0 + x[delta_])
        x[ct_] += u * (sqr(x[px_]) + sqr(x[py_])) / (2.0 * (1.0 + x[delta_]))
    else:
        u = L / get_p_s(x)
        x[ct_] += u * (1.0 + x[delta_]) - L

    x[x_] += x[px_] * u
    x[y_] += x[py_] * u
    if globval.pathlength:
        x[ct_] += L


def thin_kick(Order, MB, L, h_bend, h_ref, x):
    """
    t2elem.cc

    The kick is given by

              e L       L delta    L x              e L
     Dp_x = - --- B_y + ------- - ----- ,    Dp_y = --- B_x
              p_0         rho     rho^2             p_0

    where

                           ====
                           \
      (B_y + iB_x) = B rho  >   (ia_n  + b_n ) (x + iy)^n-1
                           /
                           ====

    where

       e      1
      --- = -----
      p_0   B rho
    """

    if (h_bend != 0.0) or (1 <= Order <= HOMmax):
        x0 = [v.copy() if isinstance(v, (TPS, tpsa)) else v for v in x]

        # compute field with Horner's rule
        ByoBrho = MB[Order + HOMmax]
        BxoBrho = MB[HOMmax - Order]

        for j in range(1, Order)[::-1]:
            ByoBrho1 = x0[x_] * ByoBrho - x0[y_] * BxoBrho + MB[j + HOMmax]
            BxoBrho = x0[y_] * ByoBrho + x0[x_] * BxoBrho + MB[HOMmax - j]
            ByoBrho = ByoBrho1.copy() if isinstance(ByoBrho1, (TPS, tpsa)) else ByoBrho1

        if globval.radiation or globval.emittance:
            B = [0.0] * 3
            B[X_] = BxoBrho
            B[Y_] = ByoBrho + h_bend
            B[Z_] = 0.0
            radiate(x, L, h_ref, B)

        if h_ref != 0.0:
            x[px_] -= L * (
                ByoBrho
                + (h_bend - h_ref) / 2.0
                + h_ref * h_bend * x0[x_]
                - h_ref * x0[delta_]
            )
            x[ct_] += L * h_ref * x0[x_]
        else:
            x[px_] -= L * (ByoBrho + h_bend)
        x[py_] += L * BxoBrho


def EdgeFocus(irho, phi, gap, x):
    """"""

    if USE_ELEGANT_EDGE_FOCUS:
        px = x[px_]
        py = x[py_]
        pz = tps_sqrt((1.0 + x[delta_]) * (1.0 + x[delta_]) - px * px - py * py)
        xp = px / pz
        yp = py / pz
        x[px_] = xp
        x[py_] = yp

    x[px_] += irho * np.tan(np.deg2rad(phi)) * x[x_]

    if False:
        # warning: => diverging Taylor map (see SSC-141)
        x[py_] -= (
            irho
            * np.tan(np.deg2rad(phi) - get_psi(irho, phi, gap))
            * x[y_]
            / (1.0 + x[delta_])
        )
    else:
        x[py_] -= irho * np.tan(np.deg2rad(phi) - get_psi(irho, phi, gap)) * x[y_]

    if USE_ELEGANT_EDGE_FOCUS:
        xp = x[px_]
        yp = x[py_]
        pz = (1.0 + x[delta_]) / tps_sqrt(1.0 + xp * xp + yp * yp)
        px = xp * pz
        py = yp * pz

        x[px_] = px
        x[py_] = py


def p_rot(phi, x):
    """"""

    c = np.cos(np.deg2rad(phi))
    s = np.sin(np.deg2rad(phi))
    ps = get_p_s(x)

    if not globval.H_exact:
        x[px_] = s * ps + c * x[px_]
    else:
        x1 = [v.copy() if isinstance(v, (TPS, tpsa)) else v for v in x]
        p = c * ps - s * x1[px_]
        x[x_] = x1[x_] * ps / p
        x[px_] = s * ps + c * x1[px_]
        x[y_] += x1[x_] * x1[py_] * s / p
        x[ct_] += (1.0 + x1[delta_]) * x1[x_] * s / p


def bend_fringe(hb, x):
    """
    t2elem.cc
    """

    coeff = -hb / 2.0
    x1 = [v.copy() if isinstance(v, (TPS, tpsa)) else v for v in x]
    ps = get_p_s(x)
    ps2 = sqr(ps)
    ps3 = ps * ps2
    u = 1.0 + 4.0 * coeff * x1[px_] * x1[y_] * x1[py_] / ps3
    if isinstance(u, (TPS, tpsa)):
        x[y_] = 2.0 * x1[y_] / (1.0 + tps_sqrt(u))
        x[x_] = x1[x_] - coeff * sqr(x[y_]) * (ps2 + sqr(x1[px_])) / ps3
        x[py_] = x1[py_] + 2.0 * coeff * x1[px_] * x[y_] / ps
        x[ct_] = x1[ct_] - coeff * x1[px_] * sqr(x[y_]) * (1.0 + x1[delta_]) / ps3
    else:

        try:
            if isinstance(u, np.ndarray):
                assert np.all(u >= 0.0)
            else:
                assert u >= 0.0

            x[y_] = 2.0 * x1[y_] / (1.0 + np.sqrt(u))
            x[x_] = x1[x_] - coeff * sqr(x[y_]) * (ps2 + sqr(x1[px_])) / ps3
            x[py_] = x1[py_] + 2.0 * coeff * x1[px_] * x[y_] / ps
            x[ct_] = x1[ct_] - coeff * x1[px_] * sqr(x[y_]) * (1.0 + x1[delta_]) / ps3

        except AssertionError:
            print("bend_fringe: *** Speed of light exceeded!\n")
            x[x_] = np.nan
            x[px_] = np.nan
            x[y_] = np.nan
            x[py_] = np.nan
            x[delta_] = np.nan
            x[ct_] = np.nan


def get_p_s(x):
    """
    Based on t2elem.cc in PyTracy
    """

    if not globval.H_exact:
        p_s = 1.0 + x[delta_]
    else:
        p_s2 = sqr(1.0 + x[delta_]) - sqr(x[px_]) - sqr(x[py_])

        if isinstance(p_s2, (TPS, tpsa)):
            p_s = tps_sqrt(p_s2)
        else:
            if p_s2 >= 0.0:
                p_s = np.sqrt(p_s2)
            else:
                print("get_p_s(): *** Speed of light exceeded!\n")
                p_s = np.nan

    return p_s


def get_psi(irho, phi, gap):
    """
    Correction for magnet gap (longitudinal fringe field)

       irho h = 1/rho [1/m]
       phi  edge angle
       gap  full gap between poles

                                    2
                   K1*gap*h*(1 + sin phi)
            psi = ----------------------- * (1 - K2*g*gap*tan phi)
                        cos phi

            K1 is usually 1/2
            K2 is zero here
    """

    k1 = 0.5
    k2 = 0.0

    if phi == 0.0:
        psi = 0.0
    else:
        psi = (
            k1
            * gap
            * irho
            * (1.0 + sqr(np.sin(np.deg2rad(phi))))
            / np.cos(np.deg2rad(phi))
            * (1.0 - k2 * gap * irho * np.tan(np.deg2rad(phi)))
        )

    return psi


def get_pass_funcs_from_pylatt(pylatt_obj, nslice=4):
    """"""

    lattice = pylatt_obj

    # Convert pylatt lattice into a list of ppytracy's element pass functions
    pass_funcs = []
    spos = 0.0
    for elem in lattice.BL:
        if type(elem) in (lattice.latt.drif, lattice.latt.kick):
            pass_funcs.append(partial(Drift, elem.L))
        elif type(elem) in (
            lattice.latt.bend,
            lattice.latt.quad,
            lattice.latt.skew,
            lattice.latt.sext,
        ):
            fnum = knum = elem_index = 1
            kind, method, nslice = 1, 4, nslice  # 20 #4
            dS_x = dS_y = PdTpar = dTerror = 0.0
            Pirho = PTx1 = PTx2 = Pgap = 0.0
            if type(elem) is lattice.latt.bend:
                Pirho = elem.angle / elem.L
                PTx1 = np.rad2deg(elem.e1)
                PTx2 = np.rad2deg(elem.e2)
                nmpole = 0
                n_design = 1
                bn_an = {}
            elif type(elem) is lattice.latt.quad:
                nmpole = 1
                n_design = 2
                bn_an = {2: dict(b=elem.K1, a=0.0)}
            elif type(elem) is lattice.latt.skew:
                nmpole = 1
                n_design = 2
                bn_an = {2: dict(b=0.0, a=elem.K1)}
            elif type(elem) is lattice.latt.sext:
                nmpole = 1
                n_design = 3
                bn_an = {
                    3: dict(b=elem.K2 / 2, a=0.0)
                }  # 2! difference between ELEGANT/MAD & Tracy/AT
            else:
                raise NotImplementedError

            flat_file_elem_def_str = f"""
            {elem.name}                 {fnum}    {knum}    {elem_index}
              {kind}   {method}   {nslice}
             -1.0000000000000000e+01  1.0000000000000000e+01 -1.0000000000000000e+01  1.0000000000000000e+01
             {dS_x:.16e}  {dS_y:.16e}  {PdTpar:.16e}  {dTerror:.16e}
             {elem.L:.16e}  {Pirho:.16e}  {PTx1:.16e}  {PTx2:.16e}  {Pgap:.16e}
                 {nmpole}  {n_design}
            """
            for _n, ba_d in bn_an.items():
                flat_file_elem_def_str += f' {_n}  {ba_d["b"]:.16e}  {ba_d["a"]:.16e}\n'

            elem_index, cell = get_elem_index_and_cell(flat_file_elem_def_str)
            pass_funcs.append(partial(Mpole_Pass, cell))
        elif type(elem) in (lattice.latt.moni, lattice.latt.aper):
            pass  # TO-BE-IMPLEMENTED later
        else:
            raise NotImplementedError(elem.name)

        spos += elem.L

    return pass_funcs


def get_pass_funcs_from_ltemanager(LTE_obj, N_KICKS=None, mod_prop_dict_list=None):
    """"""

    if N_KICKS is None:
        N_KICKS = dict(CSBEND=20, KQUAD=20, KSEXT=4, KOCT=4)

    LTE = LTE_obj

    if mod_prop_dict_list is not None:
        LTE.modify_elem_properties(mod_prop_dict_list)
    d = LTE.get_persistent_used_beamline_element_defs(
        used_beamline_name=LTE.used_beamline_name
    )
    # d = LTE.get_used_beamline_element_defs(used_beamline_name=LTE.used_beamline_name)

    elem_defs = {}
    for elem_name, elem_type, prop_str in d["elem_defs"]:
        elem_defs[elem_name] = {"elem_type": elem_type}
        elem_defs[elem_name].update(LTE.parse_elem_properties(prop_str))

    # Convert pylatt lattice into a list of ppytracy's element pass functions
    pass_funcs = []
    spos = 0.0
    for elem_name in d["flat_used_elem_names"]:
        elem = elem_defs[elem_name]
        elem_type = elem["elem_type"]

        if elem_type in ("DRIF", "KICK", "HKICK", "VKICK", "EDRIFT"):

            pass_funcs.append(partial(Drift, elem.get("L", 0.0)))

        elif elem_type in ("CSBEND", "KQUAD", "KSEXT", "KOCT"):

            fnum = knum = elem_index = 1
            kind, method, nslice = 1, 4, N_KICKS.get(elem_type, 20)
            dS_x = dS_y = PdTpar = dTerror = 0.0
            Pirho = PTx1 = PTx2 = Pgap = 0.0
            L = elem.get("L", 0.0)
            # "nmpole" refers to the number of defined multipole component strength values
            if elem_type == "CSBEND":
                Pirho = elem["ANGLE"] / L
                PTx1 = np.rad2deg(elem.get("E1", 0.0))
                PTx2 = np.rad2deg(elem.get("E2", 0.0))
                n_design = 1
                if elem.get("K1", 0.0) == 0.0:
                    bn_an = {}
                    nmpole = 0
                else:
                    bn_an = {2: dict(b=elem.get("K1", 0.0), a=0.0)}
                    nmpole = 1
            elif elem_type == "KQUAD":
                nmpole = 1
                n_design = 2
                bn_an = {2: dict(b=elem.get("K1", 0.0), a=0.0)}
            # elif type(elem) is lattice.latt.skew:
            # nmpole = 1
            # n_design = 2
            # bn_an = {2: dict(b=0.0, a=elem.K1)}
            elif elem_type == "KSEXT":
                nmpole = 1
                n_design = 3
                bn_an = {
                    3: dict(b=elem.get("K2", 0.0) / 2, a=0.0)
                }  # 2! difference between ELEGANT/MAD & Tracy/AT
            elif elem_type == "KOCT":
                nmpole = 1
                n_design = 4
                bn_an = {
                    4: dict(b=elem.get("K3", 0.0) / 6, a=0.0)
                }  # 3! difference between ELEGANT/MAD & Tracy/AT
            else:
                raise NotImplementedError

            flat_file_elem_def_str = f"""
            {elem_name}                 {fnum}    {knum}    {elem_index}
              {kind}   {method}   {nslice}
             -1.0000000000000000e+01  1.0000000000000000e+01 -1.0000000000000000e+01  1.0000000000000000e+01
             {dS_x:.16e}  {dS_y:.16e}  {PdTpar:.16e}  {dTerror:.16e}
             {L:.16e}  {Pirho:.16e}  {PTx1:.16e}  {PTx2:.16e}  {Pgap:.16e}
                 {nmpole}  {n_design}
            """
            for _n, ba_d in bn_an.items():
                flat_file_elem_def_str += f' {_n}  {ba_d["b"]:.16e}  {ba_d["a"]:.16e}\n'

            elem_index, cell = get_elem_index_and_cell(flat_file_elem_def_str)
            pass_funcs.append(partial(Mpole_Pass, cell))

        elif elem_type in ("KICK", "MONI", "MALIGN", "WATCH", "MARK", "MULT"):
            pass
        else:
            raise NotImplementedError((elem_name, elem_type))

        spos += elem.get("L", 0.0)

        if False:
            try:
                print(elem_name, elem_type, spos, pass_funcs[-1])
            except:
                print(elem_name, elem_type, spos)

    return pass_funcs


def get_elem_index_and_cell(flat_file_elem_def_str):
    """"""

    lines = [
        line.strip()
        for line in flat_file_elem_def_str.split("\n")
        if line.strip() != ""
    ]
    iLine = 0

    cell = SimpleNamespace()
    Elem = cell.Elem = SimpleNamespace()
    M = Elem.M = SimpleNamespace()

    # Default values
    cell.dS = [0.0, 0.0]
    cell.dT = [1.0, 0.0]

    tokens = [s.strip() for s in lines[iLine].split() if s.strip() != ""]
    iLine += 1

    Elem.PName = tokens[0]
    Elem.Fnum = int(tokens[1])
    Elem.Knum = int(tokens[2])
    elem_index = int(tokens[3])

    kind, method, n = [int(s.strip()) for s in lines[iLine].split() if s.strip() != ""]
    iLine += 1

    if kind == 1:
        # ### Initialization within "Mpole_Alloc()" in "t2elem.cc" ###
        M.Pmethod = Method.Meth_Fourth
        M.PN = 0
        # Displacement errors
        M.PdSsys = [0.0, 0.0]
        M.PdSrms = [0.0, 0.0]
        M.PdSrnd = [0.0, 0.0]
        M.PdTpar = 0.0  # Roll angle
        M.PdTsys = 0.0  # systematic Roll errors
        M.PdTrms = 0.0  # random Roll errors
        M.PdTrnd = 0.0  # random seed
        M.PB = [0.0] * (HOMmax * 2 + 1)
        M.PBpar = [0.0] * (HOMmax * 2 + 1)
        M.PBsys = [0.0] * (HOMmax * 2 + 1)
        M.PBrms = [0.0] * (HOMmax * 2 + 1)
        M.PBrnd = [0.0] * (HOMmax * 2 + 1)
        M.Porder = 0
        M.n_design = 0
        M.Pirho = 0.0  # inverse of curvature radius
        M.PTx1 = 0.0  # Entrance angle
        M.PTx2 = 0.0  # Exit angle
        M.Pgap = 0.0  # Gap for fringe field ???
        M.Pc0 = M.Pc1 = M.Ps1 = 0.0

        Elem.Pkind = PartsKind.Mpole
        M.Pthick = ThickType.thick
    else:
        raise ValueError

    cell.maxampl = [[-10.0, +10.0], [-10.0, +10.0]]  # default

    (
        cell.maxampl[X_][0],
        cell.maxampl[X_][1],
        cell.maxampl[Y_][0],
        cell.maxampl[Y_][1],
    ) = [float(s.strip()) for s in lines[iLine].split() if s.strip() != ""]
    iLine += 1

    Elem.PL = 0.0  # default

    if Elem.Pkind == PartsKind.Mpole:
        M.Pmethod = method
        M.PN = n

        if M.Pthick == ThickType.thick:
            cell.dS[X_], cell.dS[Y_], M.PdTpar, dTerror = [
                float(s.strip()) for s in lines[iLine].split() if s.strip() != ""
            ]
            iLine += 1
            cell.dT[X_] = np.cos(np.deg2rad(dTerror + M.PdTpar))
            cell.dT[Y_] = np.sin(np.deg2rad(dTerror + M.PdTpar))

            Elem.PL, M.Pirho, M.PTx1, M.PTx2, M.Pgap = [
                float(s.strip()) for s in lines[iLine].split() if s.strip() != ""
            ]
            iLine += 1

            if M.Pirho != 0.0:
                M.Porder = 1
        else:  # Not Thick
            cell.dS[X_], cell.dS[Y_], dTerror = np.nan, np.nan, np.nan
            cell.dT[X_] = np.cos(np.deg2rad(dTerror))
            cell.dT[Y_] = np.sin(np.deg2rad(dTerror))

        M.PdTsys = dTerror

        M.Pc0 = np.sin(Elem.PL * M.Pirho / 2.0)
        M.Pc1 = np.cos(np.deg2rad(M.PdTpar)) * M.Pc0
        M.Ps1 = np.sin(np.deg2rad(M.PdTpar)) * M.Pc0

        nmpole, M.n_design = [
            int(s.strip()) for s in lines[iLine].split() if s.strip() != ""
        ]
        iLine += 1

        for j in range(1, nmpole + 1):
            tokens = [s.strip() for s in lines[iLine].split() if s.strip() != ""]
            iLine += 1

            n = int(tokens[0])
            M.PB[HOMmax + n] = float(tokens[1])
            M.PB[HOMmax - n] = float(tokens[2])

            M.PBpar[HOMmax + n] = M.PB[HOMmax + n]
            M.PBpar[HOMmax - n] = M.PB[HOMmax - n]
            M.Porder = max([n, M.Porder])
    else:
        raise ValueError

    return elem_index, cell
