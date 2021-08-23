from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.special import binom, factorial
from six.moves import cPickle as pickle
import itertools
import time

import warnings

warnings.filterwarnings("error")
# warnings.filterwarnings('ignore')

import cytpsa
from cytpsa import TPS

# ----------------------------------------------------------------------
def test_TPS_addition():
    """"""

    import pprint

    from sympy import symbols, Poly, diff

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    f1 = lambda x, p: 1.5 * x ** 2 - 2.4 * p
    f2 = lambda x, p: 3.3 * x + 2.3 * p - 1.1 * p ** 2

    s1 = f1(x, p)
    s2 = f2(x, p)

    s_ans = s1 + s2

    a = Poly(s_ans, x, p)
    pprint.pprint(a.terms())

    norder = 3
    t_x = TPS(var_names, norder)
    t_p = TPS(var_names, norder)

    t_x.set_to_var("x")
    t_p.set_to_var("p")

    t1 = f1(t_x, t_p)
    t2 = f2(t_x, t_p)

    for _s, _t in [(s1, t1), (s2, t2), (s1 + s2, t1 + t2)]:
        a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = 0.0
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t.get_Taylor_coeff(exp) - s_c)
            np.testing.assert_almost_equal(_t.get_Taylor_coeff(exp), s_c, decimal=16)

    _s = s1 * s2
    _t = t1 * t2
    #
    a = Poly(_s, x, p)
    all_exponents = _t.T["exponents"].tolist()
    s_monoms = a.monoms()
    for m in s_monoms:
        m = list(m)
        if m not in all_exponents:
            all_exponents.append(m)
    #
    for exp in all_exponents:
        if tuple(exp) not in s_monoms:
            s_c = 0.0
        else:
            s_c = a.coeffs()[a.monoms().index(tuple(exp))]

        print(exp, _t.get_Taylor_coeff(exp), s_c)

    all_taylor_coeffs = _t.get_Taylor_coeffs()
    for i, _exp in enumerate(_t.T["exponents"]):
        np.testing.assert_almost_equal(
            _t.get_Taylor_coeff(_exp), all_taylor_coeffs[i], decimal=16
        )

    # Test 4-variable cases

    x, px, y, py = symbols(r"x px y py", real=True)

    var_names = ["x", "px", "y", "py"]

    f1 = (
        lambda x, px, y, py: 1.5 * x ** 2
        - 2.4 * px
        + 0.8 * x * py
        - 14.0 * py ** 3
        + 2 * x ** 4
    )
    f2 = (
        lambda x, px, y, py: 3.3 * x
        + 2.3 * px
        - 1.1 * px ** 2
        + 8.9 * x ** 2 * px * py
        - 9.1 * x * px * y * py
    )
    f3 = lambda x, px, y, py: (1.2 * x + 2.2 * px - 3.4 * y - 4.7 * py) ** 4

    s1 = f1(x, px, y, py)
    s2 = f2(x, px, y, py)
    s3 = f3(x, px, y, py)

    s_ans = s1 + s2

    a = Poly(s_ans, x, px, y, py)
    pprint.pprint(a.terms())

    s_ans = s1 * s2

    a = Poly(s_ans, x, px, y, py)
    pprint.pprint(a.terms())

    norder = 4
    t_x = TPS(var_names, norder)
    t_px = TPS(var_names, norder)
    t_y = TPS(var_names, norder)
    t_py = TPS(var_names, norder)

    t_x.set_to_var("x")
    t_px.set_to_var("px")
    t_y.set_to_var("y")
    t_py.set_to_var("py")

    t1 = f1(t_x, t_px, t_y, t_py)
    t2 = f2(t_x, t_px, t_y, t_py)
    t3 = f3(t_x, t_px, t_y, t_py)

    for _s, _t in [
        (s1, t1),
        (s2, t2),
        (s1 + s2, t1 + t2),
        (s1 * s2, t1 * t2),
        (s3, t3),
    ]:
        a = Poly(_s, x, px, y, py)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = 0.0
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t.get_Taylor_coeff(exp) - s_c)
            np.testing.assert_almost_equal(_t.get_Taylor_coeff(exp), s_c, decimal=12)

    # Test differentiation
    for _s, _t in [
        (s1, t1),
        (s2, t2),
    ]:  # You cannot test "s1 * s2" as its degree is 8 > norder
        s_diff = diff(_s, x)
        t_diff = _t.diff("x")

        a = Poly(s_diff, x, px, y, py)
        s_monoms = a.monoms()
        for exp in t_diff.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = 0.0
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, t_diff.get_Taylor_coeff(exp) - s_c)
            np.testing.assert_almost_equal(
                t_diff.get_Taylor_coeff(exp), s_c, decimal=14
            )

    # Test substitution
    _s = s3
    _t = t3
    #
    for _s_sub, _t_sub in [
        [
            _s.subs(px, 4.4),
            _t.subs(
                [
                    "px",
                ],
                [4.4],
            ),
        ],
        [_s.subs(x, -1.2).subs(px, 4.4), _t.subs(["x", "px"], [-1.2, 4.4])],
    ]:
        a = Poly(_s_sub, x, px, y, py)
        s_monoms = a.monoms()
        for exp in _t_sub.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = 0.0
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t_sub.get_Taylor_coeff(exp) - s_c)
            np.testing.assert_almost_equal(
                _t_sub.get_Taylor_coeff(exp), s_c, decimal=10
            )

    # Speed check

    var_names = ["x", "px", "y", "py", "cdt", "delta"]

    f1 = (
        lambda x, px, y, py, cdt, delta: 1.5 * x ** 2
        - 2.4 * px
        + 0.8 * x * py
        - 14.0 * py ** 3
        + 2 * x ** 4
        + 1.25 * cdt ** 3 * delta
        - 4.3 * delta ** 3
    )
    f2 = (
        lambda x, px, y, py, cdt, delta: 3.3 * x
        + 2.3 * px
        - 1.1 * px ** 2
        + 8.9 * x ** 2 * px * py
        - 9.1 * x * px * y * py
        - 14.9 * cdt * x * px * y
        + 9.97 * delta * 2 * y ** 2
    )

    _t0 = time.time()

    norder = 7
    t_x = TPS(var_names, norder)
    t_px = TPS(var_names, norder)
    t_y = TPS(var_names, norder)
    t_py = TPS(var_names, norder)
    t_cdt = TPS(var_names, norder)
    t_delta = TPS(var_names, norder)

    t_x.set_to_var("x")
    t_px.set_to_var("px")
    t_y.set_to_var("y")
    t_py.set_to_var("py")
    t_cdt.set_to_var("cdt")
    t_delta.set_to_var("delta")

    t1 = f1(t_x, t_px, t_y, t_py, t_cdt, t_delta)
    t2 = f2(t_x, t_px, t_y, t_py, t_cdt, t_delta)

    _t = t1 * t2

    print("Elapsed [s]: {0:.6f}".format(time.time() - _t0))


# ----------------------------------------------------------------------
def test_TPS_division(complex_coeffs):
    """"""

    import pprint

    from sympy import symbols, Poly, diff, S

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 9
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    if not complex_coeffs:
        t_ans["s1"] = 1 / (1 - t_x)  # = 1 + x + x**2 + x**3 + x**4 + ...
        # print(t_ans['s1'].get_Taylor_coeffs())

        t_ans["s1_in_place"] = t_x.copy()
        t_ans["s1_in_place"].set_to_one()
        t_ans["s1_in_place"] /= 1 - t_x

        s_ans["s1"] = (1 / (1 - x)).series(n=norder + 1)
        s_ans["s1_in_place"] = s_ans["s1"]

        t_ans["s2"] = (1 + 3 * t_x) / (1 - 4 * t_x ** 2)
        s_ans["s2"] = ((1 + 3 * x) / (1 - 4 * x ** 2)).series(n=norder + 1)

        t_ans["s2_in_place"] = 1 + 3 * t_x
        t_ans["s2_in_place"] /= 1 - 4 * t_x ** 2
        s_ans["s2_in_place"] = s_ans["s2"]
    else:
        t_ans["s1"] = 1 / (1j - t_x)

        t_ans["s1_in_place"] = t_x.copy()
        t_ans["s1_in_place"].set_to_one()
        t_ans["s1_in_place"] /= 1j - t_x

        s_ans["s1"] = (1 / (1j - x)).series(n=norder + 1)
        s_ans["s1_in_place"] = s_ans["s1"]

        t_ans["s2"] = (1 + 3j * t_x) / (1j - 4 * t_x ** 2)
        s_ans["s2"] = ((1 + 3j * x) / (1j - 4 * x ** 2)).series(n=norder + 1)

    for k in sorted(list(s_ans)):
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            ##print(exp, _t.get_Taylor_coeff(exp) - s_c)
            # np.testing.assert_almost_equal(_t.get_Taylor_coeff(exp), s_c, decimal=16)

            try:
                print(
                    "abs. diff: {:.4e}".format(
                        _t.get_Taylor_coeff(exp) - complex(s_c.evalf())
                    )
                )
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), complex(s_c.evalf()), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                print(
                    "rel. diff: {:.4e}".format(
                        np.abs(_t.get_Taylor_coeff(exp) / complex(s_c.evalf())) - 1.0
                    )
                )
                rel_tol = 1e-12
                assert (
                    np.abs(_t.get_Taylor_coeff(exp) / complex(s_c.evalf()) - 1.0)
                    < rel_tol
                )

    f1 = lambda x, p: 1.5 * x ** 2 * p + 9.0 * x + 12.0 * x ** 3
    f2 = lambda x, p: 3 * x

    s1 = f1(x, p)
    s2 = f2(x, p)

    # s_ans = s1 / s2

    # a = Poly(s_ans, x, p)
    # pprint.pprint(a.terms())

    norder = 3
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    t1 = f1(t_x, t_p)
    t2 = f2(t_x, t_p)

    div_1_in_place = t1.copy()
    div_1_in_place /= 1.5

    div_2_in_place = t1.copy()
    div_2_in_place /= t2

    for _s, _t in [
        (s1, t1),
        (s2, t2),
        (s1 / 1.5, t1 / 1.5),
        (s1 / s2, t1 / t2),
        (s1 / 1.5, div_1_in_place),
        (s1 / s2, div_2_in_place),
    ]:
        a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            ##print(exp, _t.get_Taylor_coeff(exp) - s_c)
            # np.testing.assert_almost_equal(_t.get_Taylor_coeff(exp), s_c, decimal=16)

            try:
                print(
                    "abs. diff: {:.4e}".format(
                        _t.get_Taylor_coeff(exp) - complex(s_c.evalf())
                    )
                )
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), complex(s_c.evalf()), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                print(
                    "rel. diff: {:.4e}".format(
                        np.abs(_t.get_Taylor_coeff(exp) / complex(s_c.evalf())) - 1.0
                    )
                )
                rel_tol = 1e-12
                assert (
                    np.abs(_t.get_Taylor_coeff(exp) / complex(s_c.evalf()) - 1.0)
                    < rel_tol
                )

    f2 = lambda x, p: 3 * x + p

    t2 = f2(t_x, t_p)

    try:
        t_ans = t1 / t2
    except ValueError:
        print("Successfully ValueError was raised")

    t_ans = t1.copy()
    try:
        t_ans /= t2
    except ValueError:
        print("Successfully ValueError was raised")


# ----------------------------------------------------------------------
def test_TPS_power(complex_coeffs):
    """"""

    import pprint

    from sympy import symbols, Poly, diff

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 9
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    t_ans["s1"] = (1 + t_x + 2 * t_p) ** 0
    s_ans["s1"] = ((1 + x + 2 * p) ** 0).series(n=norder + 1)

    t_ans["s2"] = (1 + t_x + 2 * t_p) ** 1
    s_ans["s2"] = (1 + x + 2 * p) ** 1

    t_ans["s3"] = (1 + t_x + 2 * t_p) ** 3
    s_ans["s3"] = (1 + x + 2 * p) ** 3

    t_ans["s4"] = (1 - t_x) ** (-1)
    s_ans["s4"] = ((1 - x) ** (-1)).series(n=norder + 1)

    t_ans["s5"] = (1 - t_x) ** (-3)
    s_ans["s5"] = ((1 - x) ** (-3)).series(n=norder + 1)

    for k in sorted(list(s_ans)):
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = 0.0
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            ##print(exp, _t.get_Taylor_coeff(exp) - s_c)
            # np.testing.assert_almost_equal(_t.get_Taylor_coeff(exp), s_c, decimal=16)

            try:
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), s_c, decimal=16
                )
            except AssertionError:
                # Try checking relative error:
                assert np.abs(_t.get_Taylor_coeff(exp) / s_c - 1.0) < 1e-12

    try:
        t_ans = t_x ** (-1)
    except ValueError:
        print("Successfully ValueError was raised")

    try:
        t_ans = t_x ** (-3)
    except ValueError:
        print("Successfully ValueError was raised")


# ----------------------------------------------------------------------
def test_TPS_exponential(complex_coeffs):
    """"""

    import pprint

    from sympy import symbols, Poly, diff, S
    from sympy import exp as sym_exp

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 9
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    if not complex_coeffs:
        t_ans["s1"] = cytpsa.exp(t_x)
        s_ans["s1"] = (sym_exp(x)).series(n=norder + 1)

        t_ans["s2"] = 1.4 * cytpsa.exp(3 * t_x + 1) - 5.0
        s_ans["s2"] = (1.4 * sym_exp(3 * x + 1) - 5.0).series(n=norder + 1)

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa.exp(3 * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5 * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_exp(3 * x + 1) + 9.0 * (p ** 2) - 4.5 * (x ** 4) * p
        ).series(x, n=norder + 1)
    else:
        t_ans["s1"] = cytpsa.exp(t_x)
        s_ans["s1"] = (sym_exp(x)).series(n=norder + 1)

        t_ans["s2"] = 1.4j * cytpsa.exp(3 * t_x + 1) - 5.0j
        s_ans["s2"] = (1.4j * sym_exp(3 * x + 1) - 5.0j).series(n=norder + 1)

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa.exp(3j * t_x + 1)
            + 9.0j * (t_p ** 2)
            - 4.5 * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_exp(3j * x + 1) + 9.0j * (p ** 2) - 4.5 * (x ** 4) * p
        ).series(x, n=norder + 1)

    for k in sorted(list(s_ans)):
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t.get_Taylor_coeff(exp) - s_c.evalf())
            try:
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), s_c.evalf(), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                assert np.abs(_t.get_Taylor_coeff(exp) / s_c.evalf() - 1.0) < 1e-12

    print("\n* Finished test successfully.")


# ----------------------------------------------------------------------
def test_TPS_log(complex_coeffs):
    """"""

    import pprint

    from sympy import symbols, Poly, diff, S
    from sympy import ln as sym_log

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 9
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    if not complex_coeffs:
        t_ans["s1"] = cytpsa.log(1 + t_x)
        s_ans["s1"] = (sym_log(1 + x)).series(n=norder + 1)

        t_ans["s2"] = 1.5 * cytpsa.log((1 + t_x) / (1 - t_x)) - 5.0
        s_ans["s2"] = (1.5 * sym_log((1 + x) / (1 - x)) - 5.0).series(n=norder + 1)

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa.log(3 * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5 * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_log(3 * x + 1) + 9.0 * (p ** 2) - 4.5 * (x ** 4) * p
        ).series(x, n=norder + 1)
    else:
        t_ans["s1"] = cytpsa.log(1 + t_x * 1j)
        s_ans["s1"] = (sym_log(1 + x * 1j)).series(n=norder + 1)

        t_ans["s2"] = 1.5 * cytpsa.log((1 + t_x * 3j) / (1 - t_x)) - 5.0j
        s_ans["s2"] = (1.5 * sym_log((1 + x * 3j) / (1 - x)) - 5.0j).series(
            n=norder + 1
        )

        t_ans["s3"] = (
            -3.0j * t_p * cytpsa.log(3j * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5j * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0j * p * sym_log(3j * x + 1) + 9.0 * (p ** 2) - 4.5j * (x ** 4) * p
        ).series(x, n=norder + 1)

    for k in sorted(list(s_ans)):
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t.get_Taylor_coeff(exp) - s_c.evalf())
            try:
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), s_c.evalf(), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                assert np.abs(_t.get_Taylor_coeff(exp) / s_c.evalf() - 1.0) < 1e-12

    try:
        t_ans = cytpsa.log(t_x)  # case of "a0 = 0"
    except AssertionError:
        print("Successfully AssertionError was raised")

    try:
        t_ans = cytpsa.log(t_x - 1e-6)  # case of "a0 < 0"
    except AssertionError:
        print("Successfully AssertionError was raised")

    print("\n* Finished test successfully.")


# ----------------------------------------------------------------------
def test_TPS_sqrt(complex_coeffs):
    """"""

    import pprint

    from sympy import symbols, Poly, diff, S
    from sympy import sqrt as sym_sqrt

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 9
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    if not complex_coeffs:
        t_ans["s1"] = cytpsa.sqrt(1 + t_x)
        s_ans["s1"] = (sym_sqrt(1 + x)).series(n=norder + 1)

        t_ans["s2"] = 1.5 * cytpsa.sqrt((3 + t_x * 2) / (1 - 4 * t_x ** 3)) - 5.0
        s_ans["s2"] = (1.5 * sym_sqrt((3 + x * 2) / (1 - 4 * x ** 3)) - 5.0).series(
            n=norder + 1
        )

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa.sqrt(3 * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5 * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_sqrt(3 * x + 1) + 9.0 * (p ** 2) - 4.5 * (x ** 4) * p
        ).series(x, n=norder + 1)
    else:
        t_ans["s1"] = cytpsa.sqrt(1j + t_x)
        s_ans["s1"] = (sym_sqrt(1j + x)).series(n=norder + 1)

        t_ans["s2"] = 1.5 * cytpsa.sqrt((3 + t_x * 2j) / (1 - 4j * t_x ** 3)) - 5.0j
        s_ans["s2"] = (1.5 * sym_sqrt((3 + x * 2j) / (1 - 4j * x ** 3)) - 5.0j).series(
            n=norder + 1
        )

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa.sqrt(3j * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5j * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_sqrt(3j * x + 1) + 9.0 * (p ** 2) - 4.5j * (x ** 4) * p
        ).series(x, n=norder + 1)

    for k in sorted(list(s_ans)):
        print("# Case: {}".format(k))
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t.get_Taylor_coeff(exp) - s_c.evalf())
            try:
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), s_c.evalf(), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                assert np.abs(_t.get_Taylor_coeff(exp) / s_c.evalf() - 1.0) < 1e-12

    try:
        t_ans = cytpsa.sqrt(t_x)  # case of "a0 = 0"
    except AssertionError:
        print("Successfully AssertionError was raised")

    try:
        t_ans = cytpsa.sqrt(t_x - 1e-6)  # case of "a0 < 0"
    except AssertionError:
        print("Successfully AssertionError was raised")

    print("\n* Finished test successfully.")


# ----------------------------------------------------------------------
def test_TPS_sin(complex_coeffs):
    """"""

    _test_TPS_trig("sin", complex_coeffs)


# ----------------------------------------------------------------------
def test_TPS_cos(complex_coeffs):
    """"""

    _test_TPS_trig("cos", complex_coeffs)


# ----------------------------------------------------------------------
def test_TPS_tan(complex_coeffs):
    """"""

    _test_TPS_trig("tan", complex_coeffs)


# ----------------------------------------------------------------------
def _test_TPS_trig(trig_name, complex_coeffs):
    """"""

    import pprint

    from sympy import symbols, Poly, diff, S

    if trig_name == "sin":
        from sympy import sin as sym_trig

        cytpsa_trig = cytpsa.sin
    elif trig_name == "cos":
        from sympy import cos as sym_trig

        cytpsa_trig = cytpsa.cos
    elif trig_name == "tan":
        from sympy import tan as sym_trig

        cytpsa_trig = cytpsa.tan
    else:
        raise ValueError('Invalid "trig_name"')

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 9
    t_x = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_p = TPS(var_names, norder, complex_coeffs=complex_coeffs)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    if not complex_coeffs:
        t_ans["s1"] = cytpsa_trig(t_x)
        s_ans["s1"] = (sym_trig(x)).series(n=norder + 1)

        t_ans["s2"] = 1.5 * cytpsa_trig((1 + t_x) / (1 - t_x)) - 5.0
        s_ans["s2"] = (1.5 * sym_trig((1 + x) / (1 - x)) - 5.0).series(n=norder + 1)

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa_trig(3 * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5 * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_trig(3 * x + 1) + 9.0 * (p ** 2) - 4.5 * (x ** 4) * p
        ).series(x, n=norder + 1)
    else:
        t_ans["s1"] = cytpsa_trig(t_x * 2j)
        s_ans["s1"] = (sym_trig(x * 2j)).series(n=norder + 1)

        t_ans["s2"] = 1.5 * cytpsa_trig((1 + t_x * 3j) / (1j - t_x)) - 5.0j
        s_ans["s2"] = (1.5 * sym_trig((1 + x * 3j) / (1j - x)) - 5.0j).series(
            n=norder + 1
        )

        t_ans["s3"] = (
            -3.0 * t_p * cytpsa_trig(3j * t_x + 1)
            + 9.0 * (t_p ** 2)
            - 4.5j * ((t_x ** 4) * t_p)
        )
        s_ans["s3"] = (
            -3.0 * p * sym_trig(3j * x + 1) + 9.0 * (p ** 2) - 4.5j * (x ** 4) * p
        ).series(x, n=norder + 1)

    for k in sorted(list(s_ans)):
        print("# Case: {}".format(k))
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            # print(exp, _t.get_Taylor_coeff(exp) - s_c.evalf())
            try:
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), s_c.evalf(), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                if trig_name != "tan":
                    rel_tol = 1e-12
                else:
                    rel_tol = 1e-10
                assert np.abs(_t.get_Taylor_coeff(exp) / s_c.evalf() - 1.0) < rel_tol

    print("\n* Finished test successfully.")


# ----------------------------------------------------------------------
def test_TPS_complex_coeff():
    """"""

    import pprint

    import cytpsa

    from sympy import symbols, Poly, diff, S
    from sympy import conjugate
    from sympy import Order

    x, p = symbols(r"x p", real=True)

    var_names = ["x", "p"]

    t_ans, s_ans = {}, {}

    norder = 5
    t_x = TPS(var_names, norder, complex_coeffs=True)
    t_p = TPS(var_names, norder, complex_coeffs=True)
    t_x.set_to_var("x")
    t_p.set_to_var("p")

    nth_power = 5

    mu = np.exp(1j * 0.205 * (2 * np.pi))

    counter = 0
    t_z = t_x - 1j * t_p
    t_ans["s{:02d}".format(counter)] = t_z
    counter += 1
    for _ in range(nth_power):
        # t_ans['s{:02d}'.format(counter)] = cytpsa.conjugate(t_z)
        # counter += 1
        # t_ans['s{:02d}'.format(counter)] = t_z + cytpsa.conjugate(t_z)
        # counter += 1
        # t_ans['s{:02d}'.format(counter)] = (t_z + cytpsa.conjugate(t_z))**2
        # counter += 1

        t_z = mu * (t_z - (t_z + cytpsa.conjugate(t_z)) ** 2)
        t_ans["s{:02d}".format(counter)] = t_z
        counter += 1

    truncate_sympy = False
    # truncate_sympy = True

    counter = 0
    z = (x - 1j * p).expand()
    s_ans["s{:02d}".format(counter)] = z
    counter += 1
    for n in range(nth_power):
        # s_ans['s{:02d}'.format(counter)] = conjugate(z)
        # counter += 1
        # s_ans['s{:02d}'.format(counter)] = z + conjugate(z)
        # counter += 1
        # s_ans['s{:02d}'.format(counter)] = (z + conjugate(z))**2
        # counter += 1

        z = (mu * (z - (z + conjugate(z)) ** 2)).expand()
        # z = z.simplify()
        s_ans["s{:02d}".format(counter)] = z
        counter += 1

        if truncate_sympy:
            z_temp = z + Order(x ** (nth_power + 1))
            z = z_temp.removeO().expand()

    n_compared_monoms = {}
    for k in sorted(list(s_ans)):
        print("# Case: {}".format(k))
        _s = s_ans[k]
        _t = t_ans[k]
        if hasattr(_s, "removeO"):
            a = Poly(_s.removeO(), x, p)
        else:
            a = Poly(_s, x, p)
        s_monoms = a.monoms()
        n_compared_monoms[k] = len(s_monoms)
        for exp in _t.T["exponents"]:
            if tuple(exp) not in s_monoms:
                s_c = S.Zero  # Here symbolic zero is needed in order to be able
                # use .evalf() later, rather than a floating point zero value.
            else:
                s_c = a.coeffs()[a.monoms().index(tuple(exp))]

            print(exp, _t.get_Taylor_coeff(exp), s_c.evalf())
            try:
                print(
                    "abs. diff: {:.4e}".format(
                        _t.get_Taylor_coeff(exp) - complex(s_c.evalf())
                    )
                )
                np.testing.assert_almost_equal(
                    _t.get_Taylor_coeff(exp), complex(s_c.evalf()), decimal=16
                )
                # .evalf() is needed to convert sympy's "E", which is a symbol for
                # np.exp(1), into a floating number for comparison.
            except AssertionError:
                # Try checking relative error:
                print(
                    "rel. diff: {:.4e}".format(
                        np.abs(_t.get_Taylor_coeff(exp) / complex(s_c.evalf())) - 1.0
                    )
                )
                rel_tol = 1e-12
                assert (
                    np.abs(_t.get_Taylor_coeff(exp) / complex(s_c.evalf()) - 1.0)
                    < rel_tol
                )

    print("\nn_compared_monoms:")
    print(n_compared_monoms)

    print("\n* Finished test successfully.")


if __name__ == "__main__":

    if False:
        test_TPS_addition()
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_division(complex_coeffs)
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_power(complex_coeffs)
    elif False:
        complex_coeffs = False
        # complex_coeffs = True
        test_TPS_exponential(complex_coeffs)
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_log(complex_coeffs)  # natural log
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_sqrt(complex_coeffs)
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_sin(complex_coeffs)
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_cos(complex_coeffs)
    elif False:
        # complex_coeffs = False
        complex_coeffs = True
        test_TPS_tan(complex_coeffs)

    elif True:
        test_TPS_complex_coeff()
