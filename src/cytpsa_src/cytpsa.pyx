'''
Author: Yoshiteru Hidaka (yhidaka@bnl.gov)
'''
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals

cimport cython # to be able to use cython decorators

import numpy as np
from scipy.special import binom, factorial, factorial2, comb
from six.moves import cPickle as pickle
import itertools
import time

#----------------------------------------------------------------------
def test_gen_tpsa_index_table(norder, nvar):
    """"""

    n = norder
    v = nvar

    assert np.mod(v, 2) == 0
    assert v <= 4

    from sympy import symbols

    symbolic_vars = symbols(r'x p_x y p_y', real=True)
    symbolic_vars = symbolic_vars[:v]

    order_list = []
    exp_list = []
    C1_list, C2_list = [], []
    for iOrder in range(n+1):
        print(list(itertools.combinations_with_replacement(symbolic_vars, iOrder)))

        for vec in itertools.combinations_with_replacement(symbolic_vars, iOrder):
            exponents = [vec.count(_var) for _var in symbolic_vars]

            C1 = np.sum([exponents[k      ] * ((n+1)**k) for k in range(v/2)])
            C2 = np.sum([exponents[k + v/2] * ((n+1)**k) for k in range(v/2)])

            print(exponents, C1, C2)

            order_list.append(iOrder)
            exp_list.append(exponents)
            C1_list.append(C1)
            C2_list.append(C2)

    order_list = np.array(order_list)
    exp_list = np.array(exp_list)
    C1_list = np.array(C1_list)
    C2_list = np.array(C2_list)

    # Placeholders for sorted
    orders, exponents, C1s, C2s = [], [], [], []

    _zero_C2 = (C2_list == 0)

    for i in np.lexsort((C1_list[_zero_C2], order_list[_zero_C2])):
        orders.append(order_list[_zero_C2][i])
        exponents.append(exp_list[_zero_C2][i])
        C1s.append(C1_list[_zero_C2][i])
        C2s.append(C2_list[_zero_C2][i])

    _C2_rest = (C2_list != 0)

    for min_order in range(n+1):

        matched = (order_list[_C2_rest] == min_order)
        for _existing_C2 in np.unique(C2s):
            matched = np.logical_and(matched, C2_list[_C2_rest] != _existing_C2)

        if np.sum(matched) == 0:
            continue

        matched_sorted_C2s = np.unique(C2_list[_C2_rest][matched]) # sorting by np.unique() is critical here
        for sel_C2 in matched_sorted_C2s:
            _C2_matched = (C2_list == sel_C2)
            for i in np.lexsort((C1_list[_C2_matched], order_list[_C2_matched])):
                orders.append(order_list[_C2_matched][i])
                exponents.append(exp_list[_C2_matched][i])
                C1s.append(C1_list[_C2_matched][i])
                C2s.append(C2_list[_C2_matched][i])

    I_M = range(1, len(C1_list)+1)

    #zip(I_M, orders, exponents, C1s, C2s)

    D1s, D2s = [], []
    D_inds = range(np.max(C1s + C2s) + 1)
    for sel_C in D_inds:
        if sel_C in C1s:
            D1s.append(I_M[C1s.index(sel_C)])
        else:
            D1s.append(0)
        if sel_C in C2s:
            D2s.append(I_M[C2s.index(sel_C)] - 1)
        else:
            D2s.append(0)

    #zip(D_inds, D1s, D2s)

    return dict(orders=orders, exponents=exponents,
                I_M=I_M, C1s=C1s, C2s=C2s, D1s=D1s, D2s=D2s)

#----------------------------------------------------------------------
cdef long get_I_MN(long[:] C1, long[:] C2, long[:] D1, long[:] D2,
                   long I_M, long I_N):
    """
    I_M & I_N are 1-based indexes for self.coeffs.

    Return I_{M*N} as 1-based index
    """

    return D1[C1[I_M-1] + C1[I_N-1]] + \
           D2[C2[I_M-1] + C2[I_N-1]]

#----------------------------------------------------------------------
@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
cdef long unsafe_get_I_MN(
    long[:] C1, long[:] C2, long[:] D1, long[:] D2, long I_M, long I_N):
    """
    I_M & I_N are 1-based indexes for self.coeffs.

    Return I_{M*N} as 1-based index
    """

    return D1[C1[I_M-1] + C1[I_N-1]] + \
           D2[C2[I_M-1] + C2[I_N-1]]

#----------------------------------------------------------------------
@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
cdef unsafe_tps_times_tps(
    long n_nz_inds, long norder, long[:] nz_inds_view,
    long[:] a_orders, long[:] b_orders,
    long[:] C1, long[:] C2, long[:] D1, long[:] D2,
    double[:] c_F, double[:] a_coeffs, double[:] b_coeffs, double[:] c_coeffs):
    """
    """

    cdef long i, j
    cdef long new_order
    cdef long a_i, b_i, I_MN1, I_MN0

    for i in range(n_nz_inds):
        a_i = nz_inds_view[i]
        for j in range(n_nz_inds):
            b_i = nz_inds_view[j]

            new_order = a_orders[a_i] + b_orders[b_i]
            if new_order > norder:
                continue

            I_MN1 = unsafe_get_I_MN(C1, C2, D1, D2, a_i+1, b_i+1) # 1-based index
            I_MN0 = I_MN1 - 1 # 0-based index

            c_coeffs[I_MN0] += (
                c_F[I_MN0] * a_coeffs[a_i] * b_coeffs[b_i] / c_F[a_i] / c_F[b_i])

#----------------------------------------------------------------------
@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
cdef unsafe_tps_times_tps_complex(
    long n_nz_inds, long norder, long[:] nz_inds_view,
    long[:] a_orders, long[:] b_orders,
    long[:] C1, long[:] C2, long[:] D1, long[:] D2,
    double complex[:] c_F, double complex[:] a_coeffs,
    double complex[:] b_coeffs, double complex[:] c_coeffs):
    """
    """

    cdef long i, j
    cdef long new_order
    cdef long a_i, b_i, I_MN1, I_MN0

    for i in range(n_nz_inds):
        a_i = nz_inds_view[i]
        for j in range(n_nz_inds):
            b_i = nz_inds_view[j]

            new_order = a_orders[a_i] + b_orders[b_i]
            if new_order > norder:
                continue

            I_MN1 = unsafe_get_I_MN(C1, C2, D1, D2, a_i+1, b_i+1) # 1-based index
            I_MN0 = I_MN1 - 1 # 0-based index

            #c_coeffs[I_MN0] += (
                #c_F[I_MN0] * a_coeffs[a_i] * b_coeffs[b_i] / c_F[a_i] / c_F[b_i])
            c_coeffs[I_MN0] = c_coeffs[I_MN0] + (
                c_F[I_MN0] * a_coeffs[a_i] * b_coeffs[b_i] / c_F[a_i] / c_F[b_i])

#----------------------------------------------------------------------
@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
cdef unsafe_calc_C_from_exponent_array(
    long[:] exponent_array, long norder, long[2] C1_C2):
    """
    """

    # Skip the check, even though unsafe:
    #    assert np.mod(len(exponent_array), 2) == 0

    cdef long nvar_half = len(exponent_array) // 2

    cdef long C1, C2, ithExp, _exp, common_fac

    C1 = exponent_array[0]
    C2 = exponent_array[nvar_half]

    common_fac = 1
    for ithExp in range(1, nvar_half):

        common_fac *= (norder + 1)

        _exp = exponent_array[ithExp]
        C1 += _exp * common_fac

        _exp = exponent_array[nvar_half + ithExp]
        C2 += _exp * common_fac

    C1_C2[0] = C1
    C1_C2[1] = C2

#----------------------------------------------------------------------
@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
cdef long unsafe_get_I_M_from_exponent_array(
    long[:] exponent_array, long norder, long[:] D1, long[:] D2):
    """
    Return I_{M} as 1-based index
    """

    cdef long C1, C2
    cdef long[2] C1_C2

    unsafe_calc_C_from_exponent_array(exponent_array, norder, C1_C2)

    C1 = C1_C2[0]
    C2 = C1_C2[1]

    return D1[C1] + D2[C2]


########################################################################
class TPS():
    """"""

    #----------------------------------------------------------------------
    def __init__(self, var_names, norder, print_order='Berz', safe_mult=True,
        div_tol_nonzero_coeff=None, complex_coeffs=False,
        repr_type='compact_table', repr_min_abs_polynom_coeff=0.0):
        """Constructor"""

        self.safe_mult = safe_mult

        if repr_type not in ('full_table', 'compact_table', 'no_table'):
            raise ValueError('Invalid "repr_type": {}'.format(repr_type))
        else:
            self.repr_type = repr_type

        if repr_min_abs_polynom_coeff < 0.0:
            raise ValueError('"repr_min_abs_polynom_coeff" must be a positive float value.')
        else:
            self.repr_min_abs_polynom_coeff = repr_min_abs_polynom_coeff

        self.div_tol_nonzero_coeff = div_tol_nonzero_coeff

        nvar = len(var_names)

        # Confirm that there is no duplicate variable name
        assert nvar == len(np.unique(var_names))

        if np.mod(nvar, 2) != 0:
            var_names = list(var_names) + ['']
            nvar += 1

        self.var_names = list(var_names)

        self.norder = norder
        self.nvar   = nvar

        assert np.mod(self.nvar, 2) == 0

        self.size = 1
        for k in range(1, norder + 1):
            self.size += factorial(nvar + k -1, exact=1) / factorial(
                nvar - 1, exact=1) / factorial(k, exact=1)
        self.size = int(self.size)

        self.binom_size = 1
        for k in range(1, norder + 1):
            self.binom_size += binom(nvar + k - 1, nvar - 1)

        assert self.size == self.binom_size
        # Note that "self.size" here is equal to (Omega + 1) in A. Chao's TPSA note

        self.coeffs = np.zeros(self.size)

        self.complex_coeffs = complex_coeffs
        if complex_coeffs:
            self.coeffs = np.zeros(self.size, dtype=complex)
        else:
            self.coeffs = np.zeros(self.size)

        self.gen_table(norder, nvar)

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""

        return self.__str__()

    #----------------------------------------------------------------------
    def __str__(self, orders=None):
        """"""

        # TODO
        'Graded lexicographic order: grlex: degree lexicographic order'

        if self.repr_type == 'full_table':
            out = ['variable name orders: {0}'.format(', '.join(self.var_names)),
                   '(I_M) : (Order) : (Exponents) : (Raw Coeff.) : (Coeff.) : (C1) : (C2)']
        elif self.repr_type == 'compact_table':
            out = ['variable name orders: {0}'.format(', '.join(self.var_names)),
                   '(I_M) : (Order) : (Exponents) : (Coeff.)']
        elif self.repr_type == 'no_table':
            out = ['{:d} Variables: {}'.format(self.nvar, ', '.join(self.var_names)),
                   'Order = {:d}; Number of Coefficients = {:d}'.format(
                    self.norder, len(self.coeffs))
            ]
            return '\n'.join(out)
        else:
            raise ValueError('Invalid "repr_type": {}'.format(self.repr_type))

        assert len(self.T['I_M']) == len(self.T['C1']) == len(self.T['C2']) \
               == len(self.T['exponents']) == len(self.coeffs)

        width_I_M = int(np.ceil(np.log10(self.size + 1)))

        if orders is None:
            all_disp_orders = np.unique(np.sum(self.T['exponents'], axis=1))
        else:
            all_disp_orders = orders

        for I_M, C1, C2, _ex, _c in zip(
            self.T['I_M'], self.T['C1'], self.T['C2'], self.T['exponents'],
            self.coeffs):

            _order = np.sum(_ex)

            if _order not in all_disp_orders:
                continue

            polynom_coeff = self.get_polynom_coeff(_ex)

            if self.repr_type == 'compact_table':

                if np.abs(polynom_coeff) < self.repr_min_abs_polynom_coeff:
                    continue

                out.append((
                    '{0:%dd} : {1:d} : ({2}) : {3:+12.6e}' % width_I_M).format(
                        I_M, _order, ', '.join(['{0:d}'.format(_i) for _i in _ex]),
                        polynom_coeff
                ))
            elif self.repr_type == 'full_table':
                out.append((
                    #'{0:%dd} : {1:d} : ({2}) : {3:+12.6g} : {4:+12.6g} : {5:d} : {6:d}' % width_I_M).format(
                    '{0:%dd} : {1:d} : ({2}) : {3:+12.6e} : {4:+12.6e} : {5:d} : {6:d}' % width_I_M).format(
                        I_M, _order, ', '.join(['{0:d}'.format(_i) for _i in _ex]),
                        _c, polynom_coeff, C1, C2
                ))

        return '\n'.join(out)

    #----------------------------------------------------------------------
    def print(self, orders=None):
        """"""

        print(self.__str__(orders=orders))

    #----------------------------------------------------------------------
    def set_to_one(self):
        """"""

        self.set_to_zero()

        # Index for the constant term is always 0.
        self.coeffs[0] = 1.0

    #----------------------------------------------------------------------
    def set_to_zero(self):
        """"""

        if self.complex_coeffs:
            self.coeffs = np.zeros(self.coeffs.shape, dtype=complex)
        else:
            self.coeffs = np.zeros(self.coeffs.shape)

    #----------------------------------------------------------------------
    def set_to_var(self, var_name):
        """"""

        var_ind0 = self.var_names.index(var_name) # 0-based index
        var_ind1 = var_ind0 + 1 # 1-based index

        if var_ind1 <= self.nvar // 2:
            C1 = (self.norder + 1)**var_ind0
            C2 = 0
        else:
            C1 = 0
            C2 = (self.norder + 1)**(var_ind0 - self.nvar // 2)

        I_MN = self.T['D1'][C1] + self.T['D2'][C2] - 1
        # ^ I_M is a 1-based index in Berz' paper. So, we must subtract 1 here.

        self.set_to_zero()

        self.coeffs[I_MN] = 1.0

    #----------------------------------------------------------------------
    def copy(self):
        """"""

        return pickle.loads(pickle.dumps(self, protocol=-1))

    #----------------------------------------------------------------------
    def dumps(self):
        """"""

        return pickle.dumps(self, protocol=-1)

    #----------------------------------------------------------------------
    @staticmethod
    def copy_from_dumps(dumps):
        """"""

        return pickle.loads(dumps)

    #----------------------------------------------------------------------
    def gen_table(self, norder, nvar):
        """"""

        n = norder
        v = nvar

        var_inds = range(v)

        order_list = []
        exp_list = []
        C1_list, C2_list = [], []
        for iOrder in range(n+1):
            #print(list(itertools.combinations_with_replacement(var_inds, iOrder)))

            for vec in itertools.combinations_with_replacement(var_inds, iOrder):
                exponents = [vec.count(_var) for _var in var_inds]

                C1 = np.sum([exponents[k       ] * ((n+1)**k) for k in range(v//2)])
                C2 = np.sum([exponents[k + v//2] * ((n+1)**k) for k in range(v//2)])

                #print(exponents, C1, C2)

                order_list.append(iOrder)
                exp_list.append(exponents)
                C1_list.append(C1)
                C2_list.append(C2)

        order_list = np.array(order_list)
        exp_list = np.array(exp_list)
        C1_list = np.array(C1_list)
        C2_list = np.array(C2_list)

        # Placeholders for sorted
        orders, exponents, C1s, C2s = [], [], [], []

        _zero_C2 = (C2_list == 0)

        for i in np.lexsort((C1_list[_zero_C2], order_list[_zero_C2])):
            orders.append(order_list[_zero_C2][i])
            exponents.append(exp_list[_zero_C2][i])
            C1s.append(C1_list[_zero_C2][i])
            C2s.append(C2_list[_zero_C2][i])

        _C2_rest = (C2_list != 0)

        for min_order in range(n+1):

            matched = (order_list[_C2_rest] == min_order)
            for _existing_C2 in np.unique(C2s):
                matched = np.logical_and(matched, C2_list[_C2_rest] != _existing_C2)

            if np.sum(matched) == 0:
                continue

            matched_sorted_C2s = np.unique(C2_list[_C2_rest][matched]) # sorting by np.unique() is critical here
            for sel_C2 in matched_sorted_C2s:
                _C2_matched = (C2_list == sel_C2)
                for i in np.lexsort((C1_list[_C2_matched], order_list[_C2_matched])):
                    orders.append(order_list[_C2_matched][i])
                    exponents.append(exp_list[_C2_matched][i])
                    C1s.append(C1_list[_C2_matched][i])
                    C2s.append(C2_list[_C2_matched][i])

        I_M = range(1, len(C1_list)+1)

        exponents = np.array(exponents)

        #zip(I_M, orders, exponents, C1s, C2s)

        D1s, D2s = [], []
        D_inds = range(np.max(C1s + C2s) + 1)
        for sel_C in D_inds:
            if sel_C in C1s:
                D1s.append(I_M[C1s.index(sel_C)])
            else:
                D1s.append(0)
            if sel_C in C2s:
                D2s.append(I_M[C2s.index(sel_C)] - 1)
            else:
                D2s.append(0)

        #zip(D_inds, D1s, D2s)

        ex_kw = dict(exact=False)

        Fs = factorial(exponents[:,0], **ex_kw)
        for i in range(1, exponents.shape[1]):
            Fs *= factorial(exponents[:,i], **ex_kw)

        if self.complex_coeffs:
            Fs = Fs.astype(complex)

        self.T = dict(orders=np.array(orders), exponents=exponents,
                      I_M=I_M, C1=np.array(C1s), C2=np.array(C2s),
                      D1=np.array(D1s), D2=np.array(D2s), F=Fs)

    #----------------------------------------------------------------------
    def get_polynom_coeffs(self):
        """"""

        assert len(self.T['orders']) == len(self.coeffs)

        tot_partial_deriv_orders = self.T['orders']

        n = tot_partial_deriv_orders.copy()
        partial_deriv_facs = np.ones(n.shape)
        for icol in range(self.T['exponents'].shape[1]):
            _exp = self.T['exponents'][:,icol]
            partial_deriv_facs *= comb(n, _exp, exact=False)
            n -= _exp

        return (
            self.coeffs / factorial(tot_partial_deriv_orders, exact=False)
            * partial_deriv_facs)

    #----------------------------------------------------------------------
    def get_polynom_coeff(self, exponent_array):
        """"""

        I_M1 = self.get_I_M_from_exponent_array(exponent_array) # 1-based index
        I_M0 = I_M1 - 1 # 0-based index

        tot_partial_deriv_order = self.T['orders'][I_M0]

        n = tot_partial_deriv_order
        partial_deriv_fac = 1
        for _exp in exponent_array:
            partial_deriv_fac *= comb(n, _exp, exact=False)
            n -= _exp

        return (
            self.coeffs[I_M0]
            / factorial(tot_partial_deriv_order, exact=False) * partial_deriv_fac
        )

    #----------------------------------------------------------------------
    def convert_polynom_coeff_to_raw_coeff(self, polynom_coeff, exponent_array):
        """"""

        I_M1 = self.get_I_M_from_exponent_array(exponent_array) # 1-based index
        I_M0 = I_M1 - 1 # 0-based index

        tot_partial_deriv_order = self.T['orders'][I_M0]

        n = tot_partial_deriv_order
        partial_deriv_fac = 1
        for _exp in exponent_array:
            partial_deriv_fac *= comb(n, _exp, exact=False)
            n -= _exp

        return polynom_coeff / partial_deriv_fac * factorial(tot_partial_deriv_order, exact=False)

    #----------------------------------------------------------------------
    def replace_polynom_coeff(self, new_polynom_coeff, exponent_array):
        """"""

        new_raw_coeff = self.convert_polynom_coeff_to_raw_coeff(
            new_polynom_coeff, exponent_array)

        I_M1 = self.get_I_M_from_exponent_array(exponent_array) # 1-based index
        I_M0 = I_M1 - 1 # 0-based index

        self.coeffs[I_M0] = new_raw_coeff

    #----------------------------------------------------------------------
    def __neg__(self):
        """"""

        return self.copy() * (-1.0)

    #----------------------------------------------------------------------
    def __add__(self, other):
        """"""

        copy = self.copy()

        if isinstance(other, (int, float, complex)):
            copy.coeffs[0] += other
        else:
            copy.coeffs += other.coeffs

        return copy

    #----------------------------------------------------------------------
    def __iadd__(self, other):
        """"""

        if isinstance(other, (int, float, complex)):
            self.coeffs[0] += other
        else:
            self.coeffs += other.coeffs

        return self

    #----------------------------------------------------------------------
    def __radd__(self, left):
        """"""
        return self.__add__(left)

    #----------------------------------------------------------------------
    def get_nonzero_coeff_inds(self):
        """"""

        if not self.complex_coeffs:
            return self.coeffs != 0.0
        else:
            return np.abs(self.coeffs) != 0.0

    #----------------------------------------------------------------------
    def get_I_MN(self, I_M, I_N):
        """
        I_M & I_N are 1-based indexes for self.coeffs.

        Return I_{M*N} as 1-based index
        """

        return self.T['D1'][self.T['C1'][I_M-1] + self.T['C1'][I_N-1]] + \
               self.T['D2'][self.T['C2'][I_M-1] + self.T['C2'][I_N-1]]

    #----------------------------------------------------------------------
    def get_C_from_exponent_array(self, exponent_array):
        """"""

        assert np.mod(len(exponent_array), 2) == 0

        nvar_half = len(exponent_array) // 2

        C1 = 0
        for ithExp, exp in enumerate(exponent_array[:nvar_half]):
            C1 += exp * (self.norder + 1)**ithExp
        C2 = 0
        for ithExp, exp in enumerate(exponent_array[nvar_half:]):
            C2 += exp * (self.norder + 1)**ithExp

        return C1, C2

    #----------------------------------------------------------------------
    def get_I_M_from_exponent_array(self, exponent_array):
        """"""

        C1, C2 = self.get_C_from_exponent_array(exponent_array)

        return self.T['D1'][C1] + self.T['D2'][C2]

    #----------------------------------------------------------------------
    def __mul__(self, other):
        """"""

        a = self
        b = other

        c = self.copy()

        cdef long i, j, n_nz_inds
        cdef long new_order
        cdef long norder = self.norder
        #cdef Py_ssize_t a_i, b_i, I_MN1, I_MN0
        cdef long a_i, b_i, I_MN1, I_MN0

        cdef long[:] nz_inds_view, a_orders, b_orders, C1, C2, D1, D2
        cdef double[:] c_F, a_coeffs, b_coeffs, c_coeffs
        cdef double complex[:] c_F_complex, a_coeffs_complex, b_coeffs_complex, c_coeffs_complex

        if isinstance(b, TPS):

            c.set_to_zero()

            nonzeros = np.logical_or(a.get_nonzero_coeff_inds(),
                                     b.get_nonzero_coeff_inds())
            nz_inds = np.where(nonzeros)[0]

            #for i, a_i in enumerate(nz_inds):
                #for b_i in nz_inds:
                    #print(a_i, b_i, c.T['orders'][a_i] + c.T['orders'][b_i])

            n_nz_inds = len(nz_inds)
            nz_inds_view = nz_inds
            a_orders = a.T['orders']
            b_orders = b.T['orders']
            C1 = c.T['C1']
            C2 = c.T['C2']
            D1 = c.T['D1']
            D2 = c.T['D2']
            if self.complex_coeffs:
                c_F_complex = c.T['F']
                a_coeffs_complex = a.coeffs
                b_coeffs_complex = b.coeffs
                c_coeffs_complex = c.coeffs

                if self.safe_mult:

                    for i in range(n_nz_inds):
                        a_i = nz_inds_view[i]
                        for j in range(n_nz_inds):
                            b_i = nz_inds_view[j]

                            new_order = a_orders[a_i] + b_orders[b_i]
                            if new_order > norder:
                                continue
                            I_MN1 = get_I_MN(C1, C2, D1, D2, a_i+1, b_i+1) # 1-based index
                            I_MN0 = I_MN1 - 1 # 0-based index
                            #c_coeffs_complex[I_MN0] += \
                                #(c_F_complex[I_MN0] *
                                #a_coeffs_complex[a_i] * b_coeffs_complex[b_i]
                                #/ c_F_complex[a_i] / c_F_complex[b_i])
                            c_coeffs_complex[I_MN0] = c_coeffs_complex[I_MN0] + \
                                (c_F_complex[I_MN0] *
                                a_coeffs_complex[a_i] * b_coeffs_complex[b_i]
                                / c_F_complex[a_i] / c_F_complex[b_i])
                else:
                    unsafe_tps_times_tps_complex(
                        n_nz_inds, norder, nz_inds_view, a_orders, b_orders,
                        C1, C2, D1, D2, c_F_complex,
                        a_coeffs_complex, b_coeffs_complex, c_coeffs_complex
                    )
            else:
                c_F = c.T['F']
                a_coeffs = a.coeffs
                b_coeffs = b.coeffs
                c_coeffs = c.coeffs

                if self.safe_mult:

                    for i in range(n_nz_inds):
                        a_i = nz_inds_view[i]
                        for j in range(n_nz_inds):
                            b_i = nz_inds_view[j]

                            new_order = a_orders[a_i] + b_orders[b_i]
                            if new_order > norder:
                                continue
                            I_MN1 = get_I_MN(C1, C2, D1, D2, a_i+1, b_i+1) # 1-based index
                            I_MN0 = I_MN1 - 1 # 0-based index
                            c_coeffs[I_MN0] += (c_F[I_MN0] *
                                                a_coeffs[a_i] * b_coeffs[b_i]
                                                / c_F[a_i] / c_F[b_i])
                else:
                    unsafe_tps_times_tps(
                        n_nz_inds, norder, nz_inds_view, a_orders, b_orders,
                        C1, C2, D1, D2, c_F, a_coeffs, b_coeffs, c_coeffs
                    )

        elif isinstance(b, (int, float, complex)):

            c.coeffs = a.coeffs * b

        else:
            raise TypeError('Invalid mulitiplication')

        return c

    #----------------------------------------------------------------------
    def __rmul__(self, left):
        """"""
        return self.__mul__(left)

    #----------------------------------------------------------------------
    def __imul__(self, other):
        """"""

        if isinstance(other, TPS):

            return self.__mul__(other)

        elif isinstance(other, (int, float, complex)):

            self.coeffs *= other

            return self

        else:
            raise TypeError('Invalid mulitiplication')

    #----------------------------------------------------------------------
    def __truediv__(self, other):
        """"""

        a = self # numerator
        b = other # denominator

        c = self.copy()

        cdef long i, j, n_nz_inds
        cdef long new_order
        cdef long norder = self.norder
        cdef long b_i, c_i, I_MN1, I_MN0

        cdef long[:] nz_inds_view, b_orders, c_orders, C1, C2, D1, D2
        cdef double[:] a_F, a_coeffs, b_coeffs, c_coeffs
        cdef double complex[:] a_F_complex, a_coeffs_complex, b_coeffs_complex, c_coeffs_complex
        cdef long n_coeffs

        cdef double denom_polynom_coeff
        cdef double complex denom_polynom_coeff_complex
        cdef double numer_before_div_polynom_coeff, numer_after_div_polynom_coeff
        cdef double complex numer_before_div_polynom_coeff_complex, numer_after_div_polynom_coeff_complex
        cdef double[:,:] M
        cdef double complex[:,:] M_complex
        cdef long[:] denom_exponent, denom_nonzero_coeff_inds, numer_nonzero_coeff_inds
        cdef long[:,:] numer_exponents_before_div, numer_exponents_after_div
        cdef long[:] a_exp, c_exp
        cdef long n_numer_nonzero_coeffs, n_monoms, I_M_c, i_exp
        cdef bint are_after_div_numer_exponents_all_positive = True

        if isinstance(b, TPS):

            assert a.size == b.size
            n_coeffs = a.size

            if b.div_tol_nonzero_coeff is None:
                if not b.complex_coeffs:
                    denom_nonzero_coeff_inds = np.where(b.coeffs != 0.0)[0]
                else:
                    denom_nonzero_coeff_inds = np.where(np.abs(b.coeffs) != 0.0)[0]
            else:
                denom_nonzero_coeff_inds = np.where(
                    np.abs(b.coeffs) > b.div_tol_nonzero_coeff)[0]

            # If the denominator has only single non-zero-coefficient term,
            # try to see if exact division is possible.
            if len(denom_nonzero_coeff_inds) == 1:
                denom_exponent = b.T['exponents'][denom_nonzero_coeff_inds[0]]

                #print('Denominator is a single non-zero monomial. Will check if numerator can be exactly divided.')

                # Check to see all numerator non-zero-coefficient terms stay
                # positive exponents after division
                if a.div_tol_nonzero_coeff is None:
                    if not a.complex_coeffs:
                        numer_nonzero_coeff_inds = np.where(a.coeffs != 0.0)[0]
                    else:
                        numer_nonzero_coeff_inds = np.where(np.abs(a.coeffs) != 0.0)[0]
                else:
                    numer_nonzero_coeff_inds = np.where(
                        np.abs(a.coeffs) > a.div_tol_nonzero_coeff)[0]

                #print('# of non-zero numerator terms = {0:d}'.format(
                    #len(numer_nonzero_coeff_inds)))

                n_numer_nonzero_coeffs = len(numer_nonzero_coeff_inds)
                n_monoms = a.T['exponents'].shape[1]

                numer_exponents_before_div = \
                    a.T['exponents'][numer_nonzero_coeff_inds]
                #numer_exponents_after_div = \
                    #numer_exponents_before_div - denom_exponent
                numer_exponents_after_div = \
                    np.zeros((n_numer_nonzero_coeffs, n_monoms)).astype(int)
                for i in range(n_numer_nonzero_coeffs):
                    for j in range(n_monoms):
                        numer_exponents_after_div[i,j] = \
                            numer_exponents_before_div[i,j] - denom_exponent[j]
                        if numer_exponents_after_div[i,j] < 0:
                            are_after_div_numer_exponents_all_positive = False
                            break

                #print('numer_exponents_before_div', numer_exponents_before_div)
                #print('numer_exponents_after_div', numer_exponents_after_div)

                #if np.all(numer_exponents_after_div.flatten() >= 0):
                if are_after_div_numer_exponents_all_positive:
                    # Exact division is feasible

                    #print('Exact division is feasible')

                    c.set_to_zero()

                    if c.complex_coeffs:
                        denom_polynom_coeff_complex = b.get_polynom_coeff(denom_exponent)

                        #print('denom_polynom_coeff_complex', denom_polynom_coeff_complex)

                        c_coeffs_complex = c.coeffs

                        #for c_exp, a_exp in zip(
                            #numer_exponents_after_div, numer_exponents_before_div):
                        c_exp = np.zeros(n_monoms).astype(int)
                        a_exp = np.zeros(n_monoms).astype(int)
                        for i_exp in range(n_numer_nonzero_coeffs):
                            for j in range(n_monoms):
                                c_exp[j] = numer_exponents_after_div[i_exp, j]
                                a_exp[j] = numer_exponents_before_div[i_exp, j]

                            numer_before_div_polynom_coeff_complex = a.get_polynom_coeff(a_exp)
                            numer_after_div_polynom_coeff_complex = \
                                numer_before_div_polynom_coeff_complex / denom_polynom_coeff_complex

                            I_M_c = c.get_I_M_from_exponent_array(c_exp) # 1-indexed
                            #print('I_M_c', I_M_c)

                            c_coeffs_complex[I_M_c - 1] = c.convert_polynom_coeff_to_raw_coeff(
                                numer_after_div_polynom_coeff_complex, c_exp)
                    else:
                        denom_polynom_coeff = b.get_polynom_coeff(denom_exponent)

                        #print('denom_polynom_coeff', denom_polynom_coeff)

                        c_coeffs = c.coeffs

                        #for c_exp, a_exp in zip(
                            #numer_exponents_after_div, numer_exponents_before_div):
                        c_exp = np.zeros(n_monoms).astype(int)
                        a_exp = np.zeros(n_monoms).astype(int)
                        for i_exp in range(n_numer_nonzero_coeffs):
                            for j in range(n_monoms):
                                c_exp[j] = numer_exponents_after_div[i_exp, j]
                                a_exp[j] = numer_exponents_before_div[i_exp, j]

                            numer_before_div_polynom_coeff = a.get_polynom_coeff(a_exp)
                            numer_after_div_polynom_coeff = \
                                numer_before_div_polynom_coeff / denom_polynom_coeff

                            I_M_c = c.get_I_M_from_exponent_array(c_exp) # 1-indexed
                            #print('I_M_c', I_M_c)

                            c_coeffs[I_M_c - 1] = c.convert_polynom_coeff_to_raw_coeff(
                                numer_after_div_polynom_coeff, c_exp)

                    return c

            # Make sure the denominator's constant term must be non-zero
            # for division to be valid
            if np.abs(b.coeffs[0]) == 0.0:
                print('The constant term in denominator cannot be zero, unless the denominator becomes one after division.')
                raise ValueError('Zero-constant polynomial in denominator not allowed for division')

            nonzeros = b.get_nonzero_coeff_inds()
            nz_inds = np.where(nonzeros)[0]

            n_nz_inds = len(nz_inds)
            nz_inds_view = nz_inds
            b_orders = b.T['orders']
            c_orders = c.T['orders']
            C1 = a.T['C1']
            C2 = a.T['C2']
            D1 = a.T['D1']
            D2 = a.T['D2']

            if a.complex_coeffs or b.complex_coeffs:
                a_F_complex = a.T['F']
                a_coeffs_complex = a.coeffs
                b_coeffs_complex = b.coeffs

                M_complex = np.zeros((n_coeffs, n_coeffs), dtype=complex) # square matrix

                for i in range(n_nz_inds):
                    b_i = nz_inds_view[i]
                    for c_i in range(n_coeffs):
                        new_order = b_orders[b_i] + c_orders[c_i]
                        if new_order > norder:
                            continue
                        I_MN1 = get_I_MN(C1, C2, D1, D2, b_i+1, c_i+1) # 1-based index
                        I_MN0 = I_MN1 - 1 # 0-based index
                        M_complex[I_MN0, c_i] = \
                            b_coeffs_complex[b_i] / a_F_complex[b_i] / a_F_complex[c_i]

                #c_coeffs = np.linalg.inv(M_complex).dot(a_coeffs / a_F)
                c.coeffs = np.linalg.inv(M_complex).dot(a.coeffs / a.T['F'])

            else:
                a_F = a.T['F']
                a_coeffs = a.coeffs
                b_coeffs = b.coeffs

                M = np.zeros((n_coeffs, n_coeffs)) # square matrix

                for i in range(n_nz_inds):
                    b_i = nz_inds_view[i]
                    for c_i in range(n_coeffs):
                        new_order = b_orders[b_i] + c_orders[c_i]
                        if new_order > norder:
                            continue
                        I_MN1 = get_I_MN(C1, C2, D1, D2, b_i+1, c_i+1) # 1-based index
                        I_MN0 = I_MN1 - 1 # 0-based index
                        M[I_MN0, c_i] = b_coeffs[b_i] / a_F[b_i] / a_F[c_i]

                #c_coeffs = np.linalg.inv(M).dot(a_coeffs / a_F)
                c.coeffs = np.linalg.inv(M).dot(a.coeffs / a.T['F'])

        elif isinstance(b, (int, float, complex)):
            c.coeffs = a.coeffs / b
        else:
            raise TypeError('Invalid division')

        return c

    #----------------------------------------------------------------------
    def __div__(self, other):
        """For Python 2 backward compatibility"""

        return self.__truediv__(other)

    #----------------------------------------------------------------------
    def __rtruediv__(self, left):
        """"""

        if isinstance(left, TPS):
            left_TPS = left
        elif isinstance(left, (int, float, complex)):
            left_TPS = self.copy()
            left_TPS.set_to_zero()
            left_TPS += left
        else:
            raise TypeError('wrong type for __rtruediv__')

        return left_TPS.__truediv__(self)

    #----------------------------------------------------------------------
    def __rdiv__(self, left):
        """For Python 2 backward compatibility"""

        return self.__rtruediv__(left)

    #----------------------------------------------------------------------
    def __itruediv__(self, other):
        """"""

        if isinstance(other, TPS):
            return self.__truediv__(other)

        elif isinstance(other, (int, float, complex)):

            self.coeffs /= other

            return self

        else:
            raise TypeError('Invalid division')

    #----------------------------------------------------------------------
    def __idiv__(self, other):
        """For Python 2 backward compatibility"""

        return self.__itruediv__(other)

    #----------------------------------------------------------------------
    def __sub__(self, other):
        """"""

        return self.__add__( -1.0 * other )

    #----------------------------------------------------------------------
    def __rsub__(self, left):
        """"""

        c = self.__neg__()
        c += left

        return c

    #----------------------------------------------------------------------
    def __isub__(self, other):
        """"""

        if isinstance(other, (int, float, complex)):
            self.coeffs[0] -= other
        else:
            self.coeffs -= other.coeffs

        return self

    #----------------------------------------------------------------------
    def __pow__(self, exponent):
        """"""

        assert isinstance(exponent, int)

        if exponent == 0:
            copy = self.copy()
            copy = copy * 0.0 + 1.0
        elif exponent > 0:
            copy = self.copy()
            if exponent != 1:
                for i in range(exponent-1):
                    copy = copy.__mul__(self)
        else: # negative exponent
            copy = 1.0 / self
            if exponent != -1:
                copy = copy.__pow__(-exponent)

        return copy

    #----------------------------------------------------------------------
    def diff(self, var_name):
        """
        Partial differentiation w.r.t. the specified variable.
        """

        varind = self.var_names.index(var_name)

        nonzero_inds = (self.T['exponents'][:, varind] != 0)
        nonzero_exponents_before = self.T['exponents'][nonzero_inds, :]
        tot_partial_deriv_orders_before = self.T['orders'][nonzero_inds]

        nonzero_exponents_after = nonzero_exponents_before.copy()
        nonzero_exponents_after[:, varind] -= 1
        tot_partial_deriv_orders_after = tot_partial_deriv_orders_before - 1

        n = tot_partial_deriv_orders_before.copy()
        partial_deriv_facs_before = np.ones(n.shape)
        for icol in range(nonzero_exponents_before.shape[1]):
            _exp = nonzero_exponents_before[:,icol]
            partial_deriv_facs_before *= comb(n, _exp, exact=False)
            n -= _exp

        n = tot_partial_deriv_orders_after.copy()
        partial_deriv_facs_after = np.ones(n.shape)
        for icol in range(nonzero_exponents_after.shape[1]):
            _exp = nonzero_exponents_after[:,icol]
            partial_deriv_facs_after *= comb(n, _exp, exact=False)
            n -= _exp

        nvar_half = self.T['exponents'].shape[1] // 2

        C1 = np.zeros(np.sum(nonzero_inds)).astype(int)
        C2 = np.zeros(np.sum(nonzero_inds)).astype(int)
        for i in range(nvar_half):
            C1 += nonzero_exponents_before[:, i] * ((self.norder + 1)**i)
            C2 += nonzero_exponents_before[:, i+nvar_half] * ((self.norder + 1)**i)

        I_M1 = self.T['D1'][C1] + self.T['D2'][C2]
        I_M0_before = I_M1 - 1

        C1 = np.zeros(np.sum(nonzero_inds)).astype(int)
        C2 = np.zeros(np.sum(nonzero_inds)).astype(int)
        for i in range(nvar_half):
            C1 += nonzero_exponents_after[:, i] * ((self.norder + 1)**i)
            C2 += nonzero_exponents_after[:, i+nvar_half] * ((self.norder + 1)**i)

        I_M1 = self.T['D1'][C1] + self.T['D2'][C2]
        I_M0_after = I_M1 - 1

        c = self.copy()

        c.set_to_zero()

        c.coeffs[I_M0_after] = (
            self.coeffs[I_M0_before] /
            factorial(tot_partial_deriv_orders_before) *
            partial_deriv_facs_before *
            factorial(tot_partial_deriv_orders_after) /
            partial_deriv_facs_after) * nonzero_exponents_before[:, varind]

        return c

    #----------------------------------------------------------------------
    def subs(self, var_names, var_values, reduce_dim=False):
        """"""

        assert len(var_names) == len(var_values)

        copy = self.copy()

        var_inds = [self.var_names.index(name) for name in var_names]

        nonvar_loginds = np.array([True if name not in var_names else False
                                   for name in self.var_names])
        # ^ Must be a numpy array, NOT a list

        nonzero_exp_loginds = np.any(self.T['exponents'][:,var_inds] != 0, axis=1)
        zero_exp_inds = np.where(~nonzero_exp_loginds)[0]

        for i, v in zip(var_inds, var_values):
            copy.coeffs[nonzero_exp_loginds] *= v**self.T['exponents'][nonzero_exp_loginds,i]

        nonzero_exponents = self.T['exponents'][nonzero_exp_loginds, :]
        zero_exponents = self.T['exponents'][~nonzero_exp_loginds, :]

        tot_partial_deriv_orders = self.T['orders'][nonzero_exp_loginds]

        n = tot_partial_deriv_orders.copy()
        partial_deriv_facs = np.ones(n.shape)
        for icol in range(nonzero_exponents.shape[1]):
            _exp = nonzero_exponents[:,icol]
            partial_deriv_facs *= comb(n, _exp, exact=False)
            n -= _exp

        zero_exp_tot_partial_deriv_orders = self.T['orders'][zero_exp_inds]

        n = zero_exp_tot_partial_deriv_orders.copy()
        zero_exp_partial_deriv_facs = np.ones(n.shape)
        for icol in range(zero_exponents.shape[1]):
            _exp = zero_exponents[:,icol]
            zero_exp_partial_deriv_facs *= comb(n, _exp, exact=False)
            n -= _exp

        nonmod_exponents = nonzero_exponents[:, nonvar_loginds]

        for i, _fac_fac, _part_fac in zip(
            zero_exp_inds, factorial(zero_exp_tot_partial_deriv_orders),
            zero_exp_partial_deriv_facs):

            _full_exp = self.T['exponents'][i]

            matched = np.all(nonmod_exponents == _full_exp[nonvar_loginds], axis=1)

            copy.coeffs[i] += np.sum(
                copy.coeffs[nonzero_exp_loginds][matched]
                / factorial(tot_partial_deriv_orders[matched])
                * partial_deriv_facs[matched]
                * _fac_fac / _part_fac
            )

        copy.coeffs[nonzero_exp_loginds] = 0.0

        cdef long i_full, i_red, n_orig_terms, norder
        cdef long[:,:] exponents_red_mv
        cdef double[:] coeffs, coeffs_red
        cdef long[:] D1_red, D2_red

        if reduce_dim:
            rem_var_names = np.array(self.var_names)[nonvar_loginds].tolist()
            if ('' in rem_var_names) and (np.mod(len(rem_var_names), 2) == 1):
                # Remove the dummpy variable added to the TPS to make its number
                # of variables even, as it is no longer needed once shrunk.
                rem_var_names.remove('')
                nonvar_loginds[self.var_names.index('')] = False

            norder = self.norder

            red_c = TPS(rem_var_names, norder, safe_mult=self.safe_mult)
            coeffs_red = red_c.coeffs

            D1_red = red_c.T['D1']
            D2_red = red_c.T['D2']

            n_orig_terms = copy.size
            coeffs = copy.coeffs
            exponents_red_mv = copy.T['exponents'][:, nonvar_loginds]


            for i_full in range(n_orig_terms):

                if coeffs[i_full] != 0.0:
                    i_red = unsafe_get_I_M_from_exponent_array(
                        exponents_red_mv[i_full,:], norder, D1_red, D2_red) # 1-based index
                    i_red -= 1 # convert to 0-based index

                    coeffs_red[i_red] += coeffs[i_full]

            copy = red_c

        return copy


#########################################################################
#class TPS():
    #""""""

    ##----------------------------------------------------------------------
    #def __init__(self, monoms, var_names=None, n_trunc=20, trunc_meth=0,
                 #min_term_norm_order=None, max_abs_values=None,
                 #max_abs_coeff=None, auto_norm=False):
        #"""Constructor

        #`monoms` is a dict with tuples of degress as its keys and their
        #corresponding coefficients as its values.

        #For example, if you want to create a polynomial of
            #1.5 * x**2 + 3 * x * y - 4.2 * y**2,
        #then you should set `monoms` as
            #{(2,0): 1.5, (1,1): 3, (0,2): -4.2}
        #when you set "var_names = ['x', 'y']".

        #"""

        #self.DEBUG_print = True
        ##self.DEBUG_print = False

        #self.n_trunc = n_trunc
        #self.trunc_meth = trunc_meth


        #n_vars_list = [len(tup) for tup in monoms.keys()]
        #assert np.unique(n_vars_list).size == 1
        #n_vars = n_vars_list[0]

        #if var_names is None:
            #self.var_names = ['x{0:3d}'.format(i) for i in range(1, n_vars+1)]
        #else:
            #assert len(var_names) == n_vars
            #self.var_names = var_names
        ##self.var_names = sorted(self.var_names)

        #if min_term_norm_order is None:
            #self.min_term_norm_order = None
            #self.max_abs_values = None
        #else:
            #self.min_term_norm_order = min_term_norm_order * (-1)
            #if max_abs_values is None:
                #max_abs_values = np.array([1e-2]*len(self.var_names))
            #try:
                #len(max_abs_values) # max_abs_values is vector?
                #self.max_abs_values = np.array(max_abs_values)
            #except: # max_abs_values is scalar
                #self.max_abs_values = np.array([max_abs_values]*len(self.var_names))

        #self.max_abs_coeff = max_abs_coeff

        #self.monoms = {deg_tup: coeff for deg_tup, coeff in monoms.iteritems()
                       #if sum(deg_tup) <= self.n_trunc}

        #self.sort_var_names()

        #self.norm_fac = 1.0
        #self.auto_norm = auto_norm

    ##----------------------------------------------------------------------
    #def sort_var_names(self):
        #""""""

        #copy = self.copy()

        #var_names_old = copy.var_names[:]
        #var_names_new = sorted(var_names_old)

        #n_var_new = len(var_names_new)

        #_map = [var_names_new.index(vname) for vname in var_names_old]
        #old_monoms = copy.monoms
        #new_monoms = {}
        #for old_deg_tup, coeff in old_monoms.iteritems():
            #if self._get_trunc_deg(old_deg_tup) > copy.n_trunc:
                #if self.DEBUG_print: print('* Add (left): Truncating', old_deg_tup)
                #continue
            #new_degs = np.zeros(n_var_new).astype(int)
            #new_degs[_map] = np.array(old_deg_tup)
            #new_monoms[tuple(new_degs)] = coeff

        #self.monoms = new_monoms

        #self.var_names = var_names_new

    ##----------------------------------------------------------------------
    #def normalize(self, force=False, extra=10.0):
        #"""
        #If maximum coefficient value is less than 1, this function will not
        #normalize by default. However, if you do want to force normalization
        #even in this case, set "force=True".
        #"""

        #monoms_values = self.monoms.values()
        #new_norm_fac = np.max(np.abs(monoms_values))
        #if new_norm_fac < 1.0:
            #if not force:
                #return
        #elif new_norm_fac == 1.0:
            #return

        #new_norm_fac *= extra

        #deg_tuples = self.monoms.keys()
        #norm_coeffs = np.array(monoms_values) / new_norm_fac

        #self.monoms = {k: v for k, v in zip(deg_tuples, norm_coeffs)}
        #self.norm_fac *= new_norm_fac

    ##----------------------------------------------------------------------
    #def unnormalize(self, in_place=False):
        #""""""

        #if in_place:
            #if self.norm_fac != 1.0:
                #deg_tuples = self.monoms.keys()
                #unnorm_coeffs = np.array(self.monoms.values()) * self.norm_fac

                #self.monoms = {k: v for k, v in zip(deg_tuples, unnorm_coeffs)}
                #self.norm_fac = 1.0
        #else:
            #copy = self.copy()

            #if copy.norm_fac != 1.0:
                #deg_tuples = copy.monoms.keys()
                #unnorm_coeffs = np.array(copy.monoms.values()) * copy.norm_fac

                #copy.monoms = {k: v for k, v in zip(deg_tuples, unnorm_coeffs)}
                #copy.norm_fac = 1.0

            #return copy

    ##----------------------------------------------------------------------
    #def copy(self):
        #""""""

        #return pickle.loads(pickle.dumps(self, protocol=-1))

    ##----------------------------------------------------------------------
    #def differentiate_mult(self):
        #""""""

        #return [self.differentiate(var_name) for var_name in self.var_names]

    ##----------------------------------------------------------------------
    #def differentiate(self, var_name):
        #""""""

        #copy = self.copy()
        #copy.monoms = {}

        #varind = copy.var_names.index(var_name)

        #for deg_tup, v in self.monoms.iteritems():
            #var_deg = deg_tup[varind]
            #if var_deg == 0: continue
            #else:
                #new_deg_tup = np.array(deg_tup)
                #new_deg_tup[varind] -= 1
                #copy.monoms[tuple(new_deg_tup)] = v * var_deg

        #if copy.monoms == {}:
            #copy.monoms[(0,)*len(copy.var_names)] = 0.0


        #return copy

    ##----------------------------------------------------------------------
    #def remove_large_coeff_terms(self, max_abs_coeff):
        #""""""

        #copy = self.copy()
        #copy.monoms = {}

        #abs_coeffs = np.abs(self.monoms.values())
        #for k in np.array(self.monoms.keys())[abs_coeffs < max_abs_coeff]:
            #copy.monoms[tuple(k)] = self.monoms[tuple(k)]

        #return copy

    ##----------------------------------------------------------------------
    #def __repr__(self):
        #""""""

        #return self.__str__()

    ##----------------------------------------------------------------------
    #def __str__(self):
        #""""""

        #out = [self.var_names.__str__(),
               #'* Normalization Factor: {0:.6g} *'.format(self.norm_fac)]

        #sorted_deg_tups = sorted(self.monoms.keys())
        #for k in sorted_deg_tups:
            #try:
                #out.append('{0} : {1: .9g}'.format(k.__str__(),
                                                   #self.monoms[k] * self.norm_fac))
            #except ValueError: # If sympy symbols are being used
                #out.append('{0} : {1}'.format(
                    #k.__str__(), (self.monoms[k] * self.norm_fac).__repr__()))

        #return '\n'.join(out)

    ##----------------------------------------------------------------------
    #def sympy_simplify_monoms(self):
        #""""""

        #for k, v in self.monoms.items():
            #self.monoms[k] = v.simplify()

    ##----------------------------------------------------------------------
    #def _get_trunc_deg(self, deg_tup):
        #""""""

        #if self.trunc_meth == 0:
            #try:
                #return np.sum(deg_tup, axis=1)
            #except:
                #return np.sum(deg_tup)
        #elif self.trunc_meth == 1:
            #try:
                #return np.max(deg_tup, axis=1)
            #except:
                #return np.max(deg_tup)
        #else:
            #raise ValueError()

    ##----------------------------------------------------------------------
    #def __add__(self, other):
        #""""""

        #copy = self.copy()

        #if isinstance(other, TPS):
            #copy.n_trunc = min([copy.n_trunc, other.n_trunc])
            ##if self.DEBUG_print: print('Add: n_trunc =', copy.n_trunc)

            #n_var_old = len(copy.var_names)
            #var_names_old = copy.var_names[:]

            #copy.var_names = np.unique(copy.var_names + other.var_names).tolist()
            #n_var_new = len(copy.var_names)
            #var_names_new = copy.var_names[:]

            #if n_var_new != n_var_old:
                #_map = [var_names_new.index(vname) for vname in var_names_old]
                #old_monoms = copy.monoms
                #new_monoms = {}
                #for old_deg_tup, coeff in old_monoms.iteritems():
                    #if self._get_trunc_deg(old_deg_tup) > copy.n_trunc:
                        #if self.DEBUG_print: print('* Add (left): Truncating', old_deg_tup)
                        #continue
                    #new_degs = np.zeros(n_var_new).astype(int)
                    #new_degs[_map] = np.array(old_deg_tup)
                    #new_monoms[tuple(new_degs)] = coeff

                #copy.monoms = new_monoms

            #_map = [var_names_new.index(vname) for vname in other.var_names]
            #norm_conv_fac = other.norm_fac / copy.norm_fac
            #for old_deg_tup, coeff in other.monoms.iteritems():
                #if self._get_trunc_deg(old_deg_tup) > copy.n_trunc:
                    #if self.DEBUG_print: print('* Add (right): Truncating', old_deg_tup)
                    #continue
                #new_degs = np.zeros(n_var_new).astype(int)
                #new_degs[_map] = np.array(old_deg_tup)
                #try: copy.monoms[tuple(new_degs)] += coeff * norm_conv_fac
                #except KeyError: copy.monoms[tuple(new_degs)] = coeff * norm_conv_fac

        #else:
            #const_key = tuple([0]*len(copy.var_names))
            #try: copy.monoms[const_key] += other / copy.norm_fac
            #except KeyError: copy.monoms[const_key] = other / copy.norm_fac

        #zero_inds = (np.array(copy.monoms.values()) == 0.0)
        #for key in np.array(copy.monoms.keys())[zero_inds]:
            #if len(copy.monoms) >= 2:
                #del copy.monoms[tuple(key)]

        #if self.auto_norm:
            #copy.normalize(force=False)

        #return copy

    ##----------------------------------------------------------------------
    #def __radd__(self, left):
        #""""""
        #return self.__add__(left)

    ##----------------------------------------------------------------------
    #def __mul__(self, other):
        #""""""

        #copy = self.copy()

        #if isinstance(other, TPS):
            #copy.n_trunc = min([copy.n_trunc, other.n_trunc])
            ##if self.DEBUG_print: print ('Mul: n_trunc =', copy.n_trunc)

            #n_var_old = len(copy.var_names)
            #var_names_old = copy.var_names[:]

            #copy.var_names = np.unique(copy.var_names + other.var_names).tolist()
            #n_var_new = len(copy.var_names)
            #var_names_new = copy.var_names[:]

            #if n_var_new != n_var_old:
                #_map = [var_names_new.index(vname) for vname in var_names_old]
                #old_monoms = copy.monoms
                #new_monoms = {}
                #for old_deg_tup, coeff in old_monoms.iteritems():
                    #if self._get_trunc_deg(old_deg_tup) > copy.n_trunc:
                        #if self.DEBUG_print: print('Mul (left): Truncating', old_deg_tup)
                        #continue
                    #new_degs = np.zeros(n_var_new).astype(int)
                    #new_degs[_map] = np.array(old_deg_tup)
                    #new_monoms[tuple(new_degs)] = coeff

                #copy.monoms = new_monoms

            #other_copy = other.copy()

            #_map = [var_names_new.index(vname) for vname in other_copy.var_names]
            #new_monoms = {}
            #for old_deg_tup, coeff in other_copy.monoms.iteritems():
                #if self._get_trunc_deg(old_deg_tup) > copy.n_trunc:
                    #if self.DEBUG_print: print('Mul (right): Truncating', old_deg_tup)
                    #continue
                #new_degs = np.zeros(n_var_new).astype(int)
                #new_degs[_map] = np.array(old_deg_tup)
                #new_monoms[tuple(new_degs)] = coeff

            #other_copy.monoms = new_monoms

            #if False:
                #v3 = TPS({(0,)*len(var_names_new): 0.0},
                         #var_names=var_names_new, n_trunc=copy.n_trunc)
                #other_copy_monom_degs = np.array(other_copy.monoms.keys())
                #v2_coeffs = np.array(other_copy.monoms.values())
                #v3_degs   = None
                #v3_coeffs = None
                #for v1_degs, v1_coeff in copy.monoms.iteritems():
                    #_degs = other_copy_monom_degs + np.array(v1_degs)
                    #above_trunc = self._get_trunc_deg(_degs) > v3.n_trunc
                    #if np.any(above_trunc):
                        #if self.DEBUG_print: print('Mul (combined): Truncating', _degs[above_trunc,:])
                        #_degs = _degs[~above_trunc,:]
                        #if _degs.size == 0: continue
                    #if v3_degs is not None:
                        #v3_degs = np.vstack((v3_degs, _degs))
                        #v3_coeffs = np.append(v3_coeffs, v2_coeffs * v1_coeff)
                    #else:
                        #v3_degs = _degs
                        #v3_coeffs = v2_coeffs * v1_coeff
                #sub_codings = np.array(
                    #[(v3.n_trunc+1)**v for v in range(len(v3.var_names))]).reshape((1,-1))
                #sub_codings = np.ones((v3_degs.shape[0], 1)).astype(int).dot(sub_codings)
                #v3_deg_codings = np.sum(v3_degs * sub_codings, axis=1)
                #u_rows, u_inds = np.unique(v3_deg_codings, return_index=True)
            #else:
                #v3 = TPS({(0,)*len(var_names_new): 0.0},
                         #var_names=var_names_new, n_trunc=copy.n_trunc)
                #v1_degs = np.array(copy.monoms.keys())
                #v2_degs = np.array(other_copy.monoms.keys())
                #nV1keys = len(v1_degs)
                #nV2keys = len(v2_degs)
                #v3_degs = np.zeros((nV1keys*nV2keys, n_var_new)).astype(int)
                #for i, (v1_deg_tup, v1_coeff) in enumerate(copy.monoms.iteritems()):
                    #v3_degs[(i*nV2keys):((i+1)*nV2keys), :] = \
                        #v2_degs + np.array(v1_deg_tup)
                #below_trunc = self._get_trunc_deg(v3_degs) <= v3.n_trunc
                #if np.any(~below_trunc):
                    #print('** Truncating {0:d} terms during multiplication **'.format(
                        #np.sum(~below_trunc)))
                #v3_degs = v3_degs[below_trunc]
                #nontrunc_v1v2inds = np.where(below_trunc)[0]
                #nontrunc_v1inds = np.divide(nontrunc_v1v2inds, nV2keys)
                #nontrunc_v2inds = np.fmod  (nontrunc_v1v2inds, nV2keys)
                #v1_coeffs = np.array(copy.monoms.values())[nontrunc_v1inds]
                #v2_coeffs = np.array(other_copy.monoms.values())[nontrunc_v2inds]
                #v3_coeffs = v1_coeffs * v2_coeffs
                #sub_codings = np.array(
                    #[(v3.n_trunc+1)**v for v in range(len(v3.var_names))]).reshape((1,-1))
                #sub_codings = np.ones((v3_degs.shape[0], 1)).astype(int).dot(sub_codings)
                #v3_deg_codings = np.sum(v3_degs * sub_codings, axis=1)
                #u_rows, u_inds = np.unique(v3_deg_codings, return_index=True)




            #v3.monoms = {} # Remove artificially added term for (0,)*len(v3.var_names)
            #t0 = time.time()
            #if True:
                #if True:
                    #for u_code, u_rowind in zip(u_rows, u_inds):
                        #match = np.where(v3_deg_codings == u_code)[0]
                        ##print(len(match))

                        #key = tuple(v3_degs[u_rowind, :])
                        #_summed_v3_coeff = np.sum(v3_coeffs[match])
                        #if _summed_v3_coeff != 0.0:
                            #v3.monoms[key] = _summed_v3_coeff
                #else:
                    #for u_code, u_rowind in zip(u_rows, u_inds):
                        #matched_loginds = (v3_deg_codings == u_code)
                        #match = np.where(matched_loginds)[0]

                        #key = tuple(v3_degs[u_rowind, :])
                        #_summed_v3_coeff = np.sum(v3_coeffs[match])
                        #if _summed_v3_coeff != 0.0:
                            #v3.monoms[key] = _summed_v3_coeff

                        #v3_deg_codings = v3_deg_codings[~matched_loginds]
                        #v3_coeffs = v3_coeffs[~matched_loginds]
            #else:
                #if False:
                    #summed_v3_coeffs = np.fromiter([
                        #np.sum(v3_coeffs[np.where(v3_deg_codings == u_code)[0]])
                        #for u_code in u_rows], np.float)
                #else:
                    #def test(u_code):
                        #return np.sum(v3_coeffs[np.where(v3_deg_codings == u_code)[0]])
                    #vf = np.vectorize(test, otypes=[np.float])
                    #summed_v3_coeffs = vf(u_rows)
                #v3_keys = np.array([tuple(v3_degs[u_rowind, :]) for u_rowind in u_inds])
                #nonzero_coeffs = summed_v3_coeffs != 0.0
                #for coeff, key in zip(summed_v3_coeffs[nonzero_coeffs],
                                      #v3_keys[nonzero_coeffs]):
                    #v3.monoms[tuple(key)] = coeff
            ##print('Elapsed [s]: {0:.3f}'.format(time.time()-t0))

            #if v3.monoms == {}: # Put back zero TPS if v3.monoms is empty
                #v3.monoms[(0,)*len(v3.var_names)] = 0.0

            #if False and (len(v3.monoms) > 1):

                #n_before = len(v3.monoms)

                #all_true = np.array([True]*n_before)

                #v3_keys = v3.monoms.keys()
                #v3_vals = np.array(v3.monoms.values())

                #if self.max_abs_coeff is not None:
                    #abs_coeffs = np.abs(v3_vals) * self.norm_fac * other.norm_fac
                    #coeff_ok = abs_coeffs < self.max_abs_coeff
                #else:
                    #coeff_ok = all_true

                #if self.min_term_norm_order is not None:
                    #test_values = []
                    #fac = self.norm_fac * other.norm_fac
                    #for deg_tup in v3_keys:
                        #monom_val = np.product(self.max_abs_values ** np.array(deg_tup))
                        #test_values.append(v3.monoms[deg_tup] * fac * monom_val)
                    #abs_test_values = np.abs(test_values)
                    #max_test_val = np.max(abs_test_values)

                    #if max_test_val != 0.0:
                        #test_values = abs_test_values / max_test_val
                        #non_negligible = np.log10(test_values) > self.min_term_norm_order
                    #else:
                        #non_negligible = all_true
                #else:
                    #non_negligible = all_true

                #keep = np.logical_and(coeff_ok, non_negligible)
                #if not np.all(keep):
                    #v3.monoms = {tuple(k): v for k, v in zip(
                        #np.array(v3_keys,dtype=object)[keep], v3_vals[keep])}

                    #n_after = len(v3.monoms)
                    #print('* Truncated {0:d} out of {1:d} terms'.format(
                        #n_before - n_after, n_before))


            #copy = v3
            #copy.norm_fac = self.norm_fac * other.norm_fac

            #if self.auto_norm:
                #copy.normalize(force=False)

        #else:
            #if other != 0.0:
                #copy.norm_fac *= other
            #else:
                #for k in copy.monoms.keys():
                    #copy.monoms[k] *= 0.0
                #copy.norm_fac = 1.0


        #return copy

    ##----------------------------------------------------------------------
    #def __rmul__(self, left):
        #""""""
        #return self.__mul__(left)

    ##----------------------------------------------------------------------
    #def __sub__(self, other):
        #""""""

        #return self.__add__( -1.0 * other )

    ##----------------------------------------------------------------------
    #def __pow__(self, exponent):
        #""""""

        #assert isinstance(exponent, int)
        #assert exponent >= 0

        #copy = self.copy()
        #if exponent == 0:
            #copy = copy * 0.0 + 1.0
        #elif exponent == 1:
            #pass
        #else:
            #for i in range(exponent-1):
                #copy = copy.__mul__(self)

        #return copy

    ##----------------------------------------------------------------------
    #def subs(self, var_name, var_value):
        #""""""

        #copy = self.copy()

        #var_ind = copy.var_names.index(var_name)

        #deg_array = np.array(copy.monoms.keys())
        #coeff_vec = np.array(copy.monoms.values())

        #substitutable_row_inds = deg_array[:, var_ind] >= 1
        #coeff_vec[substitutable_row_inds] *= (
            #var_value ** deg_array[substitutable_row_inds, var_ind])

        #remaining = range(deg_array.shape[1])
        #remaining.remove(var_ind)
        #new_deg_array = np.array(
            #[tuple(deg_array[i,remaining]) for i in range(deg_array.shape[0])])

        #new_n_var = len(copy.var_names) - 1

        #sub_codings = np.array(
            #[(copy.n_trunc+1)**v for v in range(new_n_var)]).reshape((1,-1))
        #sub_codings = np.ones((new_deg_array.shape[0], 1)).astype(int).dot(sub_codings)
        #deg_codings = np.sum(new_deg_array * sub_codings, axis=1)
        #u_rows, u_inds = np.unique(deg_codings, return_index=True)

        #new_monoms = {}
        #for u_code, u_rowind in zip(u_rows, u_inds):
            #match = np.where(deg_codings == u_code)[0]

            #key = tuple(new_deg_array[u_rowind, :])
            #_summed_coeff = np.sum(coeff_vec[match])
            #if _summed_coeff != 0.0:
                #new_monoms[key] = _summed_coeff

        #if new_monoms == {}: # add zero TPS if new_monoms is empty
            #new_monoms[(0,)*new_n_var] = 0.0

        #copy.monoms = new_monoms

        #copy.var_names.pop(var_ind)

        #return copy

#----------------------------------------------------------------------
def exp(input_tps):
    """"""

    a = input_tps

    cdef bint complex_coeffs = a.complex_coeffs

    # constant term
    cdef double a0 = 0.0
    cdef double complex a0_complex = 0.0j
    if not complex_coeffs:
        a0 = a.coeffs[0]
    else:
        a0_complex = a.coeffs[0]

    cdef long k, Omega
    cdef double prev_f, new_f
    cdef double complex prev_f_complex, new_f_complex

    # First term := 1 / (0!) * (0, a1, a2, ..., a_Omega)**0 = 1
    c = a.copy()
    c.set_to_one()

    # "const_zeroed_a" := (0, a1, a2, ..., a_Omega)
    const_zeroed_a = a.copy()
    const_zeroed_a.coeffs[0] = 0.0

    # Add 2nd term.
    # Second term := 1 / (1!) * (0, a1, a2, ..., a_Omega)**1 = "const_zeroed_a"
    c += const_zeroed_a

    if not complex_coeffs:
        prev_f = 1.0 # = 1!
    else:
        prev_f_complex = 1.0 + 0j # = 1!

    prev_term = const_zeroed_a.copy()

    Omega = a.size - 1
    # ^ Note that "a.size" here is equal to (Omega + 1) in A. Chao's TPSA note

    if not complex_coeffs:
        for k in range(2, Omega + 1):
            # Add k-th term.
            # k-th term := 1 / (k!) * (0, a1, a2, ..., a_Omega)**k
            #           := (0, a1, a2, ..., a_Omega)**(k-1) * (0, a1, a2, ..., a_Omega) / (k!)
            #           := "prev_term" * "const_zeroed_a" / (k!)
            new_term = prev_term * const_zeroed_a

            #c += new_term / float(factorial(k, exact=False))
            new_f = prev_f * k
            c += new_term / new_f

            prev_term = new_term
            prev_f = new_f
    else:
        for k in range(2, Omega + 1):
            # Add k-th term.
            # k-th term := 1 / (k!) * (0, a1, a2, ..., a_Omega)**k
            #           := (0, a1, a2, ..., a_Omega)**(k-1) * (0, a1, a2, ..., a_Omega) / (k!)
            #           := "prev_term" * "const_zeroed_a" / (k!)
            new_term = prev_term * const_zeroed_a

            #c += new_term / float(factorial(k, exact=False))
            new_f_complex = prev_f_complex * k
            c += new_term / new_f_complex

            prev_term = new_term
            prev_f_complex = new_f_complex

    c *= np.exp(a0)

    return c

#----------------------------------------------------------------------
def log(input_tps):
    """
    Natural log
    """

    a = input_tps

    cdef bint complex_coeffs = a.complex_coeffs

    cdef long k, Omega
    cdef double prev_f, new_f
    cdef double complex prev_f_complex, new_f_complex

    # constant term
    cdef double a0 = 0.0
    cdef double complex a0_complex = 0.0j
    if not complex_coeffs:
        a0 = a.coeffs[0]

        assert a0 > 0.0

        # First term := (ln(a0), 0, 0, ..., 0)
        c = a.copy()
        c.set_to_one()
        c *= np.log(a0)

        # "const_zeroed_a" := (0, a1/a0, a2/a0, ..., a_Omega/a0)
        const_zeroed_a = a.copy()
        const_zeroed_a.coeffs[0] = 0.0
        const_zeroed_a /= a0
    else:
        a0_complex = a.coeffs[0]

        assert abs(a0_complex) != 0.0

        # First term := (ln(a0), 0, 0, ..., 0)
        c = a.copy()
        c.set_to_one()
        c *= np.log(a0_complex)

        # "const_zeroed_a" := (0, a1/a0, a2/a0, ..., a_Omega/a0)
        const_zeroed_a = a.copy()
        const_zeroed_a.coeffs[0] = 0.0
        const_zeroed_a /= a0_complex

    # Add 2nd term.
    # Second term := (-1)**(1+1) / 1 * (0, a1/a0, a2/a0, ..., a_Omega/a0)**1 = "const_zeroed_a"
    c += const_zeroed_a

    if not complex_coeffs:
        prev_f = (-1)**(1+1) # = (-1)**(k+1)
    else:
        prev_f_complex = (-1)**(1+1) + 0j # = (-1)**(k+1)

    prev_term = const_zeroed_a.copy()

    Omega = a.size - 1
    # ^ Note that "a.size" here is equal to (Omega + 1) in A. Chao's TPSA note

    if not complex_coeffs:
        for k in range(2, Omega + 1):
            # Add k-th term.
            # k-th term := (-1)**(k+1) / k * (0, a1/a0, a2/a0, ..., a_Omega/a0)**k
            #           := (0, a1/a0, a2/a0, ..., a_Omega/a0)**(k-1) * (0, a1/a0, a2/a0, ..., a_Omega/a0) * (-1)**(k+1) / k
            #           := "prev_term" * "const_zeroed_a" * (-1)**(k+1) / k
            new_term = prev_term * const_zeroed_a

            #fac = ((-1)**(k+1)) / k
            #c += new_term * fac
            new_f = prev_f * (-1)
            c += new_term * (new_f / k)

            prev_term = new_term
            prev_f = new_f
    else:
        for k in range(2, Omega + 1):
            # Add k-th term.
            # k-th term := (-1)**(k+1) / k * (0, a1/a0, a2/a0, ..., a_Omega/a0)**k
            #           := (0, a1/a0, a2/a0, ..., a_Omega/a0)**(k-1) * (0, a1/a0, a2/a0, ..., a_Omega/a0) * (-1)**(k+1) / k
            #           := "prev_term" * "const_zeroed_a" * (-1)**(k+1) / k
            new_term = prev_term * const_zeroed_a

            #fac = ((-1)**(k+1)) / k
            #c += new_term * fac
            new_f_complex = prev_f_complex * (-1)
            c += new_term * (new_f_complex / k)

            prev_term = new_term
            prev_f_complex = new_f_complex

    return c

#----------------------------------------------------------------------
def sqrt(input_tps):
    """
    """

    a = input_tps

    # First term := (1, 0, 0, ..., 0)
    c = a.copy()
    c.set_to_one()

    cdef bint complex_coeffs = a.complex_coeffs

    cdef long k, Omega
    cdef double prev_f, new_f
    cdef double complex prev_f_complex, new_f_complex

    # constant term
    cdef double a0 = 0.0
    cdef double complex a0_complex = 0.0j
    if not complex_coeffs:
        a0 = a.coeffs[0]

        assert a0 > 0.0

        # "const_zeroed_a" := (0, a1/a0, a2/a0, ..., a_Omega/a0)
        const_zeroed_a = a.copy()
        const_zeroed_a.coeffs[0] = 0.0
        const_zeroed_a /= a0
    else:
        a0_complex = a.coeffs[0]

        assert abs(a0_complex) != 0.0

        # "const_zeroed_a" := (0, a1/a0, a2/a0, ..., a_Omega/a0)
        const_zeroed_a = a.copy()
        const_zeroed_a.coeffs[0] = 0.0
        const_zeroed_a /= a0_complex

    # Add 2nd term.
    # Second term := 1 / 2 * (0, a1/a0, a2/a0, ..., a_Omega/a0) = "const_zeroed_a" / 2
    c += const_zeroed_a / 2.0

    if not complex_coeffs:
        prev_f = -1.0 / 2
    else:
        prev_f_complex = (-1.0 + 0j) / 2

    prev_term = const_zeroed_a.copy()

    Omega = a.size - 1
    # ^ Note that "a.size" here is equal to (Omega + 1) in A. Chao's TPSA note

    if not complex_coeffs:
        for k in range(2, Omega + 1):
            # Subtract k-th term.
            # k-th term := (-1)**k (2*k-3)!! / (2*k)!! * (0, a1/a0, a2/a0, ..., a_Omega/a0)**k
            #           := (0, a1/a0, a2/a0, ..., a_Omega/a0)**(k-1) * (0, a1/a0, a2/a0, ..., a_Omega/a0) * (-1)**k (2*k-3)!! / (2*k)!!
            #           := "prev_term" * "const_zeroed_a" * (-1)**k (2*k-3)!! / (2*k)!!
            new_term = prev_term * const_zeroed_a

            #fac = ((-1)**k) * float(
                #factorial2(2*k-3, exact=False) / factorial2(2*k, exact=False))
            #c -= new_term * fac
            new_f = prev_f * (-1) * (2*k - 3) / (2*k)
            c -= new_term * new_f

            prev_term = new_term
            prev_f = new_f
    else:
        for k in range(2, Omega + 1):
            # Subtract k-th term.
            # k-th term := (-1)**k (2*k-3)!! / (2*k)!! * (0, a1/a0, a2/a0, ..., a_Omega/a0)**k
            #           := (0, a1/a0, a2/a0, ..., a_Omega/a0)**(k-1) * (0, a1/a0, a2/a0, ..., a_Omega/a0) * (-1)**k (2*k-3)!! / (2*k)!!
            #           := "prev_term" * "const_zeroed_a" * (-1)**k (2*k-3)!! / (2*k)!!
            new_term = prev_term * const_zeroed_a

            #fac = ((-1)**k) * float(
                #factorial2(2*k-3, exact=False) / factorial2(2*k, exact=False))
            #c -= new_term * fac
            new_f_complex = prev_f_complex * (-1) * (2*k - 3) / (2*k)
            c -= new_term * new_f_complex

            prev_term = new_term
            prev_f_complex = new_f_complex

    if not complex_coeffs:
        c *= np.sqrt(a0)
    else:
        c *= np.sqrt(a0_complex)

    return c

#----------------------------------------------------------------------
def sin(input_tps):
    """
    """

    return _sin_cos(input_tps, True)

#----------------------------------------------------------------------
def cos(input_tps):
    """
    """

    return _sin_cos(input_tps, False)

#----------------------------------------------------------------------
def tan(input_tps):
    """
    """

    return sin(input_tps) / cos(input_tps)

#----------------------------------------------------------------------
def _sin_cos(input_tps, is_sine):
    """
    """

    a = input_tps

    # constant term
    cdef double a0 = 0.0
    cdef double complex a0_complex = 0.0j
    if not a.complex_coeffs:
        a0 = a.coeffs[0]
    else:
        a0_complex = a.coeffs[0]

    cdef double prev_f1, prev_f2, new_f1, new_f2
    cdef double complex prev_f1_complex, prev_f2_complex, new_f1_complex, new_f2_complex
    cdef long k, Omega

    # "const_zeroed_a" := (0, a1, a2, ..., a_Omega)
    const_zeroed_a = a.copy()
    const_zeroed_a.coeffs[0] = 0.0

    # First term of 1st Summation
    #     := (-1)**0 / (2*0)! * (0, a1, a2, ..., a_Omega)**(2*0)
    #      = (1, 0, 0, ..., 0)
    c1 = a.copy()
    c1.set_to_one()

    # First term of 2nd Summation
    #     := (-1)**0 / (2*0+1)! * (0, a1, a2, ..., a_Omega)**(2*0+1)
    #      = (0, a1, a2, ..., a_Omega) = "const_zeroed_a"
    c2 = const_zeroed_a.copy()

    # Note that "a.size" here is equal to (Omega + 1) in A. Chao's TPSA note
    Omega = a.size - 1
    #print('Omega = ', Omega)

    prev_term1 = c1.copy()
    prev_term2 = c2.copy()

    const_zeroed_a_sq = const_zeroed_a * const_zeroed_a

    if not a.complex_coeffs:

        prev_f1 = 1.0 # := (-1)**0 / (2*0)!
        prev_f2 = 1.0 # := (-1)**0 / (2*0+1)!

        k = 1
        while True:

            new_f1 = prev_f1 * (-1) / (2 * k) / (2 * k - 1)
            new_f2 = prev_f2 * (-1) / (2 * k + 1) / (2 * k)

            new_term1 = prev_term1 * const_zeroed_a_sq
            new_term2 = prev_term2 * const_zeroed_a_sq

            if 2 * k > Omega:
                break
            else:
                # Add k-th term of 1st Summation
                c1 += new_term1 * new_f1
                #print(2 * k)

            if 2 * k + 1 > Omega:
                break
            else:
                # Add k-th term of 2nd Summation
                c2 += new_term2 * new_f2
                #print(2 * k + 1)

            prev_f1 = new_f1
            prev_f2 = new_f2

            prev_term1 = new_term1
            prev_term2 = new_term2

            k += 1

    else:

        prev_f1_complex = 1.0 + 0j # := (-1)**0 / (2*0)!
        prev_f2_complex = 1.0 + 0j # := (-1)**0 / (2*0+1)!

        k = 1
        while True:

            new_f1_complex = prev_f1_complex * (-1) / (2 * k) / (2 * k - 1)
            new_f2_complex = prev_f2_complex * (-1) / (2 * k + 1) / (2 * k)

            new_term1 = prev_term1 * const_zeroed_a_sq
            new_term2 = prev_term2 * const_zeroed_a_sq

            if 2 * k > Omega:
                break
            else:
                # Add k-th term of 1st Summation
                c1 += new_term1 * new_f1_complex
                #print(2 * k)

            if 2 * k + 1 > Omega:
                break
            else:
                # Add k-th term of 2nd Summation
                c2 += new_term2 * new_f2_complex
                #print(2 * k + 1)

            prev_f1_complex = new_f1_complex
            prev_f2_complex = new_f2_complex

            prev_term1 = new_term1
            prev_term2 = new_term2

            k += 1


    if is_sine:
        if not a.complex_coeffs:
            c1 *= np.sin(a0)
            c2 *= np.cos(a0)
        else:
            c1 *= np.sin(a0_complex)
            c2 *= np.cos(a0_complex)
    else:
        if not a.complex_coeffs:
            c1 *= np.cos(a0)
            c2 *= -np.sin(a0)
        else:
            c1 *= np.cos(a0_complex)
            c2 *= -np.sin(a0_complex)

    c1 += c2

    return c1

#----------------------------------------------------------------------
def conjugate(input_tps):
    """
    """

    c = input_tps.copy()
    c.coeffs = np.conjugate(c.coeffs)

    return c
