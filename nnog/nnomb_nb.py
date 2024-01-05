"""
Numba reimplementation of "Non-Negative Orthogonal Greedy Algorithms"
(Nguyen et al., 2019) (https://ieeexplore.ieee.org/abstract/document/8847410).
"""

import numba as nb
import numpy as np

# Note that to compile, you may need the latest version of numba, in order that
# numba supports np.append and np.delete.


@nb.jit(nopython=True, nogil=True)
def delete_row(A, j):
    m = A.shape[0]
    n = A.shape[1]
    B = np.empty((m - 1, n), dtype=A.dtype)
    B[:j] = A[:j]
    B[j:] = A[j + 1:]
    return B


@nb.jit(nopython=True, nogil=True)
def delete_col(A, j):
    m = A.shape[0]
    n = A.shape[1]
    B = np.empty((m, n - 1), dtype=A.dtype)
    B[:, :j] = A[:, :j]
    B[:, j:] = A[:, j + 1:]
    return B


@nb.jit(nopython=True, nogil=True)
def block2x2(A, B, C, D):
    m1 = A.shape[0]
    m2 = C.shape[0]
    n1 = A.shape[1]
    n2 = B.shape[1]
    m = m1 + m2
    n = n1 + n2
    P = np.empty((m, n), dtype=A.dtype)
    P[:m1, :n1] = A[:, :]
    P[:m1, n1:] = B[:, :]
    P[m1:, :n1] = C[:, :]
    P[m1:, n1:] = D[:, :]
    return P


@nb.jit(nopython=True, nogil=True)
def setdiff1d(T, S):
    T = np.sort(T)
    S = np.sort(S)
    m = T.shape[0]
    n = S.shape[0]
    i = 0
    j = 0
    k = 0
    D = np.empty_like(T)
    while i < m and j < n:
        if T[i] < S[j]:
            D[k] = T[i]
            k += 1
            i += 1
        elif T[i] > S[j]:
            j += 1
        else:
            i += 1
            j += 1
    while i < m:
        D[k] = T[i]
        k += 1
        i += 1
    R = np.empty(k, dtype=D.dtype)
    R[:k] = D[:k]
    return R


@nb.jit(nopython=True, nogil=True)
def max_argmax(values, indices):
    j = np.argmax(values)
    return values[j], indices[j], j


@nb.jit(nopython=True, nogil=True)
def min_argmin(values, indices):
    j = np.argmin(values)
    return values[j], indices[j], j


@nb.jit(nopython=True, nogil=True)
def gram_nnomp(Hy, HH, K):
    """
    The NNOMP algorithm on a precomputed Gram matrix. I didn't reimplement the
    estimation of the residual term (a.k.a. the squared error), because I
    didn't fully understant the formula. The accuracy of the solution, though,
    is guaranteed by tests.

    :param Hy: H'*y of shape (n,)
    :param HH: H'*H of shape (n, n)
    :param K: the sparsity
    :return: of shape (n,)
    """
    n = Hy.shape[0]
    x = np.zeros(n, dtype=HH.dtype)
    T = np.arange(n, dtype=np.int32)
    # Numba won't allow me to write np.array([], dtype=...) here.
    S = np.empty(0, dtype=T.dtype)
    Sbar = T  # =np.setdiff1d(T, S)
    Hr = Hy
    theta = np.empty((0, 0), dtype=Hy.dtype)
    max_prod, l, _ = max_argmax(Hr[Sbar], Sbar)
    while S.shape[0] < K and max_prod > 0:
        x, S, theta = as_nnls(Hy, HH, l, S, x, theta)
        Sbar = setdiff1d(T, S)
        Hr = Hy - np.dot(HH, x)
        max_prod, l, _ = max_argmax(Hr[Sbar], Sbar)
    return x


@nb.jit(nopython=True, nogil=True)
def as_nnls(Hy, HH, l, V, x, theta):
    """
    Active-set algorithm to solve the NNLS problem related to support T,
    starting from a positive support V. The target set T is V union {l}.

    :param Hy: of shape (n,)
    :param HH: of shape (n, n)
    :param l: scalar
    :param V: the initial support
    :param x: the ULS minimizer on support V
    :param theta: the inverse of the Gram matrix related to the support V
    :return: the NNLS minimizer on support T, the support of the minimizer,
             the inverse of the Gram matrix related to that support, and
             updated error
    """
    T = np.append(V, l)
    # Numba let me add this line
    x = np.ascontiguousarray(x)
    Hr = Hy - np.dot(HH, x)
    Vbar = np.array([l])  # =setdiff1d(T, V)
    max_prod, l_plus, _ = max_argmax(Hr[Vbar], Vbar)
    xV = x
    while max_prod > 0:
        xV, V, theta = re_uls_1(Hy, HH, V, l_plus, xV, theta)
        b = xV < 0
        while np.any(b):
            # Somehow `tmp` always becomes float64 regardless of the type of
            # x and xV, so I have to manually cast its type.
            tmp = np.where(b[V], x[V] / (x[V] - xV[V]), np.inf).astype(x.dtype)
            alpha, l_minus, j = min_argmin(tmp, V)
            x += alpha * (xV - x)
            xV, V, theta = re_uls_0(V, l_minus, j, xV, theta)
            b = xV < 0
        x = xV
        # Numba let me add this line
        x = np.ascontiguousarray(x)
        Hr = Hy - np.dot(HH, x)
        Vbar = setdiff1d(T, V)
        if Vbar.shape[0] == 0:
            break
        max_prod, l_plus, _ = max_argmax(Hr[Vbar], Vbar)
    return x, V, theta


@nb.jit(nopython=True, nogil=True)
def re_uls_1(Hy, HH, V, l, x, theta):
    """
    Recursive Unconstrained Least Square, with added regressor. Return the ULS
    solution on support V union {l}, the new support, and the updated Gram
    matrix inverse on that support.
    """
    x = np.copy(x)
    new_V = np.append(V, l)
    if V.shape[0] == 0:
        # Typo in the paper! Corrected here.
        # Why to use reciprocal rather than 1/...? Because 1/... cast the type
        # to np.float64 whatever the type of ... is, which confuses numba.
        theta = np.reciprocal(HH[l, l])
        x[l] = theta * Hy[l]
        # so that `thetaphi` below can be computed by matmul
        theta = np.array([[theta]])
    else:
        phi = HH[V, l]
        thetaphi = np.dot(theta, phi)
        # Typo in the paper! Corrected here.
        # Why to use reciprocal rather than 1/...? Because 1/... cast the type
        # to np.float64 whatever the type of ... is, which confuses numba.
        delta = np.reciprocal(HH[l, l] - np.dot(phi, thetaphi))
        m1 = np.array([-1], dtype=theta.dtype)
        thetaphi_m1 = np.concatenate((thetaphi, m1))
        _0s = np.zeros(theta.shape[0], dtype=theta.dtype)
        # See https://math.stackexchange.com/a/182319/1057593
        _B = np.expand_dims(_0s, 1)
        _C = np.expand_dims(_0s, 0)
        _D = np.zeros((1, 1), dtype=theta.dtype)
        theta = (
            block2x2(theta, _B, _C, _D)
            + delta * np.outer(thetaphi_m1, thetaphi_m1))
        x[new_V] = np.dot(theta, Hy[new_V])
    return x, new_V, theta


@nb.jit(nopython=True, nogil=True)
def re_uls_0(V, l, j, x, theta):
    """
    Recursive Unconstrained Least Square, with removed regressor. Return the
    ULS solution on support V union {l}, the new support, and the updated Gram
    matrix inverse on that support.
    """
    x = np.copy(x)
    thetaj = theta[:, j]
    x[V] -= x[l] / thetaj[j] * thetaj
    theta = delete_col(delete_row(theta, j), j)
    theta_rmj = np.delete(thetaj, j)
    theta -= np.outer(theta_rmj, theta_rmj) / thetaj[j]
    new_V = np.delete(V, j)
    return x, new_V, theta
