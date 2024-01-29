"""
Reimplementation of "Non-Negative Orthogonal Greedy Algorithms"
(Nguyen et al., 2019) (https://ieeexplore.ieee.org/abstract/document/8847410).
"""

import numpy as np


def max_argmax(values, indices):
    j = np.argmax(values)
    return values[j], indices[j], j


def min_argmin(values, indices):
    j = np.argmin(values)
    return values[j], indices[j], j


def gram_nnomp(Hy, HH, yy, K, tol=0.0):
    """
    The NNOMP algorithm on a precomputed Gram matrix.

    Note that, although this function takes Gram matrix as input, really, it
    is not a batch algorithm like "Efficient Implementation of the K-SVD
    Algorithm using Batch Orthogonal Matching Pursuit"
    (Rubinstein et al., 2008) in any sense.

    :param Hy: H'*y of shape (n,)
    :param HH: H'*H of shape (n, n)
    :param yy: y'*y, a scalar
    :param K: the target sparsity
    :param tol: the target residual norm (the error)
    :return: the solution of shape (n,) and the final squared error
    """
    n = Hy.shape[0]
    x = np.zeros(n)
    S = np.array([], dtype=int)
    T = np.arange(n)
    Sbar = T  # =np.setdiff1d(T, S)
    Hr = Hy
    theta = None
    err2 = yy
    tol2 = tol**2
    max_prod, l, _ = max_argmax(Hr[Sbar], Sbar)
    while S.shape[0] < K and err2 > tol2 and max_prod > 0:
        x, S, theta, err2 = as_nnls(Hy, HH, l, S, x, theta, err2)
        Sbar = np.setdiff1d(T, S)
        Hr = Hy - np.matmul(HH, x)
        max_prod, l, _ = max_argmax(Hr[Sbar], Sbar)
    return x, err2


def as_nnls(Hy, HH, l, V, x, theta, err2):
    """
    Active-set algorithm to solve the NNLS problem related to support T,
    starting from a positive support V. The target set T is V union {l}.

    :param Hy: of shape (n,)
    :param HH: of shape (n, n)
    :param l: scalar
    :param V: the initial support
    :param x: the ULS minimizer on support V
    :param theta: the inverse of the Gram matrix related to the support V
    :param err2: the squared error of ``x``
    :return: the NNLS minimizer on support T, the support of the minimizer,
             the inverse of the Gram matrix related to that support, and
             updated error
    """
    T = np.append(V, l)
    Hr = Hy - np.matmul(HH, x)
    Vbar = np.array([l])  # =setdiff1d(T, V)
    max_prod, l_plus, _ = max_argmax(Hr[Vbar], Vbar)
    xV = x
    while max_prod > 0:
        xV, V, theta, err2 = re_uls_1(Hy, HH, V, l_plus, xV, theta, err2)
        b = xV < 0
        while np.any(b[V]):
            alpha, l_minus, j = min_argmin(
                np.where(b[V], x[V] / (x[V] - xV[V]), np.inf), V)
            x += alpha * (xV - x)
            xV, V, theta, err2 = re_uls_0(V, l_minus, j, xV, theta, err2)
            b = xV < 0
        x = xV
        Hr = Hy - np.matmul(HH, x)
        Vbar = np.setdiff1d(T, V)
        if Vbar.shape[0] == 0:
            break
        max_prod, l_plus, _ = max_argmax(Hr[Vbar], Vbar)
    return x, V, theta, err2


def re_uls_1(Hy, HH, V, l, x, theta, err2):
    """
    Recursive Unconstrained Least Square, with added regressor. Return the ULS
    solution on support V union {l}, the new support, the updated Gram
    matrix inverse on that support and the updated squared error.
    """
    x = np.copy(x)
    new_V = np.append(V, l)
    if V.shape[0] == 0:
        # Typo in the paper! Corrected here.
        theta = 1 / HH[l, l]
        x[l] = theta * Hy[l]
        # So that `thetaphi` below can be computed by matmul
        theta = np.expand_dims(np.expand_dims(theta, 0), 0)
        # Intentionally not using '+=' operator
        err2 = err2 + -2 * x[l] * Hy[l] + HH[l, l] * x[l]**2
    else:
        phi = HH[V, l]
        thetaphi = np.matmul(theta, phi)
        # Typo in the paper! Corrected here.
        delta = 1 / (HH[l, l] - np.dot(phi, thetaphi))
        m1 = np.array([-1])
        thetaphi_m1 = np.concatenate([thetaphi, m1])
        _0s = np.zeros(theta.shape[0])
        # See https://math.stackexchange.com/a/182319/1057593
        theta = (
            np.block([[theta, np.expand_dims(_0s, 1)],
                      [np.expand_dims(_0s, 0),
                       np.zeros((1, 1))]])
            + delta * np.outer(thetaphi_m1, thetaphi_m1))
        # I'm not using the x and err update rule in the paper because it
        # doesn't pass the tests. Instead, I use a little slower and less
        # "elegant" method.
        e_V = -2 * x[V].dot(Hy[V]) + x[V].dot(HH[V][:, V]).dot(x[V])
        x[new_V] = np.matmul(theta, Hy[new_V])
        e_Vl = (-2 * x[new_V].dot(Hy[new_V])
                + x[new_V].dot(HH[new_V][:, new_V]).dot(x[new_V]))
        # Intentionally not using '+=' operator
        err2 = err2 + e_Vl - e_V
    return x, new_V, theta, err2


def re_uls_0(V, l, j, x, theta, err2):
    """
    Recursive Unconstrained Least Square, with removed regressor. Return the
    ULS solution on support V union {l}, the new support, the updated Gram
    matrix inverse on that support and the updated squared error.
    """
    x = np.copy(x)
    thetaj = theta[:, j]
    # Intentionally not using '+=' operator
    err2 = err2 + x[l]**2 / thetaj[j]
    x[V] -= x[l] / thetaj[j] * thetaj
    theta = np.delete(np.delete(theta, j, 0), j, 1)
    theta_rmj = np.delete(thetaj, j)
    theta -= np.outer(theta_rmj, theta_rmj) / thetaj[j]
    new_V = np.delete(V, j)
    return x, new_V, theta, err2
