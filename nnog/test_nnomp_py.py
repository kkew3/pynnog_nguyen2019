import typing as ty

import numpy as np
from scipy.optimize import nnls

from nnog import nnomp_py as m


def solve_S_support_nnls(A, b, S):
    """
    Find the NNLS problem (A, b) on support S using scipy's ``nnls``.
    Return the solution.
    """
    S = np.asarray(S)
    x = np.zeros(A.shape[1])
    if S.size == 0:
        return x
    x[S], _ = nnls(A[:, S], b)
    return x


def solve_S_support_lstsq(A, b, S):
    S = np.asarray(S)
    x = np.zeros(A.shape[1])
    x[S], _, _, _ = np.linalg.lstsq(A[:, S], b, rcond=None)
    return x


def prepare_data(
    n_samples: int,
    n_features: int,
    K: int,
    noise_std: float,
    initial_support,
    extra_support: ty.Optional[int],
):
    H = np.random.randn(n_samples, n_features)
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    z_true = np.zeros(n_features)
    k_ind = np.random.permutation(n_features)[:K]
    k_val = np.random.rand(K) + 1e-3
    z_true[k_ind] = k_val
    y = np.matmul(H, z_true) + np.random.randn(n_samples) * noise_std

    Hy = np.matmul(H.T, y)
    HH = np.matmul(H.T, H)

    if initial_support is None or extra_support is None:
        return (
            H,
            y,
            z_true,
            Hy,
            HH,
        )

    # initial support
    S = np.asarray(initial_support, dtype=int)
    # target support
    T = np.append(S, extra_support)

    last_z = solve_S_support_nnls(H, y, S)
    if S.size == 0:
        last_theta = np.ones((0, 0))
    else:
        last_theta = np.linalg.inv(np.matmul(H[:, S].T, H[:, S]))
    last_err2 = np.linalg.norm(y - np.matmul(H, last_z))**2

    updated_z_true = solve_S_support_nnls(H, y, T)
    updated_err2_true = np.linalg.norm(y - np.matmul(H, updated_z_true))**2

    return (
        H,
        y,
        z_true,
        Hy,
        HH,
        last_z,
        last_theta,
        last_err2,
        updated_z_true,
        updated_err2_true,
    )


def ismember(s, a):
    s = np.asarray(s)
    a = np.asarray(a)
    if a.size == 0:
        return 0, -1

    if s.ndim == 0:
        s_is_scalar = True
        s = np.expand_dims(s, 0)
    else:
        s_is_scalar = False
    b = np.expand_dims(s, 1) == np.expand_dims(a, 0)
    tf = np.any(b, 1)
    s_idx = np.where(tf, np.argmax(b, 1), -1)
    if s_is_scalar:
        tf = np.squeeze(tf)
        s_idx = np.squeeze(s_idx)
        assert tf.ndim == 0
        assert s_idx.ndim == 0
    return tf, s_idx


def test_re_uls_1():
    n_features = 7
    l = 2
    V = np.array([1, 5])
    np.random.seed(123)
    V_addl = np.append(V, l)
    (
        H,
        y,
        _,
        Hy,
        HH,
        _,
        _,
        _,
        _,
        _,
    ) = prepare_data(10, n_features, 4, 0.1, V, l)
    last_x = solve_S_support_lstsq(H, y, V)
    last_err2 = np.linalg.norm(y - np.dot(H, last_x))**2
    last_theta = np.linalg.inv(np.matmul(H[:, V].T, H[:, V]))
    upd_theta_true = np.linalg.inv(np.matmul(H[:, V_addl].T, H[:, V_addl]))
    x_pred, _, upd_theta, upd_err2 = m.re_uls_1(Hy, HH, V, l, last_x,
                                                last_theta, last_err2)
    x_true = solve_S_support_lstsq(H, y, V_addl)
    upd_err2_true = np.linalg.norm(y - np.dot(H, x_true))**2
    assert np.allclose(x_pred, x_true)
    assert np.allclose(upd_theta, upd_theta_true)
    assert np.isclose(upd_err2, upd_err2_true)

    V = np.array([], dtype=int)
    np.random.seed(124)
    V_addl = np.append(V, l)
    (
        H,
        y,
        _,
        Hy,
        HH,
        _,
        _,
        _,
        _,
        _,
    ) = prepare_data(10, n_features, 4, 0.1, V, l)
    last_x = solve_S_support_lstsq(H, y, V)
    last_err2 = np.linalg.norm(y - np.dot(H, last_x))**2
    assert np.isclose(last_err2, np.dot(y, y))
    last_theta = np.linalg.inv(np.matmul(H[:, V].T, H[:, V]))
    upd_theta_true = np.linalg.inv(np.matmul(H[:, V_addl].T, H[:, V_addl]))
    x_pred, _, upd_theta, upd_err2 = m.re_uls_1(Hy, HH, V, l, last_x,
                                                last_theta, last_err2)
    x_true = solve_S_support_lstsq(H, y, V_addl)
    upd_err2_true = np.linalg.norm(y - np.dot(H, x_true))**2
    assert np.allclose(x_pred, x_true)
    assert np.allclose(upd_theta, upd_theta_true)
    assert np.isclose(upd_err2, upd_err2_true)


def test_re_uls_0():
    n_features = 7
    l = 2
    V = np.array([2, 1, 5])
    V_rml = np.array([1, 5])
    j = 0
    (
        H,
        y,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = prepare_data(10, n_features, 4, 0.1, V, l)
    last_x = solve_S_support_lstsq(H, y, V)
    last_err2 = np.linalg.norm(y - np.dot(H, last_x))**2
    last_theta = np.linalg.inv(np.matmul(H[:, V].T, H[:, V]))
    upd_theta_true = np.linalg.inv(np.matmul(H[:, V_rml].T, H[:, V_rml]))
    x_pred, _, upd_theta, upd_err2 = m.re_uls_0(V, l, j, last_x, last_theta,
                                                last_err2)
    x_true = solve_S_support_lstsq(H, y, V_rml)
    upd_err2_true = np.linalg.norm(y - np.dot(H, x_true))**2
    assert np.allclose(x_pred, x_true)
    assert np.allclose(upd_theta, upd_theta_true)
    assert np.isclose(upd_err2, upd_err2_true)


class TestAsNnls:
    def test_from_null_support(self):
        n_features = 7
        l = 2
        V = np.array([], dtype=int)
        T = np.append(V, l)
        np.random.seed(347)
        (
            H,
            y,
            _,
            Hy,
            HH,
            last_z,
            last_theta,
            last_err2,
            upd_z_true,
            upd_err2_true,
        ) = prepare_data(10, n_features, 4, 0.1, V, l)
        upd_z, upd_V, upd_theta, upd_err2 = m.as_nnls(Hy, HH, l, V, last_z,
                                                      last_theta, last_err2)
        im, _ = ismember(upd_V, T)
        assert np.all(im)
        assert np.allclose(upd_z, upd_z_true)
        assert np.isclose(upd_err2, upd_err2_true)
        upd_theta_true = np.linalg.inv(np.matmul(H[:, upd_V].T, H[:, upd_V]))
        assert np.allclose(upd_theta, upd_theta_true)

    def test_from_random_support1(self):
        n_features = 7
        K = 4
        np.random.seed(7)
        perm = np.random.permutation(n_features)
        V = perm[:K - 1]
        l = int(perm[K - 1])
        T = perm[:K]
        np.random.seed(258)
        (
            H,
            y,
            _,
            Hy,
            HH,
            last_z,
            last_theta,
            last_err2,
            upd_z_true,
            upd_err2_true,
        ) = prepare_data(10, n_features, K, 0.1, V, l)
        upd_z, upd_V, upd_theta, upd_err2 = m.as_nnls(Hy, HH, l, V, last_z,
                                                      last_theta, last_err2)
        im, _ = ismember(upd_V, T)
        assert np.all(im)
        assert np.allclose(upd_z, upd_z_true)
        assert np.isclose(upd_err2, upd_err2_true)
        upd_theta_true = np.linalg.inv(np.matmul(H[:, upd_V].T, H[:, upd_V]))
        assert np.allclose(upd_theta, upd_theta_true)

    def test_from_random_support2(self):
        n_features = 7
        K = 4
        np.random.seed(8)
        perm = np.random.permutation(n_features)
        V = perm[:K - 1]
        l = int(perm[K - 1])
        T = perm[:K]
        np.random.seed(259)
        (
            H,
            y,
            _,
            Hy,
            HH,
            last_z,
            last_theta,
            last_err2,
            upd_z_true,
            upd_err2_true,
        ) = prepare_data(10, n_features, K, 0.1, V, l)
        upd_z, upd_V, upd_theta, upd_err2 = m.as_nnls(Hy, HH, l, V, last_z,
                                                      last_theta, last_err2)
        im, _ = ismember(upd_V, T)
        assert np.all(im)
        assert np.allclose(upd_z, upd_z_true)
        assert np.isclose(upd_err2, upd_err2_true)
        upd_theta_true = np.linalg.inv(np.matmul(H[:, upd_V].T, H[:, upd_V]))
        assert np.allclose(upd_theta, upd_theta_true)


class TestNnomp:
    def test_nnomp_small(self):
        n_features = 7
        K = 4
        np.random.seed(14)
        (
            H,
            y,
            z_true,
            Hy,
            HH,
        ) = prepare_data(10, n_features, K, 0.0, None, None)
        z_pred, err2 = m.gram_nnomp(Hy, HH, y.dot(y), K)
        assert np.allclose(z_pred, z_true)
        assert np.isclose(err2, 0) or err2 >= 0
        # test the case when the tol is loose
        z_pred, err2 = m.gram_nnomp(Hy, HH, y.dot(y), K, 1.0)
        assert not np.allclose(z_pred, z_true)
        assert np.isclose(err2, 0) or err2 >= 0

    def test_nnomp_large(self):
        n_features = 100
        K = 5
        np.random.seed(14)
        (
            H,
            y,
            z_true,
            Hy,
            HH,
        ) = prepare_data(1000, n_features, K, 1e-4, None, None)
        z_pred, err2 = m.gram_nnomp(Hy, HH, y.dot(y), K, 1e-4)
        assert np.linalg.norm(z_pred - z_true) < 1e-3
        assert np.isclose(err2, 0) or err2 >= 0
