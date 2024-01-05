import time

import numpy as np
from joblib import delayed, Parallel

from nnog import gram_nnomp_py, gram_nnomp


def profile(n_samples, n_features, K):
    H = np.random.randn(n_samples, n_features)
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    z_true = np.zeros(n_features)
    k_ind = np.random.permutation(n_features)[:K]
    k_val = np.random.rand(K) + 1e-3
    z_true[k_ind] = k_val
    y = np.matmul(H, z_true) + np.random.randn(n_samples) * 0.01

    Hy = np.matmul(H.T, y)
    HH = np.matmul(H.T, H)

    # warm-up
    _ = gram_nnomp(Hy, HH, K)

    n_runs = 100

    print('---')
    tic = time.perf_counter()
    for _ in range(n_runs):
        _ = gram_nnomp_py(Hy, HH, K)
    toc = time.perf_counter()
    print(f'(n_features={n_features}) '
          f'python took {(toc - tic) / n_runs:.6f} sec on average')

    tic = time.perf_counter()
    for _ in range(n_runs):
        _ = gram_nnomp(Hy, HH, K)
    toc = time.perf_counter()
    print(f'(n_features={n_features})  '
          f'numba took {(toc - tic) / n_runs:.6f} sec on average')

    tic = time.perf_counter()
    _ = Parallel(
        n_jobs=2, backend='loky')(
            delayed(gram_nnomp_py)(Hy, HH, K) for _ in range(n_runs))
    toc = time.perf_counter()
    print(f'(n_features={n_features}) '
          f'python (parallel) took {toc - tic:.6f} sec in total')

    tic = time.perf_counter()
    _ = Parallel(
        n_jobs=2, backend='threading')(
            delayed(gram_nnomp)(Hy, HH, K) for _ in range(n_runs))
    toc = time.perf_counter()
    print(f'(n_features={n_features})  '
          f'numba (parallel) took {toc - tic:.6f} sec in total')


if __name__ == '__main__':
    profile(100000, 100, 5)
    profile(100000, 2000, 100)
