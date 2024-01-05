# Numba reimplementation of "Non-Negative Orthogonal Greedy Algorithms" (Nguyen et al., 2019)

## Introduction

This package contains [numba](https://numba.pydata.org/) and Python3 reimplementation of the gram-NNOMP solver in ([Nguyen et al., 2019](https://hal.science/hal-02049424/document)).
It has been thoroughly tested.
It has fixed various bugs that show up in the paper and the authors' [official implementation](https://codeocean.com/capsule/1591546/tree/v1).

Note that this is not a verbatim translation from the authors' Matlab code.

## About the gram-NNOMP solver

NNOMP is the acryonym of "non-negative orthogonal matching pursuit".
The solver reimplemented in this package takes only the Gram matrix, thus able to handle large-scale problem.
For orthogonal matching pursuit *without* non-negativity constraint, see, e.g., [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp_gram.html).

## Prerequisite

To use the library without tests and profiling, only those packages listed in `requirements.txt` are required:

```bash
pip install -r requirements.txt
```

or with conda:

```bash
conda install numpy numba
```

To run tests (with [pytest](https://docs.pytest.org/en/7.4.x/)) and the profiling code (`profile.py`), you'll need `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

or with conda:

```bash
conda install numpy numba scipy pytest joblib
```

## Installation

First clone this repository.
Then `cd` into this repository and run:

```bash
pip install .
```

It's recommended to run this in a virtual environment.

## Usage

To solve $\min \|y-Hx\|_2^2$ for $x$ with constraints $x \ge 0$ (elementwise) and $\|x\|_0 \le K$ given $H^\top H$ and $H^\top y$, assuming $H$ has been normalized columnwise:

```python
import numpy as np

from nnog import gram_nnomp

# Prepare H'*H and H'*y
H = np.random.rand(1000, 100)
H /= np.linalg.norm(H, axis=1, keepdims=True)
y = np.random.rand(1000)
HH = np.matmul(H.T, H)
Hy = np.dot(H.T, y)

# Specify the sparsity
K = 5

# Solve the problem.
# You may want to convert the result `x` into scipy sparse matrix afterward.
x = gram_nnomp(Hy, HH, K)
```

The above example use the `numba` version.
To use the Python version, import with `from nnog import gram_nnomp_py`

With `numba` version, you may run multiple instances of NNOMP in parallel, with multi-threading.
Usually, to run something in parallel in Python, multiprocessing is required due to the GIL.
But the `numba` code in this repo drops the GIL.

## Comparing Python and numba version

Running `profile.py`, you'll get output like this (won't be exactly the same):

```
---
(n_features=100) python took 0.001573 sec on average
(n_features=100)  numba took 0.000095 sec on average
(n_features=100) python (parallel) took 0.553898 sec in total
(n_features=100)  numba (parallel) took 0.037113 sec in total
---
(n_features=2000) python took 0.562712 sec on average
(n_features=2000)  numba took 0.496468 sec on average
(n_features=2000) python (parallel) took 44.130961 sec in total
(n_features=2000)  numba (parallel) took 37.828639 sec in total
```

Note that the above result does not take into account the compiling overhead of `numba`.
