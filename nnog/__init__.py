# Profiling shows that numba version generally may be better, so I take it as
# the default. But for small-scale problem, you may alternatively use the
# python version to save the warm-up time.
from .nnomp_nb import gram_nnomp as gram_nnomp
from .nnomp_py import gram_nnomp as gram_nnomp_py
