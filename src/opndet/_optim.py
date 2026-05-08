"""Tiny optim helpers — replaces opndet's scipy dependency with smaller-blast-
radius alternatives. scipy reaches deep into numpy's private API and breaks on
every numpy minor bump (e.g. numpy 2.4's `_center` removal broke scipy 1.17 in
April 2026, taking opndet down on Colab). These replacements are:

  - `linear_sum_assignment(cost)` — wraps `lap.lapjv` (gatagat/lap). C-compiled
    binary that talks to numpy via the stable ndarray interface only, so numpy
    private-API churn doesn't reach it. ~1 MB install vs scipy's ~70 MB.

  - `minimize_scalar_bounded(fn, lo, hi)` — pure-numpy golden-section search.
    ~30 LOC. Used for fitting the Platt-scaling temperature in calibrate.py.
    No external dep at all.

Both expose scipy-compatible signatures so the call sites barely change.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bipartite minimum-cost assignment. scipy-compatible signature.

    Returns `(row_idx, col_idx)` such that `cost[row_idx, col_idx].sum()` is
    minimized over all 1-to-1 assignments. For rectangular cost matrices, only
    the smaller dimension of pairs is returned (same as scipy).

    Implementation: `lap.lapjv` (Jonker-Volgenant). For our typical N (10-200
    boxes per image) this runs in microseconds.
    """
    import lap   # lazy — keeps `import opndet` cheap

    cost_arr = np.asarray(cost, dtype=np.float64)
    if cost_arr.size == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    if cost_arr.ndim != 2:
        raise ValueError(f"cost must be 2D, got shape {cost_arr.shape}")
    h, w = cost_arr.shape

    # extend_cost=True handles rectangular matrices by padding with a large
    # constant before solving. Returns assignments for the smaller dim only.
    _, x, _ = lap.lapjv(cost_arr, extend_cost=True)
    # x[i] = column assigned to row i, or -1 if unassigned (rectangular case)
    rows = np.arange(h, dtype=np.int64)
    valid = x >= 0
    return rows[valid], x[valid].astype(np.int64)


def minimize_scalar_bounded(fn: Callable[[float], float], lo: float, hi: float,
                            xatol: float = 1e-4, max_iter: int = 100) -> float:
    """Golden-section search for the minimum of a unimodal scalar function on
    [lo, hi]. scipy.optimize.minimize_scalar(method="bounded") replacement.

    Returns the x value (not the SciPy `OptimizeResult` wrapper). Call sites
    in opndet only used `.x` so this matches the contract they care about.

    Algorithm: golden-section. O(log((hi-lo)/xatol)) iterations; fully
    pure-Python, no numpy/scipy needed for the search itself (fn may use them).
    """
    if not (lo < hi):
        raise ValueError(f"lo must be < hi, got lo={lo} hi={hi}")
    phi = (1.0 + 5.0 ** 0.5) / 2.0     # golden ratio
    inv_phi = 1.0 / phi                # ~0.618
    inv_phi2 = 1.0 / (phi * phi)       # ~0.382

    a, b = float(lo), float(hi)
    h = b - a
    c = a + inv_phi2 * h
    d = a + inv_phi * h
    fc = fn(c)
    fd = fn(d)

    for _ in range(max_iter):
        if abs(b - a) <= xatol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            h = b - a
            c = a + inv_phi2 * h
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            h = b - a
            d = a + inv_phi * h
            fd = fn(d)
    return (a + b) * 0.5
