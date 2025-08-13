"""Metrics for PGC."""

from __future__ import annotations

import numpy as np


def _gini_unweighted(values: np.ndarray) -> float:
    """Fast Gini for non-negative, unweighted values.

    Uses the formula:
        G = (2 * sum(i * x_i) / (n * sum_x)) - (n + 1) / n
    where x is sorted ascending and i is 1..n.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    if np.amin(x) < 0:
        raise ValueError("Values must be non-negative")
    total = x.sum()
    if total == 0.0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (idx * xs).sum() / (n * total)) - (n + 1) / n)


def compute_gini(pop: np.ndarray, val: np.ndarray, return_lorenz: bool = False):
    """Compute weighted Gini and (optionally) Lorenz curves (MATLAB-compatible).

    Parameters
    ----------
    pop:
        Population sizes (non-negative weights) for each class.
    val:
        Measurement per class (e.g., income per capita), non-negative.
    return_lorenz:
        If True, also return the normalized Lorenz curve L (Nx2) and the
        absolute cumulative curve A (Nx2), including the prepended (0,0) row.

    Returns
    -------
    g: float
        Gini coefficient in [0, 1].
    L: np.ndarray, optional
        Normalized Lorenz curve columns [relpop, relz]. Returned if
        return_lorenz=True.
    A: np.ndarray, optional
        Absolute cumulative columns [pop_cumsum, z_cumsum]. Returned if
        return_lorenz=True.
    """
    wts = np.asarray(pop, dtype=float).reshape(-1)
    vals = np.asarray(val, dtype=float).reshape(-1)
    if wts.shape != vals.shape:
        raise ValueError("pop and val must have the same shape")
    if np.any(wts < 0) or np.any(vals < 0):
        raise ValueError("pop and val must be non-negative")

    pop0 = np.concatenate(([0.0], wts))
    val0 = np.concatenate(([0.0], vals))
    z = val0 * pop0

    order = np.argsort(val0)
    pop_cum = np.cumsum(pop0[order])
    z_cum = np.cumsum(z[order])

    if z_cum[-1] == 0.0 or pop_cum[-1] == 0.0:
        g = 0.0
        if return_lorenz:
            relpop = np.zeros_like(pop_cum)
            relz = np.zeros_like(z_cum)
            L = np.column_stack((relpop, relz))
            A = np.column_stack((pop_cum, z_cum))
            return g, L, A
        return g

    relpop = pop_cum / pop_cum[-1]
    relz = z_cum / z_cum[-1]
    g = float(1.0 - np.sum((relz[:-1] + relz[1:]) * np.diff(relpop)))

    if return_lorenz:
        L = np.column_stack((relpop, relz))
        A = np.column_stack((pop_cum, z_cum))
        return g, L, A
    return g


def gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute the (possibly weighted) Gini coefficient.

    Supports both unweighted and weighted calculations. For weighted
    calculations, this is equivalent to ``compute_gini(pop=weights, val=values)``.

    Parameters
    ----------
    values:
        One-dimensional array of non-negative values.
    weights:
        Optional non-negative weights corresponding to ``values`` (population
        sizes). When omitted, all values are weighted equally.
    """
    vals = np.asarray(values, dtype=float)

    if weights is None:
        return _gini_unweighted(vals)

    return float(compute_gini(pop=np.asarray(weights, dtype=float), val=vals))


def rmsd(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Root mean square deviation between two curves."""
    c1 = np.asarray(curve1, dtype=float)
    c2 = np.asarray(curve2, dtype=float)
    if c1.shape != c2.shape:
        raise ValueError("Curves must have the same shape")
    return float(np.sqrt(np.mean((c1 - c2) ** 2)))
