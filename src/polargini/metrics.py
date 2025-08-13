"""Metrics for PGC."""

from __future__ import annotations

import numpy as np


def gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute the (possibly weighted) Gini coefficient.

    Parameters
    ----------
    values:
        One-dimensional array of non-negative values.
    weights:
        Optional non-negative weights corresponding to ``values``. When
        omitted, all values are weighted equally.
    """
    vals = np.asarray(values, dtype=float)
    if np.amin(vals) < 0:
        raise ValueError("Values must be non-negative")
    if weights is None:
        wts = np.ones_like(vals)
    else:
        wts = np.asarray(weights, dtype=float)
        if wts.shape != vals.shape:
            raise ValueError("Weights must have the same shape as values")
        if np.amin(wts) < 0:
            raise ValueError("Weights must be non-negative")
    if np.sum(vals * wts) == 0:
        return 0.0

    # Implementation follows the MATLAB ``computeGini`` script. Prepend a
    # zero to simplify the cumulative calculations.
    pop = np.concatenate(([0.0], wts))
    val = np.concatenate(([0.0], vals))
    z = val * pop
    order = np.argsort(val)
    pop = np.cumsum(pop[order])
    z = np.cumsum(z[order])
    relpop = pop / pop[-1]
    relz = z / z[-1]
    return 1.0 - np.sum((relz[:-1] + relz[1:]) * np.diff(relpop))


def rmsd(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Root mean square deviation between two curves."""
    c1 = np.asarray(curve1, dtype=float)
    c2 = np.asarray(curve2, dtype=float)
    if c1.shape != c2.shape:
        raise ValueError("Curves must have the same shape")
    return float(np.sqrt(np.mean((c1 - c2) ** 2)))
