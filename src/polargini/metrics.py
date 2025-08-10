"""Metrics for PGC."""

from __future__ import annotations

import numpy as np


def gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for a 1-D array of non-negative values."""
    vals = np.asarray(values, dtype=float)
    if np.amin(vals) < 0:
        raise ValueError("Values must be non-negative")
    if np.sum(vals) == 0:
        return 0.0
    diff = np.abs(np.subtract.outer(vals, vals)).sum()
    return diff / (2 * len(vals) * np.sum(vals))


def rmsd(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """Root mean square deviation between two curves."""
    c1 = np.asarray(curve1, dtype=float)
    c2 = np.asarray(curve2, dtype=float)
    if c1.shape != c2.shape:
        raise ValueError("Curves must have the same shape")
    return float(np.sqrt(np.mean((c1 - c2) ** 2)))
