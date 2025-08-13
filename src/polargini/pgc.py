"""Compute PGCs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .metrics import gini


def _gini_unweighted_columns(vals: np.ndarray) -> np.ndarray:
    """Compute unweighted Gini per column for a non-negative matrix.

    Each column is treated as an independent sample. Uses the closed-form
    formula after sorting each column ascending.
    """
    if vals.size == 0:
        return np.array([], dtype=float)

    xs = np.sort(vals, axis=0)
    n = xs.shape[0]
    if n == 0:
        return np.zeros(xs.shape[1], dtype=float)
    idx = np.arange(1, n + 1, dtype=float)[:, None]
    totals = xs.sum(axis=0)

    mask = totals > 0
    out = np.zeros(xs.shape[1], dtype=float)
    if np.any(mask):
        out[mask] = (2.0 * (idx * xs[:, mask]).sum(axis=0) / (n * totals[mask])) - (
            (n + 1) / n
        )
    return out


def polar_gini_curve(
    points: np.ndarray, labels: np.ndarray, num_angles: int = 360
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute Polar Gini Curves for one or more groups of points."""
    pts = np.asarray(points, dtype=float)
    lbls = np.asarray(labels)
    if pts.shape[1] != 2:
        raise ValueError("Points must be two-dimensional")
    if len(lbls) != len(pts):
        raise ValueError("Labels length mismatch")

    uniq = np.unique(lbls)
    k = int(num_angles)
    angles = np.linspace(0.0, 2 * np.pi, k, endpoint=False)

    dirs = np.vstack((np.cos(angles), np.sin(angles)))
    projections = pts @ dirs

    curves: list[np.ndarray] = []
    for label in uniq:
        mask = lbls == label
        vals = projections[mask, :]
        vals = vals - vals.min(axis=0, keepdims=True)
        curve = _gini_unweighted_columns(vals)
        curves.append(curve)

    return angles, curves
