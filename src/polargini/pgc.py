"""Compute Polar Gini Curves."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .metrics import gini


def polar_gini_curve(
    points: np.ndarray, labels: np.ndarray, num_angles: int = 180
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute Polar Gini Curves for two groups of points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2) with x and y coordinates.
    labels : np.ndarray
        Array of length n with two unique labels identifying groups.
    num_angles : int
        Number of angles between 0 and pi to evaluate.

    Returns
    -------
    angles : np.ndarray
        Angles used for projection.
    curves : List[np.ndarray]
        Gini curves for each label in ascending order.
    """

    pts = np.asarray(points, dtype=float)
    lbls = np.asarray(labels)
    if pts.shape[1] != 2:
        raise ValueError("Points must be two-dimensional")
    if len(lbls) != len(pts):
        raise ValueError("Labels length mismatch")
    uniq = np.unique(lbls)
    if len(uniq) != 2:
        raise ValueError("Labels must contain exactly two groups")

    angles = np.linspace(0.0, np.pi, num_angles, endpoint=False)
    curves = {label: np.zeros_like(angles) for label in uniq}
    for idx, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection = pts @ direction
        for label in uniq:
            vals = projection[lbls == label]
            # Shift to non-negative range for Gini computation
            vals = vals - vals.min()
            curves[label][idx] = gini(vals)
    return angles, [curves[label] for label in uniq]
