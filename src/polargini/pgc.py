"""Compute PGCs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .metrics import gini


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
    angles = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False)
    curves = {label: np.zeros_like(angles) for label in uniq}

    for idx, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection = pts @ direction
        for label in uniq:
            vals = projection[lbls == label]
            vals = vals - vals.min()
            curves[label][idx] = gini(vals)

    return angles, [curves[label] for label in uniq]
