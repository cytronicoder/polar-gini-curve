"""Preprocessing utilities."""

from __future__ import annotations

import numpy as np


def normalize(points: np.ndarray) -> np.ndarray:
    """Normalize points to [0, 1] range per axis."""
    pts = np.asarray(points, dtype=float)
    min_vals = pts.min(axis=0)
    max_vals = pts.max(axis=0)
    return (pts - min_vals) / (max_vals - min_vals)
