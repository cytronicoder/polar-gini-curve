"""Tests for the PGC package."""

import numpy as np

from polargini.pgc import polar_gini_curve


def test_polar_gini_curve():
    """Test the PGC calculation."""
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    angles, curves = polar_gini_curve(points, labels, num_angles=4)
    assert len(angles) == 4
    assert len(curves) == 3
    for curve in curves:
        assert curve.shape == (4,)
