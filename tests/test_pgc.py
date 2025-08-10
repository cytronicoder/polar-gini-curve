import numpy as np

from polargini.pgc import polar_gini_curve


def test_polar_gini_curve():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    labels = np.array([0, 0, 1, 1])
    angles, curves = polar_gini_curve(points, labels, num_angles=4)
    assert len(angles) == 4
    assert len(curves) == 2
    for curve in curves:
        assert curve.shape == (4,)
