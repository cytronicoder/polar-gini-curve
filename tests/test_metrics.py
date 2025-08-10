import numpy as np

from polargini.metrics import gini, rmsd


def test_gini_equal_values():
    assert gini(np.array([1, 1, 1])) == 0


def test_gini_simple():
    assert np.isclose(gini(np.array([0, 1])), 0.5)


def test_rmsd():
    assert rmsd(np.array([1, 2]), np.array([1, 2])) == 0
    expected = np.sqrt(((1 - 2) ** 2 + (2 - 4) ** 2) / 2)
    assert np.isclose(rmsd(np.array([1, 2]), np.array([2, 4])), expected)
