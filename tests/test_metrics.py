"""Tests for metrics."""

import numpy as np

from polargini.metrics import gini, rmsd


def test_gini_equal_values():
    """Test Gini coefficient for equal values."""
    assert gini(np.array([1, 1, 1])) == 0


def test_gini_simple():
    """Test Gini coefficient for 0 and 1. Should be 0.5."""
    assert np.isclose(gini(np.array([0, 1])), 0.5)


def test_rmsd():
    """Test RMSD calculation."""
    assert rmsd(np.array([1, 2]), np.array([1, 2])) == 0
    expected = np.sqrt(((1 - 2) ** 2 + (2 - 4) ** 2) / 2)
    assert np.isclose(rmsd(np.array([1, 2]), np.array([2, 4])), expected)
