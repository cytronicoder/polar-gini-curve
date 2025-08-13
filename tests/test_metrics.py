"""Tests for metric utilities."""

import numpy as np
import pytest

from polargini.metrics import gini


def test_gini_weighted_matches_unweighted():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert gini(values) == pytest.approx(gini(values, weights))


def test_gini_weighted_value():
    values = np.array([1.0, 1.0, 3.0])
    weights = np.array([1.0, 1.0, 2.0])
    assert gini(values, weights) == pytest.approx(0.25)
