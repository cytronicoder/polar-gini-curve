"""Tests for statistical utilities."""

import numpy as np

from polargini.stats import compute_pvalues


def test_compute_pvalues():
    rmsd = np.array([[1.0, 2.0], [2.0, 0.5]])
    perc = np.array([[0.2, 0.05], [0.3, 0.2]])
    normalized, pval = compute_pvalues(rmsd, perc)
    expected_norm = np.array([[0.5, 1.0], [1.0, 1.0]])
    expected_p = np.array([[0.15865525, 1.0], [0.84134475, 1.0]])
    assert np.allclose(normalized, expected_norm)
    assert np.allclose(pval, expected_p)
    assert ((pval >= 0) & (pval <= 1)).all()
