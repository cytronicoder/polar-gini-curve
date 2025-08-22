"""
Extra lightweight tests to exercise edge-case branches for coverage.
"""

from __future__ import annotations

import subprocess
import sys
import numpy as np
import pytest

from polargini.metrics import gini, compute_gini


def test_gini_edge_cases():
    assert gini(np.array([])) == 0.0
    assert gini(np.array([0.0, 0.0, 0.0])) == 0.0
    with pytest.raises(ValueError):
        gini(np.array([1.0, -0.5]))


def test_compute_gini_lorenz_and_zero_total():
    pop = np.array([4.0, 2.0, 1.0])
    val = np.array([3.0, 2.0, 1.0])

    g, L, A = compute_gini(pop, val, return_lorenz=True)
    assert 0.0 <= g <= 1.0
    assert L.shape[0] == pop.size + 1 and L.shape[1] == 2
    assert A.shape == L.shape

    g2, L2, A2 = compute_gini(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), return_lorenz=True
    )
    assert g2 == 0.0
    assert np.all(L2 == 0.0)
    assert A2.shape[0] == 3


def test_cli_with_clusters(tmp_path):
    """Exercise --clusters path in CLI (two explicit clusters)."""
    csv = tmp_path / "clusters.csv"
    csv.write_text("x,y,cluster\n0,0,5\n1,0,5\n0,1,7\n1,1,7\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "polargini.cli",
            "--csv",
            str(csv),
            "--clusters",
            "5",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Curve 0" in result.stdout
    assert "Curve 1" in result.stdout


def test_cli_clusters_invalid(tmp_path):
    """Exercise error branch where specified cluster not found."""
    csv = tmp_path / "clusters.csv"
    csv.write_text("x,y,cluster\n0,0,1\n1,0,1\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "polargini.cli",
            "--csv",
            str(csv),
            "--clusters",
            "1",
            "9",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not found" in (result.stderr + result.stdout)
