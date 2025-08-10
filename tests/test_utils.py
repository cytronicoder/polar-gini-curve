"""Test utilities."""

import numpy as np
import matplotlib

from polargini.io import load_csv
from polargini.preprocessing import normalize
from polargini.plotting import plot_pgc

matplotlib.use("Agg")


def test_utils(tmp_path):
    """Test preprocessing and plotting utilities."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,label\n0,0,0\n1,1,1\n")
    points, _ = load_csv(csv_path)
    norm = normalize(points)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)
    angles = np.linspace(0, np.pi, 3, endpoint=False)
    plot_pgc(angles, np.array([0, 0.5, 1]), np.array([0.2, 0.3, 0.4]))
