"""Plotting utilities for Polar Gini Curves."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_pgc(angles: np.ndarray, *curves: np.ndarray) -> None:
    """Plot one or more Polar Gini Curves."""
    for curve in curves:
        plt.plot(angles, curve)
    plt.xlabel("Angle (radians)")
    plt.ylabel("Gini coefficient")
    plt.title("Polar Gini Curve")
    plt.legend([f"curve {i+1}" for i in range(len(curves))])
    plt.tight_layout()
    plt.show()
