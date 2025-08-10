"""Plotting utilities for PGCs."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_pgc(
    angles: np.ndarray,
    *curves: np.ndarray,
    ax: Optional[plt.Axes] = None,
    labels: Optional[list[str]] = None,
) -> None:
    """Plot one or more PGCs."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        show_plot = True
    else:
        show_plot = False

    plot_angles = angles
    plot_curves = list(curves)
    if len(plot_angles) > 0:
        closed_angles = np.concatenate([plot_angles, [2 * np.pi]])
        closed_curves = []

        for curve in plot_curves:
            closed_curve = np.concatenate([curve, [curve[0]]])
            closed_curves.append(closed_curve)

        plot_angles = closed_angles
        plot_curves = closed_curves

    if labels is None:
        if len(curves) == 1:
            curve_labels = ["PGC"]
        else:
            curve_labels = [f"Group {i+1}" for i in range(len(curves))]
    else:
        curve_labels = labels

    for i, curve in enumerate(plot_curves):
        label = curve_labels[i] if i < len(curve_labels) else f"Group {i+1}"
        ax.plot(plot_angles, curve, label=label, linewidth=2)

    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Polar Gini Curve", pad=20)

    ax.set_theta_direction(-1)  # type: ignore[attr-defined]
    ax.set_theta_zero_location("E")  # type: ignore[attr-defined]

    if len(plot_curves) > 0:
        max_gini: float = max(np.max(curve) for curve in plot_curves)
        min_gini: float = min(np.min(curve) for curve in plot_curves)
        ax.set_ylim(min_gini * 0.95, max_gini * 1.05)

    ax.grid(True, alpha=0.3)

    if len(plot_curves) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0.1, 1.1))

    if show_plot:
        plt.tight_layout()
        plt.show()
