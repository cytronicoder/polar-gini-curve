"""Plotting utilities for PGCs."""

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


def plot_embedding_and_pgc(
    points: np.ndarray,
    labels: np.ndarray,
    angles: np.ndarray,
    *curves: np.ndarray,
    cluster_labels: Optional[list[str]] = None,
    scatter_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Plot the 2D embedding and PGC side by side."""
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121)
    unique_labels = np.unique(labels)

    if cluster_labels is None:
        cluster_labels = [f"Cluster {label}" for label in unique_labels]

    colors = plt.cm.tab10(np.arange(len(unique_labels)))  # type: ignore[attr-defined]

    allowed_keys = {"s", "alpha", "linewidths"}
    base_scatter: dict[str, Any] = {
        "alpha": 0.7,
        "s": 20.0,
        "linewidths": 0.0,
    }
    user_raster = False
    if scatter_kwargs:
        user_raster = bool(scatter_kwargs.get("rasterized", False))
        for k in allowed_keys:
            if k in scatter_kwargs:
                base_scatter[k] = scatter_kwargs[k]

    auto_raster = bool(points.shape[0] > 50000)
    raster_flag = bool(user_raster or auto_raster)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = points[mask]
        sc = ax1.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[i]],
            label=cluster_labels[i] if i < len(cluster_labels) else f"Cluster {label}",
            **base_scatter,
        )
        if raster_flag:
            sc.set_rasterized(True)

    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.set_title("2D Embedding")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(122, projection="polar")
    plot_pgc(angles, *curves, ax=ax2, labels=cluster_labels[: len(curves)])

    plt.tight_layout()
    plt.show()


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
    ax.set_ylabel("Gini coefficient", labelpad=30)
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
