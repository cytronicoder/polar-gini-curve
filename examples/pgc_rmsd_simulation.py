#!/usr/bin/env python3
"""Simulate PGC RMSD for random foreground selections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent.parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polargini.pgc import polar_gini_curve  # noqa: E402
from polargini.metrics import rmsd  # noqa: E402
from polargini.plotting import plot_pgc  # noqa: E402


N_POINTS = 5000
PERCENTAGES = np.arange(5, 100, 5)
REPEATS = 1000
NUM_ANGLES = 360


def sample_unit_disk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` points uniformly from the unit disk."""
    r = np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def main() -> None:
    rng = np.random.default_rng(0)

    points = sample_unit_disk(N_POINTS, rng)

    rmsd_values: dict[int, np.ndarray] = {}

    example_angles: np.ndarray | None = None
    example_curves: tuple[np.ndarray, np.ndarray] | None = None

    for m in PERCENTAGES:
        k = int(N_POINTS * m / 100)
        vals = np.empty(REPEATS, dtype=float)

        for i in range(REPEATS):
            fg_idx = rng.choice(N_POINTS, size=k, replace=False)
            labels = np.zeros(N_POINTS, dtype=int)
            labels[fg_idx] = 1

            angles, curves = polar_gini_curve(points, labels, num_angles=NUM_ANGLES)
            bg_curve, fg_curve = curves[0], curves[1]
            vals[i] = rmsd(bg_curve, fg_curve)

            if m == 50 and i == 0:
                example_angles = angles
                example_curves = (bg_curve, fg_curve)

        rmsd_values[m] = vals

    means = []
    stds = []
    lowers = []
    uppers = []
    for m in PERCENTAGES:
        v = rmsd_values[m]
        means.append(float(np.mean(v)))
        stds.append(float(np.std(v, ddof=1)))
        q = np.quantile(v, [0.025, 0.975])
        lowers.append(float(q[0]))
        uppers.append(float(q[1]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(PERCENTAGES, means, label="Mean RMSD")
    ax.fill_between(PERCENTAGES, lowers, uppers, alpha=0.2, label="2.5–97.5%")
    ax.set_xlabel("Expressing percentage (%)")
    ax.set_ylabel("RMSD")
    ax.set_title("RMSD between foreground and background PGCs")
    ax.legend()
    fig.tight_layout()

    if example_angles is not None and example_curves is not None:
        fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
        plot_pgc(
            example_angles,
            example_curves[0],
            example_curves[1],
            ax=ax2,
            labels=["Background", "Foreground"],
        )
        ax2.set_title("Example PGC (m=50%)", pad=20)
        fig2.tight_layout()

    plt.show()

    print("m%\tmean\tstd\t2.5%\t97.5%")
    for m, mean, std, lo, hi in zip(PERCENTAGES, means, stds, lowers, uppers):
        print(f"{m:2d}\t{mean:.6f}\t{std:.6f}\t{lo:.6f}\t{hi:.6f}")


if __name__ == "__main__":
    main()
