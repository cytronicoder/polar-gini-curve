"""Command-line interface for PGC."""

from __future__ import annotations

import argparse

from .io import load_csv
from .pgc import polar_gini_curve
from .plotting import plot_pgc


def main() -> None:
    """
    Main entry point for the CLI. Parses command-line arguments
    and executes the polar Gini curve calculation.
    """
    parser = argparse.ArgumentParser(description="Compute Polar Gini Curves.")
    parser.add_argument("--csv", required=True, help="CSV file with x,y,label columns.")
    parser.add_argument(
        "--angles", type=int, default=180, help="Number of angles to compute."
    )
    parser.add_argument("--plot", action="store_true", help="Display plot.")
    args = parser.parse_args()

    points, labels = load_csv(args.csv)
    angles, curves = polar_gini_curve(points, labels, num_angles=args.angles)
    for idx, curve in enumerate(curves):
        print(f"Curve {idx}: first five Gini coefficients {curve[:5]}")
    if args.plot:
        plot_pgc(angles, *curves)


if __name__ == "__main__":
    main()
