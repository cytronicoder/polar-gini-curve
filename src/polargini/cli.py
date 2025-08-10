"""Command-line interface for PGC."""

from __future__ import annotations

import argparse

import numpy as np

from .io import load_csv
from .pgc import polar_gini_curve
from .plotting import plot_embedding_and_pgc


def main() -> None:
    """
    Main entry point for the CLI. Parses command-line arguments
    and executes the PGC calculation.
    """
    parser = argparse.ArgumentParser(description="Compute Polar Gini Curves.")
    parser.add_argument("--csv", required=True, help="CSV file with x,y,label columns.")
    parser.add_argument(
        "--angles", type=int, default=180, help="Number of angles to compute."
    )
    parser.add_argument("--plot", action="store_true", help="Display plot.")
    parser.add_argument(
        "--clusters",
        nargs=2,
        type=int,
        help=(
            "Two cluster IDs to compare (e.g., --clusters 1 2). "
            "If not specified, uses first two clusters found."
        ),
    )
    args = parser.parse_args()

    points, labels = load_csv(args.csv)
    unique_labels = np.unique(labels)
    if args.clusters:
        selected_clusters = args.clusters
        if (
            selected_clusters[0] not in unique_labels
            or selected_clusters[1] not in unique_labels
        ):
            available = sorted(unique_labels)
            raise ValueError(
                f"Specified clusters {selected_clusters} not found. "
                f"Available clusters: {available}"
            )
    else:
        if len(unique_labels) < 2:
            raise ValueError(
                f"Need at least 2 clusters, found {len(unique_labels)}: {unique_labels}"
            )
        selected_clusters = unique_labels[:2]
        print(
            f"Multiple clusters found {sorted(unique_labels)}. "
            f"Using first two: {selected_clusters}"
        )

    mask = np.isin(labels, selected_clusters)
    filtered_points = points[mask]
    filtered_labels = labels[mask]

    angles, curves = polar_gini_curve(
        filtered_points, filtered_labels, num_angles=args.angles
    )
    for idx, curve in enumerate(curves):
        print(f"Curve {idx}: first five Gini coefficients {curve[:5]}")
    if args.plot:
        cluster_names = [f"Cluster {label}" for label in selected_clusters]
        plot_embedding_and_pgc(
            filtered_points,
            filtered_labels,
            angles,
            *curves,
            cluster_labels=cluster_names,
        )


if __name__ == "__main__":
    main()
