"""
Loading and converting legacy datasets from the original MATLAB
implementation to formats usable with the new Python package.
"""

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

script_dir = Path(__file__).parent.parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from polargini import (
        convert_rsmd_results,
        legacy_to_csv,
        load_legacy_dataset,
        load_legacy_for_pgc,
        polar_gini_curve,
    )
    from polargini.plotting import plot_pgc
except ImportError as e:
    print(f"Error importing polargini: {e}")
    print("Please install with: pip install -e .[legacy]")
    print(f"Script directory: {script_dir}")
    print(f"Source directory: {src_dir}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def convert_full_dataset(legacy_dir: str, output_dir: str) -> None:
    """Convert a complete legacy dataset to CSV files."""
    print(f"Converting legacy dataset from {legacy_dir} to {output_dir}")

    try:
        legacy_to_csv(legacy_dir, output_dir)
        print("✓ Main dataset converted successfully")

        rsmd_output = Path(output_dir) / "rsmd_results"
        convert_rsmd_results(legacy_dir, str(rsmd_output))
        print("✓ RSMD results converted successfully")

    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        print(f"✗ Conversion failed: {e}")
        return

    print(f"\nConversion complete! Files saved to: {output_dir}")
    print("You can now use the CSV files with polargini.load_csv()")


def analyze_gene_expression(
    legacy_dir: str, gene_name: str, output_file: str = None
) -> None:
    """Analyze a specific gene using PGC."""
    print(f"Analyzing gene '{gene_name}' from {legacy_dir}")

    try:
        coordinates, expression = load_legacy_for_pgc(legacy_dir, gene_name=gene_name)
        print(f"✓ Loaded {len(coordinates)} cells")
        print(f"  Expression range: {expression.min():.3f} - {expression.max():.3f}")

        median_expr = np.median(expression)
        binary_labels = (expression > median_expr).astype(int)

        unique_labels = np.unique(binary_labels)
        if len(unique_labels) < 2:
            print("⚠ Warning: All cells have same expression level, cannot compute PGC")
            return

        angles, curves = polar_gini_curve(coordinates, binary_labels)
        print("✓ Computed PGC")

        if len(curves) > 0:
            mean_gini = np.mean([np.mean(curve) for curve in curves])
            min_gini = np.min([np.min(curve) for curve in curves])
            avg_curve = np.mean(curves, axis=0)
            optimal_angle_idx = np.argmin(avg_curve)
            optimal_angle = np.degrees(angles[optimal_angle_idx])

            print(f"  Mean Gini: {mean_gini:.4f}")
            print(f"  Min Gini: {min_gini:.4f}")
            print(f"  Optimal angle: {optimal_angle:.2f}°")

        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(
            coordinates[:, 0], coordinates[:, 1], c=expression, cmap="viridis", s=20
        )
        ax1.set_title(f"Spatial Expression: {gene_name}")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        plt.colorbar(scatter, ax=ax1, label="Expression")

        ax2 = fig.add_subplot(122, projection="polar")
        plot_pgc(angles, *curves, ax=ax2, labels=["High Expression", "Low Expression"])
        ax2.set_title(f"PGC: {gene_name} (High vs Low)", pad=20)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"✓ Plot saved to {output_file}")
        else:
            plt.show()

    except (FileNotFoundError, KeyError, ValueError, ImportError) as e:
        print(f"✗ Analysis failed: {e}")
        traceback.print_exc()


def analyze_clusters(
    legacy_dir: str, clusters: list = None, output_file: str = None
) -> None:
    """Analyze cluster spatial distribution using PGC."""
    print(f"Analyzing cluster distribution from {legacy_dir}")

    try:
        if clusters:
            coordinates, labels = load_legacy_for_pgc(
                legacy_dir, cluster_filter=clusters
            )
            print(f"✓ Loaded {len(coordinates)} cells from clusters {clusters}")
        else:
            coordinates, labels = load_legacy_for_pgc(legacy_dir)
            unique_clusters = np.unique(labels[labels != -1])
            print(f"✓ Loaded {len(coordinates)} cells from all clusters")
            print(f"  Clusters found: {unique_clusters}")

        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            print("⚠ Warning: Need at least 2 clusters for PGC analysis")
            return
        elif len(unique_labels) == 2:
            pgc_labels = labels
            cluster1, cluster2 = unique_labels
            print(f"  Comparing cluster {cluster1} vs cluster {cluster2}")
        else:
            cluster_counts = {c: np.sum(labels == c) for c in unique_labels}
            sorted_clusters = sorted(
                cluster_counts.items(), key=lambda x: x[1], reverse=True
            )
            cluster1, cluster2 = sorted_clusters[0][0], sorted_clusters[1][0]

            mask = (labels == cluster1) | (labels == cluster2)
            coordinates = coordinates[mask]
            pgc_labels = labels[mask]

            print(
                f"  Comparing largest clusters: {cluster1} "
                f"({cluster_counts[cluster1]} cells) vs {cluster2} "
                f"({cluster_counts[cluster2]} cells)"
            )

        angles, curves = polar_gini_curve(coordinates, pgc_labels)
        print("✓ Computed PGC")

        if len(curves) > 0:
            mean_gini = np.mean([np.mean(curve) for curve in curves])
            min_gini = np.min([np.min(curve) for curve in curves])
            avg_curve = np.mean(curves, axis=0)
            optimal_angle_idx = np.argmin(avg_curve)
            optimal_angle = np.degrees(angles[optimal_angle_idx])

            print(f"  Mean Gini: {mean_gini:.4f}")
            print(f"  Min Gini: {min_gini:.4f}")
            print(f"  Optimal angle: {optimal_angle:.2f}°")

        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(121)
        if len(unique_labels) > 2:
            full_coords, full_labels = load_legacy_for_pgc(legacy_dir)
            plot_coords = full_coords
            plot_labels = full_labels
            all_unique_labels = np.unique(full_labels)
        else:
            plot_coords = coordinates
            plot_labels = pgc_labels
            all_unique_labels = unique_labels

        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(all_unique_labels)))

        for i, cluster_id in enumerate(all_unique_labels):
            mask = plot_labels == cluster_id
            alpha = 1.0 if cluster_id in [cluster1, cluster2] else 0.3
            ax1.scatter(
                plot_coords[mask, 0],
                plot_coords[mask, 1],
                c=[colors[i]],
                label=f"Cluster {cluster_id}",
                s=20,
                alpha=alpha,
            )

        ax1.set_title("Spatial Cluster Distribution")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        ax2 = fig.add_subplot(122, projection="polar")
        plot_pgc(
            angles,
            *curves,
            ax=ax2,
            labels=[f"Cluster {cluster1}", f"Cluster {cluster2}"],
        )
        ax2.set_title(f"PGC: Cluster {cluster1} vs {cluster2}", pad=20)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"✓ Plot saved to {output_file}")
        else:
            plt.show()
    except (FileNotFoundError, KeyError, ValueError, ImportError) as e:
        print(f"✗ Analysis failed: {e}")
        traceback.print_exc()


def inspect_dataset(legacy_dir: str) -> None:
    """Inspect the contents of a legacy dataset."""
    print(f"Inspecting legacy dataset: {legacy_dir}")

    try:
        data = load_legacy_dataset(legacy_dir)

        print("\nDataset Summary:")
        print(f"  Coordinates shape: {data['coordinates'].shape}")
        print(f"  Expression shape: {data['expression'].shape}")
        print(f"  Number of genes: {len(data['genes'])}")
        print(f"  Number of cells: {len(data['clusters'])}")

        unique_clusters = np.unique(data["clusters"])
        cluster_counts = {c: np.sum(data["clusters"] == c) for c in unique_clusters}
        print("\nCluster Information:")
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                print(f"  Non-clusterable cells: {count}")
            else:
                print(f"  Cluster {cluster_id}: {count} cells")

        print("\nGene Expression Statistics:")
        mean_expr = np.mean(data["expression"], axis=0)
        max_expr_gene_idx = np.argmax(mean_expr)
        min_expr_gene_idx = np.argmin(mean_expr)

        print(
            f"  Expression range: {data['expression'].min():.3f} - "
            f"{data['expression'].max():.3f}"
        )
        print(
            f"  Highest expressing gene: {data['genes'][max_expr_gene_idx]} "
            f"(mean: {mean_expr[max_expr_gene_idx]:.3f})"
        )
        print(
            f"  Lowest expressing gene: {data['genes'][min_expr_gene_idx]} "
            f"(mean: {mean_expr[min_expr_gene_idx]:.3f})"
        )

        print("\nFirst 10 genes:")
        for i, gene in enumerate(data["genes"][:10]):
            print(f"  {i+1:2d}. {gene}")
        if len(data["genes"]) > 10:
            print(f"  ... and {len(data['genes']) - 10} more")

    except (FileNotFoundError, KeyError, ValueError, ImportError, OSError) as e:
        print(f"✗ Inspection failed: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Convert and analyze legacy MATLAB datasets with polargini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a legacy dataset
  python legacy_converter.py inspect legacy/

  # Convert full dataset to CSV
  python legacy_converter.py convert legacy/ output/

  # Analyze specific gene
  python legacy_converter.py gene legacy/ Actc1 --output gene_analysis.png

  # Analyze specific clusters
  python legacy_converter.py clusters legacy/ --clusters 1 2 3 \\
    --output cluster_analysis.png

  # Analyze all clusters
  python legacy_converter.py clusters legacy/ --output all_clusters.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    inspect_parser = subparsers.add_parser("inspect", help="Inspect legacy dataset")
    inspect_parser.add_argument("legacy_dir", help="Path to legacy dataset directory")
    convert_parser = subparsers.add_parser(
        "convert", help="Convert legacy dataset to CSV"
    )
    convert_parser.add_argument("legacy_dir", help="Path to legacy dataset directory")
    convert_parser.add_argument("output_dir", help="Output directory for CSV files")
    gene_parser = subparsers.add_parser("gene", help="Analyze gene expression")
    gene_parser.add_argument("legacy_dir", help="Path to legacy dataset directory")
    gene_parser.add_argument("gene_name", help="Name of gene to analyze")
    gene_parser.add_argument("--output", "-o", help="Output file for plot")
    cluster_parser = subparsers.add_parser(
        "clusters", help="Analyze cluster distribution"
    )
    cluster_parser.add_argument("legacy_dir", help="Path to legacy dataset directory")
    cluster_parser.add_argument(
        "--clusters", nargs="+", type=int, help="Specific clusters to analyze"
    )
    cluster_parser.add_argument("--output", "-o", help="Output file for plot")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "inspect":
        inspect_dataset(args.legacy_dir)
    elif args.command == "convert":
        convert_full_dataset(args.legacy_dir, args.output_dir)
    elif args.command == "gene":
        analyze_gene_expression(args.legacy_dir, args.gene_name, args.output)
    elif args.command == "clusters":
        analyze_clusters(args.legacy_dir, args.clusters, args.output)


if __name__ == "__main__":
    main()
