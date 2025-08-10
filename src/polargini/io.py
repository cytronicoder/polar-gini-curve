"""Input utilities for PGC."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import scipy.io

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_csv(
    path: str, x_col: str = "x", y_col: str = "y", label_col: str = "label"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load coordinates and labels from a CSV file."""
    df = pd.read_csv(path)
    points = df[[x_col, y_col]].to_numpy()
    labels = df[label_col].to_numpy()
    return points, labels


def load_mat_file(
    mat_path: str, variable_name: Optional[str] = None
) -> Union[np.ndarray, Dict]:
    """Load data from a MATLAB .mat file."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for loading .mat files. Install with: pip install scipy"
        )

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    try:
        mat_data = scipy.io.loadmat(mat_path)
        data_keys = [key for key in mat_data.keys() if not key.startswith("__")]

        if variable_name:
            if variable_name not in mat_data:
                raise KeyError(
                    f"Variable '{variable_name}' not found in {mat_path}. "
                    f"Available variables: {data_keys}"
                )
            return mat_data[variable_name]
        else:
            return {key: mat_data[key] for key in data_keys}

    except NotImplementedError as exc:
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for MATLAB v7.3 files. Install with: pip install h5py"
            ) from exc

        with h5py.File(mat_path, "r") as f:
            if variable_name:
                if variable_name not in f:
                    available = list(f.keys())
                    raise KeyError(
                        f"Variable '{variable_name}' not found in {mat_path}. "
                        f"Available variables: {available}"
                    ) from None
                return np.array(f[variable_name])
            else:
                return {key: np.array(f[key]) for key in f.keys()}


def load_legacy_dataset(
    legacy_dir: str,
    coordinate_file: str = "coordinate.mat",
    expression_file: str = "Expression.mat",
    gene_list_file: str = "geneList.mat",
    cluster_file: str = "ClusterID.mat",
) -> Dict[str, Union[np.ndarray, List[str]]]:
    """Load a complete legacy dataset from MATLAB files."""
    legacy_path = Path(legacy_dir)

    coord_path = legacy_path / coordinate_file
    coordinates = load_mat_file(str(coord_path))
    if isinstance(coordinates, dict):
        coord_key = list(coordinates.keys())[0]
        coordinates = coordinates[coord_key]

    expr_path = legacy_path / expression_file
    expression = load_mat_file(str(expr_path))
    if isinstance(expression, dict):
        expr_key = list(expression.keys())[0]
        expression = expression[expr_key]

    gene_path = legacy_path / gene_list_file
    gene_data = load_mat_file(str(gene_path))
    if isinstance(gene_data, dict):
        gene_key = list(gene_data.keys())[0]
        gene_data = gene_data[gene_key]

    if gene_data.dtype == "O":
        genes = []
        for gene in gene_data.flatten():
            if hasattr(gene, "__len__") and not isinstance(gene, str):
                genes.append(str(gene[0]) if len(gene) > 0 else "")
            else:
                genes.append(str(gene))
    else:
        genes = [str(gene) for gene in gene_data.flatten()]

    cluster_path = legacy_path / cluster_file
    clusters = load_mat_file(str(cluster_path))
    if isinstance(clusters, dict):
        cluster_key = list(clusters.keys())[0]
        clusters = clusters[cluster_key]
    clusters = clusters.flatten()

    return {
        "coordinates": coordinates,
        "expression": expression,
        "genes": genes,
        "clusters": clusters,
    }


def legacy_to_csv(
    legacy_dir: str,
    output_dir: str,
    coordinate_file: str = "coordinate.mat",
    expression_file: str = "Expression.mat",
    gene_list_file: str = "geneList.mat",
    cluster_file: str = "ClusterID.mat",
) -> None:
    """Convert legacy MATLAB dataset to CSV files."""
    data = load_legacy_dataset(
        legacy_dir, coordinate_file, expression_file, gene_list_file, cluster_file
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    coord_df = pd.DataFrame(data["coordinates"], columns=["x", "y"])
    coord_df["cluster"] = data["clusters"]
    coord_df.to_csv(output_path / "coordinates.csv", index=False)

    expr_df = pd.DataFrame(data["expression"], columns=data["genes"])
    expr_df["cluster"] = data["clusters"]
    expr_df.to_csv(output_path / "expression.csv", index=False)

    gene_df = pd.DataFrame({"gene": data["genes"]})
    gene_df.to_csv(output_path / "genes.csv", index=False)

    cluster_df = pd.DataFrame({"cluster": data["clusters"]})
    cluster_df.to_csv(output_path / "clusters.csv", index=False)

    print(f"Converted legacy dataset to CSV files in: {output_dir}")
    print("Files created: coordinates.csv, expression.csv, genes.csv, clusters.csv")


def load_legacy_for_pgc(
    legacy_dir: str,
    gene_name: Optional[str] = None,
    cluster_filter: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load legacy data formatted for polar gini curve analysis."""
    data = load_legacy_dataset(legacy_dir)
    coordinates = data["coordinates"]

    if gene_name:
        if gene_name not in data["genes"]:
            raise ValueError(
                f"Gene '{gene_name}' not found. Available genes: "
                f"{data['genes'][:10]}..."
            )

        gene_idx = data["genes"].index(gene_name)
        labels = data["expression"][:, gene_idx]
    else:
        labels = data["clusters"]

    if cluster_filter is not None:
        mask = np.isin(data["clusters"], cluster_filter)
        coordinates = coordinates[mask]
        labels = labels[mask]

        if -1 not in cluster_filter:
            valid_mask = labels != -1
            coordinates = coordinates[valid_mask]
            labels = labels[valid_mask]
    else:
        valid_mask = data["clusters"] != -1
        coordinates = coordinates[valid_mask]
        labels = labels[valid_mask]

    return coordinates, labels


def convert_rsmd_results(legacy_dir: str, output_dir: str) -> None:
    """Convert RSMD Excel results to CSV format."""
    legacy_path = Path(legacy_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rsmd_files = list(legacy_path.glob("RSMD_cluster*.xlsx"))

    for excel_file in rsmd_files:
        cluster_name = excel_file.stem
        try:
            df = pd.read_excel(excel_file)
            csv_path = output_path / f"{cluster_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Converted {excel_file.name} to {csv_path.name}")
        except (FileNotFoundError, PermissionError, ValueError, pd.errors.EmptyDataError) as e:
            print(f"Error converting {excel_file.name}: {e}")
