"""Top-level package for PGC."""

from .pgc import polar_gini_curve
from .metrics import gini, rmsd
from .stats import compute_pvalues
from .plotting import plot_pgc, plot_embedding_and_pgc
from .io import (
    load_csv,
    load_legacy_dataset,
    legacy_to_csv,
    load_legacy_for_pgc,
    convert_rsmd_results,
)

__all__ = [
    "polar_gini_curve",
    "gini",
    "rmsd",
    "compute_pvalues",
    "plot_pgc",
    "plot_embedding_and_pgc",
    "load_csv",
    "load_legacy_dataset",
    "legacy_to_csv",
    "load_legacy_for_pgc",
    "convert_rsmd_results",
]

__version__ = "0.1.0"
