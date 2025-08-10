"""Top-level package for PGC."""

from .pgc import polar_gini_curve
from .metrics import gini, rmsd
from .plotting import plot_pgc
from .io import load_csv

__all__ = [
    "polar_gini_curve",
    "gini",
    "rmsd",
    "plot_pgc",
    "load_csv",
]

__version__ = "0.1.0"
