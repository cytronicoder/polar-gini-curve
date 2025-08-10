"""Input utilities for Polar Gini Curve."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def load_csv(
    path: str, x_col: str = "x", y_col: str = "y", label_col: str = "label"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load coordinates and labels from a CSV file."""
    df = pd.read_csv(path)
    points = df[[x_col, y_col]].to_numpy()
    labels = df[label_col].to_numpy()
    return points, labels
