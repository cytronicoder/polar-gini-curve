"""Statistical utilities for PGC."""

from __future__ import annotations

from math import erf, sqrt
from typing import Tuple

import numpy as np


def compute_pvalues(
    rmsd_matrix: np.ndarray, expression_percent: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize RMSD scores and compute p-values.

    Parameters
    ----------
    rmsd_matrix:
        Matrix of raw RMSD scores where rows correspond to genes and columns to
        clusters.
    expression_percent:
        Percentage of cells expressing each gene in each cluster. Only entries
        greater than ``0.1`` are considered when computing statistics.

    Returns
    -------
    normalized:
        RMSD scores normalized by the maximum score of each cluster.
    p_values:
        Normal-distribution based p-values matching the shape of ``rmsd_matrix``.
    """
    rmsd = np.asarray(rmsd_matrix, dtype=float)
    perc = np.asarray(expression_percent, dtype=float)
    if rmsd.shape != perc.shape:
        raise ValueError("Input matrices must have the same shape")

    normalized = np.ones_like(rmsd)
    p_values = np.ones_like(rmsd)

    for j in range(rmsd.shape[1]):
        all_cluster = rmsd[:, j]
        index = np.where(perc[:, j] > 0.1)[0]
        if index.size == 0:
            continue
        values = all_cluster[index]
        max_score = values.max()
        normalized[index, j] = all_cluster[index] / max_score
        values = values / max_score
        mu = values.mean()
        sigma = values.std()
        if sigma == 0:
            p = np.where(values >= mu, 1.0, 0.0)
        else:
            z = (values - mu) / (sigma * sqrt(2.0))
            p = 0.5 * (1.0 + np.vectorize(erf)(z))
        p_values[index, j] = p

    return normalized, p_values
