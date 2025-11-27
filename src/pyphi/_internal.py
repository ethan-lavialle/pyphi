"""
Private helper functions for PyPhi.

This module contains internal implementation details that are not part
of the public API. These functions may change without notice between
minor versions.

Functions prefixed with underscore (_) are strictly internal.
"""

from __future__ import annotations

import numpy as np


def _Ab_btbinv(A: np.ndarray, b: np.ndarray, A_not_nan_map: np.ndarray) -> np.ndarray:
    """Project A onto b with missing data handling: c = Ab / (b'b).

    Computes the projection coefficient for each row of A onto vector b,
    accounting for missing data indicated by A_not_nan_map.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (i, j) to project.
    b : np.ndarray
        Vector of shape (j, 1) to project onto.
    A_not_nan_map : np.ndarray
        Binary matrix of shape (i, j) where 1 indicates non-missing data.

    Returns
    -------
    np.ndarray
        Column vector of shape (i, 1) with projection coefficients.
    """
    b_mat = np.tile(b.T, (A.shape[0], 1))
    c = (np.sum(A * b_mat, axis=1)) / (np.sum((b_mat * A_not_nan_map) ** 2, axis=1))
    return c.reshape(-1, 1)
