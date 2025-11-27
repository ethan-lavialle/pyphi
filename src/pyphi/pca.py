"""
Principal Component Analysis (PCA) module for PyPhi.

This minimal version contains only the hott2 function required
for JRPLS/TPLS diagnostics.
"""

from __future__ import annotations

import numpy as np


def hott2(
    mvmobj: dict,
    *,
    Xnew: np.ndarray | bool = False,
    Tnew: np.ndarray | bool = False,
) -> np.ndarray:
    """Calculate Hotelling's T² statistic.

    Parameters
    ----------
    mvmobj : dict
        PCA or PLS model object containing 'T' (scores matrix).
    Xnew : np.ndarray or False, default=False
        New X data to project and calculate T².
        Not supported in this minimal version.
    Tnew : np.ndarray or False, default=False
        Pre-calculated scores for new data.

    Returns
    -------
    np.ndarray
        Hotelling's T² values for each observation.

    Notes
    -----
    In this minimal version, only Tnew is supported. If neither Xnew nor Tnew
    is provided, T² is calculated for the training data scores in mvmobj['T'].
    """
    if isinstance(Xnew, bool) and not isinstance(Tnew, bool):
        # Use provided Tnew
        var_t = (mvmobj["T"].T @ mvmobj["T"]) / mvmobj["T"].shape[0]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew, axis=1)
    elif isinstance(Tnew, bool) and not isinstance(Xnew, bool):
        # Project Xnew - not supported in minimal version
        raise NotImplementedError(
            "Xnew projection not supported in minimal version. Use Tnew instead."
        )
    elif isinstance(Xnew, bool) and isinstance(Tnew, bool):
        # Use training data
        var_t = (mvmobj["T"].T @ mvmobj["T"]) / mvmobj["T"].shape[0]
        Tnew = mvmobj["T"]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew, axis=1)
    return hott2_
