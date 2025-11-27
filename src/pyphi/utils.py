"""Utility helpers for PyPhi.

This module provides core utility functions used throughout PyPhi.
Functions have been simplified to use standard library equivalents
from NumPy and SciPy where possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist


def mean(X: np.ndarray) -> np.ndarray:
    """Column-wise mean that ignores NaN values.

    Parameters
    ----------
    X : np.ndarray
        Input array (2D).

    Returns
    -------
    np.ndarray
        Row vector of column means with shape (1, n_cols).
    """
    return np.nanmean(X, axis=0, keepdims=True)


def std(X: np.ndarray) -> np.ndarray:
    """Column-wise sample standard deviation ignoring NaN values.

    Uses Bessel's correction (ddof=1) for sample standard deviation.

    Parameters
    ----------
    X : np.ndarray
        Input array (2D).

    Returns
    -------
    np.ndarray
        Row vector of column standard deviations with shape (1, n_cols).
    """
    return np.nanstd(X, axis=0, keepdims=True, ddof=1)


def meancenterscale(
    X: np.ndarray, *, mcs: bool | str = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-center and/or scale a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix (2D).
    mcs : bool or str, default=True
        Scaling mode:
        - True: mean-center and autoscale (standardize)
        - False: no transformation
        - "center": mean-center only
        - "autoscale": scale by std only (no centering)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - X_proc: Transformed data matrix
        - x_mean: Mean values used (or np.nan if not centered)
        - x_std: Std values used (or np.nan if not scaled)
    """
    if isinstance(mcs, bool):
        if mcs:
            x_mean = np.nanmean(X, axis=0, keepdims=True)
            x_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
            X_proc = (X - x_mean) / x_std
        else:
            X_proc = X
            x_mean = np.nan
            x_std = np.nan
    elif mcs == "center":
        x_mean = np.nanmean(X, axis=0, keepdims=True)
        x_std = np.ones((1, X.shape[1]))
        X_proc = X - x_mean
    elif mcs == "autoscale":
        x_mean = np.zeros((1, X.shape[1]))
        x_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
        X_proc = X / x_std
    else:
        X_proc = X
        x_mean = np.nan
        x_std = np.nan
    return X_proc, x_mean, x_std


def f95(dfn: float, dfd: float) -> float:
    """F-distribution 95th percentile (critical value for alpha=0.05).

    Parameters
    ----------
    dfn : float
        Numerator degrees of freedom.
    dfd : float
        Denominator degrees of freedom.

    Returns
    -------
    float
        Critical value at 95th percentile.
    """
    return f_dist.ppf(0.95, dfn, dfd)


def f99(dfn: float, dfd: float) -> float:
    """F-distribution 99th percentile (critical value for alpha=0.01).

    Parameters
    ----------
    dfn : float
        Numerator degrees of freedom.
    dfd : float
        Denominator degrees of freedom.

    Returns
    -------
    float
        Critical value at 99th percentile.
    """
    return f_dist.ppf(0.99, dfn, dfd)


# =============================================================================
# NaN Utilities
# =============================================================================


def n2z(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert NaN values to zeros for computation.

    This function replaces NaN values with zeros and returns a map
    of where the NaN values were located.

    Parameters
    ----------
    X : np.ndarray
        Input array that may contain NaN values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - X: Modified array with NaN replaced by zeros (modified in place)
        - X_nan_map: Binary array where 1 indicates original NaN positions
    """
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map = X_nan_map.astype(int)
        X[X_nan_map == 1] = 0
    else:
        X_nan_map = X_nan_map.astype(int)
    return X, X_nan_map


# =============================================================================
# Statistical Confidence Intervals
# =============================================================================


def spe_ci(spe: np.ndarray) -> tuple[float, float]:
    """Calculate SPE (Squared Prediction Error) confidence intervals.

    Uses the chi-squared distribution approximation for SPE limits
    based on Box's method.

    Parameters
    ----------
    spe : np.ndarray
        Array of SPE values from observations.

    Returns
    -------
    tuple[float, float]
        - lim95: 95% confidence limit
        - lim99: 99% confidence limit
    """
    spe_mean = np.mean(spe)
    if spe_mean > 1e-16:
        spe_var = np.var(spe, ddof=1)
        g = spe_var / (2 * spe_mean)
        h = (2 * spe_mean**2) / spe_var
        lim95 = g * chi2.ppf(0.95, h)
        lim99 = g * chi2.ppf(0.99, h)
    else:
        lim95 = 0.0
        lim99 = 0.0
    return lim95, lim99


# =============================================================================
# Array Utilities
# =============================================================================


def unique(df: pd.DataFrame, colid: str) -> list:
    """Return unique values in a DataFrame column in order of occurrence.

    Unlike `np.unique()`, this preserves the order in which values
    first appear in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    colid : str
        Column identifier to extract unique values from.

    Returns
    -------
    list
        List of unique values in order of first occurrence.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 1, 3, 2]})
    >>> unique(df, 'A')
    [1, 2, 3]
    """
    return pd.unique(df[colid]).tolist()


# =============================================================================
# DataFrame Row/Column Reconciliation
# =============================================================================


def isin_ordered_col0(df: pd.DataFrame, alist: list) -> pd.DataFrame:
    """Filter and reorder DataFrame rows to match a list, using first column.

    Filters the DataFrame to include only rows where the first column
    value is in the provided list, then reorders rows to match the
    list order.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where first column contains identifiers.
    alist : list
        List of values to filter by and order by.

    Returns
    -------
    pd.DataFrame
        Filtered and reordered DataFrame.
    """
    df_ = df.copy()
    df_ = df_[df_[df_.columns[0]].isin(alist)]
    df_ = df_.set_index(df_.columns[0])
    df_ = df_.reindex(alist)
    df_ = df_.reset_index()
    return df_


def reconcile_rows(df_list: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Reconcile observations across multiple DataFrames.

    Returns a list of DataFrames that all have exactly the same
    observation names (from first column) in the same order.
    Only observations present in ALL DataFrames are kept.

    Useful when analyzing data for TPLS, LPLS, and JRPLS models.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        List of DataFrames to reconcile. Each DataFrame should have
        observation IDs in the first column.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames with reconciled rows.
    """
    # Collect all row identifiers
    all_rows = []
    for df in df_list:
        all_rows.extend(df[df.columns[0]].values.tolist())
    all_rows = np.unique(all_rows)

    # Keep only rows present in ALL DataFrames
    for df in df_list:
        rows = df[df.columns[0]].values.tolist()
        rows_ = []
        for r in all_rows:
            if r in rows:
                rows_.append(r)
        all_rows = rows_.copy()

    # Reorder each DataFrame to match the common rows
    new_df_list = []
    for df in df_list:
        df_ordered = isin_ordered_col0(df, all_rows)
        new_df_list.append(df_ordered)

    return new_df_list


def reconcile_rows_to_columns(
    df_list_r: list[pd.DataFrame], df_list_c: list[pd.DataFrame]
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Reconcile rows of one DataFrame list with columns of another.

    For each pair of DataFrames, aligns the row identifiers in df_list_r
    with the column identifiers in df_list_c. Only keeps identifiers
    present in both rows AND columns.

    Useful to align X-R datasets for TPLS, LPLS, and JRPLS models.

    Parameters
    ----------
    df_list_r : list[pd.DataFrame]
        List of DataFrames with row identifiers in first column.
    df_list_c : list[pd.DataFrame]
        List of DataFrames with column identifiers to align with.

    Returns
    -------
    tuple[list[pd.DataFrame], list[pd.DataFrame]]
        - df_list_r_out: Reconciled row DataFrames
        - df_list_c_out: Reconciled column DataFrames
    """
    df_list_r_o = []
    df_list_c_o = []

    for dfr, dfc in zip(df_list_r, df_list_c):
        # Get all identifiers from both sources
        all_ids = dfc.columns[1:].tolist()
        all_ids.extend(dfr[dfr.columns[0]].values.tolist())
        all_ids = np.unique(all_ids)

        # Keep only IDs present in rows
        rows = dfr[dfr.columns[0]].values.tolist()
        cols = dfc.columns[1:].tolist()

        all_ids_ = []
        for i in all_ids:
            if i in rows:
                all_ids_.append(i)

        # Keep only IDs present in columns
        all_ids_final = []
        for i in all_ids_:
            if i in cols:
                all_ids_final.append(i)

        # Reorder row DataFrame
        dfr_ = isin_ordered_col0(dfr, all_ids_final)

        # Reorder column DataFrame
        dfc_ = dfc[all_ids_final].copy()
        dfc_.insert(0, dfc.columns[0], dfc[dfc.columns[0]].values.tolist())

        df_list_r_o.append(dfr_)
        df_list_c_o.append(dfc_)

    return df_list_r_o, df_list_c_o


# =============================================================================
# JRPLS/TPLS Data Parsing
# =============================================================================


def parse_materials(
    filename: str, sheetname: str
) -> tuple[list[pd.DataFrame], list[str]]:
    """Build R matrices for JRPLS model from linear table.

    Routine to parse out compositions from linear table.
    This reads an excel file with four columns:
        'Finished Product Lot', 'Material Lot', 'Ratio or Quantity', 'Material'

    where the usage per batch of finished product is recorded. e.g.

    Finished Product Lot | Material Lot | Ratio or Quantity | Material
    A001                 | A            | 0.75              | Drug
    A001                 | B            | 0.25              | Drug
    A001                 | Z            | 1.0               | Excipient

    Parameters
    ----------
    filename : str
        Name of excel workbook containing the data.
    sheetname : str
        Name of the sheet in the workbook with the data.

    Returns
    -------
    tuple[list[pd.DataFrame], list[str]]
        - JR: Joint R matrix of material consumption, list of dataframes
        - materials_used: Names of materials
    """
    materials = pd.read_excel(filename, sheet_name=sheetname)

    ok = True
    for lot in unique(materials, "Finished Product Lot"):
        this_lot = materials[materials["Finished Product Lot"] == lot]
        for mt, m in zip(
            this_lot["Material"].values, this_lot["Material Lot"].values
        ):
            try:
                if np.isnan(m):
                    print("Lot " + lot + " has no Material Lot for " + mt)
                    ok = False
                    break
            except:
                d = 1
        if not (ok):
            break
        print(
            "Lot :"
            + lot
            + " ratio/qty adds to "
            + str(np.sum(this_lot["Ratio or Quantity"].values))
        )

    if ok:
        JR = []
        materials_used = unique(materials, "Material")
        fp_lots = unique(materials, "Finished Product Lot")
        for m in materials_used:
            r_mat = []
            mat_lots = np.unique(
                materials["Material Lot"][materials["Material"] == m]
            ).tolist()
            for lot in fp_lots:
                rvec = np.zeros(len(mat_lots))
                this_lot_this_mat = materials[
                    (materials["Finished Product Lot"] == lot)
                    & (materials["Material"] == m)
                ]
                for l, r in zip(
                    this_lot_this_mat["Material Lot"].values,
                    this_lot_this_mat["Ratio or Quantity"].values,
                ):
                    rvec[mat_lots.index(l)] = r
                r_mat.append(rvec)
            r_mat_pd = pd.DataFrame(np.array(r_mat), columns=mat_lots)
            r_mat_pd.insert(0, "FPLot", fp_lots)
            JR.append(r_mat_pd)
        return JR, materials_used
