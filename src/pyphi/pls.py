"""
Partial Least Squares (PLS) module for PyPhi.

This minimal version implements PLS regression with NIPALS algorithm only,
as required for JRPLS/TPLS models.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from .utils import (
    meancenterscale,
    n2z,
    std,
    f95,
    f99,
    spe_ci,
)
from .pca import hott2


def pls(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcsX: bool | str = True,
    mcsY: bool | str = True,
    force_nipals: bool = True,
    shush: bool = False,
) -> dict:
    """Create a Projection to Latent Structures (PLS) model.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Predictor data. If DataFrame, first column is observation IDs.
    Y : np.ndarray or pd.DataFrame
        Response data. If DataFrame, first column is observation IDs.
    A : int
        Number of Latent Variables to calculate.
    mcsX : bool or str, default=True
        X preprocessing: True (center+scale), False (none), 'center', 'autoscale'.
    mcsY : bool or str, default=True
        Y preprocessing: True (center+scale), False (none), 'center', 'autoscale'.
    force_nipals : bool, default=True
        Force NIPALS even for complete data.
    shush : bool, default=False
        Suppress output.

    Returns
    -------
    dict
        PLS model object containing:
        - T, P, Q, W, Ws, U: Model matrices
        - r2x, r2y, r2xpv, r2ypv: R² statistics
        - mx, sx, my, sy: Preprocessing parameters
        - T2, speX, speY: Diagnostics with confidence limits
        - type: "pls"

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 10)
    >>> Y = np.random.randn(50, 2)
    >>> model = pls(X, Y, 3)
    >>> model['T'].shape
    (50, 3)
    """
    plsobj = pls_(
        X, Y, A,
        mcsX=mcsX, mcsY=mcsY,
        force_nipals=force_nipals,
        shush=shush,
    )
    plsobj["type"] = "pls"
    return plsobj


def pls_(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcsX: bool | str = True,
    mcsY: bool | str = True,
    force_nipals: bool = True,
    shush: bool = False,
) -> dict:
    """Core PLS algorithm implementation.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Predictor data matrix.
    Y : np.ndarray or pd.DataFrame
        Response data matrix.
    A : int
        Number of Latent Variables to calculate.
    mcsX : bool or str, default=True
        X preprocessing mode.
    mcsY : bool or str, default=True
        Y preprocessing mode.
    force_nipals : bool, default=True
        Force NIPALS even for complete data.
    shush : bool, default=False
        Suppress output.

    Returns
    -------
    dict
        PLS model object.
    """
    # Extract X data and identifiers
    if isinstance(X, np.ndarray):
        X_ = X.copy()
        obsidX = False
        varidX = False
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        obsidX = X.values[:, 0].astype(str).tolist()
        varidX = X.columns.values[1:].tolist()

    # Extract Y data and identifiers
    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)
        obsidY = Y.values[:, 0].astype(str).tolist()
        varidY = Y.columns.values[1:].tolist()

    # Preprocess X
    if isinstance(mcsX, bool):
        if mcsX:
            X_, x_mean, x_std = meancenterscale(X_)
        else:
            x_mean = np.zeros((1, X_.shape[1]))
            x_std = np.ones((1, X_.shape[1]))
    elif mcsX == "center":
        X_, x_mean, x_std = meancenterscale(X_, mcs="center")
    elif mcsX == "autoscale":
        X_, x_mean, x_std = meancenterscale(X_, mcs="autoscale")

    # Preprocess Y
    if isinstance(mcsY, bool):
        if mcsY:
            Y_, y_mean, y_std = meancenterscale(Y_)
        else:
            y_mean = np.zeros((1, Y_.shape[1]))
            y_std = np.ones((1, Y_.shape[1]))
    elif mcsY == "center":
        Y_, y_mean, y_std = meancenterscale(Y_, mcs="center")
    elif mcsY == "autoscale":
        Y_, y_mean, y_std = meancenterscale(Y_, mcs="autoscale")

    # Generate missing data maps
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map).astype(int)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (~Y_nan_map).astype(int)

    # Use NIPALS algorithm
    return _pls_nipals(
        X_, Y_, A, x_mean, x_std, y_mean, y_std,
        not_Xmiss, not_Ymiss, obsidX, varidX, obsidY, varidY, shush
    )


def _pls_nipals(
    X_: np.ndarray,
    Y_: np.ndarray,
    A: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    not_Xmiss: np.ndarray,
    not_Ymiss: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    obsidY: list | bool,
    varidY: list | bool,
    shush: bool,
) -> dict:
    """PLS using NIPALS algorithm (handles missing data)."""
    if not shush:
        print(f"phi.pls using NIPALS executed on: {datetime.datetime.now()}")

    X_, dummy = n2z(X_.copy())
    Y_, dummy = n2z(Y_.copy())
    epsilon = 1e-9
    maxit = 2000

    TSSX = np.sum(X_**2)
    TSSXpv = np.sum(X_**2, axis=0)
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)

    for a in range(A):
        # Initial guess: column with largest variance in Y
        ui = Y_[:, [np.argmax(std(Y_))]]
        converged = False
        num_it = 0

        while not converged:
            # Step 1: w = X'u / u'u
            uimat = np.tile(ui, (1, X_.shape[1]))
            wi = np.sum(X_ * uimat, axis=0) / np.sum((uimat * not_Xmiss) ** 2, axis=0)

            # Step 2: Normalize w
            wi = wi / np.linalg.norm(wi)

            # Step 3: t = Xw / w'w
            wimat = np.tile(wi, (X_.shape[0], 1))
            ti = X_ @ wi.T
            wtw = np.sum((wimat * not_Xmiss) ** 2, axis=1)
            ti = ti / wtw
            ti = ti.reshape(-1, 1)
            wi = wi.reshape(-1, 1)

            # Step 4: q = Y't / t't
            timat = np.tile(ti, (1, Y_.shape[1]))
            qi = np.sum(Y_ * timat, axis=0) / np.sum((timat * not_Ymiss) ** 2, axis=0)

            # Step 5: u_new = Yq / q'q
            qimat = np.tile(qi, (Y_.shape[0], 1))
            qi = qi.reshape(-1, 1)
            un = Y_ @ qi
            qtq = np.sum((qimat * not_Ymiss) ** 2, axis=1)
            qtq = qtq.reshape(-1, 1)
            un = un / qtq
            un = un.reshape(-1, 1)

            if abs(np.linalg.norm(ui) - np.linalg.norm(un)) / np.linalg.norm(ui) < epsilon:
                converged = True
            if num_it > maxit:
                converged = True

            if converged:
                if np.var(ti[ti < 0]) > np.var(ti[ti >= 0]):
                    ti = -ti
                    wi = -wi
                    un = -un
                    qi = -qi

                if not shush:
                    print(f"# Iterations for LV #{a + 1}: {num_it}")

                # Calculate P for deflation
                timat = np.tile(ti, (1, X_.shape[1]))
                pi = np.sum(X_ * timat, axis=0) / np.sum((timat * not_Xmiss) ** 2, axis=0)
                pi = pi.reshape(-1, 1)

                # Deflate
                X_ = (X_ - ti @ pi.T) * not_Xmiss
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss

                if a == 0:
                    T = ti
                    P = pi
                    W = wi
                    U = un
                    Q = qi
                    r2X = 1 - np.sum(X_**2) / TSSX
                    r2Xpv = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                    r2Y = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
                else:
                    T = np.hstack((T, ti.reshape(-1, 1)))
                    U = np.hstack((U, un.reshape(-1, 1)))
                    P = np.hstack((P, pi))
                    W = np.hstack((W, wi))
                    Q = np.hstack((Q, qi))

                    r2X_ = 1 - np.sum(X_**2) / TSSX
                    r2Xpv_ = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                    r2X = np.hstack((r2X, r2X_))
                    r2Xpv = np.hstack((r2Xpv, r2Xpv_))

                    r2Y_ = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv_ = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
                    r2Y = np.hstack((r2Y, r2Y_))
                    r2Ypv = np.hstack((r2Ypv, r2Ypv_))
            else:
                num_it += 1
                ui = un

        if a == 0:
            numIT = num_it
        else:
            numIT = np.hstack((numIT, num_it))

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2X[a] = r2X[a] - r2X[a - 1]
        r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]

    Ws = W @ np.linalg.pinv(P.T @ W)
    Ws[:, 0] = W[:, 0]

    eigs = np.var(T, axis=0)
    r2xc = np.cumsum(r2X)
    r2yc = np.cumsum(r2Y)

    if not shush:
        print("-" * 62)
        print("LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)")
        if A > 1:
            for a in range(A):
                print(
                    f"LV #{a + 1}:   {eigs[a]:6.3f}    {r2X[a]:.3f}     "
                    f"{r2xc[a]:.3f}      {r2Y[a]:.3f}     {r2yc[a]:.3f}"
                )
        else:
            print(
                f"LV #1:   {eigs[0]:6.3f}    {r2X:.3f}     "
                f"{r2xc[0]:.3f}      {r2Y:.3f}     {r2yc[0]:.3f}"
            )
        print("-" * 62)

    pls_obj = {
        "T": T, "P": P, "Q": Q, "W": W, "Ws": Ws, "U": U,
        "r2x": r2X, "r2xpv": r2Xpv, "mx": x_mean, "sx": x_std,
        "r2y": r2Y, "r2ypv": r2Ypv, "my": y_mean, "sy": y_std,
    }

    if not isinstance(obsidX, bool):
        pls_obj["obsidX"] = obsidX
        pls_obj["varidX"] = varidX
    if not isinstance(obsidY, bool):
        pls_obj["obsidY"] = obsidY
        pls_obj["varidY"] = varidY

    # Add diagnostics
    _add_pls_diagnostics(pls_obj, T, X_, Y_, A)

    return pls_obj


def _add_pls_diagnostics(
    pls_obj: dict,
    T: np.ndarray,
    X_residual: np.ndarray,
    Y_residual: np.ndarray,
    A: int,
) -> None:
    """Add diagnostic statistics to PLS object."""
    T2 = hott2(pls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f99(A, n - A)
    T2_lim95 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f95(A, n - A)

    speX = np.sum(X_residual**2, axis=1, keepdims=True)
    speX_lim95, speX_lim99 = spe_ci(speX)
    speY = np.sum(Y_residual**2, axis=1, keepdims=True)
    speY_lim95, speY_lim99 = spe_ci(speY)

    pls_obj["T2"] = T2
    pls_obj["T2_lim99"] = T2_lim99
    pls_obj["T2_lim95"] = T2_lim95
    pls_obj["speX"] = speX
    pls_obj["speX_lim99"] = speX_lim99
    pls_obj["speX_lim95"] = speX_lim95
    pls_obj["speY"] = speY
    pls_obj["speY_lim99"] = speY_lim99
    pls_obj["speY_lim95"] = speY_lim95
