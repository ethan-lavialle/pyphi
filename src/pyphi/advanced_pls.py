"""
Advanced PLS Models for PyPhi.

This minimal version implements:
- JRPLS: Joint Range PLS
- TPLS: Tensor/Three-way PLS

References:
    - JRPLS/TPLS: Garcia-Munoz Chemom. Intel. Lab. Syst., 133, pp.49-62
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
from ._internal import _Ab_btbinv
from .pca import hott2
from .pls import pls


# =============================================================================
# JRPLS - Joint Range PLS
# =============================================================================


def jrpls(
    Xi: dict,
    Ri: dict,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    shush: bool = False,
) -> dict:
    """JRPLS Algorithm per Garcia-Munoz Chemom. Intel. Lab. Syst., 133, pp.49-62.

    Joint Range PLS for multi-material blending with separate properties per material.

    Parameters
    ----------
    Xi : dict
        Physical properties dictionary of DataFrames.
        Xi = {'MatA': df_with_props_for_mat_A (one row per lot, one col per property),
              'MatB': df_with_props_for_mat_B}
    Ri : dict
        Blending ratios dictionary of DataFrames.
        Ri = {'MatA': df_with_ratios_of_lots_of_A_used_per_blend,
              'MatB': df_with_ratios_of_lots_of_B_used_per_blend}
        Rows of Xi[i] must correspond to Columns of Ri[i].
    Y : np.ndarray or pd.DataFrame
        Product characteristics [b x n] of blends x product properties.
        First column of DataFrame is the observation identifier.
    A : int
        Number of latent variables.
    shush : bool, default=False
        If True, suppress output messages.

    Returns
    -------
    dict
        Dictionary with all JRPLS parameters.

    Examples
    --------
    >>> import pyphi as phi
    >>> Xi = {'MAT1': mat1_props, 'MAT2': mat2_props}
    >>> Ri = {'MAT1': mat1_ratios, 'MAT2': mat2_ratios}
    >>> jrpls_obj = phi.jrpls(Xi, Ri, Y, 4)
    """
    X = []
    varidX = []
    obsidX = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        Xaux = Xi[k]
        if isinstance(Xaux, np.ndarray):
            X_ = Xaux.copy()
            obsidX_ = False
            varidX_ = False
        elif isinstance(Xaux, pd.DataFrame):
            X_ = np.array(Xaux.values[:, 1:]).astype(float)
            obsidX_ = Xaux.values[:, 0].astype(str)
            obsidX_ = obsidX_.tolist()
            varidX_ = Xaux.columns.values
            varidX_ = varidX_[1:]
            varidX_ = varidX_.tolist()
        X.append(X_)
        varidX.append(varidX_)
        obsidX.append(obsidX_)

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)
        obsidY = Y.values[:, 0].astype(str)
        obsidY = obsidY.tolist()
        varidY = Y.columns.values
        varidY = varidY[1:]
        varidY = varidY.tolist()

    R = []
    varidR = []
    obsidR = []
    for k in materials:
        Raux = Ri[k]
        if isinstance(Raux, np.ndarray):
            R_ = Raux.copy()
            obsidR_ = False
            varidR_ = False
        elif isinstance(Raux, pd.DataFrame):
            R_ = np.array(Raux.values[:, 1:]).astype(float)
            obsidR_ = Raux.values[:, 0].astype(str)
            obsidR_ = obsidR_.tolist()
            varidR_ = Raux.columns.values
            varidR_ = varidR_[1:]
            varidR_ = varidR_.tolist()
        varidR.append(varidR_)
        obsidR.append(obsidR_)
        R.append(R_)

    x_mean = []
    x_std = []
    jr_scale = []
    r_mean = []
    r_std = []
    not_Xmiss = []
    not_Rmiss = []
    Xhat = []
    TSSX = []
    TSSXpv = []
    TSSR = []
    TSSRpv = []
    X__ = []
    R__ = []
    for X_i, R_i in zip(X, R):
        X_, x_mean_, x_std_ = meancenterscale(X_i)
        R_, r_mean_, r_std_ = meancenterscale(R_i)

        jr_scale_ = np.sqrt(X_.shape[0] * X_.shape[1])
        jr_scale_ = np.sqrt(X_.shape[1])
        X_ = X_ / jr_scale_

        x_mean.append(x_mean_)
        x_std.append(x_std_)
        jr_scale.append(jr_scale_)
        r_mean.append(r_mean_)
        r_std.append(r_std_)

        X_nan_map = np.isnan(X_)
        not_Xmiss_ = (np.logical_not(X_nan_map)) * 1
        R_nan_map = np.isnan(R_)
        not_Rmiss_ = (np.logical_not(R_nan_map)) * 1
        not_Xmiss.append(not_Xmiss_)
        not_Rmiss.append(not_Rmiss_)

        X_, dummy = n2z(X_)
        R_, dummy = n2z(R_)
        Xhat_ = np.zeros(X_.shape)
        X__.append(X_)
        R__.append(R_)
        Xhat.append(Xhat_)

        TSSX.append(np.sum(X_**2))
        TSSXpv.append(np.sum(X_**2, axis=0))
        TSSR.append(np.sum(R_**2))
        TSSRpv.append(np.sum(R_**2, axis=0))

    X = X__.copy()
    R = R__.copy()

    Y_, y_mean, y_std = meancenterscale(Y_)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map)) * 1
    Y_, dummy = n2z(Y_)
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)

    # use nipals
    if not shush:
        print("phi.jrpls using NIPALS executed on: " + str(datetime.datetime.now()))

    epsilon = 1e-9
    maxit = 2000

    for a in list(range(A)):
        # Select column with largest variance in Y as initial guess
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False
        num_it = 0
        while Converged == False:
            # _Ab_btbinv(A,b,A_not_nan_map):
            #  project c = Ab/b'b

            # Step 1. h=R'u/u'u
            hi = []
            for i, R_ in enumerate(R):
                hi_ = _Ab_btbinv(R_.T, ui, not_Rmiss[i].T)
                hi.append(hi_)
            si = []
            for i, X_ in enumerate(X):
                # Step 2. s = X'h/(h'h)
                si_ = _Ab_btbinv(X_.T, hi[i], not_Xmiss[i].T)
                si.append(si_)

            # Normalize joint s to unit length.
            js = np.array([y for x in si for y in x])  # flattening list of lists
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)

            ri = []
            for i, X_ in enumerate(X):
                # Step 3. ri= (Xs)/(s's)
                ri_ = _Ab_btbinv(X_, si[i], not_Xmiss[i])
                ri.append(ri_)

            # Calculating the Joint-r and Joint-R (hence the name of the method)
            jr = [y for x in ri for y in x]
            jr = np.array(jr).astype(float)

            for i, r_ in enumerate(R):
                if i == 0:
                    R_ = r_
                else:
                    R_ = np.hstack((R_, r_))

            for i, r_miss in enumerate(not_Rmiss):
                if i == 0:
                    not_Rmiss_ = r_miss
                else:
                    not_Rmiss_ = np.hstack((not_Rmiss_, r_miss))

            # Step 4. t = Rr/(r'r)
            ti = _Ab_btbinv(R_, jr, not_Rmiss_)

            # Step 5 q=Y't/t't
            qi = _Ab_btbinv(Y_.T, ti, not_Ymiss.T)

            # Step 5 un=(Yq)/(q'q)
            un = _Ab_btbinv(Y_, qi, not_Ymiss)

            if (
                abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui))
                < epsilon
            ):
                Converged = True

            if num_it > maxit:
                Converged = True

            if Converged:
                if not shush:
                    print("# Iterations for LV #" + str(a + 1) + ": ", str(num_it))
                pi = []
                for i, R_ in enumerate(R):
                    # Calculate P's for deflation p=R't/(t't)
                    pi_ = _Ab_btbinv(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    # Calculate v's for deflation v=Xr/(r'r)
                    vi_ = _Ab_btbinv(X_.T, ri[i], not_Xmiss[i].T)
                    vi.append(vi_)

                for i in np.arange(len(R)):
                    # Deflate X leaving missing as zeros (important!)
                    R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                    X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                    Xhat[i] = Xhat[i] + ri[i] @ vi[i].T
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss

                if a == 0:
                    T = ti
                    P = pi
                    S = si
                    U = un
                    Q = qi
                    H = hi
                    V = vi
                    Rscores = ri

                    r2X = []
                    r2Xpv = []
                    r2R = []
                    r2Rpv = []
                    for i in np.arange(len(X)):
                        r2X.append(1 - np.sum(X[i] ** 2) / TSSX[i])
                        r2Xpv_ = 1 - np.sum(X[i] ** 2, axis=0) / TSSXpv[i]
                        r2Xpv.append(r2Xpv_.reshape(-1, 1))
                        r2R.append(1 - np.sum(R[i] ** 2) / TSSR[i])
                        r2Rpv_ = 1 - np.sum(R[i] ** 2, axis=0) / TSSRpv[i]
                        r2Rpv.append(r2Rpv_.reshape(-1, 1))

                    r2Y = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv = 1 - np.sum(Y_**2, axis=0) / TSSYpv
                    r2Ypv = r2Ypv.reshape(-1, 1)

                else:
                    T = np.hstack((T, ti.reshape(-1, 1)))
                    U = np.hstack((U, un.reshape(-1, 1)))
                    Q = np.hstack((Q, qi))

                    for i in np.arange(len(P)):
                        P[i] = np.hstack((P[i], pi[i]))
                    for i in np.arange(len(S)):
                        S[i] = np.hstack((S[i], si[i]))
                    for i in np.arange(len(H)):
                        H[i] = np.hstack((H[i], hi[i]))
                    for i in np.arange(len(V)):
                        V[i] = np.hstack((V[i], vi[i]))
                    for i in np.arange(len(Rscores)):
                        Rscores[i] = np.hstack((Rscores[i], ri[i]))

                    for i in np.arange(len(X)):
                        r2X_ = 1 - np.sum(X[i] ** 2) / TSSX[i]
                        r2Xpv_ = 1 - np.sum(X[i] ** 2, axis=0) / TSSXpv[i]
                        r2Xpv_ = r2Xpv_.reshape(-1, 1)
                        r2X[i] = np.hstack((r2X[i], r2X_))
                        r2Xpv[i] = np.hstack((r2Xpv[i], r2Xpv_))

                        r2R_ = 1 - np.sum(R[i] ** 2) / TSSR[i]
                        r2Rpv_ = 1 - np.sum(R[i] ** 2, axis=0) / TSSRpv[i]
                        r2Rpv_ = r2Rpv_.reshape(-1, 1)
                        r2R[i] = np.hstack((r2R[i], r2R_))
                        r2Rpv[i] = np.hstack((r2Rpv[i], r2Rpv_))

                    r2Y_ = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv_ = 1 - np.sum(Y_**2, axis=0) / TSSYpv
                    r2Ypv_ = r2Ypv_.reshape(-1, 1)
                    r2Y = np.hstack((r2Y, r2Y_))
                    r2Ypv = np.hstack((r2Ypv, r2Ypv_))

            else:
                num_it = num_it + 1
                ui = un

        if a == 0:
            numIT = num_it
        else:
            numIT = np.hstack((numIT, num_it))
    for i in np.arange(len(Xhat)):
        Xhat[i] = (
            Xhat[i] * np.tile(x_std[i], (Xhat[i].shape[0], 1))
            + np.tile(x_mean[i], (Xhat[i].shape[0], 1))
        )

    for a in list(range(A - 1, 0, -1)):
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]
    r2xc = []
    r2rc = []
    for i in np.arange(len(X)):
        for a in list(range(A - 1, 0, -1)):
            r2X[i][a] = r2X[i][a] - r2X[i][a - 1]
            r2Xpv[i][:, a] = r2Xpv[i][:, a] - r2Xpv[i][:, a - 1]
            r2R[i][a] = r2R[i][a] - r2R[i][a - 1]
            r2Rpv[i][:, a] = r2Rpv[i][:, a] - r2Rpv[i][:, a - 1]

    for i, r in enumerate(r2Xpv):
        if i == 0:
            r2xpv_all = r
        else:
            r2xpv_all = np.vstack((r2xpv_all, r))

        r2xc.append(np.cumsum(r2X[i]))
        r2rc.append(np.cumsum(r2R[i]))

    eigs = np.var(T, axis=0)
    r2yc = np.cumsum(r2Y)
    r2rc = np.mean(np.array(r2rc), axis=0)
    r2xc = np.mean(np.array(r2xc), axis=0)
    r2x = np.mean(np.array(r2X), axis=0)
    r2r = np.mean(np.array(r2R), axis=0)

    if not shush:
        print("--------------------------------------------------------------")
        print(
            "LV #     Eig       R2X       sum(R2X)   R2R       sum(R2R)   R2Y       sum(R2Y)"
        )
        if A > 1:
            for a in list(range(A)):
                print(
                    "LV #"
                    + str(a + 1)
                    + ":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(
                        eigs[a], r2x[a], r2xc[a], r2r[a], r2rc[a], r2Y[a], r2yc[a]
                    )
                )
        else:
            d1 = eigs[0]
            d2 = r2xc[0]
            d3 = r2rc[0]
            d4 = r2yc[0]
            print(
                "LV #"
                + str(a + 1)
                + ":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(
                    d1, r2x, d2, r2r, d3, r2Y, d4
                )
            )
        print("--------------------------------------------------------------")

    jrpls_obj = {
        "T": T,
        "P": P,
        "Q": Q,
        "U": U,
        "S": S,
        "H": H,
        "V": V,
        "Rscores": Rscores,
        "r2xi": r2X,
        "r2xpvi": r2Xpv,
        "r2xpv": r2xpv_all,
        "mx": x_mean,
        "sx": x_std,
        "r2y": r2Y,
        "r2ypv": r2Ypv,
        "my": y_mean,
        "sy": y_std,
        "r2ri": r2R,
        "r2rpvi": r2Rpv,
        "mr": r_mean,
        "sr": r_std,
        "Xhat": Xhat,
        "materials": materials,
    }
    if not isinstance(obsidX, bool):
        jrpls_obj["obsidXi"] = obsidX
        jrpls_obj["varidXi"] = varidX

    varidXall = []
    for i in np.arange(len(materials)):
        for j in np.arange(len(varidX[i])):
            varidXall.append(materials[i] + ":" + varidX[i][j])
    jrpls_obj["varidX"] = varidXall

    if not isinstance(obsidR, bool):
        jrpls_obj["obsidRi"] = obsidR
        jrpls_obj["varidRi"] = varidR

    if not isinstance(obsidY, bool):
        jrpls_obj["obsidY"] = obsidY
        jrpls_obj["varidY"] = varidY

    T2 = hott2(jrpls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f99(A, (n - A))
    T2_lim95 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f95(A, (n - A))

    speX = []
    speR = []
    speX_lim95 = []
    speX_lim99 = []
    speR_lim95 = []
    speR_lim99 = []
    for i in np.arange(len(X)):
        speX.append(np.sum(X[i] ** 2, axis=1, keepdims=1))
        aux_ = np.sum(X[i] ** 2, axis=1, keepdims=1)
        speX_lim95_, speX_lim99_ = spe_ci(aux_)
        speX_lim95.append(speX_lim95_)
        speX_lim99.append(speX_lim99_)

        speR.append(np.sum(R[i] ** 2, axis=1, keepdims=1))
        aux_ = np.sum(R[i] ** 2, axis=1, keepdims=1)
        speR_lim95_, speR_lim99_ = spe_ci(aux_)
        speR_lim95.append(speR_lim95_)
        speR_lim99.append(speR_lim99_)

    speY = np.sum(Y_**2, axis=1, keepdims=1)
    speY_lim95, speY_lim99 = spe_ci(speY)

    jrpls_obj["T2"] = T2
    jrpls_obj["T2_lim99"] = T2_lim99
    jrpls_obj["T2_lim95"] = T2_lim95
    jrpls_obj["speX"] = speX
    jrpls_obj["speX_lim99"] = speX_lim99
    jrpls_obj["speX_lim95"] = speX_lim95
    jrpls_obj["speY"] = speY
    jrpls_obj["speY_lim99"] = speY_lim99
    jrpls_obj["speY_lim95"] = speY_lim95
    jrpls_obj["speR"] = speR
    jrpls_obj["speR_lim99"] = speR_lim99
    jrpls_obj["speR_lim95"] = speR_lim95

    Wsi = []
    Ws = []
    for i in np.arange(len(S)):
        Wsi.append(S[i] @ np.linalg.pinv(V[i].T @ S[i]))
        if i == 0:
            Ws = S[i] @ np.linalg.pinv(V[i].T @ S[i])
        else:
            Ws = np.vstack((Ws, S[i] @ np.linalg.pinv(V[i].T @ S[i])))
    jrpls_obj["Ssi"] = Wsi  # trick to plot the JRPLS/LPLS does not really have Ws
    jrpls_obj["Ss"] = Ws  # trick to plot the JRPLS/LPLS does not really have Ws
    # Ws=W @ np.linalg.pinv(P.T @ W)
    jrpls_obj["type"] = "jrpls"
    return jrpls_obj


def jrpls_pred(
    rnew: list | dict,
    jrplsobj: dict,
) -> dict | str:
    """Prediction with a JRPLS model.

    Parameters
    ----------
    rnew : list or dict
        New blending ratios. Can be:
        - list: List of ratio arrays per material
        - dict: Dictionary with format:
          {'matid': [(lotid, rvalue), ...], ...}

        Example for dict format:
        rnew = {
            'API': [('A0129', 0.5)],
            'Lactose': [('Lac0003', 0.1), ('Lac1010', 0.2)],
            'MgSt': [('M0012', 0.02)],
            'MCC': [('MCC0017', 0.18)],
        }

    jrplsobj : dict
        JRPLS object built with jrpls() routine.

    Returns
    -------
    dict or str
        Dictionary with:
        - Tnew: New scores
        - Yhat: Predicted Y values
        - speR: SPE for R space (per material)
        Or error message string if dimensions don't match.

    Examples
    --------
    >>> import pyphi as phi
    >>> rnew = {'MAT1': [('A0129', 0.5)], 'MAT2': [('Lac0003', 1)]}
    >>> preds = phi.jrpls_pred(rnew, jrpls_obj)
    """
    ok = True
    if isinstance(rnew, list):
        # check dimensions
        i = 0
        for r, mr, sr in zip(rnew, jrplsobj["mr"], jrplsobj["sr"]):
            if not (len(r) == len(mr[0])):
                ok = False
            np.ones(len(r))
            if i == 0:
                rnew_ = r
                mr_ = mr
                sr_ = sr
                Rscores = jrplsobj["Rscores"][i]
                P = jrplsobj["P"][i]
            else:
                rnew_ = np.hstack((rnew_, r))
                mr_ = np.hstack((mr_, mr))
                sr_ = np.hstack((sr_, sr))
                Rscores = np.vstack((Rscores, jrplsobj["Rscores"][i]))
                P = np.vstack((P, jrplsobj["P"][i]))
            i += 1

    elif isinstance(rnew, dict):
        # re-arrange
        ok = True
        rnew_ = [["*"]] * len(jrplsobj["materials"])
        for k in list(rnew.keys()):
            i = jrplsobj["materials"].index(k)
            ri = np.zeros((jrplsobj["mr"][i].shape[1]))
            for m, r in rnew[k]:
                e = jrplsobj["varidRi"][i].index(m)
                ri[e] = r
            rnew_[i] = ri

        preds = jrpls_pred(rnew_, jrplsobj)
        return preds

    if ok:
        bkzeros = 0
        selmat = []
        for i, r in enumerate(jrplsobj["Rscores"]):
            frontzeros = Rscores.shape[0] - bkzeros - r.shape[0]
            row = np.vstack(
                (
                    np.zeros((bkzeros, 1)),
                    np.ones((r.shape[0], 1)),
                    np.zeros((frontzeros, 1)),
                )
            )
            bkzeros += r.shape[0]
            selmat.append(row)

        tnew = []
        sper = []

        rnew_ = (rnew_ - mr_) / sr_
        rnew_ = rnew_.reshape(-1, 1)
        ti = []
        for a in np.arange(jrplsobj["T"].shape[1]):
            ti_ = rnew_.T @ Rscores[:, a] / (Rscores[:, a].T @ Rscores[:, a])
            ti.append(ti_[0])
            aux = ti_ * P[:, a]
            rnew_ = rnew_ - aux.reshape(-1, 1)
        tnew = np.array(ti)
        sper = []
        for row in selmat:
            sper.append(np.sum(rnew_[row == 1] ** 2))

        yhat = tnew @ jrplsobj["Q"].T
        yhat = (yhat * jrplsobj["sy"]) + jrplsobj["my"]
        preds = {"Tnew": tnew, "Yhat": yhat, "speR": sper}
        return preds
    else:
        return "dimensions of rnew did not match model"


# =============================================================================
# TPLS - Tensor/Three-way PLS
# =============================================================================


def tpls(
    Xi: dict,
    Ri: dict,
    Z: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    shush: bool = False,
) -> dict:
    """TPLS Algorithm per Garcia-Munoz Chemom. Intel. Lab. Syst., 133, pp.49-62.

    Tensor PLS for multi-material blending with process conditions.

    Parameters
    ----------
    Xi : dict
        Physical properties dictionary of DataFrames.
        Xi = {'MatA': df_with_props_for_mat_A (one row per lot, one col per property),
              'MatB': df_with_props_for_mat_B}
    Ri : dict
        Blending ratios dictionary of DataFrames.
        Ri = {'MatA': df_with_ratios_of_lots_of_A_used_per_blend,
              'MatB': df_with_ratios_of_lots_of_B_used_per_blend}
        Rows of Xi[i] must correspond to Columns of Ri[i].
    Z : np.ndarray or pd.DataFrame
        Process conditions [b x p] of blends x process variables.
        First column of DataFrame is the observation identifier.
    Y : np.ndarray or pd.DataFrame
        Product characteristics [b x n] of blends x product properties.
        First column of DataFrame is the observation identifier.
    A : int
        Number of latent variables.
    shush : bool, default=False
        If True, suppress output messages.

    Returns
    -------
    dict
        Dictionary with all TPLS parameters.

    Examples
    --------
    >>> import pyphi as phi
    >>> Xi = {'MAT1': mat1_props, 'MAT2': mat2_props}
    >>> Ri = {'MAT1': mat1_ratios, 'MAT2': mat2_ratios}
    >>> tpls_obj = phi.tpls(Xi, Ri, process, Y, 4)
    """
    X = []
    varidX = []
    obsidX = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        Xaux = Xi[k]
        if isinstance(Xaux, np.ndarray):
            X_ = Xaux.copy()
            obsidX_ = False
            varidX_ = False
        elif isinstance(Xaux, pd.DataFrame):
            X_ = np.array(Xaux.values[:, 1:]).astype(float)
            obsidX_ = Xaux.values[:, 0].astype(str)
            obsidX_ = obsidX_.tolist()
            varidX_ = Xaux.columns.values
            varidX_ = varidX_[1:]
            varidX_ = varidX_.tolist()
        X.append(X_)
        varidX.append(varidX_)
        obsidX.append(obsidX_)

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)
        obsidY = Y.values[:, 0].astype(str)
        obsidY = obsidY.tolist()
        varidY = Y.columns.values
        varidY = varidY[1:]
        varidY = varidY.tolist()

    if isinstance(Z, np.ndarray):
        Z_ = Z.copy()
        obsidZ = False
        varidZ = False
    elif isinstance(Z, pd.DataFrame):
        Z_ = np.array(Z.values[:, 1:]).astype(float)
        obsidZ = Z.values[:, 0].astype(str)
        obsidZ = obsidZ.tolist()
        varidZ = Z.columns.values
        varidZ = varidZ[1:]
        varidZ = varidZ.tolist()

    R = []
    varidR = []
    obsidR = []
    for k in materials:
        Raux = Ri[k]
        if isinstance(Raux, np.ndarray):
            R_ = Raux.copy()
            obsidR_ = False
            varidR_ = False
        elif isinstance(Raux, pd.DataFrame):
            R_ = np.array(Raux.values[:, 1:]).astype(float)
            obsidR_ = Raux.values[:, 0].astype(str)
            obsidR_ = obsidR_.tolist()
            varidR_ = Raux.columns.values
            varidR_ = varidR_[1:]
            varidR_ = varidR_.tolist()
        varidR.append(varidR_)
        obsidR.append(obsidR_)
        R.append(R_)

    x_mean = []
    x_std = []
    jr_scale = []
    r_mean = []
    r_std = []
    not_Xmiss = []
    not_Rmiss = []
    Xhat = []
    TSSX = []
    TSSXpv = []
    TSSR = []
    TSSRpv = []
    X__ = []
    R__ = []
    for X_i, R_i in zip(X, R):
        X_, x_mean_, x_std_ = meancenterscale(X_i)
        R_, r_mean_, r_std_ = meancenterscale(R_i)

        jr_scale_ = np.sqrt(X_.shape[0] * X_.shape[1])
        jr_scale_ = np.sqrt(X_.shape[1])
        X_ = X_ / jr_scale_

        x_mean.append(x_mean_)
        x_std.append(x_std_)
        jr_scale.append(jr_scale_)
        r_mean.append(r_mean_)
        r_std.append(r_std_)

        X_nan_map = np.isnan(X_)
        not_Xmiss_ = (np.logical_not(X_nan_map)) * 1
        R_nan_map = np.isnan(R_)
        not_Rmiss_ = (np.logical_not(R_nan_map)) * 1
        not_Xmiss.append(not_Xmiss_)
        not_Rmiss.append(not_Rmiss_)

        X_, dummy = n2z(X_)
        R_, dummy = n2z(R_)
        Xhat_ = np.zeros(X_.shape)
        X__.append(X_)
        R__.append(R_)
        Xhat.append(Xhat_)

        TSSX.append(np.sum(X_**2))
        TSSXpv.append(np.sum(X_**2, axis=0))
        TSSR.append(np.sum(R_**2))
        TSSRpv.append(np.sum(R_**2, axis=0))

    X = X__.copy()
    R = R__.copy()

    Y_, y_mean, y_std = meancenterscale(Y_)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map)) * 1
    Y_, dummy = n2z(Y_)
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)

    Z_, z_mean, z_std = meancenterscale(Z_)
    Z_nan_map = np.isnan(Z_)
    not_Zmiss = (np.logical_not(Z_nan_map)) * 1
    Z_, dummy = n2z(Z_)
    TSSZ = np.sum(Z_**2)
    TSSZpv = np.sum(Z_**2, axis=0)

    # use nipals
    if not shush:
        print("phi.tpls using NIPALS executed on: " + str(datetime.datetime.now()))

    epsilon = 1e-9
    maxit = 2000

    for a in list(range(A)):
        # Select column with largest variance in Y as initial guess
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False
        num_it = 0
        while Converged == False:
            # _Ab_btbinv(A,b,A_not_nan_map):
            #  project c = Ab/b'b

            # Step 2. h=R'u/u'u
            hi = []
            for i, R_ in enumerate(R):
                hi_ = _Ab_btbinv(R_.T, ui, not_Rmiss[i].T)
                hi.append(hi_)
            si = []
            for i, X_ in enumerate(X):
                # Step 3. s = X'h/(h'h)
                si_ = _Ab_btbinv(X_.T, hi[i], not_Xmiss[i].T)
                si.append(si_)

            # Step 4 Normalize joint s to unit length.
            js = np.array([y for x in si for y in x])  # flattening list of lists
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)

            # Step 5
            ri = []
            for i, X_ in enumerate(X):
                # Step 5. ri= (Xs)/(s's)
                ri_ = _Ab_btbinv(X_, si[i], not_Xmiss[i])
                ri.append(ri_)

            # Step 6
            # Calculating the Joint-r and Joint-R (hence the name of the method)
            jr = [y for x in ri for y in x]
            jr = np.array(jr).astype(float)

            for i, r_ in enumerate(R):
                if i == 0:
                    R_ = r_
                else:
                    R_ = np.hstack((R_, r_))

            for i, r_miss in enumerate(not_Rmiss):
                if i == 0:
                    not_Rmiss_ = r_miss
                else:
                    not_Rmiss_ = np.hstack((not_Rmiss_, r_miss))

            t_rx = _Ab_btbinv(R_, jr, not_Rmiss_)

            # Step 7
            # Now the process matrix
            wi = _Ab_btbinv(Z_.T, ui, not_Zmiss.T)

            # Step 8
            wi = wi / np.linalg.norm(wi)

            # Step 9
            t_z = _Ab_btbinv(Z_, wi, not_Zmiss)

            # Step 10
            Taux = np.hstack((t_rx, t_z))
            plsobj_ = pls(Taux, Y_, 1, mcsX=False, mcsY=False, shush=True, force_nipals=True)
            wt_i = plsobj_["W"]
            qi = plsobj_["Q"]
            un = plsobj_["U"]
            ti = plsobj_["T"]

            if (
                abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui))
                < epsilon
            ):
                Converged = True

            if num_it > maxit:
                Converged = True

            if Converged:
                if not shush:
                    print("# Iterations for LV #" + str(a + 1) + ": ", str(num_it))
                pi = []
                for i, R_ in enumerate(R):
                    # Calculate P's for deflation p=R't/(t't)
                    pi_ = _Ab_btbinv(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    # Calculate v's for deflation v=Xr/(r'r)
                    vi_ = _Ab_btbinv(X_.T, ri[i], not_Xmiss[i].T)
                    vi.append(vi_)

                pzi = _Ab_btbinv(Z_.T, ti, not_Zmiss.T)

                for i in np.arange(len(R)):
                    # Deflate X leaving missing as zeros (important!)
                    R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                    X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                    Xhat[i] = Xhat[i] + ri[i] @ vi[i].T

                Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                Z_ = (Z_ - ti @ pzi.T) * not_Zmiss

                if a == 0:
                    T = ti
                    P = pi
                    Pz = pzi
                    S = si
                    U = un
                    Q = qi
                    H = hi
                    V = vi
                    Rscores = ri
                    W = wi
                    Wt = wt_i

                    r2X = []
                    r2Xpv = []
                    r2R = []
                    r2Rpv = []
                    for i in np.arange(len(X)):
                        r2X.append(1 - np.sum(X[i] ** 2) / TSSX[i])
                        r2Xpv_ = 1 - np.sum(X[i] ** 2, axis=0) / TSSXpv[i]
                        r2Xpv.append(r2Xpv_.reshape(-1, 1))
                        r2R.append(1 - np.sum(R[i] ** 2) / TSSR[i])
                        r2Rpv_ = 1 - np.sum(R[i] ** 2, axis=0) / TSSRpv[i]
                        r2Rpv.append(r2Rpv_.reshape(-1, 1))

                    r2Y = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv = 1 - np.sum(Y_**2, axis=0) / TSSYpv
                    r2Ypv = r2Ypv.reshape(-1, 1)

                    r2Z = 1 - np.sum(Z_**2) / TSSZ
                    r2Zpv = 1 - np.sum(Z_**2, axis=0) / TSSZpv
                    r2Zpv = r2Zpv.reshape(-1, 1)
                else:
                    T = np.hstack((T, ti.reshape(-1, 1)))
                    U = np.hstack((U, un.reshape(-1, 1)))
                    Q = np.hstack((Q, qi))
                    W = np.hstack((W, wi))
                    Wt = np.hstack((Wt, wt_i))
                    Pz = np.hstack((Pz, pzi))

                    for i in np.arange(len(P)):
                        P[i] = np.hstack((P[i], pi[i]))
                    for i in np.arange(len(S)):
                        S[i] = np.hstack((S[i], si[i]))
                    for i in np.arange(len(H)):
                        H[i] = np.hstack((H[i], hi[i]))
                    for i in np.arange(len(V)):
                        V[i] = np.hstack((V[i], vi[i]))
                    for i in np.arange(len(Rscores)):
                        Rscores[i] = np.hstack((Rscores[i], ri[i]))

                    for i in np.arange(len(X)):
                        r2X_ = 1 - np.sum(X[i] ** 2) / TSSX[i]
                        r2Xpv_ = 1 - np.sum(X[i] ** 2, axis=0) / TSSXpv[i]
                        r2Xpv_ = r2Xpv_.reshape(-1, 1)
                        r2X[i] = np.hstack((r2X[i], r2X_))
                        r2Xpv[i] = np.hstack((r2Xpv[i], r2Xpv_))

                        r2R_ = 1 - np.sum(R[i] ** 2) / TSSR[i]
                        r2Rpv_ = 1 - np.sum(R[i] ** 2, axis=0) / TSSRpv[i]
                        r2Rpv_ = r2Rpv_.reshape(-1, 1)
                        r2R[i] = np.hstack((r2R[i], r2R_))
                        r2Rpv[i] = np.hstack((r2Rpv[i], r2Rpv_))

                    r2Y_ = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv_ = 1 - np.sum(Y_**2, axis=0) / TSSYpv
                    r2Ypv_ = r2Ypv_.reshape(-1, 1)
                    r2Y = np.hstack((r2Y, r2Y_))
                    r2Ypv = np.hstack((r2Ypv, r2Ypv_))

                    r2Z_ = 1 - np.sum(Z_**2) / TSSZ
                    r2Zpv_ = 1 - np.sum(Z_**2, axis=0) / TSSZpv
                    r2Zpv_ = r2Zpv_.reshape(-1, 1)
                    r2Z = np.hstack((r2Z, r2Z_))
                    r2Zpv = np.hstack((r2Zpv, r2Zpv_))

            else:
                num_it = num_it + 1
                ui = un
        if a == 0:
            numIT = num_it
        else:
            numIT = np.hstack((numIT, num_it))
    for i in np.arange(len(Xhat)):
        Xhat[i] = (
            Xhat[i] * np.tile(x_std[i], (Xhat[i].shape[0], 1))
            + np.tile(x_mean[i], (Xhat[i].shape[0], 1))
        )

    for a in list(range(A - 1, 0, -1)):
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]
        r2Z[a] = r2Z[a] - r2Z[a - 1]
        r2Zpv[:, a] = r2Zpv[:, a] - r2Zpv[:, a - 1]

    r2xc = []
    r2rc = []
    for i in np.arange(len(X)):
        for a in list(range(A - 1, 0, -1)):
            r2X[i][a] = r2X[i][a] - r2X[i][a - 1]
            r2Xpv[i][:, a] = r2Xpv[i][:, a] - r2Xpv[i][:, a - 1]
            r2R[i][a] = r2R[i][a] - r2R[i][a - 1]
            r2Rpv[i][:, a] = r2Rpv[i][:, a] - r2Rpv[i][:, a - 1]

    for i, r in enumerate(r2Xpv):
        if i == 0:
            r2xpv_all = r
        else:
            r2xpv_all = np.vstack((r2xpv_all, r))

        r2xc.append(np.cumsum(r2X[i]))
        r2rc.append(np.cumsum(r2R[i]))

    r2yc = np.cumsum(r2Y)
    r2zc = np.cumsum(r2Z)
    r2rc = np.mean(np.array(r2rc), axis=0)
    r2xc = np.mean(np.array(r2xc), axis=0)
    r2x = np.mean(np.array(r2X), axis=0)
    r2r = np.mean(np.array(r2R), axis=0)

    if not shush:
        print("--------------------------------------------------------------")
        print(
            "LV #     R2X       sum(R2X)   R2R       sum(R2R)   R2Z       sum(R2Z)   R2Y       sum(R2Y)"
        )
        if A > 1:
            for a in list(range(A)):
                print(
                    "LV #"
                    + str(a + 1)
                    + ":   {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(
                        r2x[a], r2xc[a], r2r[a], r2rc[a], r2Z[a], r2zc[a], r2Y[a], r2yc[a]
                    )
                )
        else:
            d1 = r2xc[0]
            d2 = r2rc[0]
            d3 = r2zc[0]
            d4 = r2yc[0]
            print(
                "LV #"
                + str(a + 1)
                + ":   {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(
                    r2x, d1, r2r, d2, r2Z, d3, r2Y, d4
                )
            )
        print("--------------------------------------------------------------")

    tpls_obj = {
        "T": T,
        "P": P,
        "Q": Q,
        "U": U,
        "S": S,
        "H": H,
        "V": V,
        "Rscores": Rscores,
        "r2xi": r2X,
        "r2xpvi": r2Xpv,
        "r2xpv": r2xpv_all,
        "mx": x_mean,
        "sx": x_std,
        "r2y": r2Y,
        "r2ypv": r2Ypv,
        "my": y_mean,
        "sy": y_std,
        "r2ri": r2R,
        "r2rpvi": r2Rpv,
        "mr": r_mean,
        "sr": r_std,
        "r2z": r2Z,
        "r2zpv": r2Zpv,
        "mz": z_mean,
        "sz": z_std,
        "Xhat": Xhat,
        "materials": materials,
        "Wt": Wt,
        "W": W,
        "Pz": Pz,
    }
    if not isinstance(obsidX, bool):
        tpls_obj["obsidXi"] = obsidX
        tpls_obj["varidXi"] = varidX

    varidXall = []
    for i in np.arange(len(materials)):
        for j in np.arange(len(varidX[i])):
            varidXall.append(materials[i] + ":" + varidX[i][j])
    tpls_obj["varidX"] = varidXall

    if not isinstance(obsidR, bool):
        tpls_obj["obsidRi"] = obsidR
        tpls_obj["varidRi"] = varidR

    if not isinstance(obsidY, bool):
        tpls_obj["obsidY"] = obsidY
        tpls_obj["varidY"] = varidY

    if not isinstance(obsidZ, bool):
        tpls_obj["obsidZ"] = obsidZ
        tpls_obj["varidZ"] = varidZ

    T2 = hott2(tpls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f99(A, (n - A))
    T2_lim95 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f95(A, (n - A))

    speX = []
    speR = []
    speX_lim95 = []
    speX_lim99 = []
    speR_lim95 = []
    speR_lim99 = []
    for i in np.arange(len(X)):
        speX.append(np.sum(X[i] ** 2, axis=1, keepdims=1))
        aux_ = np.sum(X[i] ** 2, axis=1, keepdims=1)
        speX_lim95_, speX_lim99_ = spe_ci(aux_)
        speX_lim95.append(speX_lim95_)
        speX_lim99.append(speX_lim99_)

        speR.append(np.sum(R[i] ** 2, axis=1, keepdims=1))
        aux_ = np.sum(R[i] ** 2, axis=1, keepdims=1)
        speR_lim95_, speR_lim99_ = spe_ci(aux_)
        speR_lim95.append(speR_lim95_)
        speR_lim99.append(speR_lim99_)

    speY = np.sum(Y_**2, axis=1, keepdims=1)
    speY_lim95, speY_lim99 = spe_ci(speY)

    speZ = np.sum(Z_**2, axis=1, keepdims=1)
    speZ_lim95, speZ_lim99 = spe_ci(speZ)

    tpls_obj["T2"] = T2
    tpls_obj["T2_lim99"] = T2_lim99
    tpls_obj["T2_lim95"] = T2_lim95
    tpls_obj["speX"] = speX
    tpls_obj["speX_lim99"] = speX_lim99
    tpls_obj["speX_lim95"] = speX_lim95
    tpls_obj["speY"] = speY
    tpls_obj["speY_lim99"] = speY_lim99
    tpls_obj["speY_lim95"] = speY_lim95
    tpls_obj["speR"] = speR
    tpls_obj["speR_lim99"] = speR_lim99
    tpls_obj["speR_lim95"] = speR_lim95
    tpls_obj["speZ"] = speZ
    tpls_obj["speZ_lim99"] = speZ_lim99
    tpls_obj["speZ_lim95"] = speZ_lim95

    Wsi = []
    Ws = []
    for i in np.arange(len(S)):
        Wsi.append(S[i] @ np.linalg.pinv(V[i].T @ S[i]))
        if i == 0:
            Ws = S[i] @ np.linalg.pinv(V[i].T @ S[i])
        else:
            Ws = np.vstack((Ws, S[i] @ np.linalg.pinv(V[i].T @ S[i])))
    tpls_obj["Ssi"] = Wsi  # trick to plot the JRPLS/LPLS does not really have Ws
    tpls_obj["Ss"] = Ws  # trick to plot the JRPLS/LPLS does not really have Ws
    # Ws=W @ np.linalg.pinv(P.T @ W)
    tpls_obj["type"] = "tpls"

    Ws = W @ np.linalg.pinv(Pz.T @ W)
    tpls_obj["Ws"] = Ws
    return tpls_obj


def tpls_pred(
    rnew: list | dict,
    znew: np.ndarray | list | pd.DataFrame,
    tplsobj: dict,
) -> dict | str:
    """Prediction with a TPLS model.

    Parameters
    ----------
    rnew : list or dict
        New blending ratios. Can be:
        - list: List of ratio arrays per material
        - dict: Dictionary with format:
          {'matid': [(lotid, rvalue), ...], ...}

        Example for dict format:
        rnew = {
            'API': [('A0129', 0.5)],
            'Lactose': [('Lac0003', 0.1), ('Lac1010', 0.2)],
            'MgSt': [('M0012', 0.02)],
            'MCC': [('MCC0017', 0.18)],
        }

    znew : np.ndarray, list, or pd.DataFrame
        New process conditions.
    tplsobj : dict
        TPLS object built with tpls() routine.

    Returns
    -------
    dict or str
        Dictionary with:
        - Tnew: New scores
        - Yhat: Predicted Y values
        - speR: SPE for R space (per material)
        - speZ: SPE for Z space
        Or error message string if dimensions don't match.

    Examples
    --------
    >>> import pyphi as phi
    >>> rnew = {'MAT1': [('A0129', 0.5)], 'MAT2': [('Lac0003', 1)]}
    >>> znew = process_conditions
    >>> preds = phi.tpls_pred(rnew, znew, tpls_obj)
    """
    ok = True
    if isinstance(rnew, list):
        # check dimensions
        i = 0
        for r, mr, sr in zip(rnew, tplsobj["mr"], tplsobj["sr"]):
            if not (len(r) == len(mr[0])):
                ok = False
            np.ones(len(r))
            if i == 0:
                rnew_ = r
                mr_ = mr
                sr_ = sr
                Rscores = tplsobj["Rscores"][i]
                P = tplsobj["P"][i]
            else:
                rnew_ = np.hstack((rnew_, r))
                mr_ = np.hstack((mr_, mr))
                sr_ = np.hstack((sr_, sr))
                Rscores = np.vstack((Rscores, tplsobj["Rscores"][i]))
                P = np.vstack((P, tplsobj["P"][i]))
            i += 1

    elif isinstance(rnew, dict):
        # re-arrange
        ok = True
        rnew_ = [["*"]] * len(tplsobj["materials"])
        for k in list(rnew.keys()):
            i = tplsobj["materials"].index(k)
            ri = np.zeros((tplsobj["mr"][i].shape[1]))
            for m, r in rnew[k]:
                e = tplsobj["varidRi"][i].index(m)
                ri[e] = r
            rnew_[i] = ri

        preds = tpls_pred(rnew_, znew, tplsobj)
        return preds
    if isinstance(znew, pd.DataFrame):
        znew_ = znew.values.reshape(-1)[1:].astype(float)
    elif isinstance(znew, list):
        znew_ = np.array(znew)
    elif isinstance(znew, np.ndarray):
        znew_ = znew.copy()

    if not (len(znew_) == tplsobj["mz"].shape[1]):
        ok = False

    if ok:
        bkzeros = 0
        selmat = []
        for i, r in enumerate(tplsobj["Rscores"]):
            frontzeros = Rscores.shape[0] - bkzeros - r.shape[0]
            row = np.vstack(
                (
                    np.zeros((bkzeros, 1)),
                    np.ones((r.shape[0], 1)),
                    np.zeros((frontzeros, 1)),
                )
            )
            bkzeros += r.shape[0]
            selmat.append(row)

        tnew = []
        sper = []
        spez = []

        rnew_ = (rnew_ - mr_) / sr_
        rnew_ = rnew_.reshape(-1, 1)

        znew_ = (znew_ - tplsobj["mz"]) / tplsobj["sz"]
        znew_ = znew_.reshape(-1, 1)

        tnew = []
        for a in np.arange(tplsobj["T"].shape[1]):
            ti_rx_ = rnew_.T @ Rscores[:, a] / (Rscores[:, a].T @ Rscores[:, a])

            ti_z_ = znew_.T @ tplsobj["W"][:, a]

            ti_ = np.array([ti_rx_, ti_z_]).reshape(1, -1) @ tplsobj["Wt"][:, a]

            tnew.append(ti_[0])
            aux = ti_ * P[:, a]
            rnew_ = rnew_ - aux.reshape(-1, 1)

            auxz = ti_ * tplsobj["Pz"][:, a]
            znew_ = znew_ - auxz.reshape(-1, 1)

        sper = []
        for row in selmat:
            sper.append(np.sum(rnew_[row == 1] ** 2))
        spez = np.sum(znew_**2)
        tnew = np.array(tnew)
        yhat = tnew @ tplsobj["Q"].T
        yhat = (yhat * tplsobj["sy"]) + tplsobj["my"]
        preds = {"Tnew": tnew, "Yhat": yhat, "speR": sper, "speZ": spez}
        return preds
    else:
        return "dimensions of rnew or znew did not match model"
