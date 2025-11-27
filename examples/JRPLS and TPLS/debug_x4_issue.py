#!/usr/bin/env python3
"""Debug the X[4] numerical issue at LV4."""

import sys
import os
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from pyphi.utils import meancenterscale, n2z, std
from pyphi._internal import _Ab_btbinv

import pyphi_legacy as phi_legacy
from pyphi_legacy import meancenterscale as mcs_legacy
from pyphi_legacy import n2z as n2z_legacy  
from pyphi_legacy import std as std_legacy
from pyphi_legacy import _Ab_btbinv as Ab_legacy


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "jrpls_tpls_dataset.xlsx")
    
    jr, materials = phi_legacy.parse_materials(data_file, 'Materials')
    x = []
    for m in materials:
        x_ = pd.read_excel(data_file, sheet_name=m)
        x.append(x_)
    
    xc, jrc = phi_legacy.reconcile_rows_to_columns(x, jr)
    
    quality = pd.read_excel(data_file, sheet_name='QUALITY')
    
    jrc.append(quality)
    AUX = phi_legacy.reconcile_rows(jrc)
    
    JR_ = AUX[:-1]
    quality = AUX[-1]
    
    Ri = {}
    for j, m in zip(JR_, materials):
        Ri[m] = j
    Xi = {}
    for x_, m in zip(xc, materials):
        Xi[m] = x_
    
    return Xi, Ri, quality, materials


def run_to_lv3_state(Xi, Ri, Y, use_new=True):
    """Run up to LV3 and return the X, R, Y residuals."""
    if use_new:
        from pyphi.utils import meancenterscale, n2z, std
        from pyphi._internal import _Ab_btbinv as Ab_func
    else:
        from pyphi_legacy import meancenterscale, n2z, std, _Ab_btbinv as Ab_func
    
    X = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        Xaux = Xi[k]
        X_ = np.array(Xaux.values[:, 1:]).astype(float)
        X.append(X_)

    Y_ = np.array(Y.values[:, 1:]).astype(float)

    R = []
    for k in materials:
        Raux = Ri[k]
        R_ = np.array(Raux.values[:, 1:]).astype(float)
        R.append(R_)

    not_Xmiss = []
    not_Rmiss = []
    X__ = []
    R__ = []
    
    for X_i, R_i in zip(X, R):
        X_, x_mean_, x_std_ = meancenterscale(X_i)
        R_, r_mean_, r_std_ = meancenterscale(R_i)
        jr_scale_ = np.sqrt(X_.shape[1])
        X_ = X_ / jr_scale_
        
        X_nan_map = np.isnan(X_)
        not_Xmiss_ = (np.logical_not(X_nan_map)) * 1
        R_nan_map = np.isnan(R_)
        not_Rmiss_ = (np.logical_not(R_nan_map)) * 1
        not_Xmiss.append(not_Xmiss_)
        not_Rmiss.append(not_Rmiss_)
        
        X_, dummy = n2z(X_)
        R_, dummy = n2z(R_)
        X__.append(X_)
        R__.append(R_)
        
    X = X__.copy()
    R = R__.copy()
    
    Y_, y_mean, y_std = meancenterscale(Y_)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map)) * 1
    Y_, dummy = n2z(Y_)
    
    epsilon = 1e-9
    maxit = 2000
    
    for a in range(3):
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False
        num_it = 0
        
        while not Converged:
            hi = []
            for i, R_ in enumerate(R):
                hi_ = Ab_func(R_.T, ui, not_Rmiss[i].T)
                hi.append(hi_)
            
            si = []
            for i, X_ in enumerate(X):
                si_ = Ab_func(X_.T, hi[i], not_Xmiss[i].T)
                si.append(si_)
            
            js = np.array([y for x in si for y in x])
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)
            
            ri = []
            for i, X_ in enumerate(X):
                ri_ = Ab_func(X_, si[i], not_Xmiss[i])
                ri.append(ri_)
            
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
            
            ti = Ab_func(R_, jr, not_Rmiss_)
            qi = Ab_func(Y_.T, ti, not_Ymiss.T)
            un = Ab_func(Y_, qi, not_Ymiss)
            
            if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui)) < epsilon:
                Converged = True
            if num_it > maxit:
                Converged = True
            
            if Converged:
                pi = []
                for i, R_ in enumerate(R):
                    pi_ = Ab_func(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    vi_ = Ab_func(X_.T, ri[i], not_Xmiss[i].T)
                    vi.append(vi_)
                
                for i in np.arange(len(R)):
                    R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                    X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss
            else:
                num_it = num_it + 1
                ui = un
    
    return X, R, Y_, not_Xmiss, not_Rmiss, not_Ymiss


def main():
    print("=" * 80)
    print("DEBUGGING X[4] (MAT5) NUMERICAL ISSUE")
    print("=" * 80)
    
    Xi, Ri, quality, materials = load_data()
    
    print("\nRunning both implementations through LV 1-3...")
    
    X_new, R_new, Y_new, not_Xmiss_new, not_Rmiss_new, not_Ymiss_new = run_to_lv3_state(Xi, Ri, quality, use_new=True)
    X_leg, R_leg, Y_leg, not_Xmiss_leg, not_Rmiss_leg, not_Ymiss_leg = run_to_lv3_state(Xi, Ri, quality, use_new=False)
    
    print("\n--- X[4] (MAT5) after LV3 ---")
    print(f"X[4] new shape: {X_new[4].shape}")
    print(f"X[4] new - all values:\n{X_new[4]}")
    print(f"\nX[4] new - Frobenius norm: {np.linalg.norm(X_new[4]):.2e}")
    print(f"X[4] new - max absolute value: {np.max(np.abs(X_new[4])):.2e}")
    print(f"X[4] leg - max absolute value: {np.max(np.abs(X_leg[4])):.2e}")
    
    # Check what happens when we calculate s[4] and then r[4]
    ui = Y_new[:, [np.argmax(std(Y_new))]]
    
    # Calculate h[4]
    h4_new = _Ab_btbinv(R_new[4].T, ui, not_Rmiss_new[4].T)
    h4_leg = Ab_legacy(R_leg[4].T, ui, not_Rmiss_leg[4].T)
    
    print(f"\n--- h[4] calculation ---")
    print(f"h[4] new norm: {np.linalg.norm(h4_new):.10f}")
    print(f"h[4] leg norm: {np.linalg.norm(h4_leg):.10f}")
    print(f"h[4] diff: {np.max(np.abs(h4_new - h4_leg)):.2e}")
    
    # Calculate s[4]
    s4_new = _Ab_btbinv(X_new[4].T, h4_new, not_Xmiss_new[4].T)
    s4_leg = Ab_legacy(X_leg[4].T, h4_leg, not_Xmiss_leg[4].T)
    
    print(f"\n--- s[4] calculation (BEFORE normalization) ---")
    print(f"s[4] new: {s4_new.flatten()}")
    print(f"s[4] leg: {s4_leg.flatten()}")
    print(f"s[4] new norm: {np.linalg.norm(s4_new):.2e}")
    print(f"s[4] leg norm: {np.linalg.norm(s4_leg):.2e}")
    print(f"s[4] diff: {np.max(np.abs(s4_new - s4_leg)):.2e}")
    
    # What's X[4].T @ h[4]?
    numerator_new = X_new[4].T @ h4_new
    numerator_leg = X_leg[4].T @ h4_leg
    
    print(f"\n--- Numerator: X[4].T @ h[4] ---")
    print(f"New: {numerator_new.flatten()}")
    print(f"Leg: {numerator_leg.flatten()}")
    print(f"Diff: {np.max(np.abs(numerator_new - numerator_leg)):.2e}")
    
    # What's the denominator (h[4]'h[4] weighted by not_miss)?
    h4_mat_new = np.tile(h4_new.T, (X_new[4].T.shape[0], 1))
    h4_mat_leg = np.tile(h4_leg.T, (X_leg[4].T.shape[0], 1))
    
    denom_new = np.sum((h4_mat_new * not_Xmiss_new[4].T) ** 2, axis=1)
    denom_leg = np.sum((h4_mat_leg * not_Xmiss_leg[4].T) ** 2, axis=1)
    
    print(f"\n--- Denominator: weighted h'h ---")
    print(f"New: {denom_new[:5]}...")
    print(f"Leg: {denom_leg[:5]}...")
    
    # Now calculate r[4] = X[4] @ s[4] / (s[4]'s[4])
    # But first, let's see what the joint s normalization does
    
    # Simulate full si calculation to get normalized s[4]
    hi_new = []
    hi_leg = []
    for i in range(5):
        hi_new.append(_Ab_btbinv(R_new[i].T, ui, not_Rmiss_new[i].T))
        hi_leg.append(Ab_legacy(R_leg[i].T, ui, not_Rmiss_leg[i].T))
    
    si_new = []
    si_leg = []
    for i in range(5):
        si_new.append(_Ab_btbinv(X_new[i].T, hi_new[i], not_Xmiss_new[i].T))
        si_leg.append(Ab_legacy(X_leg[i].T, hi_leg[i], not_Xmiss_leg[i].T))
    
    # Normalize
    js_new = np.array([y for x in si_new for y in x])
    js_leg = np.array([y for x in si_leg for y in x])
    
    js_norm_new = np.linalg.norm(js_new)
    js_norm_leg = np.linalg.norm(js_leg)
    
    print(f"\n--- Joint s normalization ---")
    print(f"js norm (new): {js_norm_new:.10f}")
    print(f"js norm (leg): {js_norm_leg:.10f}")
    print(f"js norm diff: {abs(js_norm_new - js_norm_leg):.2e}")
    
    for i in range(5):
        si_new[i] = si_new[i] / js_norm_new
        si_leg[i] = si_leg[i] / js_norm_leg
    
    print(f"\n--- s[4] after normalization ---")
    print(f"s[4] new: {si_new[4].flatten()}")
    print(f"s[4] leg: {si_leg[4].flatten()}")
    print(f"s[4] diff: {np.max(np.abs(si_new[4] - si_leg[4])):.2e}")
    
    # Now calculate r[4]
    r4_new = _Ab_btbinv(X_new[4], si_new[4], not_Xmiss_new[4])
    r4_leg = Ab_legacy(X_leg[4], si_leg[4], not_Xmiss_leg[4])
    
    print(f"\n--- r[4] calculation ---")
    print(f"r[4] new (first 10): {r4_new.flatten()[:10]}")
    print(f"r[4] leg (first 10): {r4_leg.flatten()[:10]}")
    print(f"r[4] new norm: {np.linalg.norm(r4_new):.6f}")
    print(f"r[4] leg norm: {np.linalg.norm(r4_leg):.6f}")
    print(f"r[4] diff: {np.max(np.abs(r4_new - r4_leg)):.2e}")
    
    # Let's trace through _Ab_btbinv for r[4] manually
    print(f"\n--- Manual _Ab_btbinv trace for r[4] ---")
    A = X_new[4]
    b = si_new[4]
    A_not_nan_map = not_Xmiss_new[4]
    
    b_mat = np.tile(b.T, (A.shape[0], 1))
    numerator = np.sum(A * b_mat, axis=1)
    denominator = np.sum((b_mat * A_not_nan_map) ** 2, axis=1)
    
    print(f"A (X[4]) max abs: {np.max(np.abs(A)):.2e}")
    print(f"b (s[4]) max abs: {np.max(np.abs(b)):.2e}")
    print(f"numerator (A*b_mat sum) - first 10: {numerator[:10]}")
    print(f"numerator max abs: {np.max(np.abs(numerator)):.2e}")
    print(f"denominator (b_mat**2 sum) - first 10: {denominator[:10]}")
    print(f"denominator min: {np.min(denominator):.2e}")
    
    # The issue is 0/small_number creates instability
    c = numerator / denominator
    print(f"\n--- Result c = numerator/denominator ---")
    print(f"c (first 10): {c[:10]}")
    print(f"c contains nan: {np.any(np.isnan(c))}")
    print(f"c contains inf: {np.any(np.isinf(c))}")


if __name__ == "__main__":
    main()
