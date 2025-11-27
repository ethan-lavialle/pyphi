#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to trace JRPLS LV #4 iteration-by-iteration.
Identifies exactly where numerical divergence begins between implementations.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

# Import internal utilities from both implementations
from pyphi.utils import meancenterscale, n2z, std
from pyphi._internal import _Ab_btbinv

import pyphi_legacy as phi_legacy
from pyphi_legacy import meancenterscale as mcs_legacy
from pyphi_legacy import n2z as n2z_legacy  
from pyphi_legacy import std as std_legacy
from pyphi_legacy import _Ab_btbinv as Ab_legacy


def load_data():
    """Load the JRPLS/TPLS dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "jrpls_tpls_dataset.xlsx")
    
    jr, materials = phi_legacy.parse_materials(data_file, 'Materials')
    x = []
    for m in materials:
        x_ = pd.read_excel(data_file, sheet_name=m)
        x.append(x_)
    
    xc, jrc = phi_legacy.reconcile_rows_to_columns(x, jr)
    
    quality = pd.read_excel(data_file, sheet_name='QUALITY')
    process = pd.read_excel(data_file, sheet_name='PROCESS')
    
    jrc.append(process)
    jrc.append(quality)
    AUX = phi_legacy.reconcile_rows(jrc)
    
    JR_ = AUX[:-2]
    process = AUX[-2]
    quality = AUX[-1]
    
    Ri = {}
    for j, m in zip(JR_, materials):
        Ri[m] = j
    Xi = {}
    for x_, m in zip(xc, materials):
        Xi[m] = x_
    
    return Xi, Ri, quality, process, materials


def run_jrpls_to_lv3_new(Xi, Ri, Y):
    """Run new implementation up to LV3, return state."""
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
    
    # Run 3 LVs
    for a in range(3):
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False
        num_it = 0
        
        while not Converged:
            hi = []
            for i, R_ in enumerate(R):
                hi_ = _Ab_btbinv(R_.T, ui, not_Rmiss[i].T)
                hi.append(hi_)
            
            si = []
            for i, X_ in enumerate(X):
                si_ = _Ab_btbinv(X_.T, hi[i], not_Xmiss[i].T)
                si.append(si_)
            
            js = np.array([y for x in si for y in x])
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)
            
            ri = []
            for i, X_ in enumerate(X):
                ri_ = _Ab_btbinv(X_, si[i], not_Xmiss[i])
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
            
            ti = _Ab_btbinv(R_, jr, not_Rmiss_)
            qi = _Ab_btbinv(Y_.T, ti, not_Ymiss.T)
            un = _Ab_btbinv(Y_, qi, not_Ymiss)
            
            if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui)) < epsilon:
                Converged = True
            if num_it > maxit:
                Converged = True
            
            if Converged:
                pi = []
                for i, R_ in enumerate(R):
                    pi_ = _Ab_btbinv(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    vi_ = _Ab_btbinv(X_.T, ri[i], not_Xmiss[i].T)
                    vi.append(vi_)
                
                for i in np.arange(len(R)):
                    R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                    X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss
            else:
                num_it = num_it + 1
                ui = un
    
    return X, R, Y_, not_Xmiss, not_Rmiss, not_Ymiss


def run_jrpls_to_lv3_legacy(Xi, Ri, Y):
    """Run legacy implementation up to LV3, return state."""
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
        X_, x_mean_, x_std_ = mcs_legacy(X_i)
        R_, r_mean_, r_std_ = mcs_legacy(R_i)
        jr_scale_ = np.sqrt(X_.shape[1])
        X_ = X_ / jr_scale_
        
        X_nan_map = np.isnan(X_)
        not_Xmiss_ = (np.logical_not(X_nan_map)) * 1
        R_nan_map = np.isnan(R_)
        not_Rmiss_ = (np.logical_not(R_nan_map)) * 1
        not_Xmiss.append(not_Xmiss_)
        not_Rmiss.append(not_Rmiss_)
        
        X_, dummy = n2z_legacy(X_)
        R_, dummy = n2z_legacy(R_)
        X__.append(X_)
        R__.append(R_)
        
    X = X__.copy()
    R = R__.copy()
    
    Y_, y_mean, y_std = mcs_legacy(Y_)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map)) * 1
    Y_, dummy = n2z_legacy(Y_)
    
    epsilon = 1e-9
    maxit = 2000
    
    # Run 3 LVs
    for a in range(3):
        ui = Y_[:, [np.argmax(std_legacy(Y_))]]
        Converged = False
        num_it = 0
        
        while not Converged:
            hi = []
            for i, R_ in enumerate(R):
                hi_ = Ab_legacy(R_.T, ui, not_Rmiss[i].T)
                hi.append(hi_)
            
            si = []
            for i, X_ in enumerate(X):
                si_ = Ab_legacy(X_.T, hi[i], not_Xmiss[i].T)
                si.append(si_)
            
            js = np.array([y for x in si for y in x])
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)
            
            ri = []
            for i, X_ in enumerate(X):
                ri_ = Ab_legacy(X_, si[i], not_Xmiss[i])
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
            
            ti = Ab_legacy(R_, jr, not_Rmiss_)
            qi = Ab_legacy(Y_.T, ti, not_Ymiss.T)
            un = Ab_legacy(Y_, qi, not_Ymiss)
            
            if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui)) < epsilon:
                Converged = True
            if num_it > maxit:
                Converged = True
            
            if Converged:
                pi = []
                for i, R_ in enumerate(R):
                    pi_ = Ab_legacy(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    vi_ = Ab_legacy(X_.T, ri[i], not_Xmiss[i].T)
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
    """Main comparison routine."""
    print("=" * 80)
    print("JRPLS LV #4 ITERATION-BY-ITERATION DEBUG")
    print("=" * 80)
    
    # Load data
    Xi, Ri, quality, process, materials = load_data()
    
    print("\nRunning both implementations through LV 1-3...")
    
    # Get state after LV 3 from both implementations
    X_new, R_new, Y_new, not_Xmiss_new, not_Rmiss_new, not_Ymiss_new = run_jrpls_to_lv3_new(Xi, Ri, quality)
    X_leg, R_leg, Y_leg, not_Xmiss_leg, not_Rmiss_leg, not_Ymiss_leg = run_jrpls_to_lv3_legacy(Xi, Ri, quality)
    
    print("\n--- State after LV 3 ---")
    print(f"Y residual norm: new={np.linalg.norm(Y_new):.10f}, legacy={np.linalg.norm(Y_leg):.10f}")
    print(f"Y residual diff: {np.max(np.abs(Y_new - Y_leg)):.2e}")
    
    for i in range(len(X_new)):
        print(f"X[{i}] residual diff: {np.max(np.abs(X_new[i] - X_leg[i])):.2e}")
        print(f"R[{i}] residual diff: {np.max(np.abs(R_new[i] - R_leg[i])):.2e}")
    
    # Check std calculations on Y residual
    print("\n--- Initial u selection for LV #4 ---")
    
    std_Y_new = std(Y_new)
    std_Y_leg = std_legacy(Y_leg)
    
    print(f"Y std values (new):    {std_Y_new.flatten()}")
    print(f"Y std values (legacy): {std_Y_leg.flatten()}")
    print(f"Std diff: {np.max(np.abs(std_Y_new - std_Y_leg)):.2e}")
    
    argmax_new = np.argmax(std_Y_new)
    argmax_leg = np.argmax(std_Y_leg)
    print(f"\nColumn with max variance: new={argmax_new}, legacy={argmax_leg}")
    
    ui_new = Y_new[:, [argmax_new]]
    ui_leg = Y_leg[:, [argmax_leg]]
    
    print(f"\nInitial ui (first 10 values):")
    print(f"  new:    {ui_new.flatten()[:10]}")
    print(f"  legacy: {ui_leg.flatten()[:10]}")
    print(f"  diff:   {np.max(np.abs(ui_new - ui_leg)):.2e}")
    
    # Run first 5 iterations of LV #4 for both
    print("\n" + "=" * 80)
    print("LV #4 ITERATION COMPARISON")
    print("=" * 80)
    
    epsilon = 1e-9
    
    for it in range(10):
        print(f"\n--- Iteration {it} ---")
        
        # Step 1: h = R'u / u'u for new
        hi_new = []
        for i, R_ in enumerate(R_new):
            hi_ = _Ab_btbinv(R_.T, ui_new, not_Rmiss_new[i].T)
            hi_new.append(hi_)
        
        hi_leg = []
        for i, R_ in enumerate(R_leg):
            hi_ = Ab_legacy(R_.T, ui_leg, not_Rmiss_leg[i].T)
            hi_leg.append(hi_)
        
        for i, (hn, hl) in enumerate(zip(hi_new, hi_leg)):
            diff = np.max(np.abs(hn - hl))
            status = "✓" if diff < 1e-10 else "❌"
            print(f"  {status} h[{i}] diff: {diff:.2e}")
        
        # Step 2: s = X'h / h'h
        si_new = []
        for i, X_ in enumerate(X_new):
            si_ = _Ab_btbinv(X_.T, hi_new[i], not_Xmiss_new[i].T)
            si_new.append(si_)
        
        si_leg = []
        for i, X_ in enumerate(X_leg):
            si_ = Ab_legacy(X_.T, hi_leg[i], not_Xmiss_leg[i].T)
            si_leg.append(si_)
        
        # Normalize
        js_new = np.array([y for x in si_new for y in x])
        js_leg = np.array([y for x in si_leg for y in x])
        
        for i in np.arange(len(si_new)):
            si_new[i] = si_new[i] / np.linalg.norm(js_new)
            si_leg[i] = si_leg[i] / np.linalg.norm(js_leg)
        
        for i, (sn, sl) in enumerate(zip(si_new, si_leg)):
            diff = np.max(np.abs(sn - sl))
            # Also check sign flip
            diff_flip = np.max(np.abs(sn + sl))
            diff = min(diff, diff_flip)
            status = "✓" if diff < 1e-10 else "❌"
            print(f"  {status} s[{i}] diff: {diff:.2e}")
        
        # Step 3: r = Xs / s's
        ri_new = []
        for i, X_ in enumerate(X_new):
            ri_ = _Ab_btbinv(X_, si_new[i], not_Xmiss_new[i])
            ri_new.append(ri_)
        
        ri_leg = []
        for i, X_ in enumerate(X_leg):
            ri_ = Ab_legacy(X_, si_leg[i], not_Xmiss_leg[i])
            ri_leg.append(ri_)
        
        for i, (rn, rl) in enumerate(zip(ri_new, ri_leg)):
            diff = np.max(np.abs(rn - rl))
            diff_flip = np.max(np.abs(rn + rl))
            diff = min(diff, diff_flip)
            status = "✓" if diff < 1e-10 else "❌"
            print(f"  {status} r[{i}] diff: {diff:.2e}")
        
        # Step 4: t = Rr / r'r
        jr_new = [y for x in ri_new for y in x]
        jr_new = np.array(jr_new).astype(float)
        jr_leg = [y for x in ri_leg for y in x]
        jr_leg = np.array(jr_leg).astype(float)
        
        for i, r_ in enumerate(R_new):
            if i == 0:
                R_cat_new = r_
            else:
                R_cat_new = np.hstack((R_cat_new, r_))
        
        for i, r_ in enumerate(R_leg):
            if i == 0:
                R_cat_leg = r_
            else:
                R_cat_leg = np.hstack((R_cat_leg, r_))
        
        for i, r_miss in enumerate(not_Rmiss_new):
            if i == 0:
                not_Rmiss_cat_new = r_miss
            else:
                not_Rmiss_cat_new = np.hstack((not_Rmiss_cat_new, r_miss))
        
        for i, r_miss in enumerate(not_Rmiss_leg):
            if i == 0:
                not_Rmiss_cat_leg = r_miss
            else:
                not_Rmiss_cat_leg = np.hstack((not_Rmiss_cat_leg, r_miss))
        
        ti_new = _Ab_btbinv(R_cat_new, jr_new, not_Rmiss_cat_new)
        ti_leg = Ab_legacy(R_cat_leg, jr_leg, not_Rmiss_cat_leg)
        
        t_diff = np.max(np.abs(ti_new - ti_leg))
        t_diff_flip = np.max(np.abs(ti_new + ti_leg))
        t_diff = min(t_diff, t_diff_flip)
        status = "✓" if t_diff < 1e-10 else "❌"
        print(f"  {status} t diff: {t_diff:.2e}")
        
        # Step 5: q = Y't / t't, u = Yq / q'q
        qi_new = _Ab_btbinv(Y_new.T, ti_new, not_Ymiss_new.T)
        qi_leg = Ab_legacy(Y_leg.T, ti_leg, not_Ymiss_leg.T)
        
        q_diff = np.max(np.abs(qi_new - qi_leg))
        q_diff_flip = np.max(np.abs(qi_new + qi_leg))
        q_diff = min(q_diff, q_diff_flip)
        status = "✓" if q_diff < 1e-10 else "❌"
        print(f"  {status} q diff: {q_diff:.2e}")
        
        un_new = _Ab_btbinv(Y_new, qi_new, not_Ymiss_new)
        un_leg = Ab_legacy(Y_leg, qi_leg, not_Ymiss_leg)
        
        u_diff = np.max(np.abs(un_new - un_leg))
        u_diff_flip = np.max(np.abs(un_new + un_leg))
        u_diff = min(u_diff, u_diff_flip)
        status = "✓" if u_diff < 1e-10 else "❌"
        print(f"  {status} u diff: {u_diff:.2e}")
        
        # Check convergence
        conv_new = abs((np.linalg.norm(ui_new) - np.linalg.norm(un_new))) / (np.linalg.norm(ui_new))
        conv_leg = abs((np.linalg.norm(ui_leg) - np.linalg.norm(un_leg))) / (np.linalg.norm(ui_leg))
        
        print(f"  Convergence criterion: new={conv_new:.2e}, legacy={conv_leg:.2e}")
        
        converged_new = conv_new < epsilon
        converged_leg = conv_leg < epsilon
        
        if converged_new or converged_leg:
            print(f"  CONVERGED: new={converged_new}, legacy={converged_leg}")
            if converged_new != converged_leg:
                print("  *** DIVERGENCE DETECTED: Different convergence! ***")
            break
        
        # Update for next iteration
        ui_new = un_new
        ui_leg = un_leg


if __name__ == "__main__":
    main()
