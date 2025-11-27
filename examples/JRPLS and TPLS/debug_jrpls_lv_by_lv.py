#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to trace JRPLS LV-by-LV calculations between new and legacy implementations.

This script runs JRPLS with verbose output at each LV to identify where divergence begins.
"""

import sys
import os
import numpy as np
import pandas as pd
import datetime

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

# Import internal utilities from both implementations
from pyphi.utils import meancenterscale, n2z, std
from pyphi._internal import _Ab_btbinv

# For legacy we need to import from the pyphi_legacy module
import pyphi_legacy as phi_legacy


def compare_arrays_brief(name: str, arr_new: np.ndarray, arr_legacy: np.ndarray, 
                         rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, float]:
    """Brief comparison of two arrays."""
    arr_new = np.atleast_1d(np.asarray(arr_new).flatten())
    arr_legacy = np.atleast_1d(np.asarray(arr_legacy).flatten())
    
    if arr_new.shape != arr_legacy.shape:
        return False, np.inf
    
    # Direct comparison
    if np.allclose(arr_new, arr_legacy, rtol=rtol, atol=atol):
        return True, np.max(np.abs(arr_new - arr_legacy))
    
    # Sign-flipped comparison
    if np.allclose(arr_new, -arr_legacy, rtol=rtol, atol=atol):
        return True, np.max(np.abs(arr_new + arr_legacy))
    
    return False, np.max(np.abs(arr_new - arr_legacy))


def load_data():
    """Load the JRPLS/TPLS dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "jrpls_tpls_dataset.xlsx")
    
    print(f"\nLoading data from: {data_file}")
    
    # Use legacy implementation for data loading
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
    
    # Build dictionaries
    Ri = {}
    for j, m in zip(JR_, materials):
        Ri[m] = j
    Xi = {}
    for x_, m in zip(xc, materials):
        Xi[m] = x_
    
    return Xi, Ri, quality, process, materials


def jrpls_debug_new(Xi, Ri, Y, A, debug_lv=None):
    """JRPLS with debug output for comparing to legacy - NEW implementation."""
    X = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        Xaux = Xi[k]
        if isinstance(Xaux, pd.DataFrame):
            X_ = np.array(Xaux.values[:, 1:]).astype(float)
        else:
            X_ = Xaux.copy()
        X.append(X_)

    if isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)
    else:
        Y_ = Y.copy()

    R = []
    for k in materials:
        Raux = Ri[k]
        if isinstance(Raux, pd.DataFrame):
            R_ = np.array(Raux.values[:, 1:]).astype(float)
        else:
            R_ = Raux.copy()
        R.append(R_)

    # Preprocessing
    x_mean = []
    x_std = []
    jr_scale = []
    r_mean = []
    r_std = []
    not_Xmiss = []
    not_Rmiss = []
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
    
    debug_info = []
    
    for a in list(range(A)):
        lv_debug = {"lv": a + 1}
        
        # Select column with largest variance in Y as initial guess
        ui = Y_[:, [np.argmax(std(Y_))]]
        lv_debug["initial_ui_norm"] = np.linalg.norm(ui)
        
        Converged = False
        num_it = 0
        
        while not Converged:
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
            js = np.array([y for x in si for y in x])
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)
            
            ri = []
            for i, X_ in enumerate(X):
                # Step 3. ri= (Xs)/(s's)
                ri_ = _Ab_btbinv(X_, si[i], not_Xmiss[i])
                ri.append(ri_)
            
            # Calculating the Joint-r and Joint-R
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
            
            if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui)) < epsilon:
                Converged = True
            
            if num_it > maxit:
                Converged = True
            
            if Converged:
                lv_debug["num_iterations"] = num_it
                lv_debug["final_ti"] = ti.flatten()[:5].tolist()  # First 5 elements
                lv_debug["final_ti_norm"] = np.linalg.norm(ti)
                lv_debug["final_qi"] = qi.flatten()[:5].tolist()
                lv_debug["final_qi_norm"] = np.linalg.norm(qi)
                
                # Store si for each material
                lv_debug["si_norms"] = [np.linalg.norm(s) for s in si]
                lv_debug["ri_norms"] = [np.linalg.norm(r) for r in ri]
                lv_debug["hi_norms"] = [np.linalg.norm(h) for h in hi]
                
                pi = []
                for i, R_ in enumerate(R):
                    pi_ = _Ab_btbinv(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    vi_ = _Ab_btbinv(X_.T, ri[i], not_Xmiss[i].T)
                    vi.append(vi_)
                
                lv_debug["pi_norms"] = [np.linalg.norm(p) for p in pi]
                lv_debug["vi_norms"] = [np.linalg.norm(v) for v in vi]
                
                for i in np.arange(len(R)):
                    R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                    X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                
                # Post-deflation residuals
                lv_debug["X_residual_norms"] = [np.linalg.norm(X_) for X_ in X]
                lv_debug["R_residual_norms"] = [np.linalg.norm(R_) for R_ in R]
                lv_debug["Y_residual_norm"] = np.linalg.norm(Y_)
                
            else:
                num_it = num_it + 1
                ui = un
        
        debug_info.append(lv_debug)
    
    return debug_info


def jrpls_debug_legacy(Xi, Ri, Y, A, debug_lv=None):
    """JRPLS with debug output for comparing - LEGACY implementation."""
    # Import legacy functions
    from pyphi_legacy import meancenterscale as mcs_legacy
    from pyphi_legacy import n2z as n2z_legacy  
    from pyphi_legacy import std as std_legacy
    from pyphi_legacy import _Ab_btbinv as Ab_legacy
    
    X = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        Xaux = Xi[k]
        if isinstance(Xaux, pd.DataFrame):
            X_ = np.array(Xaux.values[:, 1:]).astype(float)
        else:
            X_ = Xaux.copy()
        X.append(X_)

    if isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)
    else:
        Y_ = Y.copy()

    R = []
    for k in materials:
        Raux = Ri[k]
        if isinstance(Raux, pd.DataFrame):
            R_ = np.array(Raux.values[:, 1:]).astype(float)
        else:
            R_ = Raux.copy()
        R.append(R_)

    # Preprocessing
    x_mean = []
    x_std = []
    jr_scale = []
    r_mean = []
    r_std = []
    not_Xmiss = []
    not_Rmiss = []
    X__ = []
    R__ = []
    
    for X_i, R_i in zip(X, R):
        X_, x_mean_, x_std_ = mcs_legacy(X_i)
        R_, r_mean_, r_std_ = mcs_legacy(R_i)
        
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
    
    debug_info = []
    
    for a in list(range(A)):
        lv_debug = {"lv": a + 1}
        
        # Select column with largest variance in Y as initial guess
        ui = Y_[:, [np.argmax(std_legacy(Y_))]]
        lv_debug["initial_ui_norm"] = np.linalg.norm(ui)
        
        Converged = False
        num_it = 0
        
        while not Converged:
            # Step 1. h=R'u/u'u
            hi = []
            for i, R_ in enumerate(R):
                hi_ = Ab_legacy(R_.T, ui, not_Rmiss[i].T)
                hi.append(hi_)
            
            si = []
            for i, X_ in enumerate(X):
                # Step 2. s = X'h/(h'h)
                si_ = Ab_legacy(X_.T, hi[i], not_Xmiss[i].T)
                si.append(si_)
            
            # Normalize joint s to unit length.
            js = np.array([y for x in si for y in x])
            for i in np.arange(len(si)):
                si[i] = si[i] / np.linalg.norm(js)
            
            ri = []
            for i, X_ in enumerate(X):
                # Step 3. ri= (Xs)/(s's)
                ri_ = Ab_legacy(X_, si[i], not_Xmiss[i])
                ri.append(ri_)
            
            # Calculating the Joint-r and Joint-R
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
            ti = Ab_legacy(R_, jr, not_Rmiss_)
            
            # Step 5 q=Y't/t't
            qi = Ab_legacy(Y_.T, ti, not_Ymiss.T)
            
            # Step 5 un=(Yq)/(q'q)
            un = Ab_legacy(Y_, qi, not_Ymiss)
            
            if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / (np.linalg.norm(ui)) < epsilon:
                Converged = True
            
            if num_it > maxit:
                Converged = True
            
            if Converged:
                lv_debug["num_iterations"] = num_it
                lv_debug["final_ti"] = ti.flatten()[:5].tolist()
                lv_debug["final_ti_norm"] = np.linalg.norm(ti)
                lv_debug["final_qi"] = qi.flatten()[:5].tolist()
                lv_debug["final_qi_norm"] = np.linalg.norm(qi)
                
                lv_debug["si_norms"] = [np.linalg.norm(s) for s in si]
                lv_debug["ri_norms"] = [np.linalg.norm(r) for r in ri]
                lv_debug["hi_norms"] = [np.linalg.norm(h) for h in hi]
                
                pi = []
                for i, R_ in enumerate(R):
                    pi_ = Ab_legacy(R_.T, ti, not_Rmiss[i].T)
                    pi.append(pi_)
                vi = []
                for i, X_ in enumerate(X):
                    vi_ = Ab_legacy(X_.T, ri[i], not_Xmiss[i].T)
                    vi.append(vi_)
                
                lv_debug["pi_norms"] = [np.linalg.norm(p) for p in pi]
                lv_debug["vi_norms"] = [np.linalg.norm(v) for v in vi]
                
                for i in np.arange(len(R)):
                    R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                    X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                
                lv_debug["X_residual_norms"] = [np.linalg.norm(X_) for X_ in X]
                lv_debug["R_residual_norms"] = [np.linalg.norm(R_) for R_ in R]
                lv_debug["Y_residual_norm"] = np.linalg.norm(Y_)
                
            else:
                num_it = num_it + 1
                ui = un
        
        debug_info.append(lv_debug)
    
    return debug_info


def main():
    """Main comparison routine."""
    print("=" * 80)
    print("JRPLS LV-BY-LV DEBUG COMPARISON")
    print("=" * 80)
    
    # Load data
    Xi, Ri, quality, process, materials = load_data()
    
    n_components = 4
    
    print(f"\nRunning JRPLS with {n_components} components...")
    print("\n" + "-" * 80)
    
    # Run both implementations with debug output
    print("Running NEW implementation...")
    debug_new = jrpls_debug_new(Xi, Ri, quality, n_components)
    
    print("Running LEGACY implementation...")
    debug_legacy = jrpls_debug_legacy(Xi, Ri, quality, n_components)
    
    # Compare LV by LV
    print("\n" + "=" * 80)
    print("COMPARISON BY LATENT VARIABLE")
    print("=" * 80)
    
    for a in range(n_components):
        print(f"\n{'='*40}")
        print(f"LV #{a+1}")
        print(f"{'='*40}")
        
        d_new = debug_new[a]
        d_leg = debug_legacy[a]
        
        print(f"\nIterations: new={d_new['num_iterations']}, legacy={d_leg['num_iterations']}")
        print(f"Initial u norm: new={d_new['initial_ui_norm']:.10f}, legacy={d_leg['initial_ui_norm']:.10f}")
        
        # Compare ti
        ti_match, ti_diff = compare_arrays_brief("ti", d_new['final_ti'], d_leg['final_ti'])
        print(f"\nFinal t[{a+1}]:")
        print(f"  new:    {d_new['final_ti']}")
        print(f"  legacy: {d_leg['final_ti']}")
        print(f"  norm: new={d_new['final_ti_norm']:.10f}, legacy={d_leg['final_ti_norm']:.10f}")
        print(f"  Match: {ti_match}, max_diff: {ti_diff:.2e}")
        
        # Compare qi
        qi_match, qi_diff = compare_arrays_brief("qi", d_new['final_qi'], d_leg['final_qi'])
        print(f"\nFinal q[{a+1}]:")
        print(f"  new:    {d_new['final_qi']}")
        print(f"  legacy: {d_leg['final_qi']}")
        print(f"  norm: new={d_new['final_qi_norm']:.10f}, legacy={d_leg['final_qi_norm']:.10f}")
        print(f"  Match: {qi_match}, max_diff: {qi_diff:.2e}")
        
        # Compare per-material norms
        print(f"\nPer-material s norms:")
        for i, (sn, sl) in enumerate(zip(d_new['si_norms'], d_leg['si_norms'])):
            diff = abs(sn - sl)
            status = "✓" if diff < 1e-8 else "❌"
            print(f"  {status} MAT{i+1}: new={sn:.10f}, legacy={sl:.10f}, diff={diff:.2e}")
        
        print(f"\nPer-material r norms:")
        for i, (rn, rl) in enumerate(zip(d_new['ri_norms'], d_leg['ri_norms'])):
            diff = abs(rn - rl)
            status = "✓" if diff < 1e-8 else "❌"
            print(f"  {status} MAT{i+1}: new={rn:.10f}, legacy={rl:.10f}, diff={diff:.2e}")
        
        print(f"\nPer-material h norms:")
        for i, (hn, hl) in enumerate(zip(d_new['hi_norms'], d_leg['hi_norms'])):
            diff = abs(hn - hl)
            status = "✓" if diff < 1e-8 else "❌"
            print(f"  {status} MAT{i+1}: new={hn:.10f}, legacy={hl:.10f}, diff={diff:.2e}")
        
        print(f"\nPost-deflation residual norms:")
        print(f"  X residuals:")
        for i, (xn, xl) in enumerate(zip(d_new['X_residual_norms'], d_leg['X_residual_norms'])):
            diff = abs(xn - xl)
            status = "✓" if diff < 1e-8 else "❌"
            print(f"    {status} MAT{i+1}: new={xn:.10f}, legacy={xl:.10f}, diff={diff:.2e}")
        
        print(f"  R residuals:")
        for i, (rn, rl) in enumerate(zip(d_new['R_residual_norms'], d_leg['R_residual_norms'])):
            diff = abs(rn - rl)
            status = "✓" if diff < 1e-8 else "❌"
            print(f"    {status} MAT{i+1}: new={rn:.10f}, legacy={rl:.10f}, diff={diff:.2e}")
        
        yn_diff = abs(d_new['Y_residual_norm'] - d_leg['Y_residual_norm'])
        yn_status = "✓" if yn_diff < 1e-8 else "❌"
        print(f"  {yn_status} Y residual: new={d_new['Y_residual_norm']:.10f}, legacy={d_leg['Y_residual_norm']:.10f}, diff={yn_diff:.2e}")


if __name__ == "__main__":
    main()
