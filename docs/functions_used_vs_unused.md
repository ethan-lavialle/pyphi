# PyPhi Functions: Used vs Unused

Simple reference for which functions are actually used by the tutorial examples.

---

## Functions USED by Examples

### pyphi.py (54 functions)

**Always Used (5+ examples):**
- `f95`
- `f99`
- `hott2`
- `mean`
- `meancenterscale`
- `n2z`
- `spe_ci`
- `std`
- `z2n`

**Used by 2-3 examples:**
- `_Ab_btbinv` (LPLS, MBPLS, JRPLS/TPLS)
- `pls` (PCA & PLS, LWPLS, MBPLS, JRPLS/TPLS)
- `pls_` (PCA & PLS, LWPLS, MBPLS, JRPLS/TPLS)
- `pls_pred` (PCA & PLS, LWPLS, MBPLS)
- `unique` (PCA & PLS, JRPLS/TPLS)

**Used by 1 example:**
- `contributions` (PCA & PLS)
- `isin_ordered_col0` (JRPLS/TPLS)
- `jrpls` (JRPLS/TPLS)
- `jrpls_pred` (JRPLS/TPLS)
- `lpls` (LPLS)
- `lwpls` (LWPLS)
- `mbpls` (MBPLS)
- `parse_materials` (JRPLS/TPLS)
- `pca` (PCA & PLS)
- `pca_` (PCA & PLS)
- `reconcile_rows` (JRPLS/TPLS)
- `reconcile_rows_to_columns` (JRPLS/TPLS)
- `scores_conf_int_calc` (PCA & PLS)
- `single_score_conf_int` (PCA & PLS)
- `spectra_savgol` (LWPLS)
- `spectra_snv` (LWPLS)
- `tpls` (JRPLS/TPLS)
- `tpls_pred` (JRPLS/TPLS)

---

## Functions NOT USED by Any Example

### pyphi.py (50+ functions)

These functions are in `pyphi.py` but are NOT called by any of the 5 tutorial examples:

- `adapt_pls_4_pyomo`
- `bootstrap_pls`
- `bootstrap_pls_pred`
- `build_polynomial`
- `cat_2_matrix`
- `cca`
- `cca_multi`
- `clean_empty_rows`
- `clean_htmls`
- `clean_low_variances`
- `conv_pls_2_eiot`
- `evalvar`
- `export_2_gproms`
- `find`
- `findstr`
- `lpls_pred`
- `ma57_dummy_check`
- `np1D2pyomo`
- `np2D2pyomo`
- `pca_pred`
- `pls_cca`
- `prep_pca_4_MDbyNLP`
- `prep_pls_4_MDbyNLP`
- `spe`
- `spectra_autoscale`
- `spectra_baseline_correction`
- `spectra_mean_center`
- `spectra_msc`
- `varimax_`
- `varimax_rotation`
- `writeeq`

---

## Summary

**Used:** 70 functions total
- pyphi.py (core algorithms): 54 functions
- pyphi_plots.py (plotting): 16 functions

**Not Used:** 50+ functions
- pyphi.py: ~50 functions (safe to remove in trimmed builds)
- pyphi_plots.py: None (all used functions are included)

**Coverage:** ~58% of pyphi.py functions are exercised by the tutorial examples

---

## Plotting Functions (pyphi_plots.py)

### Always Used (4 examples)
- `loadings`
- `score_scatter`
- `vip`
- `weighted_loadings`

### Used by 3 examples
- `loadings_map` (PCA & PLS, LPLS, MBPLS)
- `r2pv` (PCA & PLS, LPLS, MBPLS)

### Used by 2 examples
- (none)

### Used by 1 example
- `_create_classid_` (PCA & PLS)
- `contributions_plot` (PCA & PLS)
- `diagnostics` (PCA & PLS)
- `mb_r2pb` (MBPLS)
- `mb_vip` (MBPLS)
- `mb_weights` (MBPLS)
- `plot_spectra` (LWPLS)
- `predvsobs` (PCA & PLS)

### Internal Helper (used by all plotting functions)
- `timestr` – Timestamp string generation

---

**Trace Method:** sys.settrace (Nov 25, 2025)  
**Examples Included:** PCA & PLS, LPLS, LWPLS, MBPLS, JRPLS/TPLS

