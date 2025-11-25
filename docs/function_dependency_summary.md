# PyPhi Function Dependency Summary

Consolidated overview of functions exercised across all tutorial examples. This document serves as a quick reference for understanding which pyphi routines are actively used and which are candidates for removal or relocation in trimmed builds.

**Last Updated:** Nov 25, 2025  
**Methodology:** `sys.settrace` function call tracking on all 4 main examples + JRPLS/TPLS reference

---

## Summary Table

| Example | pyphi.py | pyphi_plots.py | Data Format | Key Features |
|---------|----------|----------------|-------------|--------------|
| **PCA & PLS** | 18 | 12 | Excel (`.xls`) | PCA/PLS with cross-validation, extensive plotting |
| **LPLS** | 10 | 7 | Excel (`.xlsx`) | Linear PLS with constraint matrix, loadings/VIP |
| **LWPLS** | 14 | 2 | MATLAB (`.MAT`) | Spectrum preprocessing, localized predictions |
| **MBPLS** | 12 | 9 | Excel (`.xlsx`) | Multi-block decomposition, block-wise visualization |
| **JRPLS/TPLS** | 20 | — | Excel (`.xlsx`) | Hierarchical material-blend data, material reconciliation |

---

## Core pyphi.py Functions

Functions that appear across **multiple examples** (core algorithm suite):

### Statistical & Helper Functions (Used by all or most)
- `f95` – 95% F-distribution critical value
- `f99` – 99% F-distribution critical value
- `hott2` – Hotelling's T² calculation
- `mean` – Mean calculation helper
- `meancenterscale` – Mean-center and/or autoscale data
- `n2z` – Convert NaN to zero (with map)
- `spe_ci` – SPE confidence interval calculation
- `std` – Standard deviation helper
- `z2n` – Convert zero back to NaN (using map)
- `unique` – Unique value extraction

### Core Decomposition Algorithms
- `pls` – Standard PLS (called by most examples, either directly or indirectly)
- `pls_` – Core PLS algorithm (NIPALS implementation)
- `pls_pred` – PLS predictions (works on PLS, LPLS, LWPLS, MBPLS models)

---

## Example-Specific pyphi.py Functions

Functions used by specific examples only:

| Function | Example(s) | Purpose |
|----------|-----------|---------|
| `pca` | PCA & PLS | Principal Component Analysis |
| `pca_` | PCA & PLS | Core PCA algorithm (NIPALS/SVD) |
| `contributions` | PCA & PLS | Score/SPE contribution calculation |
| `scores_conf_int_calc` | PCA & PLS | Confidence interval calculation for scores |
| `single_score_conf_int` | PCA & PLS | Single score CI helper |
| `pls_cca` | PCA & PLS | PLS-CCA variant (OPLS alternative) |
| `_Ab_btbinv` | LPLS, MBPLS, JRPLS/TPLS | Matrix inversion with constraint handling |
| `lpls` | LPLS | Linear PLS with constraint matrix |
| `spectra_snv` | LWPLS | Standard Normal Variate preprocessing |
| `spectra_savgol` | LWPLS | Savitzky-Golay spectral filtering |
| `lwpls` | LWPLS | Locally weighted PLS predictions |
| `mbpls` | MBPLS | Multi-block PLS decomposition |
| `jrpls` | JRPLS/TPLS | Joint Regression PLS (hierarchical materials/blends) |
| `jrpls_pred` | JRPLS/TPLS | JRPLS prediction |
| `tpls` | JRPLS/TPLS | Trilinear PLS (temporal extension) |
| `tpls_pred` | JRPLS/TPLS | TPLS prediction |
| `parse_materials` | JRPLS/TPLS | Parse material-blend relationship matrix |
| `reconcile_rows` | JRPLS/TPLS | Align observation rows across dataframes |
| `reconcile_rows_to_columns` | JRPLS/TPLS | Map rows to column identifiers |
| `isin_ordered_col0` | JRPLS/TPLS | Check ordered membership in column 0 |

---

## Core pyphi_plots.py Functions

Functions that appear across **multiple examples**:

### Common Visualization Functions
- `loadings` – Bar plots of loadings (4/4 examples using plots)
- `loadings_map` – Heatmap visualization of loadings (3/4)
- `r2pv` – Plot R² per variable per component (3/4)
- `score_scatter` – Score scatter plots (4/4)
- `vip` – Variable Importance in Projection plot (3/4)
- `weighted_loadings` – Weighted loadings bar plots (4/4)
- `timestr` – Timestamp string generation (all examples using plots)

### Example-Specific Visualization Functions
| Function | Example(s) | Purpose |
|----------|-----------|---------|
| `score_line` | PCA & PLS | Line plot of scores with CI and color-coding |
| `diagnostics` | PCA & PLS | Hotelling's T² and SPE diagnostics |
| `contributions_plot` | PCA & PLS | Score/SPE contribution visualization |
| `predvsobs` | PCA & PLS | Predicted vs. observed scatter |
| `_create_classid_` | PCA & PLS | CLASSID processing helper |
| `plot_spectra` | LWPLS | Interactive spectral data visualization |
| `mb_r2pb` | MBPLS | Multi-block R² per block |
| `mb_weights` | MBPLS | Multi-block weights visualization |
| `mb_vip` | MBPLS | Multi-block VIP scores |

---

## Functions NOT Used by Examples

These functions in `pyphi.py` were NOT invoked during any of the 5 example runs and are candidates for removal in trimmed builds:

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

## Usage Patterns by Workflow

### Basic PLS/PCA Workflows (PCA & PLS, LPLS)
**Minimal set:** `pca`/`pls` + preprocessing + diagnostics (T², SPE) + plotting

### Spectral/Chemometric Workflows (LWPLS)
**Additions:** Spectral preprocessing (`spectra_snv`, `spectra_savgol`), localization (`lwpls`)

### Multi-Structure Data (MBPLS, JRPLS/TPLS)
**Additions:** Multi-block decomposition (`mbpls`, `jrpls`, `tpls`), data reconciliation (`reconcile_*`, `parse_materials`)

---

## Shared Dependencies (Internal Calls)

Functions that are only called **indirectly** by other functions (not directly by users):

- `_Ab_btbinv` – Helper for constraint matrix inversions (called by `lpls`, `mbpls`, `jrpls`, `tpls`)
- `mean`, `std` – Called by `meancenterscale`
- `n2z`, `z2n` – NaN handling (called by most algorithms)
- `pls_` – Called by `pls`, `jrpls`, `tpls`
- `timestr` – Called by all plotting functions
- `_create_classid_` – Called by color-coded plot functions

---

## Recommendations for Trimmed Builds

**If keeping only one example workflow:**

| Target | Keep | Remove |
|--------|------|--------|
| PCA/PLS only | `pca`, `pca_`, `pls`, `pls_pred` + core helpers | Everything else in pyphi.py |
| LPLS only | `lpls` + `pls` + core helpers | `pca*`, `jrpls*`, `tpls*`, `lwpls`, `mbpls`, spectral functions |
| LWPLS only | `lwpls`, `pls`, `pls_pred`, `spectra_snv`, `spectra_savgol` + core helpers | Remove other algorithms |
| MBPLS only | `mbpls`, `pls`, `pls_pred` + core helpers | `pca*`, `jrpls*`, `tpls*`, `lwpls`, `lpls`, spectral functions |
| JRPLS/TPLS only | `jrpls`, `jrpls_pred`, `tpls`, `tpls_pred`, `parse_materials`, `reconcile_*` + core helpers | Remove all other algorithms |

**Critical shared functions to preserve:**
- All of "Core pyphi.py Functions" section (used by all)
- `meancenterscale`, `n2z`, `z2n` (internal dependencies)
- `pls_`, `pls` (foundation algorithms)

---

## Verification Notes

- PCA & PLS, LPLS, LWPLS, MBPLS: Traced Nov 25, 2025 using `sys.settrace` method
- JRPLS/TPLS: Reference from `docs/jrpls_tpls_minimal.md` (earlier verification)
- All function lists validated against source code and cross-checked for completeness

See `docs/example_dependency_tracking.md` and `docs/example_dependency_plan.md` for detailed per-example breakdowns.

