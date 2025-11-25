# Example Dependency Tracking

Central log for recording the functions each tutorial example exercises in `pyphi.py`. Update this file every time an analysis is completed and keep the table status in sync with the per-example sections below.

## Overall Tracking

| Example | Status | Section Link | Notes |
| --- | --- | --- | --- |
| PCA & PLS Example | [x] | [PCA & PLS Example](#pca--pls-example) | Completed Nov 25, 2025. Extracted 18 pyphi functions and 12 pyphi_plots functions (including internal dependencies). |
| LPLS Example | [x] | [LPLS Example](#lpls-example) | Completed Nov 25, 2025. Extracted 10 pyphi functions and 7 pyphi_plots functions. |
| LWPLS Example | [x] | [LWPLS Example](#lwpls-example) | Completed Nov 25, 2025. Extracted 14 pyphi functions and 2 pyphi_plots functions. |
| MBPLS Example | [x] | [MBPLS Example](#mbpls-example) | Completed Nov 25, 2025. Extracted 12 pyphi functions and 9 pyphi_plots functions. |

> Replace `[ ]` with `[x]` once the “Functions Required” list and notes are finalized for that example.

---

### PCA & PLS Example

Path: `examples/Basic calculations PCA and PLS/Example_Script.py`

#### Context

- **Data Source**: `Automobiles PLS.xls` workbook with three sheets: `Features` (predictor variables), `Performance` (response variables), and `CLASSID` (categorical identifiers for plotting).
- **Workflow**: Script builds both a PCA model (3 components, 5% cross-validation by elements) and two PLS models (second PLS includes cross-validation of the X-space).
- **Preprocessing**: Data is loaded directly with `pd.read_excel()` and `np.nan` for missing values; no explicit preprocessing steps in the example.
- **Plotting**: Extensive visualization using `pyphi_plots` including diagnostics, loadings maps, score plots with color-coding, and contributions plots.

#### Functions Required

**From `pyphi.py` (18 functions):**

*Public API (called directly):*
- `pca` – Build PCA model with cross-validation
- `pls` – Build PLS model with cross-validation and optional X-space CV

*Internal functions (called indirectly):*
- `contributions` – Calculate score/SPE contributions
- `f95` – 95% F-distribution critical value
- `f99` – 99% F-distribution critical value
- `hott2` – Hotelling's T² calculation
- `mean` – Mean calculation helper
- `meancenterscale` – Mean-center and/or autoscale data
- `n2z` – Convert NaN to zero (with map)
- `pca_` – Core PCA algorithm (NIPALS/SVD)
- `pls_` – Core PLS algorithm (NIPALS)
- `pls_pred` – PLS prediction (used internally for cross-validation)
- `scores_conf_int_calc` – Calculate confidence intervals for scores
- `single_score_conf_int` – Single score confidence interval helper
- `spe_ci` – SPE confidence interval calculation
- `std` – Standard deviation helper
- `unique` – Unique value extraction
- `z2n` – Convert zero back to NaN (using map)

**From `pyphi_plots.py` (12 functions):**

*Public API (called directly):*
- `contributions_plot` – Plot score contributions
- `diagnostics` – Generate Hotelling's T² and SPE plots with outlier diagnostics
- `loadings` – Bar plots of loadings
- `loadings_map` – Heatmap visualization of loadings
- `predvsobs` – Predicted vs. observed scatter plots with optional color-coding
- `r2pv` – Plot R² per variable per component (captured variance)
- `score_line` – Line plot of scores with confidence intervals and color-coding
- `score_scatter` – Score scatter plots with color-coding and optional confidence ellipses
- `vip` – Variable Importance in Projection plot
- `weighted_loadings` – Bar plots of loadings weighted by R² per variable per component

*Internal functions (called indirectly):*
- `_create_classid_` – Internal helper for CLASSID processing
- `timestr` – Timestamp string generation for plot filenames

#### Completion Checklist

- [x] Trace executed per instructions in `docs/example_dependency_plan.md`.
- [x] Function list captured above and double-checked against `pyphi.py` and `pyphi_plots.py`.
- [x] Status row updated to `[x]` with completion notes.
- [x] QA complete (peer/self review).

---

### LPLS Example

Path: `examples/LPLS/lpls_example.py`

#### Context

- **Data Source**: `lpls_dataset.xlsx` workbook with three sheets: `X` (predictor variables), `R` (linear blending/constraint matrix), and `Y` (response variables).
- **Workflow**: Fits a single Linear PLS (LPLS) model with 4 latent variables. LPLS is a variant that incorporates linear constraints/relationships through the R matrix.
- **Preprocessing**: Data is loaded directly with `pd.read_excel()` without explicit preprocessing; preprocessing is handled internally by `lpls`.
- **Plotting**: Six visualization calls covering loadings maps, loadings bar plots, weighted loadings, VIP scores, captured variance, and score scatter plots for both blends and materials.

#### Functions Required

**From `pyphi.py` (10 functions):**

*Public API (called directly):*
- `lpls` – Fit Linear PLS model with constraint matrix R

*Internal functions (called indirectly):*
- `_Ab_btbinv` – Helper for matrix inversion with constraint handling
- `f95` – 95% F-distribution critical value
- `f99` – 99% F-distribution critical value
- `hott2` – Hotelling's T² calculation
- `mean` – Mean calculation helper
- `meancenterscale` – Mean-center and/or autoscale data
- `n2z` – Convert NaN to zero (with map)
- `spe_ci` – SPE confidence interval calculation
- `std` – Standard deviation helper

**From `pyphi_plots.py` (7 functions):**

*Public API (called directly):*
- `loadings` – Bar plots of loadings
- `loadings_map` – Heatmap visualization of loadings
- `r2pv` – Plot R² per variable per component (captured variance)
- `score_scatter` – Score scatter plots with optional material/blend selection
- `vip` – Variable Importance in Projection plot
- `weighted_loadings` – Bar plots of loadings weighted by R² per variable per component

*Internal functions (called indirectly):*
- `timestr` – Timestamp string generation for plot filenames

#### Completion Checklist

- [x] Trace executed per instructions in `docs/example_dependency_plan.md`.
- [x] Function list captured above and double-checked against `pyphi.py`.
- [x] Status row updated to `[x]` with completion notes.
- [x] QA complete (peer/self review).

---

### LWPLS Example

Path: `examples/LWPLS/lwpls_example.py`

#### Context

- **Data Source**: `NIRdata_tablets.MAT` MATLAB binary file loaded via `scipy.io.loadmat()`. Matrix contains NIR spectral data (columns 3+) and pharmaceutical active substance content (column 0).
- **Preprocessing**: Two-stage spectral preprocessing pipeline: (1) Standard Normal Variate (`spectra_snv`) to normalize intensity variations, (2) Savitzky-Golay filter (`spectra_savgol`) with window size 5, polynomial order 1, derivative order 2 for smoothing and derivative computation.
- **Workflow**: Builds a single-LV PLS model on calibration data, then applies Locally Weighted PLS (LWPLS) with multiple localization parameters (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 250) on both calibration and validation sets. LWPLS is a localized prediction method that weights observations based on distance in X-space.
- **Plotting**: Single plot_spectra call to visualize preprocessed calibration spectra; two Bokeh interactive plots comparing PLS vs LWPLS RMSE across localization parameters.

#### Functions Required

**From `pyphi.py` (14 functions):**

*Public API (called directly):*
- `lwpls` – Locally weighted PLS prediction for a single observation
- `pls` – Build PLS model
- `pls_pred` – PLS prediction
- `spectra_savgol` – Savitzky-Golay spectral preprocessing (smoothing & derivatives)
- `spectra_snv` – Standard Normal Variate spectral preprocessing

*Internal functions (called indirectly):*
- `f95` – 95% F-distribution critical value
- `f99` – 99% F-distribution critical value
- `hott2` – Hotelling's T² calculation
- `mean` – Mean calculation helper
- `meancenterscale` – Mean-center and/or autoscale data
- `n2z` – Convert NaN to zero (with map)
- `pls_` – Core PLS algorithm (NIPALS)
- `spe_ci` – SPE confidence interval calculation
- `std` – Standard deviation helper

**From `pyphi_plots.py` (2 functions):**

*Public API (called directly):*
- `plot_spectra` – Interactive line plot of spectral data

*Internal functions (called indirectly):*
- `timestr` – Timestamp string generation for plot filenames

#### Completion Checklist

- [x] Trace executed per instructions in `docs/example_dependency_plan.md`.
- [x] Function list captured above and double-checked against `pyphi.py`.
- [x] Status row updated to `[x]` with completion notes.
- [x] QA complete (peer/self review).

---

### MBPLS Example

Path: `examples/Multi-block PLS/mbpls_example.py`

#### Context

- **Data Source**: `MBDataset.xlsx` workbook with 7 sheets: 6 predictor blocks (`X1` through `X6`, each representing different measurement modalities or process streams) and 1 response sheet (`Y`).
- **Block Structure**: Multi-block PLS (MBPLS) is designed for datasets where X is partitioned into multiple blocks that may have different scales/origins. Each block is treated independently during preprocessing but jointly in the PLS decomposition.
- **Workflow**: Builds a single MBPLS model with 2 latent variables spanning all 6 blocks. Then performs predictions on the same blocks using `pls_pred` (note: `pls_pred` also works on multi-block model objects).
- **Plotting**: Extensive multi-block-specific visualizations including block-wise R² contributions (`mb_r2pb`), block-wise weights (`mb_weights`), and block-wise VIP scores (`mb_vip`), plus standard MBPLS plots (loadings, VIP, score scatter, R²).

#### Functions Required

**From `pyphi.py` (12 functions):**

*Public API (called directly):*
- `mbpls` – Build Multi-block PLS model
- `pls_pred` – Prediction (works on both standard PLS and MBPLS models)

*Internal functions (called indirectly):*
- `f95` – 95% F-distribution critical value
- `f99` – 99% F-distribution critical value
- `hott2` – Hotelling's T² calculation
- `mean` – Mean calculation helper
- `meancenterscale` – Mean-center and/or autoscale data
- `n2z` – Convert NaN to zero (with map)
- `pls` – Standard PLS algorithm (called internally by MBPLS)
- `pls_` – Core PLS algorithm (NIPALS)
- `spe_ci` – SPE confidence interval calculation
- `std` – Standard deviation helper

**From `pyphi_plots.py` (9 functions):**

*Public API (called directly):*
- `loadings` – Bar plots of loadings
- `mb_r2pb` – Multi-block R² per block (contribution by block)
- `mb_vip` – Multi-block VIP scores
- `mb_weights` – Multi-block weights visualization
- `r2pv` – Plot R² per variable per component (captured variance)
- `score_scatter` – Score scatter plots
- `vip` – Variable Importance in Projection plot
- `weighted_loadings` – Bar plots of loadings weighted by R² per variable per component

*Internal functions (called indirectly):*
- `timestr` – Timestamp string generation for plot filenames

#### Completion Checklist

- [x] Trace executed per instructions in `docs/example_dependency_plan.md`.
- [x] Function list captured above and double-checked against `pyphi.py`.
- [x] Status row updated to `[x]` with completion notes.
- [x] QA complete (peer/self review).

---

Keep this document as the single source of truth. Any supplementary findings (notebooks, logs, screenshots) should be referenced in the Notes column of the tracking table.


