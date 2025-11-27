## JRPLS/TPLS Example Requirements

**Context**  
Running `examples/JRPLS and TPLS/jrpls_tpls_example.py` exercises only a narrow slice of `pyphi.py`. Everything else in the module is unused for this workflow and can be removed or relocated if your sole goal is to keep this example operational.

**How this was verified**  
1. Created a virtual environment and installed the project in editable mode.  
2. Ran the example from its data directory so the Excel workbook resolves:  
   `python -m trace --listfuncs jrpls_tpls_example.py`  
3. Parsed `pyphi.py` with the `ast` module to catalogue every top-level function.  
4. Compared the trace output against the catalogue to identify what executed.

## Functions Required by the Example

- `_Ab_btbinv`
- `f95`
- `f99`
- `hott2`
- `jrpls`
- `jrpls_pred`
- `mean`
- `meancenterscale`
- `n2z`
- `parse_materials`
- `pls`
- `pls_`
- `reconcile_rows`
- `reconcile_rows_to_columns`
- `spe_ci`
- `std`
- `tpls`
- `tpls_pred`
- `unique`
- `isin_ordered_col0`

These functions (plus their shared imports and constants) are the minimum necessary for `jrpls_tpls_example.py` to run end-to-end and generate the plots and predictions shown in the current repo.

## Functions Not Touched by the Example

Everything else in `pyphi.py`—while useful for other workflows—was not invoked during the `jrpls_tpls_example.py` run and is a candidate for deletion or extraction if you are preparing a trimmed build:

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
- `contributions`
- `conv_pls_2_eiot`
- `evalvar`
- `export_2_gproms`
- `find`
- `findstr`
- `lpls`
- `lpls_pred`
- `lwpls`
- `ma57_dummy_check`
- `mbpls`
- `np1D2pyomo`
- `np2D2pyomo`
- `pca`
- `pca_`
- `pca_pred`
- `pls_cca`
- `pls_pred`
- `prep_pca_4_MDbyNLP`
- `prep_pls_4_MDbyNLP`
- `scores_conf_int_calc`
- `single_score_conf_int`
- `spe`
- `spectra_autoscale`
- `spectra_baseline_correction`
- `spectra_mean_center`
- `spectra_msc`
- `spectra_savgol`
- `spectra_snv`
- `varimax_`
- `varimax_rotation`
- `writeeq`
- `z2n`