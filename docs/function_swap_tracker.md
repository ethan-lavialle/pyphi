# Function Swap Tracker

## Pending Candidates

| Function | Proposed Replacement | Notes |
|----------|----------------------|-------|
| `pyphi.n2z` / `pyphi.z2n` | `numpy.nan_to_num` / `numpy.where` | Use `mask = np.isnan(X)` once, feed `np.nan_to_num(X, nan=0.0)` to the linear algebra routines, then restore missing entries with `np.where(mask, np.nan, X_clean)` to preserve current tuple contract. Audit every caller that unpacks `(X_clean, mask)` so they keep working with the new helper and ensure dtype casting stays consistent. |
| `pyphi.mean` | `numpy.nanmean` | Drop the bespoke NaN handling in favor of `np.nanmean(X, axis=0, keepdims=True)`. Verify downstream code that expects column vectors still receives a 2-D result and add guards for all-NaN columns (NumPy returns `NaN`, matching today’s silent divide-by-zero behavior). |
| `pyphi.std` | `numpy.nanstd` | Replace manual variance math with `np.nanstd(X, axis=0, ddof=1, keepdims=True)` so we retain sample-std semantics. Review any consumers that rely on the exact shape/ordering of the returned array and confirm they handle potential `NaN` outputs when a column is entirely missing. |
| `pyphi.meancenterscale` | `sklearn.preprocessing.StandardScaler` (+ `sklearn.impute.SimpleImputer`) | Fit an imputer/scaler pipeline per mode: `with_mean` / `with_std` toggled to mirror the flags, reapply the original NaN mask after scaling so downstream routines (especially cross-validation blocks that call `n2z`) still see missingness. We need to reconcile that `StandardScaler` uses population variance (ddof=0); if parity with the current sample variance is critical, extend the pipeline with a custom wrapper to rescale by `sqrt(n/(n-1))`. |


