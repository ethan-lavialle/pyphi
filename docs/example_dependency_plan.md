# Example Dependency Audit Plan

This playbook explains how to catalog the minimum set of `pyphi.py` functions exercised by the main tutorial examples. Record all findings in `docs/example_dependency_tracking.md`; use this file purely as a coordination aid.

## Prerequisites

- Python environment with `pyphi` installed in editable mode plus dependencies from `requirements.txt`.
- Access to the raw data files referenced by each example (Excel workbooks or `.MAT` binaries) in the same directories as the scripts.
- Optional but recommended: create a clean virtual environment per example to avoid cross-run artifacts.

## Standard Workflow

1. **Set up environment**: From the project root, create/activate a venv and install dependencies:
   ```bash
   cd /path/to/pyphi-min-test
   python3 -m venv venv && source venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. **Run the trace using `sys.settrace`**: The built-in `python -m trace --listfuncs` does NOT capture internal function calls reliably. Instead, use Python's `sys.settrace` to capture ALL functions (both public API and internal helpers):

   ```python
   import sys
   sys.path.insert(0, '../..')  # Adjust path to project root

   pyphi_functions = set()
   pyphi_plots_functions = set()

   def trace_calls(frame, event, arg):
       if event == 'call':
           filename = frame.f_code.co_filename
           func_name = frame.f_code.co_name
           if 'pyphi.py' in filename and not func_name.startswith('<'):
               pyphi_functions.add(func_name)
           elif 'pyphi_plots.py' in filename and not func_name.startswith('<'):
               pyphi_plots_functions.add(func_name)
       return trace_calls

   sys.settrace(trace_calls)
   
   # ... run the example script code here ...
   
   sys.settrace(None)
   
   print('pyphi.py functions:', sorted(pyphi_functions))
   print('pyphi_plots.py functions:', sorted(pyphi_plots_functions))
   ```

3. **Important**: For plotting examples, use a non-interactive matplotlib backend to avoid GUI blocking:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   plt.ioff()
   # ... run plotting calls ...
   plt.close('all')
   ```

4. **Categorize functions**: Separate the results into:
   - **Public API** (called directly in the example script)
   - **Internal functions** (called indirectly by the public API)

5. **Document the results** in `docs/example_dependency_tracking.md` under the appropriate example section and update the overall tracking table.

6. Only check off the per-example checklist once the tracking doc reflects the final findings.

## How to Update Tracking

1. Open `docs/example_dependency_tracking.md`.
2. Locate the section that matches the example you just analyzed (anchors are provided below each checklist item).
3. Paste your finalized “Functions Required” bullets directly under that section, keeping alphabetical order if practical.
4. Add contextual notes (data path quirks, preprocessing calls, etc.) in the “Context” subsection.
5. In the “Overall Tracking” table, change the checkbox from `[ ]` to `[x]`, include a short completion note, and—if needed—link to any supplemental diagnostics or notebooks.

## Example Checklists

### Example Script (PCA & PLS)
Path: `examples/Basic calculations PCA and PLS/Example_Script.py`

- [x] Run the trace from the script's directory so the Excel files resolve.
- [x] Extract the function list and summarize it in `docs/example_dependency_tracking.md#pca-and-pls-example`.
- [x] Update the tracking table row for "PCA & PLS Example" to `[x]` with notes (e.g., date, tool used).
- [x] Re-read the entry for accuracy, then mark this checklist item as complete.

**Completed Nov 25, 2025** – Used `sys.settrace` method. Found 18 pyphi functions and 12 pyphi_plots functions.

### LPLS Example
Path: `examples/LPLS/lpls_example.py`

- [x] Run the trace with access to `lpls_dataset.xlsx`.
- [x] Populate `docs/example_dependency_tracking.md#lpls-example` with context plus the required function list.
- [x] Flip the tracking table checkbox for "LPLS Example" to `[x]` and capture any caveats.
- [x] Confirm everything renders correctly in Markdown before checking this item off.

**Completed Nov 25, 2025** – Used `sys.settrace` method. Found 10 pyphi functions and 7 pyphi_plots functions. Note: `_Ab_btbinv` is an internal helper for the LPLS constraint matrix handling.

### LWPLS Example
Path: `examples/LWPLS/lwpls_example.py`

- [x] Ensure the `.MAT` file is accessible, then run the trace.
- [x] Document the invoked functions under `docs/example_dependency_tracking.md#lwpls-example`, noting preprocessing helpers (`spectra_*`, etc.).
- [x] Update the "LWPLS Example" row in the tracking table to `[x]` once complete.
- [x] Perform a quick peer review (self-check or second agent) before checking this list off.

**Completed Nov 25, 2025** – Used `sys.settrace` method. Found 14 pyphi functions and 2 pyphi_plots functions. Note: Includes spectrum preprocessing functions (`spectra_snv`, `spectra_savgol`) which are key to the example workflow.

### MBPLS Example
Path: `examples/Multi-block PLS/mbpls_example.py`

- [x] Run the trace while keeping the workbook (`MBDataset.xlsx`) in the working directory.
- [x] Summarize required functions in `docs/example_dependency_tracking.md#mbpls-example`, including any multi-block specifics.
- [x] Switch the tracking table checkbox for "MBPLS Example" to `[x]` with supporting notes.
- [x] Confirm no outstanding TODOs remain for this example, then check this item complete.

**Completed Nov 25, 2025** – Used `sys.settrace` method. Found 12 pyphi functions and 9 pyphi_plots functions. Note: Includes multi-block-specific plotting functions (`mb_r2pb`, `mb_weights`, `mb_vip`).

## QA Notes

- When uncertain about whether a helper is active, re-run the trace with `-t` to log calls or inspect the source for indirect imports.
- Keep formatting consistent with `docs/jrpls_tpls_minimal.md` to ease diff reviews.
- If data paths move, re-verify that the trace run uses the updated locations before finalizing the function list.
- Avoid deleting functions from `pyphi.py` on the basis of these audits without first backing up the full module.


