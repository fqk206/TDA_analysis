# Simplified CRLI workflow — plan

A proposal to turn the current 3-model, shared-module package into a single
**thin vertical slice** that does exactly one thing: read two-column traces,
subtract a background, and fit a CRLI hydrodynamic-radius distribution.

This document is a plan only. No code has been changed yet.

---

## 1. Goal

Today the CRLI fit is spread across two files and pulls in machinery built to
serve three different fitting models:

| File | Lines | Role |
|------|------:|------|
| `tda_processing.py` | 955 | Shared readers, preprocessing, kernels, plotting, batch runner, CLI plumbing — serves all 3 models |
| `fit_crli.py` | 246 | The CRLI fit itself, plus CLI wiring |
| `fit_dual_gaussian.py` | 210 | (other model — out of scope) |
| `fit_parametric_distribution.py` | 221 | (other model — out of scope) |

To run a CRLI fit you currently touch ~1200 lines of code, most of which exists
to keep three models interchangeable. The aim is a **single file** (~300 lines)
that you can read top to bottom and understand in one sitting, with the fit math
unchanged.

---

## 2. Decisions (from requirements discussion)

| Question | Decision | Consequence |
|----------|----------|-------------|
| Input format | **Two-column only** | Delete the FidaNeo reader and all method-table / time-unit-autodetect / metadata parsing. Temperature becomes a config constant. |
| Free-fluorophore mode | **Keep both** (`fit` and `fixed`) | The two NNLS branches in the fit loop stay. Fit math unchanged. |
| Outputs | **Keep full** (CSVs + JSON + plots) | All CSV/JSON/plot outputs stay, but only the CRLI-relevant ones. |
| Spike handling | **Drop** | Delete spike detection, the spike mask, and all spike columns from outputs. Assumes clean traces. |

---

## 3. What gets removed, and why it's safe

These are pure "book keeping" / abstraction that the CRLI-only, two-column slice does
not need:

- **FidaNeo reader + parsing** — `read_fidaneo_txt`, `get_block`,
  `parse_simple_key_values`, `parse_method_table`, `numeric_rows_from_text`,
  `fidaneo_time_scale`, `read_trace` dispatch.
  *(~120 lines.)* Not used in two-column mode.
- **Spike handling** — `detect_spikes`, `odd_window`, `contiguous_groups`, the
  `spike_*` settings, and the `spike_mask` / `spike_count` / `spike_times_s`
  fields. *(~70 lines.)*
- **Multi-model abstraction** — the `run_batch(cfg, model_name, fit_one)`
  callable-injection pattern, `make_fitter`, and the `model_name`-conditional
  logic in plotting/summary. With one model these all collapse to straight-line
  code.
- **Shared CLI plumbing** — `add_common_arguments` / `apply_common_args`
  (~80 lines) plus the per-script argparse blocks. Replaced by a tiny argparse
  (or just edit-the-constants-at-top, matching the existing "Run by editing"
  style).
- **Parity placeholder stats** — fields the CRLI fit only sets to `NaN` so its
  output matches the parametric/dual models: `n_fit_parameters`,
  `fit_center_parameter_Rh_nm`, `fit_width_parameter`,
  `fit_width_parameter_units`. Dropped.
- **Settings fields for other models** — `dual_gaussian_peak_fraction`,
  `normal_sd_bounds_nm`, `lognormal_sigma_bounds`, `n_Rh_grid_parametric`,
  `Rh_center_bounds_nm` (only used by the parametric initial-guess path),
  `fit_signal_min/max`, `max_fit_points`, `t0_half_window_s`, all `fidaneo_*`
  and `spike_*` and `two_column_*`-beyond-the-essentials.
- **Heavyweight containers** — `Trace` (4 fields), `PreparedTrace` (11 fields),
  `FitResult` (17 fields). Slimmed to only what the CRLI slice reads.

What is **kept unchanged** so results don't move:

- The Taylor kernel `rh_kernel` and its physics constants.
- `trapezoid_weights`, `second_difference_matrix`, `robust_std`,
  `quadratic_peak_time`.
- `distribution_stats` (median/mean/mode/percentiles/span) — genuinely useful,
  cheap, kept verbatim.
- The CRLI solve itself: log-spaced `Rh` grid, alpha scan, NNLS with the
  second-difference roughness penalty, and the "smoothest alpha within
  `residual_factor` × best residual" selection rule. Both `fit` and `fixed`
  free-fluorophore branches retained exactly.
- The linear edge-based **background subtraction** (`choose_baseline_window` +
  `initial_linear_baseline`) and the noise estimate.

---

## 4. Proposed structure: one file

`fit_crli_simple.py` — read top to bottom:

```text
# 1. CONFIG          module-level constants (the slimmed Settings)
# 2. READ            read_trace(path)            -> time_s, signal
# 3. PREPROCESS      prepare(time, signal)       -> baseline, corrected,
#                                                   t0, peak_height, normalized
# 4. FIT             fit_crli(prep)              -> distribution + stats
#    helpers:        rh_kernel, trapezoid_weights,
#                    second_difference_matrix, distribution_stats
# 5. OUTPUT          save plots + distribution CSV + summary row
# 6. main()          glob files, loop, write summary CSV + run_settings.json
```

No shared module, no callable injection, no `model_name` branching. The other
two fit scripts and `tda_processing.py` are left untouched on `main`; this slice
simply doesn't depend on them.

### Slimmed config (constants at top of file)

```python
# Input
DATA_PATTERN          = "*.csv"
OUTPUT_DIR            = "tda_results/crli"
TIME_COLUMN           = 0
SIGNAL_COLUMN         = 1
DELIMITER             = ","
TIME_UNIT             = "s"      # "s", "min", or "ms"
TEMPERATURE_C         = 25.0

# Physics
CAPILLARY_RADIUS_M    = 37.5e-6
VISCOSITY_PA_S        = 8.90e-4
KERNEL_REFERENCE_RH_NM = 25.0

# Background subtraction
BASELINE_MIN_WINDOW_S      = 15.0
BASELINE_MAX_WINDOW_S      = 90.0
BASELINE_FRACTION_OF_TRACE = 0.15

# Free fluorophore
FREE_RH_NM            = 0.59
FREE_FRACTION_MODE    = "fit"    # "fit" or "fixed"
FREE_FRACTION_FIXED   = 0.05

# CRLI
RH_GRID_BOUNDS_NM     = (1.5, 200.0)
N_RH_GRID             = 120
CRLI_ALPHAS           = np.logspace(-8, 8, 80)
CRLI_RESIDUAL_FACTOR  = 1.10
CRLI_FIT_STRIDE       = 4
CRLI_SIGNAL_MIN       = 0.03
CRLI_SIGNAL_MAX       = 1.02

# Output
SAVE_PLOTS            = True
```

That is ~25 knobs versus ~50 in the current `Settings`, and every one is used by
this slice.

---

## 5. The pipeline, end to end

### Step 1 — Read (two-column)
`np.genfromtxt` the file, take the time/signal columns, scale time to seconds
(`s`/`min`/`ms`), drop non-finite rows, sort by time. Returns two arrays. No
metadata dict, no format dispatch. *(~12 lines, from `read_two_column_file`.)*

### Step 2 — Background subtraction
- Pick a baseline window: `clip(fraction * duration, [min, max])`.
- Fit a straight line through the **edge medians** (first window, last window)
  and subtract it → `corrected`.
- Noise estimate = `robust_std` of corrected signal in the edge windows.
- Locate the peak: `argmax` then `quadratic_peak_time` for sub-sample `t0`.
- `peak_height = corrected(t0)`; `normalized = corrected / peak_height`.

Drops the spike mask entirely — `good` is just `isfinite`. *(~25 lines.)*

### Step 3 — CRLI fit (both free modes, unchanged)
- Build the fit mask: good points with `CRLI_SIGNAL_MIN ≤ s ≤ CRLI_SIGNAL_MAX`,
  restricted to the chosen side (`rising`/`falling`/`both`), then strided.
- Rh grid: `geomspace(bounds, N_RH_GRID)`, optionally narrowed around the
  apparent slope-based radius (keep `crli_radius_bounds` — small and useful).
- Build kernel `A_col`, free-fluorophore column `K_free`, roughness matrix `L`.
- For each `alpha`: augment with `sqrt(alpha)·L`, solve `nnls`.
  - `fixed` branch: free contribution folded in via the fixed ratio.
  - `fit` branch: free fraction is an extra NNLS column.
- Pick the smoothest alpha whose residual ≤ `RESIDUAL_FACTOR × best`.
- Normalize to a PDF, compute `distribution_stats`.

This is lifted essentially verbatim from `fit_crli.py:73-209`, minus the
parity-placeholder stat fields.

### Step 4 — Output (full)
Per file:
- `<stem>_crli_distribution.csv` — `Rh_nm, pdf_per_ln_Rh, pdf_per_nm, cdf`.
- Plots: preprocessing (raw + baseline + t0), fit overlay, residuals,
  decomposition (baseline / +free / +colloid), distribution, and the two
  CRLI alpha-scan diagnostics.
- One row appended to `crli_summary.csv`.

Run-level:
- `run_settings.json` — dump of the config constants.
- `crli_summary.csv` — one row per file with the **trimmed** stat set:

```text
file, t0_s, free_fraction, free_peak_height_fraction, colloid_peak_height_fraction,
colloid_mean_Rh_nm, colloid_median_Rh_nm, colloid_mode_Rh_nm, colloid_sd_Rh_nm, colloid_cv,
colloid_p05/p10/p90/p95_Rh_nm, colloid_span_p10_p90_nm, colloid_span_p05_p95_nm,
residual_rms_raw, residual_rms_normalized_to_peak, crli_alpha, crli_alpha_at_scan_boundary,
crli_rh_min_nm, crli_rh_max_nm, noise_raw, success, error
```

Gone from the summary: the FidaNeo metadata columns (`pressure_mbar`,
`capillary_temp_C`, `method_time_s`, `time_unit_detected`), spike columns, and
the `NaN` parametric placeholders.

---

## 6. Proposed file size

| | Current (to run CRLI) | Proposed |
|---|---:|---:|
| Files touched | 2 (+955-line shared module) | 1 |
| Lines | ~1200 | ~300 |
| Config knobs | ~50 | ~25 |
| Result/stat fields | ~30 (4 are NaN placeholders) | ~22 |
| Input readers | 2 | 1 |
| Dataclasses | 4 | 0–1 (a small result dict/namedtuple) |

---

## 7. Migration / how to land it

1. Create `fit_crli_simple.py` on the `Simplification` branch as the single
   self-contained slice described above.
2. Leave `tda_processing.py`, `fit_dual_gaussian.py`,
   `fit_parametric_distribution.py` in place for now (no need to delete to prove
   the slice works).
3. Sanity check: run both scripts on the same two-column file and confirm the
   distribution CSV and key stats match the current `fit_crli.py` output
   (the math is copied, so they should be identical to floating-point noise).
4. Once happy, decide whether to retire the old files or keep them for the other
   two models.

---

## 8. Open questions / things to confirm

- **Run style**: tiny argparse, edit-the-constants-at-top, or both? The current
  package supports both; the simplest slice is constants-only, but a 5-line
  argparse for `--data`/`--output`/`--side` is cheap. *(Proposing: constants +
  a minimal argparse for data/output/side.)*
- **Side default**: keep `rising` as the default fit side? *(Proposing: yes,
  matches current `FIT_SIDE`.)*
- **Rh-bound narrowing**: keep the slope-based `crli_radius_bounds` auto-narrow,
  or always use the full `RH_GRID_BOUNDS_NM`? It pulls in `apparent_rh_from_slope`
  + `initial_colloid_center`. *(Proposing: keep it — ~20 lines, improves the
  grid placement.)*
- **Plots**: keep all 7 plot types, or is the fit-overlay + distribution +
  alpha-scan subset enough? You said "full", so the plan keeps all of them;
  easy to trim later.
