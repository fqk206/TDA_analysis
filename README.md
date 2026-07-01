# TDA analysis — v0.2

Taylor-dispersion-analysis (TDA) fitting library for Taylorgrams: despike/bin
a raw trace, subtract a background, and recover a hydrodynamic-radius (Rh)
distribution by one of four methods (CRLI, Laplace, parametric, or a
multi-trace global-kinetics fit).

Version 0.2 is a refactor of a working set of analysis scripts into an
sklearn-style library (`tda/`), with the original scripts kept as thin
CONFIG-block wrappers so the "edit constants, press Run" workflow still
works. See [CHANGELOG](#changelog) for what changed.

## Install

```bash
pip install -e .
```

(`numpy`, `scipy`, `matplotlib` are the only runtime dependencies;
`scikit-learn` is additionally needed for `factor_series.py`.)

## Quick start

```python
from tda.models.crli import CRLIModel, CRLIParams

model = CRLIModel(params=CRLIParams(free_rh_nm=0.6, peak_fraction=0.55))
result = model.fit_file("data/sample.txt")

print(result.stats["colloid_median_Rh_nm"])
print(result.stats["free_fraction_detector_weighted"])
```

`fit_file` is `prepare(path) -> fit(prepared)` in one call. Every model
(`CRLIModel`, `LaplaceModel`, `ParametricModel`) has this same shape; a
`FitResult` always carries the same fields (`Rh_nm`, `pdf_logRh`, `stats`,
`components`, ...) regardless of which model produced it.

Because config is a plain dataclass instead of module globals, multiple
configurations can coexist — a parameter sweep is just a list comprehension,
not a loop that mutates shared state between iterations:

```python
sweep = [model.set_params(free_rh_nm=x) for x in (0.5, 0.6, 0.7)]
results = [m.fit_file("data/sample.txt") for m in sweep]
```

`GlobalSeriesModel` fits many traces (a FidaNeo time series) at once, so its
entry point is `fit_folder`/`load_traces`+`fit` instead of `fit_file`:

```python
from tda.models.global_series import GlobalSeriesModel

model = GlobalSeriesModel()
result, info = model.fit_folder("data_series/series_01")
print(result.fitted["all"].tau_h)   # relaxation time constant, hours
```

## Layout

```
tda/
  physics.py            Taylor kernel (rh_kernel) + PhysicsParams (temperature,
                         capillary radius, viscosity, kernel reference Rh)
  numerics.py            Config-free math: trapezoid_weights, second_difference_matrix,
                         robust_std, quadratic_peak_time, distribution_stats
  fitting.py              Fit-window selection (select_fit_mask) and t0 policy
                         (t0_seed / refine_t0), shared by every model
  io/
    trace.py               read_trace / ReaderParams (plain two-column or FidaNeo)
    fidaneo.py               FidaNeo .txt file parser (FidaNeoRecord, load_folder, ...)
  preprocessing/
    steps.py                Despike, Bin - composable trace-conditioning steps
    pipeline.py              PreprocessingPipeline: ordered steps + baseline/t0/normalize
  models/
    base.py                  FitModel / SingleTraceModel - the shared estimator contract
    result.py                 FitResult - one result shape for every single-trace model
    crli.py                    CRLIModel + CRLIParams
    laplace.py                  LaplaceModel + LaplaceParams
    parametric.py                 ParametricModel + ParametricParams (+ ParamPack,
                               design_matrix, solve_amplitudes as standalone functions)
    global_series.py               GlobalSeriesModel + GlobalSeriesParams
  reporting/
    plots.py                  Plot primitives (generic + per-model compositions)
    batch.py                    run_batch: glob -> prepare -> fit -> save -> summary CSV
    io.py                       safe_filename, write_rows, write_distribution_csv

fit_crli_simple.py / fit_laplace.py / fit_parametric.py / fit_global_series.py
                       Thin CONFIG-block wrappers over the library above - still the
                       "edit constants and press Run" entry points.
factor_series.py, diagnose_free_size.py, simulate_two_gaussian.py
                       Diagnostic/exploratory tools built on the library.
preprocess.py          Original standalone despike_mask/bin_trace (unchanged; the
                       canonical copies now live in tda/preprocessing/steps.py).
fidaneo.py              Original standalone FidaNeo parser (unchanged; canonical
                       copy now lives in tda/io/fidaneo.py).
tests/parity/           Scripts that diff the library's output against the
                       pre-refactor scripts on real data (see Testing below).
legacy/                 Pre-2024 archived originals, out of scope.
docs/                   Design notes (simpl.md, parametric.md, test_findings.md).
```

## Design principles

**Composition for preprocessing, inheritance for the estimator contract.**
Every model *holds* a `PreprocessingPipeline` (`model.preprocessing`) rather
than inheriting despike/bin behaviour through mixins — this mirrors
scikit-learn's own split (`Pipeline` composes transforms; `BaseEstimator` is
inherited for the shared contract). A new denoising/despiking/binning
routine is a new step class dropped into the pipeline's `steps` list; no
model code changes. What *is* inherited is `FitModel`/`SingleTraceModel`,
which give every model `prepare()`, `fit_file()`, and `config_dict()` for
free, because "is a fit model with a params object" is a genuine is-a
relationship.

**Config is a dataclass, not a module global.** `CRLIParams`, `LaplaceParams`,
`ParametricParams`, `GlobalSeriesParams` replace the ~20-50 module-level
CONFIG constants each original script used to read directly. This is what
makes sweeps, multi-instance use, and `model.get_params()`/`set_params()`
possible — none of that works with shared mutable globals.

**One `FitResult` shape for every single-trace model.** `CRLIModel`,
`LaplaceModel`, and `ParametricModel` used to return three different
containers (a dataclass and two different ad-hoc dicts). Now they all return
`tda.models.result.FitResult`, with per-model decomposition (CRLI's
free/colloid, Laplace's low/high, parametric's comp0/comp1/...) expressed as
an open `components: dict[str, np.ndarray]`. This is what lets
`tda.reporting.plots.plot_decomposition` (and the rest of the reporting
layer) be written once instead of three times.

**Physics and fit-window/t0 logic live in their own modules.** Previously
every script did `import fit_crli_simple as crli` and reached into its
namespace for the Taylor kernel, physics constants, and fit-window helpers —
an accidental "CRLI is secretly the core module" hierarchy. `tda.physics` and
`tda.fitting` now hold that logic independently, imported directly by
whichever model needs it.

**Behaviour-preserving.** Every phase of the refactor was checked against
the pre-refactor scripts on real data before being trusted — see Testing.

## Preprocessing steps

```python
from tda.preprocessing.pipeline import PreprocessingPipeline
from tda.preprocessing.steps import Despike, Bin

# Default: despike only (matches the historical DESPIKE=True, BIN=False).
pipeline = PreprocessingPipeline()

# Despike, then bin into 0.5 s windows.
pipeline = PreprocessingPipeline(steps=(Despike(window=4, n_sigma=18.0), Bin(width_s=0.5)))

model = CRLIModel(preprocessing=pipeline)
```

A step implements `apply(t, y, good) -> (t, y, good)`:
- A **filter** step (like `Despike`) returns same-length arrays and ANDs its
  own new exclusions onto the incoming `good` mask.
- A **resample** step (like `Bin`) returns a new-length grid and a fresh
  `good` mask sized to it, built only from the incoming-`good` samples.

Adding a new denoising routine means writing one class with that method and
putting it in `steps=(...)` — no model class needs to change.

## Testing

`tests/parity/` record the behaviour-preservation checks run during the
0.1 -> 0.2 migration: each script diffs the library against the pre-refactor
scripts on real files in `data/` and `data_series/series_01`, at bitwise
precision where no optimizer is involved and tight relative tolerance where
NNLS/`least_squares`/`minimize_scalar` are.

They compare against `fit_crli_simple.py`'s *old* flat API
(`crli.rh_kernel`, `crli.prepare_from_trace`, ...), which no longer exists on
`HEAD` now that `fit_crli_simple.py` itself has been rewired to the new
library (phase 6) — so they will not run as-is against the current tree.
To re-run one, check out the pre-refactor scripts into a scratch location
(e.g. `git worktree add /tmp/tda-v01 <pre-refactor commit>`) and point the
script's `sys.path` at that copy instead of the live repo root. They're kept
as a record of what was verified and a template for verifying any future
migration of this shape, not a live regression suite.

Some (CRLI/Laplace with `t0_mode="fit"`) were deliberately restricted to one
or two real files rather than the whole `data/` folder, because that mode
nests a Brent search around an 80-point alpha scan and can take minutes per
file — the t0-refinement code path is generic, not file-specific, so one
confirmed file was sufficient.

## Changelog

**0.2** — Refactor into the `tda/` library described above. `CRLIModel`,
`LaplaceModel`, `ParametricModel`, `GlobalSeriesModel` replace direct use of
`fit_crli_simple.py`'s globals; `fit_crli_simple.py`/`fit_laplace.py`/
`fit_parametric.py`/`fit_global_series.py` become thin wrappers over them.
`factor_series.py`, `diagnose_free_size.py`, `simulate_two_gaussian.py`
updated to use the library instead of reaching into `fit_crli_simple`'s
former internals. Numeric outputs unchanged from pre-refactor (see Testing).

**0.1** — Prior state: `fit_crli_simple.py`, `fit_laplace.py`,
`fit_parametric.py`, `fit_global_series.py` as independent scripts sharing
code via `import fit_crli_simple as crli`.
