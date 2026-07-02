#!/usr/bin/env python3
"""Parametric (log-normal mixture) fit of a Taylorgram - the constrained
counterpart to the free-form CRLI / Laplace inversion.

Thin CONFIG-block wrapper over :class:`tda.models.parametric.ParametricModel`.
Fits the same physics as ``fit_laplace.py`` with a *small, fixed* number of
log-normal components (default two: a small "dye" species and a colloid) via
variable projection (VARPRO). The fit math lives in :mod:`tda.models.parametric`;
this file is the "edit CONFIG and press Run" surface plus the batch loop.

The distribution plot overlays the free-form Laplace inversion for a
cross-check; the Laplace fit is a plain injected collaborator (a
:class:`~tda.models.laplace.LaplaceModel`), not a hard dependency baked into
the plotting code.

See ``docs/parametric.md`` for the design.

Run:
    python fit_parametric.py
"""

from __future__ import annotations

import warnings

import fit_crli_simple as crli   # shared data pattern / physics / preprocessing CONFIG
import fit_laplace               # for the distribution-plot overlay only

from tda.io.trace import ReaderParams
from tda.models.parametric import ParametricModel, ParametricParams
from tda.physics import PhysicsParams
from tda.preprocessing.pipeline import BaselineParams, PreprocessingPipeline
from tda.preprocessing.steps import Despike
from tda.reporting.batch import print_parametric_result, run_batch
from tda.reporting.io import write_decomposition_csv, write_distribution_csv
from tda.reporting.plots import parametric_component_role, save_parametric_plots


# =============================================================================
# CONFIG - edit these and run the script directly
# =============================================================================

N_COMPONENTS = 2
INIT_RH_NM = (0.75, 23.0)            # initial centers (one per component)
INIT_SIGMA_LN = (0.05, 0.25)         # initial log-widths
RH_BOUNDS_NM = (0.3, 200.0)          # per-center bounds
SIGMA_LN_BOUNDS = (0.02, 1.5)        # per-width bounds
FIT_T0 = True                        # refine t0 as a nuisance parameter
T0_BOUND_S = 10.0                    # t0 is bounded to prep.t0 +/- this (seconds)
RH_GRID = (0.2, 300.0, 400)          # (min_nm, max_nm, n) dense grid for the model integral

# Reuse the inversion's fit-window controls so the comparison is apples-to-apples.
PEAK_FRACTION = crli.PEAK_FRACTION   # which part of the peak to fit (see fit_crli_simple)
REFERENCE_FREE_RH_NM = 0.59          # drawn on the distribution plot for orientation
COST_SLICES = True                   # produce the (optional) cost-vs-parameter slice plot

OUTPUT_DIR = "output/parametric"


def build_model() -> ParametricModel:
    physics = PhysicsParams(temperature_c=crli.TEMPERATURE_C, capillary_radius_m=crli.CAPILLARY_RADIUS_M,
                             viscosity_pa_s=crli.VISCOSITY_PA_S, kernel_reference_rh_nm=crli.KERNEL_REFERENCE_RH_NM)
    preprocessing = PreprocessingPipeline(
        steps=(Despike(window=crli.DESPIKE_WINDOW, n_sigma=crli.DESPIKE_SIGMA),) if crli.DESPIKE else (),
        baseline=BaselineParams(crli.BASELINE_MIN_WINDOW_S, crli.BASELINE_MAX_WINDOW_S, crli.BASELINE_FRACTION_OF_TRACE),
    )
    reader = ReaderParams(file_format=crli.FILE_FORMAT, time_column=crli.TIME_COLUMN, signal_column=crli.SIGNAL_COLUMN,
                           delimiter=crli.DELIMITER, skiprows=crli.SKIPROWS, comments=crli.COMMENTS,
                           time_unit=crli.TIME_UNIT)
    params = ParametricParams(
        n_components=N_COMPONENTS, init_rh_nm=INIT_RH_NM, init_sigma_ln=INIT_SIGMA_LN, rh_bounds_nm=RH_BOUNDS_NM,
        sigma_ln_bounds=SIGMA_LN_BOUNDS, fit_t0=FIT_T0, t0_bound_s=T0_BOUND_S, rh_grid=RH_GRID,
        fit_side=crli.FIT_SIDE, peak_fraction=PEAK_FRACTION, fit_stride=crli.CRLI_FIT_STRIDE,
        signal_min=crli.CRLI_SIGNAL_MIN, signal_max=crli.CRLI_SIGNAL_MAX,
    )
    return ParametricModel(params=params, physics=physics, preprocessing=preprocessing, reader=reader)


def _laplace_overlay(prep, side: str):
    """Free-form Laplace distribution for the overlay; None if it can't be computed."""
    try:
        return fit_laplace.build_model().fit(prep, side=side)
    except Exception as exc:
        warnings.warn(f"Laplace overlay unavailable: {exc}")
        return None


def _write_csv(result, out_dir, stem) -> None:
    write_distribution_csv(result, out_dir, stem, "parametric")
    k = int(result.stats["n_components"])
    labels = {f"comp{i}": parametric_component_role(i, k) for i in range(k)}
    write_decomposition_csv(result, out_dir, stem, "parametric", component_labels=labels)


def _make_save_plots(model: ParametricModel):
    def _save_plots(result, prep, out_dir, stem) -> None:
        overlay = _laplace_overlay(prep, result.stats["fit_side"])
        save_parametric_plots(result, prep, out_dir, stem, laplace_overlay=overlay,
                               reference_free_rh_nm=REFERENCE_FREE_RH_NM, cost_slices=COST_SLICES)
    return _save_plots


def main() -> None:
    model = build_model()
    run_batch(model, crli.DATA_PATTERN, OUTPUT_DIR, "parametric_summary.csv", _make_save_plots(model), _write_csv,
              print_parametric_result, save_plots_enabled=crli.SAVE_PLOTS)


if __name__ == "__main__":
    main()
