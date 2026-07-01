#!/usr/bin/env python3
"""Pure Laplace / CRLI inversion of a Taylorgram - no separate free-fluorophore term.

Thin CONFIG-block wrapper over :class:`tda.models.laplace.LaplaceModel`. Where
``fit_crli_simple.py`` models the free fluorophore as a fixed-Rh component
bolted onto a colloid grid, this script does the opposite: it solves for a
single smooth, non-negative size distribution over a grid that reaches down
to very small Rh, and lets the inversion recover the small
(free-fluorophore-like) species on its own. The fit math lives in
:mod:`tda.models.laplace`; this file is the "edit CONFIG and press Run"
surface plus the batch loop.

Reading/preprocessing and physics constants come from the same CONFIG shape
as ``fit_crli_simple.py`` so the two stay directly comparable.

Run:
    python fit_laplace.py
"""

from __future__ import annotations

from tda.io.trace import ReaderParams
from tda.models.laplace import LaplaceModel, LaplaceParams
from tda.physics import PhysicsParams
from tda.preprocessing.pipeline import BaselineParams, PreprocessingPipeline
from tda.preprocessing.steps import Despike
from tda.reporting.batch import print_laplace_result, run_batch
from tda.reporting.io import write_distribution_csv
from tda.reporting.plots import save_laplace_plots

import fit_crli_simple as crli   # shared data pattern / physics / preprocessing CONFIG


# Fraction of the peak fit (rising side): 0.50 = rising half, 1.00 = whole peak.
PEAK_FRACTION = 0.51

# Inversion grid - starts near zero so the small species sits inside the grid.
RH_MIN_NM = 0.2
RH_MAX_NM = 200.0
N_RH_GRID = 160

# Cutoff used only for reporting/plotting the emergent "small species" component.
SMALL_CUTOFF_NM = 2.0
REFERENCE_FREE_RH_NM = 0.59          # drawn on the distribution plot for orientation

OUTPUT_DIR = "output/laplace"


def build_model() -> LaplaceModel:
    physics = PhysicsParams(temperature_c=crli.TEMPERATURE_C, capillary_radius_m=crli.CAPILLARY_RADIUS_M,
                             viscosity_pa_s=crli.VISCOSITY_PA_S, kernel_reference_rh_nm=crli.KERNEL_REFERENCE_RH_NM)
    preprocessing = PreprocessingPipeline(
        steps=(Despike(window=crli.DESPIKE_WINDOW, n_sigma=crli.DESPIKE_SIGMA),) if crli.DESPIKE else (),
        baseline=BaselineParams(crli.BASELINE_MIN_WINDOW_S, crli.BASELINE_MAX_WINDOW_S, crli.BASELINE_FRACTION_OF_TRACE),
    )
    reader = ReaderParams(file_format=crli.FILE_FORMAT, time_column=crli.TIME_COLUMN, signal_column=crli.SIGNAL_COLUMN,
                           delimiter=crli.DELIMITER, skiprows=crli.SKIPROWS, comments=crli.COMMENTS,
                           time_unit=crli.TIME_UNIT)
    params = LaplaceParams(
        fit_side=crli.FIT_SIDE, peak_fraction=PEAK_FRACTION, rh_min_nm=RH_MIN_NM, rh_max_nm=RH_MAX_NM,
        n_rh_grid=N_RH_GRID, small_cutoff_nm=SMALL_CUTOFF_NM, alphas=crli.CRLI_ALPHAS,
        residual_factor=crli.CRLI_RESIDUAL_FACTOR, fit_stride=crli.CRLI_FIT_STRIDE,
        signal_min=crli.CRLI_SIGNAL_MIN, signal_max=crli.CRLI_SIGNAL_MAX, t0_mode=crli.T0_MODE,
        t0_manual_s=crli.T0_MANUAL_S, t0_refine_bound_s=crli.T0_REFINE_BOUND_S,
    )
    return LaplaceModel(params=params, physics=physics, preprocessing=preprocessing, reader=reader)


def _write_csv(result, out_dir, stem) -> None:
    write_distribution_csv(result, out_dir, stem, "laplace")


def _save_plots(result, prep, out_dir, stem) -> None:
    save_laplace_plots(result, out_dir, stem, reference_free_rh_nm=REFERENCE_FREE_RH_NM)


def main() -> None:
    model = build_model()
    run_batch(model, crli.DATA_PATTERN, OUTPUT_DIR, "laplace_summary.csv", _save_plots, _write_csv,
              print_laplace_result, save_plots_enabled=crli.SAVE_PLOTS)


if __name__ == "__main__":
    main()
