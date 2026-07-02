#!/usr/bin/env python3
"""Simplified CRLI fit for two-column Taylor dispersion traces.

Thin CONFIG-block wrapper over :class:`tda.models.crli.CRLIModel`. Read a
two-column (time, signal) file, subtract a linear background, and solve for a
smooth, non-negative colloid hydrodynamic-radius distribution by constrained
regularized linear inversion (CRLI), plus the fixed-Rh free fluorophore
component. The fit math lives in :mod:`tda.models.crli`; this file is just the
"edit CONFIG and press Run" surface plus the batch loop.

Run by editing the CONFIG block below and pressing Run, or from the shell:

    python fit_crli_simple.py --data "*.csv" --output tda_results/crli --side rising
"""

from __future__ import annotations

import argparse

import numpy as np

from tda.io.trace import ReaderParams
from tda.models.crli import CRLIModel, CRLIParams
from tda.physics import PhysicsParams
from tda.preprocessing.pipeline import BaselineParams, PreprocessingPipeline
from tda.preprocessing.steps import Bin, Despike
from tda.reporting.batch import print_crli_result, run_batch
from tda.reporting.io import write_decomposition_csv, write_distribution_csv
from tda.reporting.plots import save_crli_plots


# =============================================================================
# CONFIG - edit these and run the script directly
# =============================================================================

# Input
DATA_PATTERN = "data/*.txt"
OUTPUT_DIR = "output/simple/"
FILE_FORMAT = "old"                  # "old" (plain two-column) or "fidaneo" (FidaNeo .txt export)
TIME_COLUMN = 0                      # "old" format only
SIGNAL_COLUMN = 1                    # "old" format only
DELIMITER: str | None = None         # None splits on any whitespace (tabs/spaces); "old" format only
SKIPROWS = 0                         # "old" format only
COMMENTS = "#"                       # "old" format only
TIME_UNIT = "min"                    # "s", "min", or "ms"; "old" format only (FidaNeo is always minutes)
TEMPERATURE_C = 25.0

# Physics
CAPILLARY_RADIUS_M = 37.5e-6         # radius, not diameter
VISCOSITY_PA_S = 8.90e-4
KERNEL_REFERENCE_RH_NM = 25.0        # numerical scale factor only

# Background subtraction
BASELINE_MIN_WINDOW_S = 15.0
BASELINE_MAX_WINDOW_S = 90.0
BASELINE_FRACTION_OF_TRACE = 0.15

# Despiking - narrow artefact spikes are removed before peak/t0 detection (see
# tda.preprocessing.steps.Despike: Hampel identifier, flags samples deviating
# from a rolling median by > DESPIKE_SIGMA local MADs). DESPIKE_WINDOW is in
# samples and must exceed the widest spike but stay well below the (broad)
# Taylor peak - 31 samples is ~2.6 s at the usual 12 Hz.
DESPIKE = True
DESPIKE_WINDOW = 8
DESPIKE_SIGMA = 8.0

# Binning - optional averaging of the raw trace into fixed-width time bins
# before the fit (tda.preprocessing.steps.Bin; opt-in, off by default). Runs
# after despiking. Keep BIN_WIDTH_S well below the (broad) Taylor peak.
BIN = False
BIN_WIDTH_S = 0.5

# Free fluorophore
FREE_RH_NM = 0.59
FREE_FRACTION_MODE = "fit_bounded"   # "fit", "fit_bounded", or "fixed"
FREE_FRACTION_FIXED = 0.05           # used when mode == "fixed"
FREE_FRACTION_BOUNDS = (0.01, 0.10)  # (lower, upper); used when mode == "fit_bounded"

# CRLI
FIT_SIDE = "rising"                  # "rising", "falling", or "both"
PEAK_FRACTION = 0.50                 # for the rising side: fraction of the peak fit;
#                                      0.50 = rising half (up to t0), 1.00 = whole peak.
RH_GRID_BOUNDS_NM = (1.5, 200.0)
RH_GRID_AUTONARROW = False           # if True, auto-zoom the grid around the apparent size
RH_CENTER_BOUNDS_NM = (1.5, 300.0)   # only used when RH_GRID_AUTONARROW is True
N_RH_GRID = 120
CRLI_ALPHAS = np.logspace(-8, 8, 80)
CRLI_RESIDUAL_FACTOR = 1.10          # smoothest alpha with residual <= factor * best
RIDGE_RHO = 0.0                      # ridge weight as a fraction of alpha (beta = RIDGE_RHO * alpha)
CRLI_FIT_STRIDE = 1
CRLI_SIGNAL_MIN = -10.00
CRLI_SIGNAL_MAX = 1.02

# t0 handling
#   "auto"   - lock t0 to the value found in preprocessing (default)
#   "manual" - lock t0 to T0_MANUAL_S, a hand-set / eyeballed value
#   "fit"    - refine t0 in the fit, bounded to the seed +/- T0_REFINE_BOUND_S
T0_MODE = "fit"
T0_MANUAL_S: float | None = None
T0_REFINE_BOUND_S = 5.0

# Output
SAVE_PLOTS = True
SHOW_PLOTS = False                   # not yet wired through the shared plotting layer (see reporting/plots.py)

# Diagnostics
TRUNCATION_RH_NM = 4.0               # truncation-test cutoff: signal carried by colloid mass below this Rh


# =============================================================================
# Wiring CONFIG -> the library
# =============================================================================

def build_model() -> CRLIModel:
    physics = PhysicsParams(temperature_c=TEMPERATURE_C, capillary_radius_m=CAPILLARY_RADIUS_M,
                             viscosity_pa_s=VISCOSITY_PA_S, kernel_reference_rh_nm=KERNEL_REFERENCE_RH_NM)
    steps: list = [Despike(window=DESPIKE_WINDOW, n_sigma=DESPIKE_SIGMA)] if DESPIKE else []
    if BIN:
        steps.append(Bin(width_s=BIN_WIDTH_S))
    preprocessing = PreprocessingPipeline(
        steps=tuple(steps),
        baseline=BaselineParams(BASELINE_MIN_WINDOW_S, BASELINE_MAX_WINDOW_S, BASELINE_FRACTION_OF_TRACE),
    )
    reader = ReaderParams(file_format=FILE_FORMAT, time_column=TIME_COLUMN, signal_column=SIGNAL_COLUMN,
                           delimiter=DELIMITER, skiprows=SKIPROWS, comments=COMMENTS, time_unit=TIME_UNIT)
    params = CRLIParams(
        free_rh_nm=FREE_RH_NM, free_fraction_mode=FREE_FRACTION_MODE, free_fraction_fixed=FREE_FRACTION_FIXED,
        free_fraction_bounds=FREE_FRACTION_BOUNDS, fit_side=FIT_SIDE, peak_fraction=PEAK_FRACTION,
        rh_grid_bounds_nm=RH_GRID_BOUNDS_NM, rh_grid_autonarrow=RH_GRID_AUTONARROW,
        rh_center_bounds_nm=RH_CENTER_BOUNDS_NM, n_rh_grid=N_RH_GRID, alphas=CRLI_ALPHAS,
        residual_factor=CRLI_RESIDUAL_FACTOR, ridge_rho=RIDGE_RHO, fit_stride=CRLI_FIT_STRIDE,
        signal_min=CRLI_SIGNAL_MIN, signal_max=CRLI_SIGNAL_MAX, t0_mode=T0_MODE, t0_manual_s=T0_MANUAL_S,
        t0_refine_bound_s=T0_REFINE_BOUND_S, truncation_rh_nm=TRUNCATION_RH_NM,
    )
    return CRLIModel(params=params, physics=physics, preprocessing=preprocessing, reader=reader)


def _write_csv(result, out_dir, stem) -> None:
    write_distribution_csv(result, out_dir, stem, "crli")
    write_decomposition_csv(result, out_dir, stem, "crli", component_labels={"free": "free_fluorophore"})


def run(data_pattern: str, output_dir: str, side: str) -> None:
    model = build_model()
    if side != model.params.fit_side:
        model = model.set_params(fit_side=side)
    run_batch(model, data_pattern, output_dir, "crli_summary.csv", save_crli_plots, _write_csv,
              print_crli_result, save_plots_enabled=SAVE_PLOTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified CRLI fit for two-column TDA traces.")
    parser.add_argument("--data", dest="data_pattern", default=DATA_PATTERN, help="Input glob, e.g. '*.csv'.")
    parser.add_argument("--output", dest="output_dir", default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--format", dest="file_format", choices=["old", "fidaneo"], default=FILE_FORMAT,
                        help="Input file format: 'old' (plain two-column) or 'fidaneo' (FidaNeo export).")
    parser.add_argument("--side", choices=["rising", "falling", "both"], default=FIT_SIDE, help="Taylorgram side to fit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global FILE_FORMAT
    FILE_FORMAT = args.file_format
    run(args.data_pattern, args.output_dir, args.side)


if __name__ == "__main__":
    main()
