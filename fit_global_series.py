#!/usr/bin/env python3
"""Global exponential-kinetics fit over a FidaNeo time series.

Thin CONFIG-block wrapper over
:class:`tda.models.global_series.GlobalSeriesModel`. A series of Taylorgrams
of the *same* sample is measured repeatedly over many hours; the slow change
in the large (colloid) species is modeled as an exponential relaxation to a
new plateau, shared across the whole series, while the dye component shape is
a shared constant and the component amplitudes (and t0) are fit per trace.
The fit math lives in :mod:`tda.models.global_series`; this file is the
"edit CONFIG and press Run" surface plus the output/plotting orchestration.

Run:
    python fit_global_series.py
    python fit_global_series.py --data data_series/series_01 --group pooled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fit_crli_simple as crli   # shared physics CONFIG (temperature, capillary, viscosity)

from tda.models.global_series import GlobalSeriesModel, GlobalSeriesParams
from tda.physics import PhysicsParams
from tda.preprocessing.pipeline import BaselineParams, PreprocessingPipeline
from tda.preprocessing.steps import Despike
from tda.reporting.io import safe_filename, write_rows
from tda.reporting.plots import (
    plot_example_fits,
    plot_kinetics,
    plot_preprocessing,
    plot_residuals_free_vs_global,
    plot_trace_diagnostics,
)


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = "data_series/series_01"
FILE_PATTERN = "*.txt"

GROUP_MODE = "pooled"                 # "pooled" (one curve) or "per_sample"
FIT_SIDE = crli.FIT_SIDE              # Taylorgram side to fit ("rising")
PEAK_FRACTION = crli.PEAK_FRACTION    # part of the peak to fit (see fit_crli_simple)

RH_GRID = (0.2, 300.0, 300)           # (min_nm, max_nm, n) dense grid for the model integral

# Shared shape initial guesses / bounds.
DYE_RH_NM = 0.75
DYE_SIGMA_LN = 0.10
COLLOID_SIGMA_LN = 0.40
RH_BOUNDS_NM = (1.0, 200.0)           # bounds on the colloid centers Rh_0 / Rh_plateau
DYE_RH_BOUNDS_NM = (0.3, 3.0)
SIGMA_LN_BOUNDS = (0.002, 1.5)
TAU_BOUNDS_H = (0.05, 200.0)

OUTPUT_DIR = "output/global_series_01"


def build_model() -> GlobalSeriesModel:
    physics = PhysicsParams(temperature_c=crli.TEMPERATURE_C, capillary_radius_m=crli.CAPILLARY_RADIUS_M,
                             viscosity_pa_s=crli.VISCOSITY_PA_S, kernel_reference_rh_nm=crli.KERNEL_REFERENCE_RH_NM)
    preprocessing = PreprocessingPipeline(
        steps=(Despike(window=crli.DESPIKE_WINDOW, n_sigma=crli.DESPIKE_SIGMA),) if crli.DESPIKE else (),
        baseline=BaselineParams(crli.BASELINE_MIN_WINDOW_S, crli.BASELINE_MAX_WINDOW_S, crli.BASELINE_FRACTION_OF_TRACE),
    )
    params = GlobalSeriesParams(
        group_mode=GROUP_MODE, fit_side=FIT_SIDE, peak_fraction=PEAK_FRACTION, fit_stride=crli.CRLI_FIT_STRIDE,
        signal_min=crli.CRLI_SIGNAL_MIN, signal_max=crli.CRLI_SIGNAL_MAX, rh_grid=RH_GRID, dye_rh_nm=DYE_RH_NM,
        dye_sigma_ln=DYE_SIGMA_LN, colloid_sigma_ln=COLLOID_SIGMA_LN, rh_bounds_nm=RH_BOUNDS_NM,
        dye_rh_bounds_nm=DYE_RH_BOUNDS_NM, sigma_ln_bounds=SIGMA_LN_BOUNDS, tau_bounds_h=TAU_BOUNDS_H,
        rh_center_bounds_nm=crli.RH_CENTER_BOUNDS_NM, t0_mode=crli.T0_MODE, t0_manual_s=crli.T0_MANUAL_S,
        t0_refine_bound_s=crli.T0_REFINE_BOUND_S,
    )
    return GlobalSeriesModel(params=params, physics=physics, preprocessing=preprocessing)


# =============================================================================
# Outputs
# =============================================================================

def kinetics_rows(res) -> list[dict]:
    rows = []
    for g in res.names:
        gp, sd = res.fitted[g], res.seeds[g]
        rows.append({
            "group": g,
            "n_traces": len(res.groups[g]),
            "Rh0_nm": gp.Rh0_nm, "Rh_plateau_nm": gp.Rhinf_nm, "tau_h": gp.tau_h,
            "delta_Rh_nm": gp.Rh0_nm - gp.Rhinf_nm,
            "colloid_sigma_ln": gp.colloid_sigma, "dye_mu_nm": gp.dye_mu_nm, "dye_sigma_ln": gp.dye_sigma,
            "seed_Rh0_nm": sd.Rh0_nm, "seed_Rh_plateau_nm": sd.Rhinf_nm, "seed_tau_h": sd.tau_h,
        })
    return rows


def save_preprocessing(res, out_dir: Path) -> None:
    for g in res.names:
        for tr in res.groups[g]:
            plot_preprocessing(tr.prep, out_dir, safe_filename(tr.record.path.stem))


def save_outputs(model: GlobalSeriesModel, res, info: dict, out_dir: Path, save_plots: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "kinetics_summary.csv", kinetics_rows(res))
    write_rows(out_dir / "per_trace.csv", res.per_trace)
    settings = model.config_dict()
    settings.update({"fit_success": res.success, "fit_cost": res.cost, "fit_nfev": res.nfev, **info})
    (out_dir / "run_settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    if save_plots:
        save_preprocessing(res, out_dir)
        plot_kinetics(res, out_dir)
        plot_residuals_free_vs_global(res, out_dir, model.physics)
        plot_example_fits(res, out_dir, model.physics)
        plot_trace_diagnostics(res, out_dir)


# =============================================================================
# Runner
# =============================================================================

def run(data_dir: str, pattern: str, group_mode: str, out_dir: str) -> None:
    model = build_model()
    if group_mode != model.params.group_mode:
        model = model.set_params(group_mode=group_mode)

    traces, info = model.load_traces(data_dir, pattern)
    print(f"Method (from first data point): {info['method']}")
    print(f"Traces: {info['n_used']} used / {info['n_loaded']} loaded "
          f"({info['n_skipped']} skipped), grouping={group_mode}")

    res = model.fit(traces)
    root = Path(out_dir)
    save_outputs(model, res, info, root, save_plots=crli.SAVE_PLOTS)
    print(f"\nConverged={res.success} cost={res.cost:.4g} nfev={res.nfev}")
    for row in kinetics_rows(res):
        print(f"  group {row['group']:12s} (n={row['n_traces']:2d}): "
              f"Rh0={row['Rh0_nm']:.2f} -> plateau={row['Rh_plateau_nm']:.2f} nm "
              f"(dRh={row['delta_Rh_nm']:+.2f}), tau={row['tau_h']:.2f} h, "
              f"colloid_sigma={row['colloid_sigma_ln']:.3f}")
    print(f"\nOutputs in {root.resolve()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global exponential-kinetics fit over a FidaNeo series.")
    ap.add_argument("--data", dest="data_dir", default=DATA_DIR, help="Folder of FidaNeo files.")
    ap.add_argument("--pattern", default=FILE_PATTERN, help="Glob within the folder.")
    ap.add_argument("--group", dest="group_mode", choices=["pooled", "per_sample"], default=GROUP_MODE)
    ap.add_argument("--output", dest="out_dir", default=OUTPUT_DIR, help="Output folder.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_dir, args.pattern, args.group_mode, args.out_dir)


if __name__ == "__main__":
    main()
