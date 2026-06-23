#!/usr/bin/env python3
"""Fit TDA taylorgrams with a fixed free fluorophore plus one colloid Gaussian.

Edit the USER SETTINGS block and press Run in your editor, or use command-line
arguments such as:

    python fit_dual_gaussian.py --data "*.txt" --output tda_results --peak-fraction 0.75
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
from scipy.optimize import least_squares

from tda_processing import (
    FitResult,
    PreparedTrace,
    Settings,
    add_common_arguments,
    apply_common_args,
    delta_distribution_stats,
    delta_like_pdf_logRh,
    initial_colloid_center,
    make_fit_mask,
    rh_kernel,
    run_batch,
    thin_mask,
    trace_temperature_K,
)


# =============================================================================
# USER SETTINGS - edit these and run the script directly
# =============================================================================

DATA_PATTERN = "*.txt"
OUTPUT_DIR = "tda_results"
INPUT_FORMAT = "fidaneo"        # "fidaneo" or "two_column"
PEAK_FRACTION = 0.50             # 0.50 = rising only, 0.75 = rising + half falling, 1.00 = full selected peak
SAVE_PLOTS = True


# =============================================================================
# Model definition and fitting
# =============================================================================

def unpack_parameters(p: np.ndarray, cfg: Settings) -> dict:
    idx = 4
    free_fraction = float(p[idx]) if cfg.free_fraction_mode == "fit" else float(cfg.free_fraction_fixed)
    idx += 1 if cfg.free_fraction_mode == "fit" else 0
    return {
        "baseline_at_tref": float(p[0]),
        "baseline_slope_per_s": float(p[1]),
        "amplitude": float(p[2]),
        "t0_s": float(p[3]),
        "free_fraction": free_fraction,
        "colloid_Rh_nm": float(p[idx]),
    }


def model_signal(time_s: np.ndarray, p: np.ndarray, t_ref_s: float, T_K: float, cfg: Settings):
    pars = unpack_parameters(p, cfg)
    baseline = pars["baseline_at_tref"] + pars["baseline_slope_per_s"] * (time_s - t_ref_s)
    free = pars["amplitude"] * pars["free_fraction"] * rh_kernel(time_s, cfg.free_Rh_nm, pars["t0_s"], T_K, cfg)
    colloid = pars["amplitude"] * (1.0 - pars["free_fraction"]) * rh_kernel(time_s, pars["colloid_Rh_nm"], pars["t0_s"], T_K, cfg)
    return baseline + free + colloid, baseline, free, colloid


def initial_guess_and_bounds(prep: PreparedTrace, fit_mask: np.ndarray, T_K: float, cfg: Settings):
    t_all, y_all = prep.trace.time_s, prep.trace.signal
    t_ref = float(np.median(t_all))
    duration = max(float(t_all[-1] - t_all[0]), 1.0)
    y_range = max(float(np.ptp(y_all[prep.good_mask])), prep.peak_height_initial, 1e-9)
    slope_bound = 20.0 * y_range / duration

    b0 = float(np.interp(t_ref, t_all, prep.baseline_initial))
    slope = float((prep.baseline_initial[-1] - prep.baseline_initial[0]) / duration)
    t0_window = cfg.t0_half_window_s or max(3.0, 0.025 * prep.t0_initial_s)
    t0_low = max(float(t_all[0]), prep.t0_initial_s - t0_window)
    t0_high = min(float(t_all[-1]), prep.t0_initial_s + t0_window)
    rh0 = initial_colloid_center(prep, "rising", T_K, cfg)

    p0 = [b0, slope, prep.peak_height_initial, prep.t0_initial_s]
    lower = [float(np.min(y_all) - 10 * y_range), -slope_bound, 0.0, t0_low]
    upper = [float(np.max(y_all) + 10 * y_range), slope_bound, 20.0 * y_range, t0_high]

    if cfg.free_fraction_mode == "fit":
        p0.append(float(np.mean(cfg.free_fraction_bounds)))
        lower.append(float(cfg.free_fraction_bounds[0]))
        upper.append(float(cfg.free_fraction_bounds[1]))

    p0.append(rh0)
    lower.append(float(cfg.Rh_center_bounds_nm[0]))
    upper.append(float(cfg.Rh_center_bounds_nm[1]))

    return np.asarray(p0), np.asarray(lower), np.asarray(upper), t_ref


def fit_dual_gaussian(prep: PreparedTrace, cfg: Settings) -> FitResult:
    """Fit one prepared taylorgram with the dual-Gaussian model."""
    T_K = trace_temperature_K(prep.trace, cfg)
    mask = make_fit_mask(prep, cfg, side="rising", peak_fraction=cfg.dual_gaussian_peak_fraction)
    mask = thin_mask(mask, cfg.max_fit_points)
    if np.sum(mask) < 30:
        raise ValueError("Too few fitting points for the dual-Gaussian fit.")

    t = prep.trace.time_s[mask]
    y = prep.trace.signal[mask]
    noise = max(prep.noise_raw, 1e-12)
    p0, lower, upper, t_ref = initial_guess_and_bounds(prep, mask, T_K, cfg)

    def residuals(p: np.ndarray) -> np.ndarray:
        fit, *_ = model_signal(t, p, t_ref, T_K, cfg)
        return (fit - y) / noise

    solution = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=20000)
    p = solution.x
    model, baseline, free, colloid = model_signal(t, p, t_ref, T_K, cfg)
    raw_residuals = model - y
    pars = unpack_parameters(p, cfg)

    rh_max = max(cfg.Rh_grid_bounds_nm[1], 1.5 * pars["colloid_Rh_nm"])
    Rh_grid = np.geomspace(cfg.Rh_grid_bounds_nm[0], rh_max, cfg.n_Rh_grid_parametric)
    pdf_log, pdf_R, cdf = delta_like_pdf_logRh(Rh_grid, pars["colloid_Rh_nm"])

    stats = delta_distribution_stats(pars["colloid_Rh_nm"])
    free_peak = float(np.interp(pars["t0_s"], t, free))
    colloid_peak = float(np.interp(pars["t0_s"], t, colloid))
    peak_sum = free_peak + colloid_peak
    rss = float(np.sum(raw_residuals**2))
    n, k = len(raw_residuals), len(p)

    stats.update({
        "success": bool(solution.success),
        "message": str(solution.message),
        "t0_s": float(pars["t0_s"]),
        "initial_t0_s": float(prep.t0_initial_s),
        "noise_raw": float(prep.noise_raw),
        "baseline_window_s": float(prep.baseline_window_s),
        "n_fit_points": int(n),
        "n_fit_parameters": int(k),
        "dual_gaussian_peak_fraction": float(cfg.dual_gaussian_peak_fraction),
        "free_Rh_nm": float(cfg.free_Rh_nm),
        "free_fraction_detector_weighted": float(pars["free_fraction"]),
        "free_peak_height_fraction": float(free_peak / peak_sum) if peak_sum > 0 else np.nan,
        "colloid_peak_height_fraction": float(colloid_peak / peak_sum) if peak_sum > 0 else np.nan,
        "fit_center_parameter_Rh_nm": float(pars["colloid_Rh_nm"]),
        "fit_width_parameter": 0.0,
        "fit_width_parameter_units": "none",
        "residual_rms_raw": float(np.sqrt(np.mean(raw_residuals**2))),
        "residual_rms_normalized_to_peak": float(np.sqrt(np.mean(raw_residuals**2)) / prep.peak_height_initial),
        "rss_raw": rss,
        "aic": float(n * math.log(max(rss / n, 1e-300)) + 2 * k),
        "bic": float(n * math.log(max(rss / n, 1e-300)) + k * math.log(n)),
    })

    return FitResult(
        model_name="dual_gaussian",
        success=bool(solution.success),
        message=str(solution.message),
        params=pars,
        stats=stats,
        time_fit_s=t,
        signal_fit_raw=y,
        signal_model_raw=model,
        baseline_model_raw=baseline,
        free_signal_raw=free,
        colloid_signal_raw=colloid,
        residuals_raw=raw_residuals,
        Rh_nm=Rh_grid,
        pdf_logRh=pdf_log,
        pdf_Rh_per_nm=pdf_R,
        cdf=cdf,
        fit_mask=mask,
        extra={"least_squares_cost": float(solution.cost)},
    )


# =============================================================================
# Script entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit TDA files with the dual-Gaussian model.")
    add_common_arguments(parser)
    parser.add_argument("--peak-fraction", type=float, help="Fraction of selected peak used in the fit: 0.50, 0.75, 1.00, or 50/75/100.")
    return parser.parse_args()


def settings_from_script_and_args() -> Settings:
    cfg = Settings(data_pattern=DATA_PATTERN, output_dir=OUTPUT_DIR, input_format=INPUT_FORMAT, save_plots=SAVE_PLOTS)
    args = parse_args()
    apply_common_args(cfg, args)
    if args.peak_fraction is not None:
        cfg.dual_gaussian_peak_fraction = args.peak_fraction / 100.0 if args.peak_fraction > 1 else args.peak_fraction
    else:
        cfg.dual_gaussian_peak_fraction = PEAK_FRACTION / 100.0 if PEAK_FRACTION > 1 else PEAK_FRACTION
    return cfg


def main() -> None:
    run_batch(settings_from_script_and_args(), "dual_gaussian", fit_dual_gaussian)


if __name__ == "__main__":
    main()
