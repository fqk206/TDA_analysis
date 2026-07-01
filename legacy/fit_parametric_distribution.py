#!/usr/bin/env python3
"""Fit TDA taylorgrams with parametric colloid Rh distributions.

The model is a fixed free fluorophore plus one continuous colloid distribution.
By default this script fits both normal and lognormal colloid distributions and
saves them in separate folders.

Example:

    python fit_parametric_distribution.py --data "*.txt" --output tda_results --distributions lognormal
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import least_squares

from legacy.tda_processing import (
    FitResult,
    PreparedTrace,
    Settings,
    add_common_arguments,
    apply_common_args,
    colloid_kernel_integral,
    colloid_pdf_logRh,
    distribution_stats,
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
INPUT_FORMAT = "fidaneo"              # "fidaneo" or "two_column"
DISTRIBUTIONS = ("normal", "lognormal")
FIT_SIDE = "rising"                   # "rising", "falling", or "both"
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
        "colloid_center_nm": float(p[idx]),
        "colloid_width": float(p[idx + 1]),
    }


def model_signal(time_s: np.ndarray, p: np.ndarray, t_ref_s: float, Rh_grid: np.ndarray, distribution: str, T_K: float, cfg: Settings):
    pars = unpack_parameters(p, cfg)
    pdf_log, pdf_R, cdf = colloid_pdf_logRh(Rh_grid, distribution, pars["colloid_center_nm"], pars["colloid_width"])
    baseline = pars["baseline_at_tref"] + pars["baseline_slope_per_s"] * (time_s - t_ref_s)
    free = pars["amplitude"] * pars["free_fraction"] * rh_kernel(time_s, cfg.free_Rh_nm, pars["t0_s"], T_K, cfg)
    colloid_kernel = colloid_kernel_integral(time_s, Rh_grid, pdf_log, pars["t0_s"], T_K, cfg)
    colloid = pars["amplitude"] * (1.0 - pars["free_fraction"]) * colloid_kernel
    return baseline + free + colloid, baseline, free, colloid, pdf_log, pdf_R, cdf


def initial_guess_and_bounds(prep: PreparedTrace, distribution: str, side: str, T_K: float, cfg: Settings):
    t_all, y_all = prep.trace.time_s, prep.trace.signal
    t_ref = float(np.median(t_all))
    duration = max(float(t_all[-1] - t_all[0]), 1.0)
    y_range = max(float(np.ptp(y_all[prep.good_mask])), prep.peak_height_initial, 1e-9)
    slope_bound = 20.0 * y_range / duration

    center0 = initial_colloid_center(prep, side, T_K, cfg)
    if distribution == "normal":
        width0 = float(np.clip(0.30 * center0, *cfg.normal_sd_bounds_nm))
        width_lower, width_upper = cfg.normal_sd_bounds_nm
    elif distribution == "lognormal":
        width0 = float(np.clip(0.25, *cfg.lognormal_sigma_bounds))
        width_lower, width_upper = cfg.lognormal_sigma_bounds
    else:
        raise ValueError("distribution must be 'normal' or 'lognormal'.")

    t0_window = cfg.t0_half_window_s or max(3.0, 0.025 * prep.t0_initial_s)
    p0 = [
        float(np.interp(t_ref, t_all, prep.baseline_initial)),
        float((prep.baseline_initial[-1] - prep.baseline_initial[0]) / duration),
        prep.peak_height_initial,
        prep.t0_initial_s,
    ]
    lower = [float(np.min(y_all) - 10 * y_range), -slope_bound, 0.0, max(float(t_all[0]), prep.t0_initial_s - t0_window)]
    upper = [float(np.max(y_all) + 10 * y_range), slope_bound, 20.0 * y_range, min(float(t_all[-1]), prep.t0_initial_s + t0_window)]

    if cfg.free_fraction_mode == "fit":
        p0.append(float(np.mean(cfg.free_fraction_bounds)))
        lower.append(float(cfg.free_fraction_bounds[0]))
        upper.append(float(cfg.free_fraction_bounds[1]))

    p0 += [center0, width0]
    lower += [float(cfg.Rh_center_bounds_nm[0]), float(width_lower)]
    upper += [float(cfg.Rh_center_bounds_nm[1]), float(width_upper)]
    return np.asarray(p0), np.asarray(lower), np.asarray(upper), t_ref


def fit_parametric(prep: PreparedTrace, cfg: Settings, distribution: str, side: str) -> FitResult:
    T_K = trace_temperature_K(prep.trace, cfg)
    mask = thin_mask(make_fit_mask(prep, cfg, side=side), cfg.max_fit_points)
    if np.sum(mask) < 30:
        raise ValueError(f"Too few fitting points for the {distribution} fit.")

    t, y = prep.trace.time_s[mask], prep.trace.signal[mask]
    Rh_grid = np.geomspace(*cfg.Rh_grid_bounds_nm, cfg.n_Rh_grid_parametric)
    p0, lower, upper, t_ref = initial_guess_and_bounds(prep, distribution, side, T_K, cfg)
    noise = max(prep.noise_raw, 1e-12)

    def residuals(p: np.ndarray) -> np.ndarray:
        fit, *_ = model_signal(t, p, t_ref, Rh_grid, distribution, T_K, cfg)
        return (fit - y) / noise

    solution = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=30000)
    p = solution.x
    model, baseline, free, colloid, pdf_log, pdf_R, cdf = model_signal(t, p, t_ref, Rh_grid, distribution, T_K, cfg)
    residual = model - y
    pars = unpack_parameters(p, cfg)
    stats = distribution_stats(Rh_grid, pdf_log)

    free_peak = float(np.interp(pars["t0_s"], t, free))
    colloid_peak = float(np.interp(pars["t0_s"], t, colloid))
    peak_sum = free_peak + colloid_peak
    rss = float(np.sum(residual**2))
    n, k = len(residual), len(p)
    model_name = f"parametric_{distribution}"

    stats.update({
        "success": bool(solution.success),
        "message": str(solution.message),
        "distribution_model": distribution,
        "fit_side": side,
        "t0_s": float(pars["t0_s"]),
        "initial_t0_s": float(prep.t0_initial_s),
        "noise_raw": float(prep.noise_raw),
        "baseline_window_s": float(prep.baseline_window_s),
        "n_fit_points": int(n),
        "n_fit_parameters": int(k),
        "free_Rh_nm": float(cfg.free_Rh_nm),
        "free_fraction_detector_weighted": float(pars["free_fraction"]),
        "free_peak_height_fraction": float(free_peak / peak_sum) if peak_sum > 0 else np.nan,
        "colloid_peak_height_fraction": float(colloid_peak / peak_sum) if peak_sum > 0 else np.nan,
        "fit_center_parameter_Rh_nm": float(pars["colloid_center_nm"]),
        "fit_width_parameter": float(pars["colloid_width"]),
        "fit_width_parameter_units": "nm" if distribution == "normal" else "sigma_ln",
        "residual_rms_raw": float(np.sqrt(np.mean(residual**2))),
        "residual_rms_normalized_to_peak": float(np.sqrt(np.mean(residual**2)) / prep.peak_height_initial),
        "rss_raw": rss,
        "aic": float(n * math.log(max(rss / n, 1e-300)) + 2 * k),
        "bic": float(n * math.log(max(rss / n, 1e-300)) + k * math.log(n)),
    })

    return FitResult(
        model_name=model_name,
        success=bool(solution.success),
        message=str(solution.message),
        params={**pars, "distribution": distribution, "side": side},
        stats=stats,
        time_fit_s=t,
        signal_fit_raw=y,
        signal_model_raw=model,
        baseline_model_raw=baseline,
        free_signal_raw=free,
        colloid_signal_raw=colloid,
        residuals_raw=residual,
        Rh_nm=Rh_grid,
        pdf_logRh=pdf_log,
        pdf_Rh_per_nm=pdf_R,
        cdf=cdf,
        fit_mask=mask,
        extra={"least_squares_cost": float(solution.cost)},
    )


def make_fitter(distribution: str, side: str):
    return lambda prep, cfg: fit_parametric(prep, cfg, distribution=distribution, side=side)


# =============================================================================
# Script entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit TDA files with normal/lognormal parametric colloid distributions.")
    add_common_arguments(parser)
    parser.add_argument("--distributions", nargs="+", choices=["normal", "lognormal"], help="One or both distributions to fit.")
    parser.add_argument("--side", choices=["rising", "falling", "both"], help="Taylorgram side to use for fitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Settings(data_pattern=DATA_PATTERN, output_dir=OUTPUT_DIR, input_format=INPUT_FORMAT, save_plots=SAVE_PLOTS)
    apply_common_args(cfg, args)
    distributions = tuple(args.distributions) if args.distributions else DISTRIBUTIONS
    side = args.side or FIT_SIDE
    for distribution in distributions:
        run_batch(cfg, f"parametric_{distribution}", make_fitter(distribution, side))


if __name__ == "__main__":
    main()
