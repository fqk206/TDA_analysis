#!/usr/bin/env python3
"""Fit TDA taylorgrams by constrained regularized linear inversion (CRLI).

The CRLI fit uses the shared taylorgram reader/preprocessor in tda_processing.py,
then solves for a smooth, non-negative colloid hydrodynamic-radius distribution
plus the fixed-Rh free fluorophore component.

Example:

    python fit_crli.py --data "*.txt" --output tda_results --side rising
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import nnls

from tda_processing import (
    FitResult,
    PreparedTrace,
    Settings,
    add_common_arguments,
    apply_common_args,
    distribution_stats,
    initial_colloid_center,
    rh_kernel,
    run_batch,
    second_difference_matrix,
    trace_temperature_K,
    trapezoid_weights,
)


# =============================================================================
# USER SETTINGS - edit these and run the script directly
# =============================================================================

DATA_PATTERN = "*.txt"
OUTPUT_DIR = "tda_results"
INPUT_FORMAT = "fidaneo"       # "fidaneo" or "two_column"
FIT_SIDE = "rising"            # "rising", "falling", or "both"
SAVE_PLOTS = True


# =============================================================================
# CRLI fitting
# =============================================================================

def crli_radius_bounds(prep: PreparedTrace, side: str, T_K: float, cfg: Settings) -> tuple[float, float]:
    """Choose practical Rh bounds from the apparent Taylor slope."""
    guess = initial_colloid_center(prep, side, T_K, cfg)
    lower = max(cfg.Rh_grid_bounds_nm[0], guess / 8.0)
    upper = min(cfg.Rh_grid_bounds_nm[1], guess * 10.0)
    return cfg.Rh_grid_bounds_nm if lower >= upper else (float(lower), float(upper))


def crli_fit_mask(prep: PreparedTrace, side: str, cfg: Settings) -> np.ndarray:
    t, s, t0 = prep.trace.time_s, prep.normalized_initial, prep.t0_initial_s
    mask = prep.good_mask & np.isfinite(s) & (s >= cfg.crli_signal_min) & (s <= cfg.crli_signal_max)
    if side == "rising":
        mask &= t <= t0
    elif side == "falling":
        mask &= t >= t0
    elif side == "both":
        pass
    else:
        raise ValueError("side must be 'rising', 'falling', or 'both'.")
    return mask


def fit_crli(prep: PreparedTrace, cfg: Settings, side: str = FIT_SIDE) -> FitResult:
    T_K = trace_temperature_K(prep.trace, cfg)
    t0 = prep.t0_initial_s
    all_indices = np.flatnonzero(crli_fit_mask(prep, side, cfg))
    indices = all_indices[:: max(1, int(cfg.crli_fit_stride))]
    if indices.size < 25:
        raise ValueError("Too few fitting points for the CRLI fit.")

    t = prep.trace.time_s[indices]
    y = prep.trace.signal[indices]
    s = prep.normalized_initial[indices]
    used_mask = np.zeros_like(prep.good_mask, dtype=bool)
    used_mask[indices] = True

    rh_min, rh_max = crli_radius_bounds(prep, side, T_K, cfg)
    Rh_grid = np.geomspace(rh_min, rh_max, cfg.n_Rh_grid_crli)
    wx = trapezoid_weights(np.log(Rh_grid))
    A_col = rh_kernel(t, Rh_grid, t0, T_K, cfg) * wx[None, :]
    K_free = rh_kernel(t, cfg.free_Rh_nm, t0, T_K, cfg)
    L = second_difference_matrix(cfg.n_Rh_grid_crli)

    solutions = []
    residual_rms = []
    roughness = []
    for alpha in cfg.crli_alphas:
        if cfg.free_fraction_mode == "fixed":
            if not 0.0 <= cfg.free_fraction_fixed < 1.0:
                raise ValueError("free_fraction_fixed must be in [0, 1).")
            ratio = cfg.free_fraction_fixed / (1.0 - cfg.free_fraction_fixed)
            A_fit = A_col + K_free[:, None] * ratio * wx[None, :]
            A_aug = np.vstack([A_fit, np.sqrt(alpha) * L])
            b_aug = np.r_[s, np.zeros(L.shape[0])]
            q_col, _ = nnls(A_aug, b_aug, maxiter=20000)
            q_free = ratio * float(np.sum(q_col * wx))
        elif cfg.free_fraction_mode == "fit":
            A_fit = np.column_stack([K_free, A_col])
            L_aug = np.column_stack([np.zeros(L.shape[0]), L])
            A_aug = np.vstack([A_fit, np.sqrt(alpha) * L_aug])
            b_aug = np.r_[s, np.zeros(L.shape[0])]
            q_all, _ = nnls(A_aug, b_aug, maxiter=20000)
            q_free = float(q_all[0])
            q_col = q_all[1:]
        else:
            raise ValueError("free_fraction_mode must be 'fit' or 'fixed'.")

        s_col = A_col @ q_col
        s_free = K_free * q_free
        s_model = s_col + s_free
        solutions.append((q_col, q_free, s_col, s_free, s_model))
        residual_rms.append(float(np.sqrt(np.mean((s_model - s) ** 2))))
        roughness.append(float(np.linalg.norm(L @ q_col)))

    residual_rms = np.asarray(residual_rms)
    roughness = np.asarray(roughness)
    best_rms = float(np.min(residual_rms))
    acceptable = np.flatnonzero(residual_rms <= cfg.crli_residual_factor * best_rms)
    chosen = int(acceptable[-1]) if acceptable.size else int(np.argmin(residual_rms))

    q_col, q_free, s_col, s_free, s_model = solutions[chosen]
    area_col = float(np.sum(q_col * wx))
    total_area = area_col + q_free
    if area_col <= 0 or total_area <= 0:
        raise ValueError("CRLI recovered zero distribution area.")

    pdf_log = q_col / area_col
    pdf_R = pdf_log / Rh_grid
    cdf = np.cumsum(pdf_log * wx)
    cdf = cdf / cdf[-1]
    baseline = np.interp(t, prep.trace.time_s, prep.baseline_initial)
    model_raw = baseline + prep.peak_height_initial * s_model
    free_raw = prep.peak_height_initial * s_free
    colloid_raw = prep.peak_height_initial * s_col
    residual_raw = model_raw - y

    free_fraction = float(q_free / total_area)
    K_free_t0 = float(rh_kernel(np.array([t0]), cfg.free_Rh_nm, t0, T_K, cfg)[0])
    K_col_t0 = rh_kernel(np.array([t0]), Rh_grid, t0, T_K, cfg)[0]
    peak_free = K_free_t0 * q_free
    peak_col = float(np.sum(K_col_t0 * q_col * wx))
    peak_sum = peak_free + peak_col

    stats = distribution_stats(Rh_grid, pdf_log)
    stats.update({
        "success": True,
        "message": "CRLI fit completed",
        "distribution_model": "smooth_crli",
        "fit_side": side,
        "t0_s": float(t0),
        "initial_t0_s": float(prep.t0_initial_s),
        "noise_raw": float(prep.noise_raw),
        "baseline_window_s": float(prep.baseline_window_s),
        "n_fit_points": int(t.size),
        "n_fit_parameters": np.nan,
        "free_Rh_nm": float(cfg.free_Rh_nm),
        "free_fraction_detector_weighted": free_fraction,
        "free_peak_height_fraction": float(peak_free / peak_sum) if peak_sum > 0 else np.nan,
        "colloid_peak_height_fraction": float(peak_col / peak_sum) if peak_sum > 0 else np.nan,
        "fit_center_parameter_Rh_nm": np.nan,
        "fit_width_parameter": np.nan,
        "fit_width_parameter_units": "none",
        "residual_rms_raw": float(np.sqrt(np.mean(residual_raw**2))),
        "residual_rms_normalized_to_peak": float(np.sqrt(np.mean((s_model - s) ** 2))),
        "rss_raw": float(np.sum(residual_raw**2)),
        "crli_alpha": float(cfg.crli_alphas[chosen]),
        "crli_alpha_at_scan_boundary": chosen in {0, len(cfg.crli_alphas) - 1},
        "crli_best_residual_rms": best_rms,
        "crli_chosen_residual_rms": float(residual_rms[chosen]),
        "crli_rh_min_nm": float(rh_min),
        "crli_rh_max_nm": float(rh_max),
    })

    return FitResult(
        model_name="crli",
        success=True,
        message="CRLI fit completed",
        params={"t0_s": float(t0), "free_fraction": free_fraction, "alpha": float(cfg.crli_alphas[chosen]), "side": side},
        stats=stats,
        time_fit_s=t,
        signal_fit_raw=y,
        signal_model_raw=model_raw,
        baseline_model_raw=baseline,
        free_signal_raw=free_raw,
        colloid_signal_raw=colloid_raw,
        residuals_raw=residual_raw,
        Rh_nm=Rh_grid,
        pdf_logRh=pdf_log,
        pdf_Rh_per_nm=pdf_R,
        cdf=cdf,
        fit_mask=used_mask,
        extra={
            "alpha_scan": cfg.crli_alphas,
            "residual_rms_scan": residual_rms,
            "roughness_scan": roughness,
            "s_fit": s,
            "s_model": s_model,
        },
    )


def make_fitter(side: str):
    return lambda prep, cfg: fit_crli(prep, cfg, side=side)


# =============================================================================
# Script entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit TDA files with CRLI.")
    add_common_arguments(parser)
    parser.add_argument("--side", choices=["rising", "falling", "both"], help="Taylorgram side to use for fitting.")
    parser.add_argument("--crli-residual-factor", type=float, help="Choose smoothest alpha with residual <= factor * best residual.")
    parser.add_argument("--alpha-min", type=float, help="Minimum CRLI alpha.")
    parser.add_argument("--alpha-max", type=float, help="Maximum CRLI alpha.")
    parser.add_argument("--alpha-count", type=int, help="Number of alpha values in the log-spaced scan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Settings(data_pattern=DATA_PATTERN, output_dir=OUTPUT_DIR, input_format=INPUT_FORMAT, save_plots=SAVE_PLOTS)
    apply_common_args(cfg, args)
    if args.crli_residual_factor is not None:
        cfg.crli_residual_factor = args.crli_residual_factor
    if args.alpha_min is not None or args.alpha_max is not None or args.alpha_count is not None:
        lo = args.alpha_min if args.alpha_min is not None else float(cfg.crli_alphas[0])
        hi = args.alpha_max if args.alpha_max is not None else float(cfg.crli_alphas[-1])
        n = args.alpha_count if args.alpha_count is not None else int(len(cfg.crli_alphas))
        cfg.crli_alphas = np.logspace(np.log10(lo), np.log10(hi), n)
    run_batch(cfg, "crli", make_fitter(args.side or FIT_SIDE))


if __name__ == "__main__":
    main()
