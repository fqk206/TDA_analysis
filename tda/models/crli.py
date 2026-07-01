#!/usr/bin/env python3
"""CRLI: constrained regularized linear inversion for a Taylorgram.

Solves for a smooth, non-negative colloid hydrodynamic-radius distribution by
NNLS + second-difference smoothness + an alpha scan, plus a fixed-Rh free
fluorophore component. Math ported unchanged from ``fit_crli_simple.fit_crli``
and its private helpers; only the CONFIG-globals -> ``CRLIParams`` fields and
free-standing-module -> class packaging changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from tda.fitting import select_fit_mask, t0_seed, t0_fit_enabled, refine_t0
from tda.numerics import second_difference_matrix, trapezoid_weights, distribution_stats
from tda.physics import PhysicsParams, apparent_rh_from_slope, rh_kernel
from tda.preprocessing.pipeline import PreparedTrace
from tda.models.base import SingleTraceModel
from tda.models.result import FitResult


@dataclass
class CRLIParams:
    # Free fluorophore
    free_rh_nm: float = 0.59
    free_fraction_mode: str = "fit_bounded"      # "fit", "fit_bounded", or "fixed"
    free_fraction_fixed: float = 0.05             # used when mode == "fixed"
    free_fraction_bounds: tuple[float, float] = (0.01, 0.10)   # used when mode == "fit_bounded"

    # CRLI
    fit_side: str = "rising"                       # "rising", "falling", or "both"
    peak_fraction: float = 0.50                     # rising side: 0.50 = rising half, 1.00 = whole peak
    rh_grid_bounds_nm: tuple[float, float] = (1.5, 200.0)
    rh_grid_autonarrow: bool = False                 # if True, auto-zoom the grid around the apparent size
    rh_center_bounds_nm: tuple[float, float] = (1.5, 300.0)   # only used when rh_grid_autonarrow is True
    n_rh_grid: int = 120
    alphas: np.ndarray = field(default_factory=lambda: np.logspace(-8, 8, 80))
    residual_factor: float = 1.10                    # smoothest alpha with residual <= factor * best
    ridge_rho: float = 0.0                             # ridge weight as a fraction of alpha
    fit_stride: int = 1
    signal_min: float = -10.00
    signal_max: float = 1.02

    # t0 handling: "auto" locks to the detected t0, "manual" locks to
    # t0_manual_s, "fit" refines t0 in the fit (bounded to seed +/- t0_refine_bound_s)
    t0_mode: str = "fit"
    t0_manual_s: float | None = None
    t0_refine_bound_s: float = 5.0

    # Diagnostics
    truncation_rh_nm: float = 4.0                    # signal carried by colloid mass below this Rh


def crli_radius_bounds(prep: PreparedTrace, side: str, T_K: float, params: CRLIParams,
                        physics: PhysicsParams) -> tuple[float, float]:
    """Rh grid bounds for the inversion; auto-narrowed around the apparent
    Taylor-slope size when ``params.rh_grid_autonarrow`` is True."""
    if not params.rh_grid_autonarrow:
        return params.rh_grid_bounds_nm
    guess = (apparent_rh_from_slope(prep.time_s, prep.normalized, prep.good_mask, prep.t0_s, side, T_K,
                                     params.rh_center_bounds_nm, physics)
             or apparent_rh_from_slope(prep.time_s, prep.normalized, prep.good_mask, prep.t0_s, "both", T_K,
                                        params.rh_center_bounds_nm, physics)
             or 25.0)
    guess = float(np.clip(guess, *params.rh_center_bounds_nm))
    lower = max(params.rh_grid_bounds_nm[0], guess / 8.0)
    upper = min(params.rh_grid_bounds_nm[1], guess * 10.0)
    return params.rh_grid_bounds_nm if lower >= upper else (float(lower), float(upper))


def _solve_fixed(A_col, K_free, L, wx, s, alpha, fraction, ridge_rho):
    """NNLS solve with the free fraction pinned exactly to ``fraction``."""
    n = L.shape[1]
    beta = ridge_rho * alpha
    ratio = fraction / (1.0 - fraction)
    A_fit = A_col + K_free[:, None] * ratio * wx[None, :]
    A_aug = np.vstack([A_fit, np.sqrt(alpha) * L, np.sqrt(beta) * np.eye(n)])
    b_aug = np.r_[s, np.zeros(L.shape[0] + n)]
    q_col, _ = nnls(A_aug, b_aug, maxiter=20000)
    q_free = ratio * float(np.sum(q_col * wx))
    return q_col, q_free


def _solve_fit(A_col, K_free, L, wx, s, alpha, ridge_rho):
    """NNLS solve with the free amplitude as a free (non-negative) parameter."""
    n = L.shape[1]
    beta = ridge_rho * alpha
    A_fit = np.column_stack([K_free, A_col])
    L_aug = np.column_stack([np.zeros(L.shape[0]), L])
    R_aug = np.sqrt(beta) * np.column_stack([np.zeros(n), np.eye(n)])
    A_aug = np.vstack([A_fit, np.sqrt(alpha) * L_aug, R_aug])
    b_aug = np.r_[s, np.zeros(L.shape[0] + n)]
    q_all, _ = nnls(A_aug, b_aug, maxiter=20000)
    return q_all[1:], float(q_all[0])


def _free_fraction(q_col, q_free, wx) -> float:
    total = float(np.sum(q_col * wx)) + q_free
    return q_free / total if total > 0 else 0.0


def _crli_scan_at_t0(t, s, Rh_grid, wx, L, t0, T_K, free_lo, free_hi, params: CRLIParams, physics: PhysicsParams) -> dict:
    """Build the kernel at ``t0`` and run the alpha scan; return the chosen solve."""
    A_col = rh_kernel(t, Rh_grid, t0, T_K, physics) * wx[None, :]
    K_free = rh_kernel(t, params.free_rh_nm, t0, T_K, physics)
    solutions, residual_rms, roughness = [], [], []
    for alpha in params.alphas:
        if params.free_fraction_mode == "fixed":
            q_col, q_free = _solve_fixed(A_col, K_free, L, wx, s, alpha, params.free_fraction_fixed, params.ridge_rho)
        elif params.free_fraction_mode in ("fit", "fit_bounded"):
            q_col, q_free = _solve_fit(A_col, K_free, L, wx, s, alpha, params.ridge_rho)
            if params.free_fraction_mode == "fit_bounded":
                # Bounding the fraction is a linear bound on q_free; by convexity the
                # constrained optimum is either interior (keep) or on the violated
                # bound, where the problem reduces to the fixed-fraction solve.
                f = _free_fraction(q_col, q_free, wx)
                if f < free_lo:
                    q_col, q_free = _solve_fixed(A_col, K_free, L, wx, s, alpha, free_lo, params.ridge_rho)
                elif f > free_hi:
                    q_col, q_free = _solve_fixed(A_col, K_free, L, wx, s, alpha, free_hi, params.ridge_rho)
        else:
            raise ValueError("free_fraction_mode must be 'fit', 'fit_bounded', or 'fixed'.")

        s_col = A_col @ q_col
        s_free = K_free * q_free
        s_model = s_col + s_free
        solutions.append((q_col, q_free, s_col, s_free, s_model))
        residual_rms.append(float(np.sqrt(np.mean((s_model - s) ** 2))))
        roughness.append(float(np.linalg.norm(L @ q_col)))

    residual_rms = np.asarray(residual_rms)
    roughness = np.asarray(roughness)
    best_rms = float(np.min(residual_rms))
    acceptable = np.flatnonzero(residual_rms <= params.residual_factor * best_rms)
    chosen = int(acceptable[-1]) if acceptable.size else int(np.argmin(residual_rms))
    return {"A_col": A_col, "K_free": K_free, "solutions": solutions,
            "residual_rms": residual_rms, "roughness": roughness,
            "chosen": chosen, "best_rms": best_rms}


class CRLIModel(SingleTraceModel):
    def __init__(self, params: CRLIParams | None = None, physics: PhysicsParams | None = None,
                 preprocessing=None, reader=None):
        super().__init__(physics=physics, preprocessing=preprocessing, reader=reader)
        self.params = params if params is not None else CRLIParams()

    def fit(self, prepared: PreparedTrace, side: str | None = None) -> FitResult:
        p = self.params
        side = side if side is not None else p.fit_side
        T_K = self.temperature_k

        all_indices = np.flatnonzero(select_fit_mask(prepared, side, p.peak_fraction, p.signal_min, p.signal_max))
        indices = all_indices[:: max(1, int(p.fit_stride))]
        if indices.size < 25:
            raise ValueError("Too few fitting points for the CRLI fit.")

        t = prepared.time_s[indices]
        y = prepared.signal[indices]
        s = prepared.normalized[indices]

        if p.free_fraction_mode == "fixed" and not 0.0 <= p.free_fraction_fixed < 1.0:
            raise ValueError("free_fraction_fixed must be in [0, 1).")
        free_lo, free_hi = p.free_fraction_bounds
        if p.free_fraction_mode == "fit_bounded" and not 0.0 <= free_lo <= free_hi < 1.0:
            raise ValueError("free_fraction_bounds must satisfy 0 <= lower <= upper < 1.")

        rh_min, rh_max = crli_radius_bounds(prepared, side, T_K, p, self.physics)
        Rh_grid = np.geomspace(rh_min, rh_max, p.n_rh_grid)
        wx = trapezoid_weights(np.log(Rh_grid))
        L = second_difference_matrix(p.n_rh_grid)

        t0_init = t0_seed(prepared, p.t0_mode, p.t0_manual_s)
        t0 = refine_t0(
            lambda tt: _crli_scan_at_t0(t, s, Rh_grid, wx, L, tt, T_K, free_lo, free_hi, p, self.physics)["best_rms"],
            t0_init, p.t0_mode, p.t0_refine_bound_s,
        )
        scan = _crli_scan_at_t0(t, s, Rh_grid, wx, L, t0, T_K, free_lo, free_hi, p, self.physics)
        A_col, K_free = scan["A_col"], scan["K_free"]
        solutions, residual_rms, roughness = scan["solutions"], scan["residual_rms"], scan["roughness"]
        chosen, best_rms = scan["chosen"], scan["best_rms"]

        q_col, q_free, s_col, s_free, s_model = solutions[chosen]
        area_col = float(np.sum(q_col * wx))
        total_area = area_col + q_free
        if area_col <= 0 or total_area <= 0:
            raise ValueError("CRLI recovered zero distribution area.")

        pdf_log = q_col / area_col
        pdf_R = pdf_log / Rh_grid
        cdf = np.cumsum(pdf_log * wx)
        cdf = cdf / cdf[-1]
        baseline = np.interp(t, prepared.time_s, prepared.baseline)
        model_raw = baseline + prepared.peak_height * s_model
        free_raw = prepared.peak_height * s_free
        colloid_raw = prepared.peak_height * s_col
        residual_raw = model_raw - y

        free_fraction = float(q_free / total_area)
        K_free_t0 = float(rh_kernel(np.array([t0]), p.free_rh_nm, t0, T_K, self.physics)[0])
        K_col_t0 = rh_kernel(np.array([t0]), Rh_grid, t0, T_K, self.physics)[0]
        peak_free = K_free_t0 * q_free
        peak_col = float(np.sum(K_col_t0 * q_col * wx))
        peak_sum = peak_free + peak_col

        # Truncation test: signal contributed solely by colloid mass below the
        # cutoff (free fluorophore term left intact). Near noise -> inversion
        # artefact; well above noise -> real feature of the sample.
        low = Rh_grid < p.truncation_rh_nm
        s_col_low = A_col[:, low] @ q_col[low]
        trunc_low_raw = prepared.peak_height * s_col_low
        trunc_rms_raw = float(np.sqrt(np.mean(trunc_low_raw ** 2)))
        trunc_peak_raw = float(np.max(np.abs(trunc_low_raw)))
        mass_below = float(np.sum(q_col[low] * wx[low]) / area_col)

        stats = distribution_stats(Rh_grid, pdf_log)
        stats.update({
            "success": True,
            "fit_side": side,
            "peak_fraction": float(p.peak_fraction),
            "t0_s": float(t0),
            "t0_mode": p.t0_mode,
            "t0_seed_s": float(t0_init),
            "t0_fitted": t0_fit_enabled(p.t0_mode),
            "t0_shift_s": float(t0 - t0_init),
            "noise_raw": float(prepared.noise_raw),
            "baseline_window_s": float(prepared.baseline_window_s),
            "n_fit_points": int(t.size),
            "free_Rh_nm": float(p.free_rh_nm),
            "free_fraction_mode": p.free_fraction_mode,
            "free_fraction_detector_weighted": free_fraction,
            "free_fraction_lower_bound": float(free_lo) if p.free_fraction_mode == "fit_bounded" else np.nan,
            "free_fraction_upper_bound": float(free_hi) if p.free_fraction_mode == "fit_bounded" else np.nan,
            "free_fraction_bound_active": (
                "none" if p.free_fraction_mode != "fit_bounded"
                else "lower" if free_fraction <= free_lo + 1e-6
                else "upper" if free_fraction >= free_hi - 1e-6
                else "none"
            ),
            "free_peak_height_fraction": float(peak_free / peak_sum) if peak_sum > 0 else np.nan,
            "colloid_peak_height_fraction": float(peak_col / peak_sum) if peak_sum > 0 else np.nan,
            "residual_rms_raw": float(np.sqrt(np.mean(residual_raw ** 2))),
            "residual_rms_normalized_to_peak": float(np.sqrt(np.mean((s_model - s) ** 2))),
            "rss_raw": float(np.sum(residual_raw ** 2)),
            "crli_alpha": float(p.alphas[chosen]),
            "crli_ridge_rho": float(p.ridge_rho),
            "crli_ridge_beta": float(p.ridge_rho * p.alphas[chosen]),
            "crli_alpha_at_scan_boundary": chosen in {0, len(p.alphas) - 1},
            "crli_best_residual_rms": best_rms,
            "crli_chosen_residual_rms": float(residual_rms[chosen]),
            "crli_rh_min_nm": float(rh_min),
            "crli_rh_max_nm": float(rh_max),
            "trunc_rh_nm": float(p.truncation_rh_nm),
            "trunc_mass_fraction_below": mass_below,
            "trunc_signal_rms_raw": trunc_rms_raw,
            "trunc_signal_peak_raw": trunc_peak_raw,
            "trunc_signal_rms_over_noise": trunc_rms_raw / prepared.noise_raw if prepared.noise_raw > 0 else np.nan,
            "trunc_signal_peak_over_noise": trunc_peak_raw / prepared.noise_raw if prepared.noise_raw > 0 else np.nan,
        })

        return FitResult(
            time_fit_s=t,
            signal_fit_raw=y,
            signal_fit_normalized=s,
            signal_model_raw=model_raw,
            signal_model_normalized=s_model,
            baseline_model_raw=baseline,
            residuals_raw=residual_raw,
            peak_height=float(prepared.peak_height),
            Rh_nm=Rh_grid,
            pdf_logRh=pdf_log,
            pdf_Rh_per_nm=pdf_R,
            cdf=cdf,
            components={"free": free_raw, "colloid": colloid_raw},
            params={"t0_s": float(t0), "free_fraction": free_fraction, "alpha": float(p.alphas[chosen]), "side": side},
            stats=stats,
            extra={
                "alpha_scan": p.alphas,
                "residual_rms_scan": residual_rms,
                "roughness_scan": roughness,
                "s_fit": s,
                "s_model": s_model,
                "trunc_rh_nm": float(p.truncation_rh_nm),
                "trunc_low_signal_raw": trunc_low_raw,
            },
        )
