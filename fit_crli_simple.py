#!/usr/bin/env python3
"""Simplified CRLI fit for two-column Taylor dispersion traces.

A single self-contained vertical slice: read a two-column (time, signal) file,
subtract a linear background, and solve for a smooth, non-negative colloid
hydrodynamic-radius distribution by constrained regularized linear inversion
(CRLI), plus the fixed-Rh free fluorophore component.

The fit math is identical to the original fit_crli.py; this version drops the
FidaNeo reader, spike handling, the multi-model abstraction, and the
parity-placeholder output fields.

Run by editing the CONFIG block below and pressing Run, or from the shell:

    python fit_crli_simple.py --data "*.csv" --output tda_results/crli --side rising
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
import argparse
import csv
import json
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann as kB
from scipy.ndimage import median_filter
from scipy.optimize import nnls


# =============================================================================
# CONFIG - edit these and run the script directly
# =============================================================================

# Input
DATA_PATTERN = "data/*.txt"
OUTPUT_DIR = "output"
TIME_COLUMN = 0
SIGNAL_COLUMN = 1
DELIMITER: str | None = None         # None splits on any whitespace (tabs/spaces)
SKIPROWS = 0
COMMENTS = "#"
TIME_UNIT = "min"                    # "s", "min", or "ms"
TEMPERATURE_C = 25.0

# Physics
CAPILLARY_RADIUS_M = 37.5e-6         # radius, not diameter
VISCOSITY_PA_S = 8.90e-4
KERNEL_REFERENCE_RH_NM = 25.0        # numerical scale factor only

# Background subtraction
BASELINE_MIN_WINDOW_S = 15.0
BASELINE_MAX_WINDOW_S = 90.0
BASELINE_FRACTION_OF_TRACE = 0.15

# Despiking - narrow artefact spikes are removed before peak/t0 detection.  A
# single spike can otherwise be the global maximum and place t0 on the artefact
# (and define the wrong peak height / normalization).  Hampel identifier: flag
# samples deviating from a rolling median by > DESPIKE_SIGMA local MADs; flagged
# samples are ignored entirely (dropped from good_mask), never smoothed.
# DESPIKE_WINDOW is in samples and must exceed the widest spike but stay well
# below the (broad) Taylor peak - 31 samples is ~2.6 s at the usual 12 Hz.
# DESPIKE_SIGMA trades off catching small spikes vs sparing genuine points on a
# very steep, low-noise peak flank (where the rolling median lags the curvature):
# 8 catches the real artefacts here while flagging at most ~1-2 real flank points
# on the steepest trace, which is harmless (and shown on the preprocessing plot).
DESPIKE = True
DESPIKE_WINDOW = 4
DESPIKE_SIGMA = 8.0

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
RIDGE_RHO = 0.0                      # ridge weight as a fraction of alpha (beta = RIDGE_RHO * alpha);
#                                      off by default - see note: a uniform ridge broadens the main peak.
CRLI_FIT_STRIDE = 1
CRLI_SIGNAL_MIN = -10.00
CRLI_SIGNAL_MAX = 1.02

# t0 handling (shared by the fitters - see t0_seed / refine_t0 below)
#   "auto"   - lock t0 to the value found in preprocessing (default; t0 rarely
#              drifts under stable chemical conditions, so this is usually right)
#   "manual" - lock t0 to T0_MANUAL_S, a hand-set / eyeballed value, bypassing the
#              peak detector entirely (for when you'd rather read it off the plot)
#   "fit"    - refine t0 in the fit, bounded to the seed +/- T0_REFINE_BOUND_S
T0_MODE = "auto"
T0_MANUAL_S: float | None = None
T0_REFINE_BOUND_S = 5.0

# Output
SAVE_PLOTS = True
SHOW_PLOTS = False

# Diagnostics
TRUNCATION_RH_NM = 4.0               # truncation-test cutoff: signal carried by colloid mass below this Rh


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class Prepared:
    path: Path
    time_s: np.ndarray
    signal: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    normalized: np.ndarray
    good_mask: np.ndarray
    baseline_window_s: float
    noise_raw: float
    t0_s: float
    peak_height: float
    spike_mask: np.ndarray


@dataclass
class FitResult:
    params: dict
    stats: dict
    time_fit_s: np.ndarray
    signal_fit_raw: np.ndarray
    signal_model_raw: np.ndarray
    baseline_model_raw: np.ndarray
    free_signal_raw: np.ndarray
    colloid_signal_raw: np.ndarray
    residuals_raw: np.ndarray
    Rh_nm: np.ndarray
    pdf_logRh: np.ndarray
    pdf_Rh_per_nm: np.ndarray
    cdf: np.ndarray
    extra: dict


# =============================================================================
# Small numerical helpers (carried over unchanged from tda_processing.py)
# =============================================================================

def robust_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float(np.std(values, ddof=1)) if values.size > 1 else 1.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values, ddof=1)
    return float(sigma if sigma > 0 else 1.0)


def despike_mask(signal: np.ndarray, window: int, n_sigma: float) -> np.ndarray:
    """Flag narrow artefact spikes with a Hampel identifier.

    Each sample is compared with a rolling-median estimate of where its
    neighbours sit (the robust version of "how far can a point jump relative to
    the points around it"); samples deviating by more than ``n_sigma`` local MADs
    are flagged.  ``window`` (in samples) must be wider than the broadest spike -
    so the median tracks the baseline straight across a spike - yet far narrower
    than the Taylor peak, which is broad and therefore left untouched.  Returns a
    boolean mask, True at spike samples (to be ignored entirely, not smoothed).
    """
    y = np.asarray(signal, dtype=float)
    if not window or window <= 2 or y.size < 3:
        return np.zeros(y.shape, dtype=bool)
    local = median_filter(y, size=int(window), mode="nearest")
    resid = y - local
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(y.shape, dtype=bool)
    return np.abs(resid) > n_sigma * scale


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two grid points.")
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def second_difference_matrix(n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("Need at least three grid points.")
    matrix = np.zeros((n - 2, n), dtype=float)
    row = np.arange(n - 2)
    matrix[row, row] = 1.0
    matrix[row, row + 1] = -2.0
    matrix[row, row + 2] = 1.0
    return matrix


def quadratic_peak_time(t: np.ndarray, y: np.ndarray, i_peak: int, half_window: int = 4) -> float:
    lo = max(0, i_peak - half_window)
    hi = min(len(t), i_peak + half_window + 1)
    if hi - lo < 3:
        return float(t[i_peak])

    x = t[lo:hi] - t[i_peak]
    a, b, _ = np.polyfit(x, y[lo:hi], deg=2)
    if a >= 0:
        return float(t[i_peak])
    dt = -b / (2 * a)
    if abs(dt) > max(abs(x[0]), abs(x[-1])):
        return float(t[i_peak])
    return float(t[i_peak] + dt)


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries to CSV without requiring pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# Step 1 - read a two-column file
# =============================================================================

_TIME_SCALES = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "min": 60.0, "minutes": 60.0, "ms": 1e-3}


def read_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=DELIMITER, comments=COMMENTS, skip_header=SKIPROWS)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] <= max(TIME_COLUMN, SIGNAL_COLUMN):
        raise ValueError(f"Could not parse requested columns from {path}")

    unit = TIME_UNIT.lower()
    if unit not in _TIME_SCALES:
        raise ValueError("TIME_UNIT must be 's', 'min', or 'ms'.")

    time_raw = data[:, TIME_COLUMN]
    signal = data[:, SIGNAL_COLUMN]
    finite = np.isfinite(time_raw) & np.isfinite(signal)
    time_s = time_raw[finite] * _TIME_SCALES[unit]
    signal = signal[finite]
    order = np.argsort(time_s)
    return time_s[order], signal[order]


# =============================================================================
# Step 2 - background subtraction and peak location
# =============================================================================

def choose_baseline_window(time_s: np.ndarray) -> float:
    duration = float(np.max(time_s) - np.min(time_s))
    return max(BASELINE_MIN_WINDOW_S, min(BASELINE_MAX_WINDOW_S, BASELINE_FRACTION_OF_TRACE * duration))


def linear_baseline(time_s: np.ndarray, signal: np.ndarray, good: np.ndarray, window_s: float) -> np.ndarray:
    start = good & (time_s <= time_s[0] + window_s)
    end = good & (time_s >= time_s[-1] - window_s)
    if np.sum(start) < 5 or np.sum(end) < 5:
        raise ValueError("Not enough good edge points for baseline estimation.")

    t_start, y_start = np.median(time_s[start]), np.median(signal[start])
    t_end, y_end = np.median(time_s[end]), np.median(signal[end])
    return y_start + (y_end - y_start) * (time_s - t_start) / (t_end - t_start)


def prepare(path: str | Path) -> Prepared:
    path = Path(path)
    t, y = read_trace(path)
    return prepare_from_trace(t, y, path)


def prepare_from_trace(time_s: np.ndarray, signal: np.ndarray, path: str | Path) -> Prepared:
    """Background-subtract and locate the peak for an already-loaded trace.

    Same as :func:`prepare`, but takes the ``(time_s, signal)`` arrays directly
    so callers that read the trace elsewhere (e.g. the FidaNeo parser) can reuse
    the identical preprocessing without going back through :func:`read_trace`.
    """
    path = Path(path)
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    spike = (despike_mask(y, DESPIKE_WINDOW, DESPIKE_SIGMA) & finite) if DESPIKE else np.zeros(y.shape, dtype=bool)
    good = finite & ~spike
    if np.sum(good) < 20:
        raise ValueError("Too few finite points in the trace.")

    window_s = choose_baseline_window(t)
    baseline = linear_baseline(t, y, good, window_s)
    corrected = y - baseline

    edge = good & ((t <= t[0] + window_s) | (t >= t[-1] - window_s))
    noise = robust_std(corrected[edge])

    t_good, y_good = t[good], corrected[good]
    local_peak_i = int(np.argmax(y_good))
    t0 = quadratic_peak_time(t_good, y_good, local_peak_i)
    peak_height = float(np.interp(t0, t_good, y_good))
    if not np.isfinite(peak_height) or peak_height <= 0:
        raise ValueError("Peak height after baseline subtraction is not positive.")

    return Prepared(
        path=path,
        time_s=t,
        signal=y,
        baseline=baseline,
        corrected=corrected,
        normalized=corrected / peak_height,
        good_mask=good,
        baseline_window_s=float(window_s),
        noise_raw=float(noise),
        t0_s=float(t0),
        peak_height=float(peak_height),
        spike_mask=spike,
    )


# =============================================================================
# Step 3 - Taylor kernel, grid narrowing, and the CRLI fit
# =============================================================================

def rh_kernel(time_s: np.ndarray, Rh_nm: np.ndarray | float, t0_s: float, T_K: float) -> np.ndarray:
    """Taylor kernel for a hydrodynamic radius or radius grid."""
    time_s = np.asarray(time_s, dtype=float)
    input_was_scalar = np.ndim(Rh_nm) == 0
    Rh_nm = np.atleast_1d(np.asarray(Rh_nm, dtype=float))
    Rh_m = Rh_nm * 1e-9
    Rh_ref_m = KERNEL_REFERENCE_RH_NM * 1e-9
    if t0_s <= 0 or np.any(Rh_m <= 0):
        raise ValueError("t0_s and Rh must be positive.")

    lam_R = 2 * kB * T_K / (np.pi * VISCOSITY_PA_S * CAPILLARY_RADIUS_M**2 * t0_s)
    tau2 = (time_s[:, None] - t0_s) ** 2
    kernel = np.sqrt(Rh_ref_m / Rh_m)[None, :] * np.exp(-lam_R * tau2 / Rh_m[None, :])
    return kernel[:, 0] if input_was_scalar else kernel


def apparent_rh_from_slope(prep: Prepared, side: str, T_K: float) -> float | None:
    t, s, t0 = prep.time_s, prep.normalized, prep.t0_s
    mask = prep.good_mask & np.isfinite(s) & (s > 0.08) & (s < 0.90)
    if side == "rising":
        mask &= t < t0
    elif side == "falling":
        mask &= t > t0
    if np.sum(mask) < 10:
        return None

    slope, _ = np.polyfit((t[mask] - t0) ** 2, np.log(s[mask]), deg=1)
    if not np.isfinite(slope) or slope >= 0:
        return None

    lam_R = 2 * kB * T_K / (np.pi * VISCOSITY_PA_S * CAPILLARY_RADIUS_M**2 * t0)
    Rh_nm = -lam_R / slope * 1e9
    low, high = RH_CENTER_BOUNDS_NM
    return float(Rh_nm) if np.isfinite(Rh_nm) and low <= Rh_nm <= high else None


def crli_radius_bounds(prep: Prepared, side: str, T_K: float) -> tuple[float, float]:
    """Rh grid bounds for the inversion.

    By default the configured RH_GRID_BOUNDS_NM is used as given.  When
    RH_GRID_AUTONARROW is True, the grid is instead zoomed to roughly
    [guess / 8, guess * 10] around the apparent Taylor-slope size (clamped to
    RH_GRID_BOUNDS_NM).  Narrowing can improve conditioning of the ill-posed
    inversion, but it overrides the configured edges - hence it is opt-in.
    """
    if not RH_GRID_AUTONARROW:
        return RH_GRID_BOUNDS_NM
    guess = apparent_rh_from_slope(prep, side, T_K) or apparent_rh_from_slope(prep, "both", T_K) or 25.0
    guess = float(np.clip(guess, *RH_CENTER_BOUNDS_NM))
    lower = max(RH_GRID_BOUNDS_NM[0], guess / 8.0)
    upper = min(RH_GRID_BOUNDS_NM[1], guess * 10.0)
    return RH_GRID_BOUNDS_NM if lower >= upper else (float(lower), float(upper))


def distribution_stats(Rh_nm: np.ndarray, pdf_logRh: np.ndarray) -> dict:
    Rh_nm = np.asarray(Rh_nm, dtype=float)
    wx = trapezoid_weights(np.log(Rh_nm))
    weights = pdf_logRh * wx
    weights = weights / np.sum(weights)
    cdf = np.cumsum(weights)
    mean = float(np.sum(Rh_nm * weights))
    sd = float(np.sqrt(max(np.sum((Rh_nm - mean) ** 2 * weights), 0.0)))
    percentile = lambda q: float(np.interp(q, cdf, Rh_nm))
    return {
        "colloid_mean_Rh_nm": mean,
        "colloid_sd_Rh_nm": sd,
        "colloid_cv": sd / mean if mean > 0 else np.nan,
        "colloid_median_Rh_nm": percentile(0.50),
        "colloid_mode_Rh_nm": float(Rh_nm[int(np.argmax(pdf_logRh))]),
        "colloid_p05_Rh_nm": percentile(0.05),
        "colloid_p10_Rh_nm": percentile(0.10),
        "colloid_p90_Rh_nm": percentile(0.90),
        "colloid_p95_Rh_nm": percentile(0.95),
        "colloid_span_p10_p90_nm": percentile(0.90) - percentile(0.10),
        "colloid_span_p05_p95_nm": percentile(0.95) - percentile(0.05),
    }


def crli_fit_mask(prep: Prepared, side: str, peak_fraction: float = 0.50) -> np.ndarray:
    """Select fit points.

    For the rising side, ``peak_fraction`` sets how much of the peak is included:
    0.50 keeps the rising half (up to t0); larger values extend the upper time
    cutoff into the falling side, with 1.00 reaching the end of the peak.
    """
    t, s, t0 = prep.time_s, prep.normalized, prep.t0_s
    mask = prep.good_mask & np.isfinite(s) & (s >= CRLI_SIGNAL_MIN) & (s <= CRLI_SIGNAL_MAX)
    if side == "rising":
        if not 0.5 <= peak_fraction <= 1.0:
            raise ValueError("PEAK_FRACTION must be in [0.5, 1.0].")
        if peak_fraction <= 0.5:
            mask &= t <= t0
        else:
            post = mask & (t >= t0)
            post_end = float(np.max(t[post])) if np.any(post) else t0
            cutoff = t0 + (2.0 * peak_fraction - 1.0) * (post_end - t0)
            mask &= t <= cutoff
    elif side == "falling":
        mask &= t >= t0
    elif side == "both":
        pass
    else:
        raise ValueError("side must be 'rising', 'falling', or 'both'.")
    return mask


def t0_seed(prep: Prepared) -> float:
    """Starting t0: the hand-set value in 'manual' mode, else the detected one."""
    if T0_MODE == "manual":
        if T0_MANUAL_S is None:
            raise ValueError("T0_MODE='manual' requires T0_MANUAL_S to be set.")
        return float(T0_MANUAL_S)
    if T0_MODE not in ("auto", "fit"):
        raise ValueError("T0_MODE must be 'auto', 'manual', or 'fit'.")
    return float(prep.t0_s)


def t0_fit_enabled() -> bool:
    return T0_MODE == "fit"


def refine_t0(objective, seed: float, bound: float | None = None) -> float:
    """Bounded 1-D refinement of t0 minimizing ``objective(t0)`` (a scalar misfit).

    Returns ``seed`` unchanged unless ``T0_MODE == 'fit'``.  ``objective`` should
    rebuild the kernel at the trial t0 and return the fit residual; the search is
    confined to ``seed +/- (bound or T0_REFINE_BOUND_S)`` so it only ever nudges
    t0, never relocates the peak.
    """
    if not t0_fit_enabled():
        return float(seed)
    from scipy.optimize import minimize_scalar
    b = T0_REFINE_BOUND_S if bound is None else bound
    result = minimize_scalar(objective, bounds=(seed - b, seed + b), method="bounded",
                             options={"xatol": 1e-3})
    return float(result.x) if result.success else float(seed)


def _solve_fixed(A_col, K_free, L, wx, s, alpha, fraction):
    """NNLS solve with the free fraction pinned exactly to `fraction`.

    Penalties: alpha * ||L q_col||^2 (second-difference smoothness) plus the
    ridge beta * ||q_col||^2 with beta = RIDGE_RHO * alpha (amplitude penalty
    that suppresses low-amplitude spurious modes the curvature penalty misses).
    """
    n = L.shape[1]
    beta = RIDGE_RHO * alpha
    ratio = fraction / (1.0 - fraction)
    A_fit = A_col + K_free[:, None] * ratio * wx[None, :]
    A_aug = np.vstack([A_fit, np.sqrt(alpha) * L, np.sqrt(beta) * np.eye(n)])
    b_aug = np.r_[s, np.zeros(L.shape[0] + n)]
    q_col, _ = nnls(A_aug, b_aug, maxiter=20000)
    q_free = ratio * float(np.sum(q_col * wx))
    return q_col, q_free


def _solve_fit(A_col, K_free, L, wx, s, alpha):
    """NNLS solve with the free amplitude as a free (non-negative) parameter.

    The smoothness and ridge penalties (see `_solve_fixed`) act on the colloid
    amplitudes only; the free fluorophore amplitude is left unpenalized.
    """
    n = L.shape[1]
    beta = RIDGE_RHO * alpha
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


def _crli_scan_at_t0(t, s, Rh_grid, wx, L, t0, T_K, free_lo, free_hi) -> dict:
    """Build the kernel at ``t0`` and run the alpha scan; return the chosen solve.

    Factored out of fit_crli so t0 can be refined by re-running it over trial t0
    values (see refine_t0).  Returns the design columns, the full scan and the
    selected index, so the caller computes outputs without rebuilding.
    """
    A_col = rh_kernel(t, Rh_grid, t0, T_K) * wx[None, :]
    K_free = rh_kernel(t, FREE_RH_NM, t0, T_K)
    solutions, residual_rms, roughness = [], [], []
    for alpha in CRLI_ALPHAS:
        if FREE_FRACTION_MODE == "fixed":
            q_col, q_free = _solve_fixed(A_col, K_free, L, wx, s, alpha, FREE_FRACTION_FIXED)
        elif FREE_FRACTION_MODE in ("fit", "fit_bounded"):
            q_col, q_free = _solve_fit(A_col, K_free, L, wx, s, alpha)
            if FREE_FRACTION_MODE == "fit_bounded":
                # Bounding the fraction is a linear bound on q_free; by convexity the
                # constrained optimum is either interior (keep) or on the violated
                # bound, where the problem reduces to the fixed-fraction solve.
                f = _free_fraction(q_col, q_free, wx)
                if f < free_lo:
                    q_col, q_free = _solve_fixed(A_col, K_free, L, wx, s, alpha, free_lo)
                elif f > free_hi:
                    q_col, q_free = _solve_fixed(A_col, K_free, L, wx, s, alpha, free_hi)
        else:
            raise ValueError("FREE_FRACTION_MODE must be 'fit', 'fit_bounded', or 'fixed'.")

        s_col = A_col @ q_col
        s_free = K_free * q_free
        s_model = s_col + s_free
        solutions.append((q_col, q_free, s_col, s_free, s_model))
        residual_rms.append(float(np.sqrt(np.mean((s_model - s) ** 2))))
        roughness.append(float(np.linalg.norm(L @ q_col)))

    residual_rms = np.asarray(residual_rms)
    roughness = np.asarray(roughness)
    best_rms = float(np.min(residual_rms))
    acceptable = np.flatnonzero(residual_rms <= CRLI_RESIDUAL_FACTOR * best_rms)
    chosen = int(acceptable[-1]) if acceptable.size else int(np.argmin(residual_rms))
    return {"A_col": A_col, "K_free": K_free, "solutions": solutions,
            "residual_rms": residual_rms, "roughness": roughness,
            "chosen": chosen, "best_rms": best_rms}


def fit_crli(prep: Prepared, side: str = FIT_SIDE) -> FitResult:
    T_K = 273.15 + TEMPERATURE_C
    all_indices = np.flatnonzero(crli_fit_mask(prep, side, PEAK_FRACTION))
    indices = all_indices[:: max(1, int(CRLI_FIT_STRIDE))]
    if indices.size < 25:
        raise ValueError("Too few fitting points for the CRLI fit.")

    t = prep.time_s[indices]
    y = prep.signal[indices]
    s = prep.normalized[indices]

    if FREE_FRACTION_MODE == "fixed" and not 0.0 <= FREE_FRACTION_FIXED < 1.0:
        raise ValueError("FREE_FRACTION_FIXED must be in [0, 1).")
    free_lo, free_hi = FREE_FRACTION_BOUNDS
    if FREE_FRACTION_MODE == "fit_bounded" and not 0.0 <= free_lo <= free_hi < 1.0:
        raise ValueError("FREE_FRACTION_BOUNDS must satisfy 0 <= lower <= upper < 1.")

    rh_min, rh_max = crli_radius_bounds(prep, side, T_K)
    Rh_grid = np.geomspace(rh_min, rh_max, N_RH_GRID)
    wx = trapezoid_weights(np.log(Rh_grid))
    L = second_difference_matrix(N_RH_GRID)

    # t0 is either locked (auto/manual) or refined here against the inversion's
    # own best residual - see T0_MODE.  The fit window above stays pinned to the
    # preprocessing t0; only the kernel t0 moves.
    t0_init = t0_seed(prep)
    t0 = refine_t0(lambda tt: _crli_scan_at_t0(t, s, Rh_grid, wx, L, tt, T_K, free_lo, free_hi)["best_rms"], t0_init)
    scan = _crli_scan_at_t0(t, s, Rh_grid, wx, L, t0, T_K, free_lo, free_hi)
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
    baseline = np.interp(t, prep.time_s, prep.baseline)
    model_raw = baseline + prep.peak_height * s_model
    free_raw = prep.peak_height * s_free
    colloid_raw = prep.peak_height * s_col
    residual_raw = model_raw - y

    free_fraction = float(q_free / total_area)
    K_free_t0 = float(rh_kernel(np.array([t0]), FREE_RH_NM, t0, T_K)[0])
    K_col_t0 = rh_kernel(np.array([t0]), Rh_grid, t0, T_K)[0]
    peak_free = K_free_t0 * q_free
    peak_col = float(np.sum(K_col_t0 * q_col * wx))
    peak_sum = peak_free + peak_col

    # Truncation test: the signal contributed solely by colloid mass below the
    # cutoff (the free fluorophore term is left intact).  If this is at the noise
    # level the low-Rh mode is an inversion artefact; if it is well above noise it
    # is a real feature of the sample.
    low = Rh_grid < TRUNCATION_RH_NM
    s_col_low = A_col[:, low] @ q_col[low]              # normalized signal from sub-cutoff mass
    trunc_low_raw = prep.peak_height * s_col_low
    trunc_rms_raw = float(np.sqrt(np.mean(trunc_low_raw**2)))
    trunc_peak_raw = float(np.max(np.abs(trunc_low_raw)))
    mass_below = float(np.sum(q_col[low] * wx[low]) / area_col)

    stats = distribution_stats(Rh_grid, pdf_log)
    stats.update({
        "success": True,
        "fit_side": side,
        "peak_fraction": float(PEAK_FRACTION),
        "t0_s": float(t0),
        "t0_mode": T0_MODE,
        "t0_seed_s": float(t0_init),
        "t0_fitted": t0_fit_enabled(),
        "t0_shift_s": float(t0 - t0_init),
        "noise_raw": float(prep.noise_raw),
        "baseline_window_s": float(prep.baseline_window_s),
        "n_fit_points": int(t.size),
        "free_Rh_nm": float(FREE_RH_NM),
        "free_fraction_mode": FREE_FRACTION_MODE,
        "free_fraction_detector_weighted": free_fraction,
        "free_fraction_lower_bound": float(free_lo) if FREE_FRACTION_MODE == "fit_bounded" else np.nan,
        "free_fraction_upper_bound": float(free_hi) if FREE_FRACTION_MODE == "fit_bounded" else np.nan,
        "free_fraction_bound_active": (
            "none" if FREE_FRACTION_MODE != "fit_bounded"
            else "lower" if free_fraction <= free_lo + 1e-6
            else "upper" if free_fraction >= free_hi - 1e-6
            else "none"
        ),
        "free_peak_height_fraction": float(peak_free / peak_sum) if peak_sum > 0 else np.nan,
        "colloid_peak_height_fraction": float(peak_col / peak_sum) if peak_sum > 0 else np.nan,
        "residual_rms_raw": float(np.sqrt(np.mean(residual_raw**2))),
        "residual_rms_normalized_to_peak": float(np.sqrt(np.mean((s_model - s) ** 2))),
        "rss_raw": float(np.sum(residual_raw**2)),
        "crli_alpha": float(CRLI_ALPHAS[chosen]),
        "crli_ridge_rho": float(RIDGE_RHO),
        "crli_ridge_beta": float(RIDGE_RHO * CRLI_ALPHAS[chosen]),
        "crli_alpha_at_scan_boundary": chosen in {0, len(CRLI_ALPHAS) - 1},
        "crli_best_residual_rms": best_rms,
        "crli_chosen_residual_rms": float(residual_rms[chosen]),
        "crli_rh_min_nm": float(rh_min),
        "crli_rh_max_nm": float(rh_max),
        "trunc_rh_nm": float(TRUNCATION_RH_NM),
        "trunc_mass_fraction_below": mass_below,
        "trunc_signal_rms_raw": trunc_rms_raw,
        "trunc_signal_peak_raw": trunc_peak_raw,
        "trunc_signal_rms_over_noise": trunc_rms_raw / prep.noise_raw if prep.noise_raw > 0 else np.nan,
        "trunc_signal_peak_over_noise": trunc_peak_raw / prep.noise_raw if prep.noise_raw > 0 else np.nan,
    })

    return FitResult(
        params={"t0_s": float(t0), "free_fraction": free_fraction, "alpha": float(CRLI_ALPHAS[chosen]), "side": side},
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
        extra={
            "alpha_scan": CRLI_ALPHAS,
            "residual_rms_scan": residual_rms,
            "roughness_scan": roughness,
            "s_fit": s,
            "s_model": s_model,
            "trunc_rh_nm": float(TRUNCATION_RH_NM),
            "trunc_low_signal_raw": trunc_low_raw,
        },
    )


# =============================================================================
# Step 4 - outputs (CSV + plots)
# =============================================================================

def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def save_preprocessing_plots(prep: Prepared, out_dir: Path, stem: str) -> None:
    t, y = prep.time_s, prep.signal

    plt.figure(figsize=(8, 4))
    plt.plot(t, y, ".", markersize=2, label="raw data")
    plt.plot(t, prep.baseline, "-", label="baseline")
    spikes = getattr(prep, "spike_mask", None)
    if spikes is not None and np.any(spikes):
        plt.plot(t[spikes], y[spikes], "x", color="red", markersize=7, markeredgewidth=1.5,
                 label=f"{int(np.sum(spikes))} spikes (ignored)")
    plt.axvline(prep.t0_s, linestyle="--", label="t0")
    plt.xlabel("time / s")
    plt.ylabel("signal")
    plt.title(f"{stem}: preprocessing")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / "preprocessing" / f"{stem}_preprocessing.svg")

    plt.figure(figsize=(8, 4))
    plt.plot(t[prep.good_mask], prep.normalized[prep.good_mask], ".", markersize=2)
    plt.axvline(prep.t0_s, linestyle="--")
    plt.xlabel("time / s")
    plt.ylabel("normalized signal")
    plt.title(f"{stem}: normalized trace")
    save_current_figure(out_dir / "preprocessing" / f"{stem}_normalized.svg")


def save_fit_plots(result: FitResult, prep: Prepared, out_dir: Path, stem: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(result.time_fit_s, result.signal_fit_raw, ".", markersize=3, label="fit data")
    plt.plot(result.time_fit_s, result.signal_model_raw, "-", label="fit")
    plt.xlabel("time / s")
    plt.ylabel("raw signal")
    plt.title(f"{stem}: CRLI fit")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_crli_fit.svg")

    plt.figure(figsize=(9, 4))
    plt.plot(result.time_fit_s, result.residuals_raw / prep.peak_height, ".", markersize=3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("time / s")
    plt.ylabel("residual / peak height")
    plt.title(f"{stem}: CRLI residuals")
    save_current_figure(out_dir / f"{stem}_crli_residuals.svg")

    plt.figure(figsize=(9, 5))
    plt.plot(result.time_fit_s, result.signal_fit_raw, ".", markersize=3, label="fit data")
    plt.plot(result.time_fit_s, result.signal_model_raw, "-", label="total fit")
    plt.plot(result.time_fit_s, result.baseline_model_raw, "--", label="baseline")
    plt.plot(result.time_fit_s, result.baseline_model_raw + result.free_signal_raw, ":", label="baseline + free fluorophore")
    plt.plot(result.time_fit_s, result.baseline_model_raw + result.colloid_signal_raw, "-.", label="baseline + colloid")
    plt.xlabel("time / s")
    plt.ylabel("raw signal")
    plt.title(f"{stem}: CRLI decomposition")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_crli_decomposition.svg")

    plt.figure(figsize=(7, 5))
    plt.semilogx(result.Rh_nm, result.pdf_logRh, "-", label="crli")
    plt.axvline(FREE_RH_NM, linestyle=":", label=f"free fluorophore {FREE_RH_NM:g} nm")
    plt.xlabel(r"$R_h$ / nm")
    plt.ylabel(r"colloid PDF per $\ln R_h$")
    plt.title(f"{stem}: colloid distribution")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_crli_distribution.svg")


def save_alpha_plots(result: FitResult, out_dir: Path, stem: str) -> None:
    alpha = result.extra["alpha_scan"]
    residual = result.extra["residual_rms_scan"]
    roughness = result.extra["roughness_scan"]
    chosen = result.stats.get("crli_alpha", np.nan)

    plt.figure(figsize=(6, 4))
    plt.loglog(alpha, residual, ".-")
    if np.isfinite(chosen):
        plt.axvline(chosen, linestyle="--", label="chosen alpha")
        plt.legend(fontsize=8)
    plt.xlabel(r"regularization $\alpha$")
    plt.ylabel("normalized residual RMS")
    plt.title(f"{stem}: CRLI alpha scan")
    save_current_figure(out_dir / f"{stem}_crli_alpha_residual_scan.svg")

    plt.figure(figsize=(6, 4))
    plt.loglog(roughness, residual, ".-")
    plt.xlabel("roughness")
    plt.ylabel("normalized residual RMS")
    plt.title(f"{stem}: CRLI roughness scan")
    save_current_figure(out_dir / f"{stem}_crli_roughness_scan.svg")


def save_truncation_plot(result: FitResult, prep: Prepared, out_dir: Path, stem: str) -> None:
    """Compare the full model with one where colloid mass below the cutoff is zeroed.

    The lower panel is the signal that mass alone contributes, shown against the
    +/- noise band so artefact (within noise) vs feature (well above noise) is
    read off directly.
    """
    cutoff = result.extra["trunc_rh_nm"]
    low_raw = result.extra["trunc_low_signal_raw"]
    t = result.time_fit_s
    noise = prep.noise_raw
    trunc_model = result.signal_model_raw - low_raw      # full model with sub-cutoff mass removed
    rms = result.stats["trunc_signal_rms_raw"]
    peak = result.stats["trunc_signal_peak_raw"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, result.signal_fit_raw, ".", markersize=3, label="fit data")
    ax1.plot(t, result.signal_model_raw, "-", label="full model")
    ax1.plot(t, trunc_model, "--", label=f"model with $R_h<{cutoff:g}$ nm removed")
    ax1.set_ylabel("raw signal")
    ax1.set_title(f"{stem}: truncation test (mass below {cutoff:g} nm)")
    ax1.legend(fontsize=8)

    ax2.plot(t, low_raw, "-", color="C3", label=f"signal from $R_h<{cutoff:g}$ nm mass")
    ax2.axhspan(-noise, noise, color="gray", alpha=0.25, label=r"$\pm$ noise")
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="k")
    ax2.set_xlabel("time / s")
    ax2.set_ylabel("signal difference")
    ax2.set_title(f"peak {peak:.3g} = {peak/noise:.2f}x noise | RMS {rms:.3g} = {rms/noise:.2f}x noise", fontsize=9)
    ax2.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_crli_truncation_test.svg")


def write_distribution_csv(result: FitResult, out_dir: Path, stem: str) -> None:
    rows = [
        {"Rh_nm": float(rh), "pdf_per_ln_Rh": float(pdf_log), "pdf_per_nm": float(pdf_R), "cdf": float(c)}
        for rh, pdf_log, pdf_R, c in zip(result.Rh_nm, result.pdf_logRh, result.pdf_Rh_per_nm, result.cdf)
    ]
    write_rows(out_dir / f"{stem}_crli_distribution.csv", rows)


def save_outputs(result: FitResult, prep: Prepared, out_dir: Path, stem: str) -> None:
    write_distribution_csv(result, out_dir, stem)
    if SAVE_PLOTS:
        save_fit_plots(result, prep, out_dir, stem)
        save_alpha_plots(result, out_dir, stem)
        save_truncation_plot(result, prep, out_dir, stem)


def config_dump() -> dict:
    keys = [
        "DATA_PATTERN", "OUTPUT_DIR", "TIME_COLUMN", "SIGNAL_COLUMN", "DELIMITER", "SKIPROWS",
        "COMMENTS", "TIME_UNIT", "TEMPERATURE_C", "CAPILLARY_RADIUS_M", "VISCOSITY_PA_S",
        "KERNEL_REFERENCE_RH_NM", "BASELINE_MIN_WINDOW_S", "BASELINE_MAX_WINDOW_S",
        "BASELINE_FRACTION_OF_TRACE", "DESPIKE", "DESPIKE_WINDOW", "DESPIKE_SIGMA",
        "FREE_RH_NM", "FREE_FRACTION_MODE", "FREE_FRACTION_FIXED",
        "FREE_FRACTION_BOUNDS",
        "FIT_SIDE", "PEAK_FRACTION", "RH_GRID_BOUNDS_NM", "RH_GRID_AUTONARROW", "RH_CENTER_BOUNDS_NM", "N_RH_GRID", "CRLI_ALPHAS",
        "CRLI_RESIDUAL_FACTOR", "RIDGE_RHO", "CRLI_FIT_STRIDE", "CRLI_SIGNAL_MIN", "CRLI_SIGNAL_MAX",
        "T0_MODE", "T0_MANUAL_S", "T0_REFINE_BOUND_S",
        "SAVE_PLOTS", "SHOW_PLOTS", "TRUNCATION_RH_NM",
    ]
    out = {}
    for key in keys:
        value = globals()[key]
        out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


def print_result(stem: str, result: FitResult) -> None:
    s = result.stats
    print(
        f"  {stem:30s} | "
        f"t0={s['t0_s']:.2f} s | "
        f"free={100 * s['free_fraction_detector_weighted']:.2f}% | "
        f"median Rh={s['colloid_median_Rh_nm']:.2f} nm | "
        f"mean Rh={s['colloid_mean_Rh_nm']:.2f} nm | "
        f"RMS/peak={s['residual_rms_normalized_to_peak']:.4g}",
        flush=True,
    )


# =============================================================================
# Step 5 - batch runner
# =============================================================================

def run(data_pattern: str, output_dir: str, side: str) -> None:
    files = sorted(Path(p) for p in glob(data_pattern, recursive=True) if Path(p).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {data_pattern!r}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_settings.json").write_text(json.dumps(config_dump(), indent=2), encoding="utf-8")

    print(f"Found {len(files)} file(s).")
    print(f"Output folder: {root.resolve()}")

    rows: list[dict] = []
    for path in files:
        print(f"\nAnalysing {path}", flush=True)
        try:
            prep = prepare(path)
            stem = safe_filename(path.stem)
            if SAVE_PLOTS:
                save_preprocessing_plots(prep, root, stem)
            result = fit_crli(prep, side=side)
            save_outputs(result, prep, root, stem)
            rows.append({"file": str(path), **result.stats})
            print_result(stem, result)
        except Exception as exc:
            warnings.warn(f"Skipped {path.name}: {exc}")
            rows.append({"file": str(path), "success": False, "error": str(exc)})

    summary_path = root / "crli_summary.csv"
    write_rows(summary_path, rows)
    print(f"\nSaved summary: {summary_path}")
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified CRLI fit for two-column TDA traces.")
    parser.add_argument("--data", dest="data_pattern", default=DATA_PATTERN, help="Input glob, e.g. '*.csv'.")
    parser.add_argument("--output", dest="output_dir", default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--side", choices=["rising", "falling", "both"], default=FIT_SIDE, help="Taylorgram side to fit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_pattern, args.output_dir, args.side)


if __name__ == "__main__":
    main()
