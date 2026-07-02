#!/usr/bin/env python3
"""Small, config-free numerical helpers shared by every fit model.

Moved out of ``fit_crli_simple.py`` verbatim (originally noted there as coming
from the now-archived ``legacy/tda_processing.py``); math is unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


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


def quadratic_peak_time(t: np.ndarray, y: np.ndarray, i_peak: int, half_window: int = 25) -> float:
    """Sub-sample peak time: a Huber-loss parabola fit through a window around
    ``i_peak``. Robust (not plain OLS) and wide (not a handful of samples) so
    that a single artefact spike near the peak - one despiking missed - is
    just one of many points rather than able to dominate the fit or, worse,
    be ``i_peak`` itself with the window centred on it."""
    lo = max(0, i_peak - half_window)
    hi = min(len(t), i_peak + half_window + 1)
    if hi - lo < 3:
        return float(t[i_peak])

    x = t[lo:hi] - t[i_peak]
    y_win = y[lo:hi]
    a0, b0, c0 = np.polyfit(x, y_win, deg=2)
    scale = robust_std(y_win - (a0 * x ** 2 + b0 * x + c0))
    fit = least_squares(lambda coef: coef[0] * x ** 2 + coef[1] * x + coef[2] - y_win,
                         x0=[a0, b0, c0], loss="huber", f_scale=scale)
    a, b, _ = fit.x
    if a >= 0:
        return float(t[i_peak])
    dt = -b / (2 * a)
    if abs(dt) > max(abs(x[0]), abs(x[-1])):
        return float(t[i_peak])
    return float(t[i_peak] + dt)


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
