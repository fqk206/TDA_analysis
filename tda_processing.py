#!/usr/bin/env python3
"""Shared Taylor Dispersion Analysis helpers.

This module contains only the instrument/file reading, taylorgram preparation,
Taylor kernels, distribution utilities, plotting, and batch-running code.  The
actual fitting models live in separate scripts:

    fit_dual_gaussian.py
    fit_crli.py
    fit_parametric_distribution.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from glob import glob
from io import StringIO
from pathlib import Path
from typing import Callable
import argparse
import csv
import json
import math
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann as kB
from scipy.ndimage import median_filter


# =============================================================================
# Settings and data containers
# =============================================================================

@dataclass
class Settings:
    """Shared user settings used by all model scripts."""

    # Input/output.  These are usually overridden at the top of each model script.
    data_pattern: str = "*.txt"
    output_dir: str = "tda_results"
    input_format: str = "fidaneo"          # "fidaneo" or "two_column"
    fidaneo_time_unit: str = "auto"        # "auto", "s", or "min"

    # Plain two-column input settings.
    two_column_time_column: int = 0
    two_column_signal_column: int = 1
    two_column_delimiter: str | None = None
    two_column_skiprows: int = 0
    two_column_comments: str = "#"
    two_column_time_unit: str = "s"        # "s", "min", or "ms"
    two_column_temperature_C: float = 25.0

    # Experimental constants.  capillary_radius_m is the radius, not diameter.
    capillary_radius_m: float = 37.5e-6
    viscosity_Pa_s: float = 8.90e-4
    default_temperature_K: float = 298.15

    # Known free fluorophore contribution.
    free_Rh_nm: float = 0.59
    free_fraction_mode: str = "fit"        # "fit" or "fixed"
    free_fraction_fixed: float = 0.05
    free_fraction_bounds: tuple[float, float] = (0.0, 0.70)

    # Initial preprocessing.
    baseline_min_window_s: float = 15.0
    baseline_max_window_s: float = 90.0
    baseline_fraction_of_trace: float = 0.15

    # Spike handling.  Spike points are excluded from preprocessing and fitting.
    despike: bool = True
    max_spikes_allowed: int = 5
    spike_threshold_mad: float = 10.0
    spike_median_window_points: int = 11
    spike_padding_points: int = 2
    spike_min_prominence_fraction_of_signal_range: float = 0.15

    # Nonlinear fit point selection.
    fit_signal_min: float = 0.03
    fit_signal_max: float = 1.10
    max_fit_points: int = 700
    t0_half_window_s: float | None = None

    # Dual-Gaussian fit only.  0.50 = rising side only; 1.00 = whole selected peak.
    dual_gaussian_peak_fraction: float = 0.50

    # Hydrodynamic-radius grids and fitted bounds for the colloid population.
    Rh_grid_bounds_nm: tuple[float, float] = (1.5, 200.0)
    Rh_center_bounds_nm: tuple[float, float] = (1.5, 300.0)
    initial_colloid_Rh_nm: float | None = None
    n_Rh_grid_parametric: int = 180
    n_Rh_grid_crli: int = 120
    normal_sd_bounds_nm: tuple[float, float] = (0.2, 250.0)
    lognormal_sigma_bounds: tuple[float, float] = (0.02, 1.8)

    # CRLI settings.
    crli_alphas: np.ndarray = field(default_factory=lambda: np.logspace(-8, 8, 80))
    crli_residual_factor: float = 1.10
    crli_fit_stride: int = 4
    crli_signal_min: float = 0.03
    crli_signal_max: float = 1.02

    # Plotting.
    save_plots: bool = True
    show_plots: bool = False

    # Numerical scale factor only.  Fitted amplitudes absorb this value.
    kernel_reference_Rh_nm: float = 25.0


@dataclass
class Trace:
    path: Path
    time_s: np.ndarray
    signal: np.ndarray
    meta: dict


@dataclass
class PreparedTrace:
    trace: Trace
    baseline_initial: np.ndarray
    signal_corrected: np.ndarray
    baseline_window_s: float
    noise_raw: float
    t0_initial_s: float
    peak_height_initial: float
    normalized_initial: np.ndarray
    good_mask: np.ndarray
    spike_mask: np.ndarray
    spike_count: int
    spike_times_s: list[float]


@dataclass
class FitResult:
    model_name: str
    success: bool
    message: str
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
    fit_mask: np.ndarray
    extra: dict | None = None


# =============================================================================
# Small general utilities
# =============================================================================

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


def settings_to_jsonable(cfg: Settings) -> dict:
    out = asdict(cfg)
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
    return out


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


# =============================================================================
# Input readers
# =============================================================================

def get_block(text: str, name: str) -> str | None:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, flags=re.IGNORECASE | re.DOTALL)
    return None if match is None else match.group(1).strip()


def parse_simple_key_values(block: str | None) -> dict:
    out = {}
    if not block:
        return out
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def parse_method_table(block: str | None) -> dict | None:
    if not block:
        return None
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) == len(headers):
            rows.append(dict(zip(headers, fields)))
    measured = [row for row in rows if row.get("Measure", "").strip().lower() == "yes"]
    return measured[-1] if measured else None


def numeric_rows_from_text(block: str) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for line in block.splitlines():
        parts = re.split(r"[\t,; ]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError("No two-column numeric data found.")
    return np.asarray(rows, dtype=float)


def read_fidaneo_txt(path: str | Path, cfg: Settings) -> Trace:
    """Read a FidaNeo .txt export with <header> and <data> blocks."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    data_block = get_block(text, "data")
    if data_block is None:
        raise ValueError(f"No <data>...</data> block found in {path}")

    data = numeric_rows_from_text(data_block)
    time_raw, signal = data[:, 0], data[:, 1]

    meta = {
        "file": str(path),
        **parse_simple_key_values(get_block(text, "sample")),
        **parse_simple_key_values(get_block(text, "timestamps")),
    }
    method_row = parse_method_table(get_block(text, "meth"))
    if method_row:
        meta["method_row"] = method_row
        for src, dest in (("Pressure [mbar]", "pressure_mbar"), ("Time [s]", "method_time_s")):
            try:
                meta[dest] = float(method_row[src])
            except Exception:
                pass
        temp_key = next((key for key in method_row if key.lower().startswith("capillary temp")), None)
        if temp_key:
            try:
                meta["capillary_temp_C"] = float(method_row[temp_key])
            except Exception:
                pass

    scale, unit = fidaneo_time_scale(time_raw, meta.get("method_time_s"), cfg.fidaneo_time_unit)
    meta.update({"time_unit_detected": unit, "time_scale_to_seconds": scale, "n_points": len(time_raw)})
    order = np.argsort(time_raw)
    return Trace(path=path, time_s=time_raw[order] * scale, signal=signal[order], meta=meta)


def fidaneo_time_scale(time_raw: np.ndarray, duration_s: float | None, setting: str) -> tuple[float, str]:
    if setting in {"s", "sec", "seconds"}:
        return 1.0, "s_forced"
    if setting in {"min", "minute", "minutes"}:
        return 60.0, "min_forced"
    if setting != "auto":
        raise ValueError("fidaneo_time_unit must be 'auto', 's', or 'min'.")

    if duration_s and duration_s > 0:
        end_raw = float(time_raw[-1])
        if abs(end_raw * 60 - duration_s) / duration_s < 0.05:
            return 60.0, "min_auto"
        if abs(end_raw - duration_s) / duration_s < 0.05:
            return 1.0, "s_auto"
    return 60.0, "min_assumed"


def read_two_column_file(path: str | Path, cfg: Settings) -> Trace:
    data = np.genfromtxt(
        path,
        delimiter=cfg.two_column_delimiter,
        comments=cfg.two_column_comments,
        skip_header=cfg.two_column_skiprows,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] <= max(cfg.two_column_time_column, cfg.two_column_signal_column):
        raise ValueError(f"Could not parse requested columns from {path}")

    time_raw = data[:, cfg.two_column_time_column]
    signal = data[:, cfg.two_column_signal_column]
    finite = np.isfinite(time_raw) & np.isfinite(signal)
    time_s, unit, scale = scale_time_to_seconds(time_raw[finite], cfg.two_column_time_unit)
    meta = {
        "file": str(path),
        "Sample name": Path(path).stem,
        "input_format": "two_column",
        "capillary_temp_C": cfg.two_column_temperature_C,
        "time_unit_detected": unit,
        "time_scale_to_seconds": scale,
        "n_points": int(np.sum(finite)),
    }
    order = np.argsort(time_s)
    return Trace(path=Path(path), time_s=time_s[order], signal=signal[finite][order], meta=meta)


def scale_time_to_seconds(time_raw: np.ndarray, unit: str) -> tuple[np.ndarray, str, float]:
    unit = unit.lower()
    scales = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "min": 60.0, "minutes": 60.0, "ms": 1e-3}
    if unit not in scales:
        raise ValueError("time unit must be 's', 'min', or 'ms'.")
    return np.asarray(time_raw, dtype=float) * scales[unit], unit, scales[unit]


def read_trace(path: str | Path, cfg: Settings) -> Trace:
    if cfg.input_format == "fidaneo":
        return read_fidaneo_txt(path, cfg)
    if cfg.input_format == "two_column":
        return read_two_column_file(path, cfg)
    raise ValueError("input_format must be 'fidaneo' or 'two_column'.")


# =============================================================================
# Spike detection and trace preparation
# =============================================================================

def odd_window(value: int, n_points: int) -> int:
    value = max(3, int(value))
    value = value + 1 if value % 2 == 0 else value
    if value > n_points:
        value = n_points if n_points % 2 == 1 else n_points - 1
    return max(3, value)


def contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [group for group in np.split(indices, cuts) if group.size]


def detect_spikes(signal: np.ndarray, cfg: Settings) -> tuple[np.ndarray, list[float]]:
    """Return a spike mask and representative spike indices.

    A spike is defined as a narrow excursion from a short running median, with
    the excursion size estimated robustly by the MAD of the median residuals.
    """
    spike_mask = np.zeros(signal.size, dtype=bool)
    if not cfg.despike or signal.size < 5:
        return spike_mask, []

    window = odd_window(cfg.spike_median_window_points, signal.size)
    local_median = median_filter(signal, size=window, mode="nearest")
    residual = signal - local_median
    sigma = robust_std(residual)
    signal_range = float(np.percentile(signal, 99) - np.percentile(signal, 1))
    min_prominence = cfg.spike_min_prominence_fraction_of_signal_range * max(signal_range, 0.0)
    threshold = max(cfg.spike_threshold_mad * sigma, min_prominence)
    outliers = np.flatnonzero(np.abs(residual) > threshold)

    max_event_width = max(3, window // 2)
    spike_events = [group for group in contiguous_groups(outliers) if group.size <= max_event_width]
    centers: list[float] = []
    pad = max(0, int(cfg.spike_padding_points))
    for group in spike_events:
        lo = max(0, int(group[0]) - pad)
        hi = min(signal.size, int(group[-1]) + pad + 1)
        spike_mask[lo:hi] = True
        centers.append(float(group[group.size // 2]))
    return spike_mask, centers


def choose_baseline_window(time_s: np.ndarray, cfg: Settings) -> float:
    duration = float(np.max(time_s) - np.min(time_s))
    return max(cfg.baseline_min_window_s, min(cfg.baseline_max_window_s, cfg.baseline_fraction_of_trace * duration))


def initial_linear_baseline(time_s: np.ndarray, signal: np.ndarray, good: np.ndarray, window_s: float) -> np.ndarray:
    start = good & (time_s <= time_s[0] + window_s)
    end = good & (time_s >= time_s[-1] - window_s)
    if np.sum(start) < 5 or np.sum(end) < 5:
        raise ValueError("Not enough good edge points for baseline estimation.")

    t_start, y_start = np.median(time_s[start]), np.median(signal[start])
    t_end, y_end = np.median(time_s[end]), np.median(signal[end])
    return y_start + (y_end - y_start) * (time_s - t_start) / (t_end - t_start)


def prepare_trace(trace: Trace, cfg: Settings) -> PreparedTrace:
    t, y = trace.time_s, trace.signal
    finite = np.isfinite(t) & np.isfinite(y)
    spike_mask, spike_index_centers = detect_spikes(y, cfg)
    spike_count = len(spike_index_centers)
    if spike_count > cfg.max_spikes_allowed:
        raise ValueError(f"Detected {spike_count} spike events; maximum allowed is {cfg.max_spikes_allowed}.")

    good = finite & ~spike_mask
    window_s = choose_baseline_window(t, cfg)
    baseline = initial_linear_baseline(t, y, good, window_s)
    corrected = y - baseline

    edge = good & ((t <= t[0] + window_s) | (t >= t[-1] - window_s))
    noise = robust_std(corrected[edge])

    good_indices = np.flatnonzero(good)
    if good_indices.size < 20:
        raise ValueError("Too few non-spike points remain after preprocessing.")
    local_peak_i = int(np.argmax(corrected[good]))
    t_good, y_good = t[good], corrected[good]
    t0 = quadratic_peak_time(t_good, y_good, local_peak_i)
    peak_height = float(np.interp(t0, t_good, y_good))
    if not np.isfinite(peak_height) or peak_height <= 0:
        raise ValueError("Initial peak height after baseline subtraction is not positive.")

    spike_times = [float(t[int(i)]) for i in spike_index_centers]
    return PreparedTrace(
        trace=trace,
        baseline_initial=baseline,
        signal_corrected=corrected,
        baseline_window_s=float(window_s),
        noise_raw=float(noise),
        t0_initial_s=float(t0),
        peak_height_initial=float(peak_height),
        normalized_initial=corrected / peak_height,
        good_mask=good,
        spike_mask=spike_mask,
        spike_count=spike_count,
        spike_times_s=spike_times,
    )


def thin_mask(mask: np.ndarray, max_points: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= max_points:
        return mask.copy()
    keep = np.linspace(0, indices.size - 1, max_points).round().astype(int)
    out = np.zeros_like(mask, dtype=bool)
    out[indices[np.unique(keep)]] = True
    return out


def normalize_fraction(value: float) -> float:
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return float(np.clip(value, 0.0, 1.0))


def make_fit_mask(prep: PreparedTrace, cfg: Settings, side: str = "rising", peak_fraction: float | None = None, include_edges: bool = True) -> np.ndarray:
    """Choose points used for raw-signal nonlinear fitting.

    For the dual-Gaussian model, peak_fraction can be used to include more than
    the rising half of the selected peak: 0.50 = rising only, 1.00 = full peak.
    """
    t, s, t0 = prep.trace.time_s, prep.normalized_initial, prep.t0_initial_s
    good = prep.good_mask & np.isfinite(s)
    edge = good & ((t <= t[0] + prep.baseline_window_s) | (t >= t[-1] - prep.baseline_window_s))
    peak = good & (s >= cfg.fit_signal_min) & (s <= cfg.fit_signal_max)

    if peak_fraction is not None:
        fraction = normalize_fraction(peak_fraction)
        if fraction <= 0.5:
            peak &= t <= t0
        else:
            post = peak & (t >= t0)
            post_end = float(np.max(t[post])) if np.any(post) else t0
            cutoff = t0 + (2.0 * fraction - 1.0) * (post_end - t0)
            peak &= t <= cutoff
    elif side == "rising":
        peak &= t <= t0
    elif side == "falling":
        peak &= t >= t0
    elif side == "both":
        pass
    else:
        raise ValueError("side must be 'rising', 'falling', or 'both'.")

    return (peak | edge) if include_edges else peak


# =============================================================================
# Taylor kernels and colloid distributions
# =============================================================================

def trace_temperature_K(trace: Trace, cfg: Settings) -> float:
    try:
        return 273.15 + float(trace.meta["capillary_temp_C"])
    except Exception:
        return cfg.default_temperature_K


def rh_kernel(time_s: np.ndarray, Rh_nm: np.ndarray | float, t0_s: float, T_K: float, cfg: Settings) -> np.ndarray:
    """Taylor kernel for a hydrodynamic radius or radius grid."""
    time_s = np.asarray(time_s, dtype=float)
    input_was_scalar = np.ndim(Rh_nm) == 0
    Rh_nm = np.atleast_1d(np.asarray(Rh_nm, dtype=float))
    Rh_m = Rh_nm * 1e-9
    Rh_ref_m = cfg.kernel_reference_Rh_nm * 1e-9
    if t0_s <= 0 or np.any(Rh_m <= 0):
        raise ValueError("t0_s and Rh must be positive.")

    lam_R = 2 * kB * T_K / (np.pi * cfg.viscosity_Pa_s * cfg.capillary_radius_m**2 * t0_s)
    tau2 = (time_s[:, None] - t0_s) ** 2
    kernel = np.sqrt(Rh_ref_m / Rh_m)[None, :] * np.exp(-lam_R * tau2 / Rh_m[None, :])
    return kernel[:, 0] if input_was_scalar else kernel


def colloid_pdf_logRh(Rh_nm: np.ndarray, distribution: str, center_nm: float, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized colloid PDF per ln(Rh), per nm, and CDF."""
    Rh_nm = np.asarray(Rh_nm, dtype=float)
    x = np.log(Rh_nm)
    wx = trapezoid_weights(x)

    if distribution == "lognormal":
        sigma = float(width)
        pdf_log = np.exp(-0.5 * ((x - np.log(center_nm)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    elif distribution == "normal":
        sd = float(width)
        z = (Rh_nm - center_nm) / sd
        pdf_R = np.exp(-0.5 * z**2) / (sd * np.sqrt(2 * np.pi))
        pdf_R[Rh_nm <= 0] = 0.0
        pdf_log = Rh_nm * pdf_R
    else:
        raise ValueError("distribution must be 'normal' or 'lognormal'.")

    area = float(np.sum(pdf_log * wx))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Distribution has zero area on the chosen Rh grid.")
    pdf_log = pdf_log / area
    pdf_R = pdf_log / Rh_nm
    cdf = np.cumsum(pdf_log * wx)
    return pdf_log, pdf_R, cdf / cdf[-1]


def delta_like_pdf_logRh(Rh_nm: np.ndarray, center_nm: float, sigma_ln: float = 0.025) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.log(Rh_nm)
    wx = trapezoid_weights(x)
    pdf_log = np.exp(-0.5 * ((x - np.log(center_nm)) / sigma_ln) ** 2) / (sigma_ln * np.sqrt(2 * np.pi))
    pdf_log = pdf_log / np.sum(pdf_log * wx)
    cdf = np.cumsum(pdf_log * wx)
    return pdf_log, pdf_log / Rh_nm, cdf / cdf[-1]


def colloid_kernel_integral(time_s: np.ndarray, Rh_nm: np.ndarray, pdf_logRh: np.ndarray, t0_s: float, T_K: float, cfg: Settings) -> np.ndarray:
    wx = trapezoid_weights(np.log(Rh_nm))
    return rh_kernel(time_s, Rh_nm, t0_s, T_K, cfg) @ (pdf_logRh * wx)


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


def delta_distribution_stats(center_nm: float) -> dict:
    return {
        "colloid_mean_Rh_nm": float(center_nm),
        "colloid_sd_Rh_nm": 0.0,
        "colloid_cv": 0.0,
        "colloid_median_Rh_nm": float(center_nm),
        "colloid_mode_Rh_nm": float(center_nm),
        "colloid_p05_Rh_nm": float(center_nm),
        "colloid_p10_Rh_nm": float(center_nm),
        "colloid_p90_Rh_nm": float(center_nm),
        "colloid_p95_Rh_nm": float(center_nm),
        "colloid_span_p10_p90_nm": 0.0,
        "colloid_span_p05_p95_nm": 0.0,
    }


def apparent_rh_from_slope(prep: PreparedTrace, side: str, T_K: float, cfg: Settings) -> float | None:
    t, s, t0 = prep.trace.time_s, prep.normalized_initial, prep.t0_initial_s
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

    lam_R = 2 * kB * T_K / (np.pi * cfg.viscosity_Pa_s * cfg.capillary_radius_m**2 * t0)
    Rh_nm = -lam_R / slope * 1e9
    low, high = cfg.Rh_center_bounds_nm
    return float(Rh_nm) if np.isfinite(Rh_nm) and low <= Rh_nm <= high else None


def initial_colloid_center(prep: PreparedTrace, side: str, T_K: float, cfg: Settings) -> float:
    if cfg.initial_colloid_Rh_nm is not None:
        return float(np.clip(cfg.initial_colloid_Rh_nm, *cfg.Rh_center_bounds_nm))
    guess = apparent_rh_from_slope(prep, side, T_K, cfg) or apparent_rh_from_slope(prep, "both", T_K, cfg) or 25.0
    return float(np.clip(guess, *cfg.Rh_center_bounds_nm))


# =============================================================================
# Output helpers
# =============================================================================

def sample_stem(prep: PreparedTrace) -> str:
    sample = prep.trace.meta.get("Sample name", prep.trace.path.stem)
    return safe_filename(f"{sample}_{prep.trace.path.stem}")


def save_current_figure(path: Path, cfg: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if cfg.show_plots:
        plt.show()
    plt.close()


def save_preprocessing_plots(prep: PreparedTrace, out_dir: Path, cfg: Settings) -> None:
    if not cfg.save_plots:
        return
    stem = sample_stem(prep)
    t, y = prep.trace.time_s, prep.trace.signal

    plt.figure(figsize=(8, 4))
    plt.plot(t[~prep.spike_mask], y[~prep.spike_mask], ".", markersize=2, label="used raw data")
    if np.any(prep.spike_mask):
        plt.plot(t[prep.spike_mask], y[prep.spike_mask], "x", markersize=4, label="excluded spike points")
    plt.plot(t, prep.baseline_initial, "-", label="initial baseline")
    plt.axvline(prep.t0_initial_s, linestyle="--", label="initial t0")
    plt.xlabel("time / s")
    plt.ylabel("signal")
    plt.title(f"{stem}: preprocessing")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / "preprocessing" / f"{stem}_preprocessing.svg", cfg)

    plt.figure(figsize=(8, 4))
    plt.plot(t[prep.good_mask], prep.normalized_initial[prep.good_mask], ".", markersize=2)
    plt.axvline(prep.t0_initial_s, linestyle="--")
    plt.xlabel("time / s")
    plt.ylabel("initial normalized signal")
    plt.title(f"{stem}: normalized trace")
    save_current_figure(out_dir / "preprocessing" / f"{stem}_normalized.svg", cfg)


def write_distribution_csv(result: FitResult, out_dir: Path, stem: str) -> None:
    rows = [
        {"Rh_nm": float(rh), "pdf_per_ln_Rh": float(pdf_log), "pdf_per_nm": float(pdf_R), "cdf": float(cdf)}
        for rh, pdf_log, pdf_R, cdf in zip(result.Rh_nm, result.pdf_logRh, result.pdf_Rh_per_nm, result.cdf)
    ]
    write_rows(out_dir / f"{stem}_{result.model_name}_distribution.csv", rows)


def save_fit_plots(result: FitResult, prep: PreparedTrace, out_dir: Path, cfg: Settings) -> None:
    if not cfg.save_plots:
        return
    stem = sample_stem(prep)

    plt.figure(figsize=(9, 5))
    plt.plot(result.time_fit_s, result.signal_fit_raw, ".", markersize=3, label="fit data")
    plt.plot(result.time_fit_s, result.signal_model_raw, "-", label="fit")
    plt.xlabel("time / s")
    plt.ylabel("raw signal")
    plt.title(f"{stem}: {result.model_name} fit")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_{result.model_name}_fit.svg", cfg)

    plt.figure(figsize=(9, 4))
    plt.plot(result.time_fit_s, result.residuals_raw / prep.peak_height_initial, ".", markersize=3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("time / s")
    plt.ylabel("residual / initial peak height")
    plt.title(f"{stem}: {result.model_name} residuals")
    save_current_figure(out_dir / f"{stem}_{result.model_name}_residuals.svg", cfg)

    plt.figure(figsize=(9, 5))
    plt.plot(result.time_fit_s, result.signal_fit_raw, ".", markersize=3, label="fit data")
    plt.plot(result.time_fit_s, result.signal_model_raw, "-", label="total fit")
    plt.plot(result.time_fit_s, result.baseline_model_raw, "--", label="baseline")
    plt.plot(result.time_fit_s, result.baseline_model_raw + result.free_signal_raw, ":", label="baseline + free fluorophore")
    plt.plot(result.time_fit_s, result.baseline_model_raw + result.colloid_signal_raw, "-.", label="baseline + colloid")
    plt.xlabel("time / s")
    plt.ylabel("raw signal")
    plt.title(f"{stem}: {result.model_name} decomposition")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_{result.model_name}_decomposition.svg", cfg)

    plt.figure(figsize=(7, 5))
    if result.model_name == "dual_gaussian":
        rh = result.stats.get("colloid_mean_Rh_nm", np.nan)
        if np.isfinite(rh):
            plt.axvline(rh, linestyle="--", label=f"colloid {rh:.2f} nm")
    else:
        plt.semilogx(result.Rh_nm, result.pdf_logRh, "-", label=result.model_name)
    plt.axvline(cfg.free_Rh_nm, linestyle=":", label=f"free fluorophore {cfg.free_Rh_nm:g} nm")
    plt.xlabel(r"$R_h$ / nm")
    plt.ylabel(r"colloid PDF per $\ln R_h$")
    plt.title(f"{stem}: colloid distribution")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / f"{stem}_{result.model_name}_distribution.svg", cfg)


def save_crli_alpha_plots(result: FitResult, prep: PreparedTrace, out_dir: Path, cfg: Settings) -> None:
    if not cfg.save_plots or result.model_name != "crli" or not result.extra:
        return
    stem = sample_stem(prep)
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
    save_current_figure(out_dir / f"{stem}_crli_alpha_residual_scan.svg", cfg)

    plt.figure(figsize=(6, 4))
    plt.loglog(roughness, residual, ".-")
    plt.xlabel("roughness")
    plt.ylabel("normalized residual RMS")
    plt.title(f"{stem}: CRLI roughness scan")
    save_current_figure(out_dir / f"{stem}_crli_roughness_scan.svg", cfg)


def result_summary_row(prep: PreparedTrace, result: FitResult) -> dict:
    trace = prep.trace
    base = {
        "file": str(trace.path),
        "sample_name": trace.meta.get("Sample name", trace.path.stem),
        "model_name": result.model_name,
        "pressure_mbar": trace.meta.get("pressure_mbar", np.nan),
        "capillary_temp_C": trace.meta.get("capillary_temp_C", np.nan),
        "method_time_s": trace.meta.get("method_time_s", np.nan),
        "time_unit_detected": trace.meta.get("time_unit_detected", ""),
        "spike_count": prep.spike_count,
        "spike_times_s": ";".join(f"{x:.6g}" for x in prep.spike_times_s),
    }
    return {**base, **result.stats}


def save_fit_outputs(result: FitResult, prep: PreparedTrace, out_dir: Path, cfg: Settings) -> None:
    stem = sample_stem(prep)
    write_distribution_csv(result, out_dir, stem)
    save_fit_plots(result, prep, out_dir, cfg)
    save_crli_alpha_plots(result, prep, out_dir, cfg)


def print_result(result: FitResult) -> None:
    s = result.stats
    print(
        f"  {result.model_name:22s} | "
        f"t0={s.get('t0_s', np.nan):.2f} s | "
        f"free={100 * s.get('free_fraction_detector_weighted', np.nan):.2f}% | "
        f"median Rh={s.get('colloid_median_Rh_nm', np.nan):.2f} nm | "
        f"mean Rh={s.get('colloid_mean_Rh_nm', np.nan):.2f} nm | "
        f"RMS/peak={s.get('residual_rms_normalized_to_peak', np.nan):.4g}",
        flush=True,
    )


def run_batch(cfg: Settings, model_name: str, fit_one: Callable[[PreparedTrace, Settings], FitResult]) -> None:
    """Read, process, fit, and save outputs for one fitting script."""
    files = sorted(Path(p) for p in glob(cfg.data_pattern, recursive=True) if Path(p).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {cfg.data_pattern!r}")

    root = Path(cfg.output_dir) / model_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_settings.json").write_text(json.dumps(settings_to_jsonable(cfg), indent=2), encoding="utf-8")

    print(f"Found {len(files)} file(s).")
    print(f"Model: {model_name}")
    print(f"Output folder: {root.resolve()}")

    rows: list[dict] = []
    for path in files:
        print(f"\nAnalysing {path}", flush=True)
        try:
            prep = prepare_trace(read_trace(path, cfg), cfg)
            save_preprocessing_plots(prep, root, cfg)
            result = fit_one(prep, cfg)
            save_fit_outputs(result, prep, root, cfg)
            rows.append(result_summary_row(prep, result))
            print_result(result)
        except Exception as exc:
            warnings.warn(f"Skipped {path.name}: {exc}")
            rows.append({"file": str(path), "model_name": model_name, "success": False, "error": str(exc)})

    summary_path = root / f"{model_name}_summary.csv"
    write_rows(summary_path, rows)
    print(f"\nSaved summary: {summary_path}")
    print("Done.")


# =============================================================================
# Shared command-line helpers
# =============================================================================

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", dest="data_pattern", help="Input glob pattern, e.g. '*.txt'.")
    parser.add_argument("--output", dest="output_dir", help="Output folder.")
    parser.add_argument("--input-format", choices=["fidaneo", "two_column"])
    parser.add_argument("--fidaneo-time-unit", choices=["auto", "s", "min"])
    parser.add_argument("--two-column-time-unit", choices=["s", "min", "ms"])
    parser.add_argument("--two-column-delimiter", help="Delimiter for two-column files; use ',' for CSV.")
    parser.add_argument("--two-column-skiprows", type=int)
    parser.add_argument("--capillary-radius-um", type=float, help="Capillary radius in micrometres.")
    parser.add_argument("--viscosity", type=float, help="Carrier liquid viscosity in Pa s.")
    parser.add_argument("--free-rh-nm", type=float, help="Known free fluorophore Rh in nm.")
    parser.add_argument("--free-fraction-mode", choices=["fit", "fixed"])
    parser.add_argument("--free-fraction-fixed", type=float)
    parser.add_argument("--no-despike", action="store_true", help="Do not exclude narrow spikes before fitting.")
    parser.add_argument("--max-spikes", type=int, help="Skip files with more than this many spike events.")
    parser.add_argument("--spike-threshold", type=float, help="Spike threshold in MAD-scaled residual units.")
    parser.add_argument("--spike-window", type=int, help="Median-filter window, in points, for spike detection.")
    parser.add_argument("--spike-padding", type=int, help="Extra points removed on each side of a spike.")
    parser.add_argument("--spike-min-prominence-fraction", type=float, help="Minimum spike prominence as a fraction of the 1st-99th percentile signal range.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV output but skip plots.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots while saving them.")


def apply_common_args(cfg: Settings, args: argparse.Namespace) -> Settings:
    for name in [
        "data_pattern",
        "output_dir",
        "input_format",
        "fidaneo_time_unit",
        "two_column_time_unit",
        "two_column_delimiter",
        "two_column_skiprows",
        "free_fraction_mode",
        "free_fraction_fixed",
    ]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)

    if getattr(args, "capillary_radius_um", None) is not None:
        cfg.capillary_radius_m = args.capillary_radius_um * 1e-6
    if getattr(args, "viscosity", None) is not None:
        cfg.viscosity_Pa_s = args.viscosity
    if getattr(args, "free_rh_nm", None) is not None:
        cfg.free_Rh_nm = args.free_rh_nm
    if getattr(args, "no_despike", False):
        cfg.despike = False
    if getattr(args, "max_spikes", None) is not None:
        cfg.max_spikes_allowed = args.max_spikes
    if getattr(args, "spike_threshold", None) is not None:
        cfg.spike_threshold_mad = args.spike_threshold
    if getattr(args, "spike_window", None) is not None:
        cfg.spike_median_window_points = args.spike_window
    if getattr(args, "spike_padding", None) is not None:
        cfg.spike_padding_points = args.spike_padding
    if getattr(args, "spike_min_prominence_fraction", None) is not None:
        cfg.spike_min_prominence_fraction_of_signal_range = args.spike_min_prominence_fraction
    if getattr(args, "no_plots", False):
        cfg.save_plots = False
    if getattr(args, "show_plots", False):
        cfg.show_plots = True
    return cfg
