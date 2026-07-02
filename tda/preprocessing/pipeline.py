#!/usr/bin/env python3
"""Preprocessing orchestration: ordered opt-in steps + a fixed baseline/t0 tail.

:class:`PreprocessingPipeline` is what every :class:`~tda.models.base.FitModel`
holds (composition, not inheritance) to turn a raw ``(time_s, signal)`` trace
into a :class:`PreparedTrace`. The opt-in, ordered part
(:mod:`tda.preprocessing.steps` - despike, bin, and any future denoising
routine) runs first; background subtraction, noise estimation, t0 detection
and normalization always run afterwards, identically for every model, since
every fit needs them regardless of which conditioning steps ran.

Note: which t0 *mode* a model's fit uses (lock to the detected value, a manual
override, or refine it further) is deliberately NOT handled here - every
:class:`PreparedTrace` carries the one detected ``t0_s`` (quadratic peak fit);
what a model does with it as a fit-time seed is that model's own concern (see
e.g. ``tda.models.crli.CRLIParams.t0_mode``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from tda.numerics import quadratic_peak_time, robust_std
from tda.preprocessing.steps import PreprocessingStep, Despike

# Median-filter window (samples) for locating the coarse peak index before
# quadratic_peak_time() refines it. argmax() alone is one noisy/spiky sample
# away from picking the wrong peak entirely on a low-SNR or under-despiked
# trace (a single sample winning by a hair is enough); a short median filter
# is immune to any one sample the way argmax isn't, regardless of whether
# despiking caught it. Independent of Despike's window/n_sigma - this runs
# whether or not despiking is even in the step list.
PEAK_LOCATE_MEDIAN_WINDOW = 9


@dataclass
class BaselineParams:
    min_window_s: float = 15.0
    max_window_s: float = 90.0
    fraction_of_trace: float = 0.15


@dataclass
class PreparedTrace:
    path: Path | None
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


def choose_baseline_window(time_s: np.ndarray, params: BaselineParams) -> float:
    duration = float(np.max(time_s) - np.min(time_s))
    return max(params.min_window_s, min(params.max_window_s, params.fraction_of_trace * duration))


def linear_baseline(time_s: np.ndarray, signal: np.ndarray, good: np.ndarray, window_s: float) -> np.ndarray:
    start = good & (time_s <= time_s[0] + window_s)
    end = good & (time_s >= time_s[-1] - window_s)
    if np.sum(start) < 5 or np.sum(end) < 5:
        raise ValueError("Not enough good edge points for baseline estimation.")

    t_start, y_start = np.median(time_s[start]), np.median(signal[start])
    t_end, y_end = np.median(time_s[end]), np.median(signal[end])
    return y_start + (y_end - y_start) * (time_s - t_start) / (t_end - t_start)


@dataclass
class PreprocessingPipeline:
    """Ordered, opt-in conditioning steps, then the fixed baseline/t0 tail.

    Default reproduces the historical ``DESPIKE=True, BIN=False`` defaults:
    one :class:`~tda.preprocessing.steps.Despike` step and no binning. To
    enable binning, add a :class:`~tda.preprocessing.steps.Bin` step after
    ``Despike``. To add a new denoising routine, implement the
    ``apply(t, y, good) -> (t, y, good)`` contract (see
    :mod:`tda.preprocessing.steps`) and drop it into ``steps``; no model class
    needs to change.
    """

    steps: tuple[PreprocessingStep, ...] = field(default_factory=lambda: (Despike(),))
    baseline: BaselineParams = field(default_factory=BaselineParams)

    def run(self, time_s: np.ndarray, signal: np.ndarray, path: str | Path | None = None) -> PreparedTrace:
        path = Path(path) if path is not None else None
        t = np.asarray(time_s, dtype=float)
        y = np.asarray(signal, dtype=float)
        good = np.isfinite(t) & np.isfinite(y)

        for step in self.steps:
            t, y, good = step.apply(t, y, good)

        # Points a same-length filter step excluded (finite but not good) are
        # the "spikes" shown on the preprocessing plot; once a resample step
        # has run the trace is finite and good=True everywhere by
        # construction, so this collapses to all-False - matching the
        # original "clean after binning" behaviour without special-casing it.
        spike_mask = np.isfinite(t) & np.isfinite(y) & ~good

        if np.sum(good) < 20:
            raise ValueError("Too few finite points in the trace.")

        window_s = choose_baseline_window(t, self.baseline)
        baseline = linear_baseline(t, y, good, window_s)
        corrected = y - baseline

        edge = good & ((t <= t[0] + window_s) | (t >= t[-1] - window_s))
        noise = robust_std(corrected[edge])

        t_good, y_good = t[good], corrected[good]
        coarse = median_filter(y_good, size=PEAK_LOCATE_MEDIAN_WINDOW, mode="nearest")
        local_peak_i = int(np.argmax(coarse))
        t0 = quadratic_peak_time(t_good, y_good, local_peak_i)
        peak_height = float(np.interp(t0, t_good, y_good))
        if not np.isfinite(peak_height) or peak_height <= 0:
            raise ValueError("Peak height after baseline subtraction is not positive.")

        return PreparedTrace(
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
            spike_mask=spike_mask,
        )
