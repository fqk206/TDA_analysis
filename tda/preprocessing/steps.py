#!/usr/bin/env python3
"""Composable raw-trace conditioning steps: despiking and time binning.

``despike_mask``/``bin_trace`` are ``preprocess.py``'s pure functions, moved
here unchanged. ``Despike``/``Bin`` wrap them as small param-dataclasses behind
a common step contract so a :class:`~tda.preprocessing.pipeline.PreprocessingPipeline`
can compose any ordered list of them - including new denoising/despiking/
binning routines added later - without any fit model needing to change.

Step contract: every step is a callable-like object with

    step.apply(t, y, good) -> (t, y, good)

- A **filter step** (e.g. :class:`Despike`) returns arrays of the same length
  and must AND its own new exclusions onto the incoming ``good`` mask - it
  must not silently drop or un-flag points the previous step already flagged.
- A **resample step** (e.g. :class:`Bin`) changes the array length (it
  resamples onto a new grid) and must return a fresh ``good`` mask sized to
  that new grid, built only from the incoming-``good`` samples.

Both kinds only need to keep ``t``, ``y`` and ``good`` the same length as each
other; the pipeline doesn't otherwise care which kind a step is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import median_filter


def despike_mask(signal: np.ndarray, window: int, n_sigma: float) -> np.ndarray:
    """Flag narrow artefact spikes with a Hampel identifier.

    Each sample is compared with a rolling-median estimate of where its
    neighbours sit; samples deviating by more than ``n_sigma`` local MADs are
    flagged. ``window`` (in samples) must be wider than the broadest spike yet
    far narrower than the (broad) Taylor peak. Returns a boolean mask, True at
    spike samples (to be ignored entirely, not smoothed).
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


def bin_trace(
    time_s: np.ndarray,
    signal: np.ndarray,
    width_s: float,
    good_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a trace into fixed-width time bins.

    Points where ``good_mask`` is False - or where either coordinate is
    non-finite - are excluded from the averages; empty bins are dropped.
    Returns ``(time_binned, signal_binned, counts)`` sorted in time.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    if not np.isfinite(width_s) or width_s <= 0:
        raise ValueError("bin width_s must be positive.")

    sel = np.isfinite(t) & np.isfinite(y)
    if good_mask is not None:
        sel &= np.asarray(good_mask, dtype=bool)
    t, y = t[sel], y[sel]
    if t.size == 0:
        raise ValueError("No finite points to bin.")

    idx = np.floor((t - t.min()) / width_s).astype(int)
    n_bins = int(idx.max()) + 1
    counts = np.bincount(idx, minlength=n_bins)
    t_sum = np.bincount(idx, weights=t, minlength=n_bins)
    y_sum = np.bincount(idx, weights=y, minlength=n_bins)

    keep = counts > 0
    return t_sum[keep] / counts[keep], y_sum[keep] / counts[keep], counts[keep]


@runtime_checkable
class PreprocessingStep(Protocol):
    def apply(self, t: np.ndarray, y: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@dataclass
class Despike:
    """Filter step: Hampel-flag narrow artefact spikes and exclude them.

    ``window`` in samples must exceed the widest spike but stay well below the
    Taylor peak. ``n_sigma`` trades off catching small spikes vs sparing
    genuine points on a steep, low-noise flank.
    """

    window: int = 4
    n_sigma: float = 18.0

    def apply(self, t: np.ndarray, y: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite = np.isfinite(t) & np.isfinite(y)
        spike = despike_mask(y, self.window, self.n_sigma) & finite
        return t, y, good & ~spike


@dataclass
class Bin:
    """Resample step: average the (already-filtered) trace into fixed-width
    time bins. Cuts the point count and lowers per-point noise by ~sqrt(N_bin);
    keep ``width_s`` well below the (broad) Taylor peak so its shape survives.
    """

    width_s: float = 0.5

    def apply(self, t: np.ndarray, y: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_b, y_b, _ = bin_trace(t, y, self.width_s, good_mask=good)
        return t_b, y_b, np.ones(t_b.shape, dtype=bool)
