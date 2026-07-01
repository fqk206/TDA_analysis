#!/usr/bin/env python3
"""Raw-trace conditioning: despiking and time binning.

Two small, pure preprocessing operations applied to a ``(time, signal)`` trace
before background subtraction and fitting.  Both are opt-in and independent, but
they have a required *order*: despike first, bin second.  A narrow artefact spike
must be flagged and excluded before any averaging, otherwise it contaminates the
whole bin it falls in - that ordering dependency is why the two live together.

Neither function reads global config; callers pass the knobs in.  The fitters
(``fit_crli_simple`` and everything that imports it as ``crli``) drive these from
their CONFIG block and apply them inside ``prepare_from_trace``.

    from preprocess import despike_mask, bin_trace

    spikes = despike_mask(y, window=4, n_sigma=18.0)      # bool mask, True at spikes
    good = np.isfinite(t) & np.isfinite(y) & ~spikes
    t_b, y_b, n_b = bin_trace(t, y, width_s=0.5, good_mask=good)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


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


def bin_trace(
    time_s: np.ndarray,
    signal: np.ndarray,
    width_s: float,
    good_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a trace into fixed-width time bins.

    Samples are grouped into consecutive ``width_s``-wide bins spanning the data;
    within each bin the time and signal are count-averaged, so the returned time
    is the mean sample time of the bin (its intensity-neutral centre).  Points
    where ``good_mask`` is False - or where either coordinate is non-finite - are
    excluded from the averages, and empty bins are dropped.  Binning both shrinks
    the point count and lowers per-point noise by roughly ``sqrt(N_bin)``; keep
    ``width_s`` well below the (broad) Taylor peak so the peak shape is preserved.

    Returns ``(time_binned, signal_binned, counts)`` sorted in time, where
    ``counts`` is the number of samples averaged into each surviving bin.
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
