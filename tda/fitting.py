#!/usr/bin/env python3
"""Fit-window selection and t0 policy, shared by every single-trace model.

Originally lived only inside ``fit_crli_simple.py`` (as ``crli_fit_mask``,
``t0_seed``, ``t0_fit_enabled``, ``refine_t0``) and every other script reused
it by reaching into that module's namespace (``crli.crli_fit_mask``, ...) even
though none of this is CRLI-specific - Laplace, parametric and global-series
all select fit points and resolve t0 the same way. Moved here so every model
imports it directly instead of through CRLI's module.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from tda.preprocessing.pipeline import PreparedTrace


def select_fit_mask(prep: PreparedTrace, side: str, peak_fraction: float,
                     signal_min: float, signal_max: float) -> np.ndarray:
    """Select fit points. For the rising side, ``peak_fraction`` sets how much
    of the peak is included: 0.50 keeps the rising half (up to t0); larger
    values extend the upper time cutoff into the falling side, with 1.00
    reaching the end of the peak."""
    t, s, t0 = prep.time_s, prep.normalized, prep.t0_s
    mask = prep.good_mask & np.isfinite(s) & (s >= signal_min) & (s <= signal_max)
    if side == "rising":
        if not 0.5 <= peak_fraction <= 1.0:
            raise ValueError("peak_fraction must be in [0.5, 1.0].")
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


def t0_seed(prep: PreparedTrace, t0_mode: str, t0_manual_s: float | None) -> float:
    """Starting t0: the hand-set value in 'manual' mode, else the detected one."""
    if t0_mode == "manual":
        if t0_manual_s is None:
            raise ValueError("t0_mode='manual' requires t0_manual_s to be set.")
        return float(t0_manual_s)
    if t0_mode not in ("auto", "fit"):
        raise ValueError("t0_mode must be 'auto', 'manual', or 'fit'.")
    return float(prep.t0_s)


def t0_fit_enabled(t0_mode: str) -> bool:
    return t0_mode == "fit"


def refine_t0(objective, seed: float, t0_mode: str, bound: float) -> float:
    """Bounded 1-D refinement of t0 minimizing ``objective(t0)`` (a scalar
    misfit). Returns ``seed`` unchanged unless ``t0_mode == 'fit'``; the
    search is confined to ``seed +/- bound`` so it only ever nudges t0, never
    relocates the peak."""
    if not t0_fit_enabled(t0_mode):
        return float(seed)
    result = minimize_scalar(objective, bounds=(seed - bound, seed + bound), method="bounded",
                              options={"xatol": 1e-3})
    return float(result.x) if result.success else float(seed)
