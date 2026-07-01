#!/usr/bin/env python3
"""Shared fit-result container for every single-trace model.

Replaces three previously-incompatible shapes: ``fit_crli_simple.FitResult``
(a dataclass with fixed ``free_signal_raw``/``colloid_signal_raw`` fields),
and ``fit_laplace.invert``/``fit_parametric.fit_parametric``'s plain dicts
(different key sets each). The per-model decomposition (CRLI's free/colloid,
Laplace's low/high, parametric's per-component) becomes the open
``components`` dict so shared plotting/CSV code (``tda.reporting``) is
written once instead of three times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    time_fit_s: np.ndarray
    signal_fit_raw: np.ndarray                # y, raw units
    signal_fit_normalized: np.ndarray          # s = corrected / peak_height
    signal_model_raw: np.ndarray               # baseline + peak_height * s_model
    signal_model_normalized: np.ndarray        # s_model
    baseline_model_raw: np.ndarray             # baseline interpolated onto the fit window
    residuals_raw: np.ndarray                  # signal_model_raw - signal_fit_raw
    peak_height: float

    Rh_nm: np.ndarray
    pdf_logRh: np.ndarray
    pdf_Rh_per_nm: np.ndarray
    cdf: np.ndarray

    components: dict[str, np.ndarray] = field(default_factory=dict)   # name -> raw signal contribution
    params: dict = field(default_factory=dict)                        # fitted nuisance/shape params
    stats: dict = field(default_factory=dict)                         # scalar summary, one row of a summary CSV
    extra: dict = field(default_factory=dict)                         # model-specific (scans, design matrices, ...)
