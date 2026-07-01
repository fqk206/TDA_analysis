#!/usr/bin/env python3
"""Taylor-dispersion physics: the Rh kernel and its physical constants.

Moved out of ``fit_crli_simple.py`` so every model (CRLI, Laplace, parametric,
global-series) can import the kernel directly instead of reaching through the
CRLI module's namespace. The math is unchanged from the original
``rh_kernel``/``apparent_rh_from_slope``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann as kB


@dataclass(frozen=True)
class PhysicsParams:
    temperature_c: float = 25.0
    capillary_radius_m: float = 37.5e-6      # radius, not diameter
    viscosity_pa_s: float = 8.90e-4
    kernel_reference_rh_nm: float = 25.0     # numerical scale factor only

    @property
    def temperature_k(self) -> float:
        return 273.15 + self.temperature_c


def rh_kernel(time_s: np.ndarray, Rh_nm: np.ndarray | float, t0_s: float, T_K: float,
              physics: PhysicsParams = PhysicsParams()) -> np.ndarray:
    """Taylor kernel for a hydrodynamic radius or radius grid."""
    time_s = np.asarray(time_s, dtype=float)
    input_was_scalar = np.ndim(Rh_nm) == 0
    Rh_nm = np.atleast_1d(np.asarray(Rh_nm, dtype=float))
    Rh_m = Rh_nm * 1e-9
    Rh_ref_m = physics.kernel_reference_rh_nm * 1e-9
    if t0_s <= 0 or np.any(Rh_m <= 0):
        raise ValueError("t0_s and Rh must be positive.")

    lam_R = 2 * kB * T_K / (np.pi * physics.viscosity_pa_s * physics.capillary_radius_m**2 * t0_s)
    tau2 = (time_s[:, None] - t0_s) ** 2
    kernel = np.sqrt(Rh_ref_m / Rh_m)[None, :] * np.exp(-lam_R * tau2 / Rh_m[None, :])
    return kernel[:, 0] if input_was_scalar else kernel


def apparent_rh_from_slope(time_s: np.ndarray, normalized: np.ndarray, good_mask: np.ndarray,
                            t0_s: float, side: str, T_K: float, bounds_nm: tuple[float, float],
                            physics: PhysicsParams = PhysicsParams()) -> float | None:
    """Quick Taylor-slope size estimate, used to seed/narrow a grid (not a fit).

    Operates on raw arrays (rather than a particular model's prepared-trace
    container) so it stays reusable across models; ``bounds_nm`` is passed in
    explicitly instead of read from a global.
    """
    t, s, t0 = time_s, normalized, t0_s
    mask = good_mask & np.isfinite(s) & (s > 0.08) & (s < 0.90)
    if side == "rising":
        mask &= t < t0
    elif side == "falling":
        mask &= t > t0
    if np.sum(mask) < 10:
        return None

    slope, _ = np.polyfit((t[mask] - t0) ** 2, np.log(s[mask]), deg=1)
    if not np.isfinite(slope) or slope >= 0:
        return None

    lam_R = 2 * kB * T_K / (np.pi * physics.viscosity_pa_s * physics.capillary_radius_m**2 * t0)
    Rh_nm = -lam_R / slope * 1e9
    low, high = bounds_nm
    return float(Rh_nm) if np.isfinite(Rh_nm) and low <= Rh_nm <= high else None
