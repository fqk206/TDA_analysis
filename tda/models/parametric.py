#!/usr/bin/env python3
"""Parametric (log-normal mixture) fit of a Taylorgram.

Where :class:`~tda.models.laplace.LaplaceModel` solves for a free, smooth size
distribution, this model fits the same physics with a small, fixed number of
log-normal components via variable projection (VARPRO): for fixed component
shapes the model is linear in the amplitudes, so those are solved by NNLS at
every step and the nonlinear optimiser only handles the shape parameters (+
optional t0 nuisance). Math ported unchanged from ``fit_parametric.py``.

``ParamPack``, ``design_matrix`` and ``solve_amplitudes`` are kept as
standalone, stateless functions/value type (not methods on
:class:`ParametricModel`) - :mod:`tda.models.global_series` reuses exactly
this machinery for its per-trace inner solve, with its own t0 policy, and
never instantiates a :class:`ParametricModel` to do so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, nnls

from tda.fitting import select_fit_mask
from tda.numerics import trapezoid_weights, distribution_stats
from tda.physics import PhysicsParams, rh_kernel
from tda.preprocessing.pipeline import PreparedTrace
from tda.models.base import SingleTraceModel
from tda.models.result import FitResult


@dataclass
class ParametricParams:
    n_components: int = 2
    init_rh_nm: tuple[float, ...] = (0.75, 23.0)          # initial centers (one per component)
    init_sigma_ln: tuple[float, ...] = (0.05, 0.25)        # initial log-widths
    rh_bounds_nm: tuple[float, float] = (0.3, 200.0)        # per-center bounds
    sigma_ln_bounds: tuple[float, float] = (0.02, 1.5)       # per-width bounds
    fit_t0: bool = True                                       # refine t0 as a nuisance parameter
    t0_bound_s: float = 10.0                                   # t0 bounded to prep.t0 +/- this (seconds)
    rh_grid: tuple[float, float, int] = (0.2, 300.0, 400)       # (min_nm, max_nm, n) dense grid for the model integral

    fit_side: str = "rising"
    peak_fraction: float = 0.50
    fit_stride: int = 1
    signal_min: float = -10.00
    signal_max: float = 1.02


# =============================================================================
# Parameter pack/unpack - a shared value type, not private to ParametricModel
# (tda.models.global_series builds/consumes ParamPack directly).
# =============================================================================

@dataclass
class ParamPack:
    ln_mu: np.ndarray        # (K,) log centers, ln(Rh / nm)
    ln_sigma: np.ndarray     # (K,) log log-widths, ln(sigma)
    t0_s: float              # per-trace (local) nuisance

    @property
    def Rh_nm(self) -> np.ndarray:
        return np.exp(self.ln_mu)

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.ln_sigma)


def pack_to_vector(pack: ParamPack, fit_t0: bool) -> np.ndarray:
    """Flatten shape params (and optionally t0) into the optimiser vector."""
    parts = np.empty(2 * pack.ln_mu.size)
    parts[0::2] = pack.ln_mu
    parts[1::2] = pack.ln_sigma
    return np.r_[parts, pack.t0_s] if fit_t0 else parts


def vector_to_pack(x: np.ndarray, k: int, fit_t0: bool, t0_fixed: float) -> ParamPack:
    shape = x[: 2 * k]
    t0 = float(x[2 * k]) if fit_t0 else t0_fixed
    return ParamPack(ln_mu=shape[0::2].copy(), ln_sigma=shape[1::2].copy(), t0_s=t0)


def parameter_bounds(t0_fixed: float, fit_t0: bool, k: int, rh_bounds_nm: tuple[float, float],
                      sigma_ln_bounds: tuple[float, float], t0_bound_s: float) -> tuple[np.ndarray, np.ndarray]:
    lo_rh, hi_rh = np.log(rh_bounds_nm[0]), np.log(rh_bounds_nm[1])
    lo_sg, hi_sg = np.log(sigma_ln_bounds[0]), np.log(sigma_ln_bounds[1])
    lo = np.tile([lo_rh, lo_sg], k)
    hi = np.tile([hi_rh, hi_sg], k)
    if fit_t0:
        lo = np.r_[lo, t0_fixed - t0_bound_s]
        hi = np.r_[hi, t0_fixed + t0_bound_s]
    return lo, hi


# =============================================================================
# Forward model and residual (pure functions - reused verbatim by GlobalSeriesModel)
# =============================================================================

def lognormal_pdf_logRh(Rh_nm: np.ndarray, mu_nm: float, sigma: float) -> np.ndarray:
    """Log-normal in Rh, expressed as a density per ln(Rh), unit area in ln(Rh)."""
    z = (np.log(Rh_nm) - np.log(mu_nm)) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def design_matrix(pack: ParamPack, t: np.ndarray, Rh: np.ndarray, wx: np.ndarray, T_K: float,
                   physics: PhysicsParams = PhysicsParams()) -> np.ndarray:
    """Per-component normalised-signal columns C[:,k] = integral g_k(Rh) K(t,Rh) dlnRh.

    The model is ``C @ a`` and is linear in the amplitudes ``a`` (VARPRO).
    """
    kernel = rh_kernel(t, Rh, pack.t0_s, T_K, physics)         # (n_t, n_grid)
    cols = [kernel @ (lognormal_pdf_logRh(Rh, mu, sg) * wx)
            for mu, sg in zip(pack.Rh_nm, pack.sigma)]
    return np.column_stack(cols)


def solve_amplitudes(C: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Non-negative amplitudes for fixed shapes (the linear VARPRO sub-problem)."""
    a, _ = nnls(C, s, maxiter=20000)
    return a


def forward_signal(pack: ParamPack, a: np.ndarray, t: np.ndarray, Rh: np.ndarray,
                    wx: np.ndarray, T_K: float, physics: PhysicsParams = PhysicsParams()) -> np.ndarray:
    """Pure forward model: params -> normalised signal(t)."""
    return design_matrix(pack, t, Rh, wx, T_K, physics) @ a


def residual_vector(x: np.ndarray, t: np.ndarray, s: np.ndarray, Rh: np.ndarray, wx: np.ndarray,
                     T_K: float, k: int, fit_t0: bool, t0_fixed: float,
                     physics: PhysicsParams = PhysicsParams()) -> np.ndarray:
    """Residual (model - data) on the normalised signal, amplitudes via NNLS inside."""
    pack = vector_to_pack(x, k, fit_t0, t0_fixed)
    C = design_matrix(pack, t, Rh, wx, T_K, physics)
    a = solve_amplitudes(C, s)
    return C @ a - s


# =============================================================================
# Model
# =============================================================================

class ParametricModel(SingleTraceModel):
    def __init__(self, params: ParametricParams | None = None, physics: PhysicsParams | None = None,
                 preprocessing=None, reader=None):
        super().__init__(physics=physics, preprocessing=preprocessing, reader=reader)
        self.params = params if params is not None else ParametricParams()

    def fit(self, prepared: PreparedTrace, side: str | None = None) -> FitResult:
        p = self.params
        side = side if side is not None else p.fit_side
        T_K = self.temperature_k
        t0_fixed = prepared.t0_s
        k = int(p.n_components)
        if len(p.init_rh_nm) != k or len(p.init_sigma_ln) != k:
            raise ValueError("init_rh_nm and init_sigma_ln must each have n_components entries.")

        indices = np.flatnonzero(select_fit_mask(prepared, side, p.peak_fraction, p.signal_min, p.signal_max))
        indices = indices[:: max(1, int(p.fit_stride))]
        if indices.size < 25:
            raise ValueError("Too few fitting points for the parametric fit.")
        t = prepared.time_s[indices]
        y = prepared.signal[indices]
        s = prepared.normalized[indices]

        rh_min, rh_max, n_grid = p.rh_grid
        Rh = np.geomspace(rh_min, rh_max, int(n_grid))
        wx = trapezoid_weights(np.log(Rh))

        init = ParamPack(ln_mu=np.log(p.init_rh_nm), ln_sigma=np.log(p.init_sigma_ln), t0_s=t0_fixed)
        x0 = pack_to_vector(init, p.fit_t0)
        lo, hi = parameter_bounds(t0_fixed, p.fit_t0, k, p.rh_bounds_nm, p.sigma_ln_bounds, p.t0_bound_s)
        x0 = np.clip(x0, lo, hi)

        args = (t, s, Rh, wx, T_K, k, p.fit_t0, t0_fixed, self.physics)
        sol = least_squares(residual_vector, x0, bounds=(lo, hi), args=args, method="trf", x_scale="jac")

        pack = vector_to_pack(sol.x, k, p.fit_t0, t0_fixed)
        C = design_matrix(pack, t, Rh, wx, T_K, self.physics)
        a = solve_amplitudes(C, s)
        s_model = C @ a
        comp_signal = a[None, :] * C                          # (n_t, K) per-component normalised signal

        # Sort components by center so comp0 is always the smallest (stable reporting).
        order = np.argsort(pack.Rh_nm)
        pack = ParamPack(pack.ln_mu[order], pack.ln_sigma[order], pack.t0_s)
        a = a[order]
        comp_signal = comp_signal[:, order]

        total_amp = float(np.sum(a))
        if total_amp <= 0:
            raise ValueError("Parametric fit recovered zero amplitude.")
        fractions = a / total_amp

        component_density = np.column_stack([
            a_k * lognormal_pdf_logRh(Rh, mu, sg) for a_k, mu, sg in zip(a, pack.Rh_nm, pack.sigma)
        ])
        dens = component_density.sum(axis=1)
        area = float(np.sum(dens * wx))
        pdf_log = dens / area
        pdf_R = pdf_log / Rh
        cdf = np.cumsum(pdf_log * wx)
        cdf = cdf / cdf[-1]

        baseline = np.interp(t, prepared.time_s, prepared.baseline)
        model_raw = baseline + prepared.peak_height * s_model
        residual_norm = s_model - s
        residual_raw = model_raw - y
        rms_norm = float(np.sqrt(np.mean(residual_norm ** 2)))
        rms_raw = float(np.sqrt(np.mean(residual_raw ** 2)))
        noise = prepared.noise_raw

        stats = {
            "success": bool(sol.success),
            "fit_side": side,
            "peak_fraction": float(p.peak_fraction),
            "n_components": k,
            "fit_t0": bool(p.fit_t0),
            "t0_s": float(pack.t0_s),
            "t0_preprocessing_s": float(t0_fixed),
            "t0_shift_s": float(pack.t0_s - t0_fixed),
            "noise_raw": float(noise),
            "n_fit_points": int(t.size),
            "residual_rms_normalized_to_peak": rms_norm,
            "residual_rms_raw": rms_raw,
            "residual_rms_over_noise": rms_raw / noise if noise > 0 else np.nan,
            "optimizer_cost": float(sol.cost),
            "optimizer_nfev": int(sol.nfev),
            "optimizer_status": int(sol.status),
        }
        for i in range(k):
            stats[f"comp{i}_Rh_nm"] = float(pack.Rh_nm[i])
            stats[f"comp{i}_sigma_ln"] = float(pack.sigma[i])
            stats[f"comp{i}_fraction"] = float(fractions[i])
            stats[f"comp{i}_amplitude"] = float(a[i])
        # Shared distribution stats on the combined parametric distribution (comparable
        # to CRLI / Laplace; "colloid_*" keys are reused verbatim for a direct line-up).
        stats.update(distribution_stats(Rh, pdf_log))

        return FitResult(
            time_fit_s=t,
            signal_fit_raw=y,
            signal_fit_normalized=s,
            signal_model_raw=model_raw,
            signal_model_normalized=s_model,
            baseline_model_raw=baseline,
            residuals_raw=residual_raw,
            peak_height=float(prepared.peak_height),
            Rh_nm=Rh,
            pdf_logRh=pdf_log,
            pdf_Rh_per_nm=pdf_R,
            cdf=cdf,
            components={f"comp{i}": prepared.peak_height * comp_signal[:, i] for i in range(k)},
            params={"t0_s": float(pack.t0_s), "side": side, "fractions": fractions.tolist(),
                    "Rh_nm": pack.Rh_nm.tolist(), "sigma": pack.sigma.tolist()},
            stats=stats,
            extra={
                "pack": pack, "a": a, "wx": wx, "T_K": T_K, "x0": x0, "lo": lo, "hi": hi,
                "component_density": component_density, "area": area, "comp_signal": comp_signal,
                "residual_norm": residual_norm, "optimizer_result": sol, "physics": self.physics,
            },
        )
