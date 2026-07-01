#!/usr/bin/env python3
"""Laplace: pure free-form NNLS inversion of a Taylorgram - no free-fluorophore term.

Where :class:`~tda.models.crli.CRLIModel` models the free fluorophore as a
fixed-Rh component bolted onto a colloid grid, this model does the opposite:
it solves for a single smooth, non-negative size distribution over a grid
that reaches down to very small Rh, letting the inversion recover the small
(free-fluorophore-like) species on its own. Math ported unchanged from
``fit_laplace.invert``/``_invert_at_t0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from tda.fitting import select_fit_mask, t0_seed, t0_fit_enabled, refine_t0
from tda.numerics import second_difference_matrix, trapezoid_weights, distribution_stats
from tda.physics import PhysicsParams, rh_kernel
from tda.preprocessing.pipeline import PreparedTrace
from tda.models.base import SingleTraceModel
from tda.models.result import FitResult


@dataclass
class LaplaceParams:
    fit_side: str = "rising"
    # Fraction of the peak fit (rising side): 0.50 = rising half, 1.00 = whole peak.
    peak_fraction: float = 0.51

    # Inversion grid - starts near zero so the small species sits inside the grid
    # (the Taylor kernel is singular at exactly Rh = 0, so not literally zero).
    rh_min_nm: float = 0.2
    rh_max_nm: float = 200.0
    n_rh_grid: int = 160

    # Cutoff used only for reporting/plotting the emergent "small species" component.
    small_cutoff_nm: float = 2.0

    alphas: np.ndarray = field(default_factory=lambda: np.logspace(-8, 8, 80))
    residual_factor: float = 1.10
    fit_stride: int = 1
    signal_min: float = -10.00
    signal_max: float = 1.02

    t0_mode: str = "fit"
    t0_manual_s: float | None = None
    t0_refine_bound_s: float = 5.0


def _invert_at_t0(t, s, Rh, wx, L, t0, T_K, params: LaplaceParams, physics: PhysicsParams) -> dict:
    """Build the kernel at ``t0`` and run the alpha scan; return the chosen solve."""
    A = rh_kernel(t, Rh, t0, T_K, physics) * wx[None, :]
    solutions, residual_rms, roughness = [], [], []
    for alpha in params.alphas:
        A_aug = np.vstack([A, np.sqrt(alpha) * L])
        b_aug = np.r_[s, np.zeros(L.shape[0])]
        q, _ = nnls(A_aug, b_aug, maxiter=20000)
        s_model = A @ q
        solutions.append(q)
        residual_rms.append(float(np.sqrt(np.mean((s_model - s) ** 2))))
        roughness.append(float(np.linalg.norm(L @ q)))
    residual_rms = np.asarray(residual_rms)
    roughness = np.asarray(roughness)
    best_rms = float(np.min(residual_rms))
    acceptable = np.flatnonzero(residual_rms <= params.residual_factor * best_rms)
    chosen = int(acceptable[-1]) if acceptable.size else int(np.argmin(residual_rms))
    return {"A": A, "q": solutions[chosen], "chosen": chosen,
            "residual_rms": residual_rms, "roughness": roughness, "best_rms": best_rms}


class LaplaceModel(SingleTraceModel):
    def __init__(self, params: LaplaceParams | None = None, physics: PhysicsParams | None = None,
                 preprocessing=None, reader=None):
        super().__init__(physics=physics, preprocessing=preprocessing, reader=reader)
        self.params = params if params is not None else LaplaceParams()

    def fit(self, prepared: PreparedTrace, side: str | None = None) -> FitResult:
        p = self.params
        side = side if side is not None else p.fit_side
        T_K = self.temperature_k

        indices = np.flatnonzero(select_fit_mask(prepared, side, p.peak_fraction, p.signal_min, p.signal_max))
        indices = indices[:: max(1, int(p.fit_stride))]
        if indices.size < 25:
            raise ValueError("Too few fitting points for the inversion.")

        t = prepared.time_s[indices]
        y = prepared.signal[indices]
        s = prepared.normalized[indices]

        Rh = np.geomspace(p.rh_min_nm, p.rh_max_nm, p.n_rh_grid)
        wx = trapezoid_weights(np.log(Rh))
        L = second_difference_matrix(p.n_rh_grid)

        # t0 locked (auto/manual) or refined against the inversion residual.
        # The fit window stays pinned to the preprocessing t0; only the kernel moves.
        t0_init = t0_seed(prepared, p.t0_mode, p.t0_manual_s)
        t0 = refine_t0(
            lambda tt: _invert_at_t0(t, s, Rh, wx, L, tt, T_K, p, self.physics)["best_rms"],
            t0_init, p.t0_mode, p.t0_refine_bound_s,
        )
        out = _invert_at_t0(t, s, Rh, wx, L, t0, T_K, p, self.physics)
        A, residual_rms, roughness = out["A"], out["residual_rms"], out["roughness"]
        chosen, best_rms = out["chosen"], out["best_rms"]

        q = out["q"]
        area = float(np.sum(q * wx))
        if area <= 0:
            raise ValueError("Inversion recovered zero distribution area.")
        pdf_log = q / area
        pdf_R = pdf_log / Rh
        cdf = np.cumsum(pdf_log * wx)
        cdf = cdf / cdf[-1]

        s_model = A @ q
        baseline = np.interp(t, prepared.time_s, prepared.baseline)
        model_raw = baseline + prepared.peak_height * s_model

        # Emergent small-species component (mass below the cutoff), for reporting/plots.
        low = Rh < p.small_cutoff_nm
        s_low = A[:, low] @ q[low]
        s_high = A[:, ~low] @ q[~low]
        small_area_fraction = float(np.sum(q[low] * wx[low]) / area)

        stats = {
            "success": True,
            "fit_side": side,
            "peak_fraction": float(p.peak_fraction),
            "t0_s": float(t0),
            "t0_mode": p.t0_mode,
            "t0_seed_s": float(t0_init),
            "t0_fitted": t0_fit_enabled(p.t0_mode),
            "t0_shift_s": float(t0 - t0_init),
            "noise_raw": float(prepared.noise_raw),
            "n_fit_points": int(t.size),
            "rh_min_nm": float(p.rh_min_nm),
            "rh_max_nm": float(p.rh_max_nm),
            "small_cutoff_nm": float(p.small_cutoff_nm),
            "small_area_fraction_below_cutoff": small_area_fraction,
            "residual_rms_normalized_to_peak": float(np.sqrt(np.mean((s_model - s) ** 2))),
            "residual_rms_raw": float(np.sqrt(np.mean((model_raw - y) ** 2))),
            "crli_alpha": float(p.alphas[chosen]),
            "crli_alpha_at_scan_boundary": chosen in {0, len(p.alphas) - 1},
        }
        stats.update({f"all_{k}": v for k, v in distribution_stats(Rh, pdf_log).items()})
        stats.update({f"above_cutoff_{k}": v for k, v in distribution_stats(Rh[~low], pdf_log[~low]).items()})

        return FitResult(
            time_fit_s=t,
            signal_fit_raw=y,
            signal_fit_normalized=s,
            signal_model_raw=model_raw,
            signal_model_normalized=s_model,
            baseline_model_raw=baseline,
            residuals_raw=model_raw - y,
            peak_height=float(prepared.peak_height),
            Rh_nm=Rh,
            pdf_logRh=pdf_log,
            pdf_Rh_per_nm=pdf_R,
            cdf=cdf,
            components={"low": prepared.peak_height * s_low, "high": prepared.peak_height * s_high},
            params={"t0_s": float(t0), "side": side},
            stats=stats,
            extra={
                "alpha_scan": p.alphas,
                "residual_scan": residual_rms,
                "roughness_scan": roughness,
                "s_fit": s,
                "s_model": s_model,
            },
        )
