#!/usr/bin/env python3
"""Global exponential-kinetics fit over a FidaNeo time series.

A series of Taylorgrams of the *same* sample is measured repeatedly over many
hours. The slow change in the large (colloid) species is modeled as an
exponential relaxation to a new plateau, shared across the whole series, while
the dye component shape is a shared constant and the component amplitudes
(and t0) are fit per trace - a VARPRO global fit. Math ported unchanged from
``fit_global_series.py``.

This model fits *many* traces at once, so unlike CRLI/Laplace/parametric it
inherits only :class:`~tda.models.base.FitModel` (physics + preprocessing
composition), not :class:`~tda.models.base.SingleTraceModel` - there is no
single ``fit_file``, only :meth:`load_traces`/:meth:`fit`/:meth:`fit_folder`
over a folder of records.

The per-trace inner solve reuses :mod:`tda.models.parametric`'s *stateless*
``ParamPack``/``design_matrix``/``solve_amplitudes`` directly - never a
``ParametricModel`` instance - because this model's per-trace t0 policy
(:func:`_resolve_trace_t0`) is independent of ``ParametricModel``'s own
t0-nuisance handling; the two never should be conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from tda.fitting import select_fit_mask, t0_seed, t0_fit_enabled, refine_t0
from tda.io import fidaneo
from tda.numerics import trapezoid_weights
from tda.physics import PhysicsParams, apparent_rh_from_slope, rh_kernel
from tda.preprocessing.pipeline import PreparedTrace
from tda.models.base import FitModel
from tda.models.parametric import ParamPack, design_matrix, solve_amplitudes

N_GROUP_PARAMS = 6   # ln_Rh0, ln_Rhinf, ln_tau, ln_colloid_sigma, ln_dye_mu, ln_dye_sigma


@dataclass
class GlobalSeriesParams:
    group_mode: str = "pooled"                # "pooled" (one curve) or "per_sample"
    fit_side: str = "rising"
    peak_fraction: float = 0.50
    fit_stride: int = 1
    signal_min: float = -10.00
    signal_max: float = 1.02

    rh_grid: tuple[float, float, int] = (0.2, 300.0, 300)   # dense grid for the model integral

    # Shared shape initial guesses / bounds.
    dye_rh_nm: float = 0.75
    dye_sigma_ln: float = 0.10
    colloid_sigma_ln: float = 0.40
    rh_bounds_nm: tuple[float, float] = (1.0, 200.0)          # bounds on colloid centers Rh_0 / Rh_plateau
    dye_rh_bounds_nm: tuple[float, float] = (0.3, 3.0)
    sigma_ln_bounds: tuple[float, float] = (0.002, 1.5)
    tau_bounds_h: tuple[float, float] = (0.05, 200.0)
    rh_center_bounds_nm: tuple[float, float] = (1.5, 300.0)    # apparent-Rh clamp (see tda.physics.apparent_rh_from_slope)

    t0_mode: str = "fit"
    t0_manual_s: float | None = None
    t0_refine_bound_s: float = 5.0


@dataclass
class GroupParams:
    Rh0_nm: float
    Rhinf_nm: float
    tau_h: float
    colloid_sigma: float
    dye_mu_nm: float
    dye_sigma: float

    def colloid_rh(self, time_h: np.ndarray | float) -> np.ndarray | float:
        return self.Rhinf_nm + (self.Rh0_nm - self.Rhinf_nm) * np.exp(-np.asarray(time_h) / self.tau_h)


def _group_vector(p: GroupParams) -> np.ndarray:
    return np.log([p.Rh0_nm, p.Rhinf_nm, p.tau_h, p.colloid_sigma, p.dye_mu_nm, p.dye_sigma])


def _group_from_vector(v: np.ndarray) -> GroupParams:
    Rh0, Rhinf, tau, csig, dmu, dsig = np.exp(v)
    return GroupParams(Rh0, Rhinf, tau, csig, dmu, dsig)


def _group_bounds(p: GlobalSeriesParams) -> tuple[np.ndarray, np.ndarray]:
    lo = np.log([p.rh_bounds_nm[0], p.rh_bounds_nm[0], p.tau_bounds_h[0], p.sigma_ln_bounds[0],
                 p.dye_rh_bounds_nm[0], p.sigma_ln_bounds[0]])
    hi = np.log([p.rh_bounds_nm[1], p.rh_bounds_nm[1], p.tau_bounds_h[1], p.sigma_ln_bounds[1],
                 p.dye_rh_bounds_nm[1], p.sigma_ln_bounds[1]])
    return lo, hi


@dataclass
class Trace:
    record: "fidaneo.FidaNeoRecord"
    prep: PreparedTrace
    time_h: float
    group: str
    t_fit: np.ndarray
    s_fit: np.ndarray
    apparent_rh_nm: float | None
    time0_s: float                       # kernel t0 used for this trace


@dataclass
class GlobalSeriesResult:
    names: list[str]
    groups: dict[str, list[Trace]]
    fitted: dict[str, GroupParams]
    seeds: dict[str, GroupParams]
    Rh: np.ndarray
    wx: np.ndarray
    T_K: float
    per_trace: list[dict]
    success: bool
    cost: float
    nfev: int
    status: int
    x0: np.ndarray
    x: np.ndarray


def _resolve_trace_t0(prep: PreparedTrace, t_fit: np.ndarray, s_fit: np.ndarray, apparent_rh: float | None,
                       T_K: float, dye_rh_nm: float, t0_mode: str, t0_manual_s: float | None,
                       t0_refine_bound_s: float, physics: PhysicsParams) -> float:
    """Per-trace kernel t0: locked to the seed (auto/manual), or refined once
    against a quick two-point (dye + colloid) NNLS misfit at load time, then
    held fixed during the global fit (t0 is a local nuisance, not shared)."""
    seed = t0_seed(prep, t0_mode, t0_manual_s)
    if not t0_fit_enabled(t0_mode):
        return seed
    rh_col = apparent_rh if (apparent_rh and apparent_rh > 0) else 20.0
    sizes = np.array([dye_rh_nm, rh_col])

    def objective(t0: float) -> float:
        kernel = rh_kernel(t_fit, sizes, t0, T_K, physics)
        a = solve_amplitudes(kernel, s_fit)
        return float(np.sqrt(np.mean((kernel @ a - s_fit) ** 2)))

    return refine_t0(objective, seed, t0_mode, t0_refine_bound_s)


def solve_trace(gp: GroupParams, tr: Trace, Rh: np.ndarray, wx: np.ndarray, T_K: float, physics: PhysicsParams) -> dict:
    """Per-trace inner solve: build the two columns at this trace's time, NNLS amplitudes."""
    rh_col = float(gp.colloid_rh(tr.time_h))
    pack = ParamPack(
        ln_mu=np.log([gp.dye_mu_nm, rh_col]),
        ln_sigma=np.log([gp.dye_sigma, gp.colloid_sigma]),
        t0_s=tr.time0_s,
    )
    C = design_matrix(pack, tr.t_fit, Rh, wx, T_K, physics)
    a = solve_amplitudes(C, tr.s_fit)
    s_model = C @ a
    return {"rh_col": rh_col, "a": a, "s_model": s_model, "resid": s_model - tr.s_fit, "pack": pack, "C": C}


def independent_colloid_rh(gp: GroupParams, tr: Trace, Rh: np.ndarray, wx: np.ndarray, T_K: float,
                            rh_bounds_nm: tuple[float, float], physics: PhysicsParams) -> float:
    """Colloid centre fit to this trace ALONE - holds the shared shapes fixed
    at the global values and frees only the colloid Rh for this one trace."""
    def objective(ln_rc: float) -> float:
        pack = ParamPack(
            ln_mu=np.array([np.log(gp.dye_mu_nm), ln_rc]),
            ln_sigma=np.log([gp.dye_sigma, gp.colloid_sigma]),
            t0_s=tr.time0_s,
        )
        C = design_matrix(pack, tr.t_fit, Rh, wx, T_K, physics)
        a = solve_amplitudes(C, tr.s_fit)
        return float(np.mean((C @ a - tr.s_fit) ** 2))

    res = minimize_scalar(objective, bounds=(np.log(rh_bounds_nm[0]), np.log(rh_bounds_nm[1])), method="bounded")
    return float(np.exp(res.x))


def trace_residual_at_rh(gp: GroupParams, tr: Trace, rh_col: float, Rh: np.ndarray, wx: np.ndarray,
                          T_K: float, physics: PhysicsParams) -> np.ndarray:
    """Normalised residual (model - data) with the colloid at ``rh_col``."""
    pack = ParamPack(
        ln_mu=np.log([gp.dye_mu_nm, rh_col]),
        ln_sigma=np.log([gp.dye_sigma, gp.colloid_sigma]),
        t0_s=tr.time0_s,
    )
    C = design_matrix(pack, tr.t_fit, Rh, wx, T_K, physics)
    a = solve_amplitudes(C, tr.s_fit)
    return C @ a - tr.s_fit


def make_groups(traces: list[Trace]) -> dict[str, list[Trace]]:
    groups: dict[str, list[Trace]] = {}
    for tr in traces:
        groups.setdefault(tr.group, []).append(tr)
    return dict(sorted(groups.items()))


def seed_group(group_traces: list[Trace], p: GlobalSeriesParams) -> GroupParams:
    """Initial guess for a group from quick apparent-Rh estimates vs time."""
    times = np.array([t.time_h for t in group_traces])
    sizes = np.array([t.apparent_rh_nm if t.apparent_rh_nm else np.nan for t in group_traces])
    order = np.argsort(times)
    sizes = sizes[order]
    valid = sizes[np.isfinite(sizes)]
    if valid.size >= 2:
        n = max(1, valid.size // 3)
        Rh0 = float(np.clip(np.median(valid[:n]), *p.rh_bounds_nm))
        Rhinf = float(np.clip(np.median(valid[-n:]), *p.rh_bounds_nm))
    else:
        Rh0 = Rhinf = 20.0
    span = float(np.max(times) - np.min(times)) or 1.0
    tau = float(np.clip(span / 3.0, *p.tau_bounds_h))
    return GroupParams(Rh0, Rhinf, tau, np.exp(p.colloid_sigma_ln), p.dye_rh_nm, np.exp(p.dye_sigma_ln))


class GlobalSeriesModel(FitModel):
    def __init__(self, params: GlobalSeriesParams | None = None, physics: PhysicsParams | None = None,
                 preprocessing=None):
        super().__init__(physics=physics, preprocessing=preprocessing)
        self.params = params if params is not None else GlobalSeriesParams()

    def load_traces(self, data_dir: str | Path, pattern: str = "*.txt") -> tuple[list[Trace], dict]:
        """Parse the folder, keep the first trace's method, and prepare every trace."""
        p = self.params
        records = fidaneo.load_folder(data_dir, pattern)
        if not records:
            raise FileNotFoundError(f"No FidaNeo files under {data_dir!r} matching {pattern!r}")
        records = fidaneo.with_method_of_first(records)
        method = records[0].method
        origin = records[0].measurement_start
        T_K = self.temperature_k

        if p.t0_mode == "manual":
            warnings.warn("t0_mode='manual' applies one t0 to every trace in the series; "
                          "that is rarely right for a kinetic run - prefer 'auto' or 'fit'.")

        traces: list[Trace] = []
        skipped: list[dict] = []
        for rec in records:
            try:
                prep = self.preprocessing.run(rec.time_s, rec.signal, rec.path)
                mask = select_fit_mask(prep, p.fit_side, p.peak_fraction, p.signal_min, p.signal_max)
                idx = np.flatnonzero(mask)[:: max(1, int(p.fit_stride))]
                if idx.size < 25:
                    raise ValueError("too few fit points after windowing")
                time_h = (rec.measurement_start - origin).total_seconds() / 3600.0
                group = "all" if p.group_mode == "pooled" else rec.sample_name
                t_fit, s_fit = prep.time_s[idx], prep.normalized[idx]
                apparent = apparent_rh_from_slope(prep.time_s, prep.normalized, prep.good_mask, prep.t0_s,
                                                   p.fit_side, T_K, p.rh_center_bounds_nm, self.physics)
                t0 = _resolve_trace_t0(prep, t_fit, s_fit, apparent, T_K, p.dye_rh_nm, p.t0_mode, p.t0_manual_s,
                                       p.t0_refine_bound_s, self.physics)
                traces.append(Trace(
                    record=rec, prep=prep, time_h=time_h, group=group,
                    t_fit=t_fit, s_fit=s_fit, apparent_rh_nm=apparent, time0_s=t0,
                ))
            except Exception as exc:
                warnings.warn(f"Skipped {rec.path.name}: {exc}")
                skipped.append({"file": rec.path.name, "error": str(exc)})

        if not traces:
            raise RuntimeError("No traces survived preprocessing.")
        info = {"method": method, "origin": origin.isoformat(), "n_loaded": len(records),
                "n_used": len(traces), "n_skipped": len(skipped), "skipped": skipped}
        return traces, info

    def fit(self, traces: list[Trace]) -> GlobalSeriesResult:
        p = self.params
        T_K = self.temperature_k
        rh_min, rh_max, n_grid = p.rh_grid
        Rh = np.geomspace(rh_min, rh_max, int(n_grid))
        wx = trapezoid_weights(np.log(Rh))

        groups = make_groups(traces)
        names = list(groups.keys())
        seeds = {g: seed_group(groups[g], p) for g in names}
        x0 = np.concatenate([_group_vector(seeds[g]) for g in names])
        lo_g, hi_g = _group_bounds(p)
        lo = np.tile(lo_g, len(names))
        hi = np.tile(hi_g, len(names))
        x0 = np.clip(x0, lo, hi)

        def slice_for(i: int) -> slice:
            return slice(i * N_GROUP_PARAMS, (i + 1) * N_GROUP_PARAMS)

        def residual(x: np.ndarray) -> np.ndarray:
            out = []
            for i, g in enumerate(names):
                gp = _group_from_vector(x[slice_for(i)])
                for tr in groups[g]:
                    out.append(solve_trace(gp, tr, Rh, wx, T_K, self.physics)["resid"])
            return np.concatenate(out)

        sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", x_scale="jac")

        fitted = {g: _group_from_vector(sol.x[slice_for(i)]) for i, g in enumerate(names)}
        per_trace = []
        for g in names:
            for tr in groups[g]:
                r = solve_trace(fitted[g], tr, Rh, wx, T_K, self.physics)
                a = r["a"]
                total = float(np.sum(a)) or 1.0
                resid_raw = tr.prep.peak_height * r["resid"]
                per_trace.append({
                    "file": tr.record.path.name,
                    "sample": tr.record.sample_name,
                    "group": g,
                    "method": tr.record.method,
                    "measurement_start": tr.record.measurement_start.isoformat(),
                    "time_h": tr.time_h,
                    "t0_s": tr.time0_s,
                    "t0_detected_s": tr.prep.t0_s,
                    "t0_shift_s": tr.time0_s - tr.prep.t0_s,
                    "apparent_rh_nm": tr.apparent_rh_nm,
                    "indep_colloid_rh_nm": independent_colloid_rh(fitted[g], tr, Rh, wx, T_K, p.rh_bounds_nm, self.physics),
                    "model_colloid_rh_nm": r["rh_col"],
                    "dye_fraction": float(a[0] / total),
                    "colloid_fraction": float(a[1] / total),
                    "dye_amplitude": float(a[0]),
                    "colloid_amplitude": float(a[1]),
                    "n_fit_points": int(tr.s_fit.size),
                    "residual_rms_norm": float(np.sqrt(np.mean(r["resid"] ** 2))),
                    "residual_rms_raw": float(np.sqrt(np.mean(resid_raw ** 2))),
                    "residual_rms_over_noise": float(np.sqrt(np.mean(resid_raw ** 2)) / tr.prep.noise_raw)
                    if tr.prep.noise_raw > 0 else np.nan,
                })

        return GlobalSeriesResult(
            names=names, groups=groups, fitted=fitted, seeds=seeds,
            Rh=Rh, wx=wx, T_K=T_K, per_trace=per_trace,
            success=bool(sol.success), cost=float(sol.cost), nfev=int(sol.nfev),
            status=int(sol.status), x0=x0, x=sol.x,
        )

    def fit_folder(self, data_dir: str | Path, pattern: str = "*.txt") -> tuple[GlobalSeriesResult, dict]:
        traces, info = self.load_traces(data_dir, pattern)
        return self.fit(traces), info
