#!/usr/bin/env python3
"""Global exponential-kinetics fit over a FidaNeo time series.

A series of Taylorgrams of the *same* sample is measured repeatedly over many
hours. We model the slow change in the large (colloid) species as an exponential
relaxation to a new plateau,

    Rh_colloid(t) = Rh_plateau + (Rh_0 - Rh_plateau) * exp(-t / tau),

shared across the whole series (the kinetic parameters Rh_0, Rh_plateau, tau and
the colloid log-width are *global*), while the dye component shape is a shared
constant and the component amplitudes (and t0) are fit *per trace*. This is a
variable-projection (VARPRO) global fit: for fixed shapes every trace's
amplitudes come from NNLS, so the nonlinear optimiser only handles the handful of
shared shape/kinetic parameters.

Infrastructure built on:
  * ``fidaneo``            - parse files, resolve method + measurement time point,
                             and select the traces sharing the first trace's method.
  * ``fit_crli_simple``    - data prep (baseline, peak, noise), Taylor kernel, NNLS,
                             fit-window controls, distribution stats, CSV helper.
  * ``fit_parametric``     - the per-trace log-normal forward model (design matrix,
                             amplitude solve) reused as the inner model here.

Grouping is configurable (``GROUP_MODE``): "pooled" shares one kinetic curve over
all traces (correct when the vials are the same sample), "per_sample" gives each
distinct sample name its own curve (for future series of different samples).

Run:
    python fit_global_series.py
    python fit_global_series.py --data data_series/series_01 --group pooled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import fidaneo
import fit_crli_simple as crli
import fit_parametric as fp


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = "data_series/series_01"
FILE_PATTERN = "*.txt"

GROUP_MODE = "pooled"                 # "pooled" (one curve) or "per_sample"
FIT_SIDE = crli.FIT_SIDE              # Taylorgram side to fit ("rising")
PEAK_FRACTION = crli.PEAK_FRACTION    # part of the peak to fit (see fit_crli_simple)

# Despiking of the injection-spike artefacts is handled centrally in
# fit_crli_simple.prepare_from_trace (DESPIKE / DESPIKE_WINDOW / DESPIKE_SIGMA).

RH_GRID = (0.2, 300.0, 300)           # (min_nm, max_nm, n) dense grid for the model integral

# Shared shape initial guesses / bounds.
DYE_RH_NM = 0.75
DYE_SIGMA_LN = 0.10
COLLOID_SIGMA_LN = 0.40
RH_BOUNDS_NM = (1.0, 200.0)           # bounds on the colloid centers Rh_0 / Rh_plateau
DYE_RH_BOUNDS_NM = (0.3, 3.0)
SIGMA_LN_BOUNDS = (0.002, 1.5)
TAU_BOUNDS_H = (0.05, 200.0)

OUTPUT_DIR = "output/global_series_01"


# =============================================================================
# Per-group parameter pack (the shared/global parameters)
# =============================================================================

N_GROUP_PARAMS = 6   # ln_Rh0, ln_Rhinf, ln_tau, ln_colloid_sigma, ln_dye_mu, ln_dye_sigma


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


def _group_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.log([RH_BOUNDS_NM[0], RH_BOUNDS_NM[0], TAU_BOUNDS_H[0], SIGMA_LN_BOUNDS[0],
                 DYE_RH_BOUNDS_NM[0], SIGMA_LN_BOUNDS[0]])
    hi = np.log([RH_BOUNDS_NM[1], RH_BOUNDS_NM[1], TAU_BOUNDS_H[1], SIGMA_LN_BOUNDS[1],
                 DYE_RH_BOUNDS_NM[1], SIGMA_LN_BOUNDS[1]])
    return lo, hi


# =============================================================================
# Trace container (one prepared Taylorgram + its time point and fit window)
# =============================================================================

@dataclass
class Trace:
    record: fidaneo.FidaNeoRecord
    prep: crli.Prepared
    time_h: float
    group: str
    t_fit: np.ndarray
    s_fit: np.ndarray
    apparent_rh_nm: float | None
    time0_s: float                       # kernel t0 used for this trace (see T0_MODE)


def _resolve_trace_t0(prep, t_fit, s_fit, apparent_rh, T_K) -> float:
    """Per-trace kernel t0: locked to the seed (auto/manual), or refined once.

    In 'fit' mode t0 is refined against a quick two-point (dye + colloid) NNLS
    misfit at this trace's apparent size - cheap, done once at load time, then
    held fixed during the global fit (t0 is a local nuisance, not a shared param).
    """
    seed = crli.t0_seed(prep)
    if not crli.t0_fit_enabled():
        return seed
    rh_col = apparent_rh if (apparent_rh and apparent_rh > 0) else 20.0
    sizes = np.array([DYE_RH_NM, rh_col])

    def objective(t0: float) -> float:
        kernel = crli.rh_kernel(t_fit, sizes, t0, T_K)
        a = fp.solve_amplitudes(kernel, s_fit)
        return float(np.sqrt(np.mean((kernel @ a - s_fit) ** 2)))

    return crli.refine_t0(objective, seed)


def load_traces(data_dir: str, pattern: str, group_mode: str) -> tuple[list[Trace], dict]:
    """Parse the folder, keep the first trace's method, and prepare every trace."""
    records = fidaneo.load_folder(data_dir, pattern)
    if not records:
        raise FileNotFoundError(f"No FidaNeo files under {data_dir!r} matching {pattern!r}")
    records = fidaneo.with_method_of_first(records)
    method = records[0].method
    origin = records[0].measurement_start
    T_K = 273.15 + crli.TEMPERATURE_C

    if crli.T0_MODE == "manual":
        warnings.warn("T0_MODE='manual' applies one t0 to every trace in the series; "
                      "that is rarely right for a kinetic run - prefer 'auto' or 'fit'.")

    traces: list[Trace] = []
    skipped: list[dict] = []
    for rec in records:
        try:
            prep = crli.prepare_from_trace(rec.time_s, rec.signal, rec.path)
            mask = crli.crli_fit_mask(prep, FIT_SIDE, PEAK_FRACTION)
            idx = np.flatnonzero(mask)[:: max(1, int(crli.CRLI_FIT_STRIDE))]
            if idx.size < 25:
                raise ValueError("too few fit points after windowing")
            time_h = (rec.measurement_start - origin).total_seconds() / 3600.0
            group = "all" if group_mode == "pooled" else rec.sample_name
            t_fit, s_fit = prep.time_s[idx], prep.normalized[idx]
            apparent = crli.apparent_rh_from_slope(prep, FIT_SIDE, T_K)
            traces.append(Trace(
                record=rec, prep=prep, time_h=time_h, group=group,
                t_fit=t_fit, s_fit=s_fit, apparent_rh_nm=apparent,
                time0_s=_resolve_trace_t0(prep, t_fit, s_fit, apparent, T_K),
            ))
        except Exception as exc:
            warnings.warn(f"Skipped {rec.path.name}: {exc}")
            skipped.append({"file": rec.path.name, "error": str(exc)})

    if not traces:
        raise RuntimeError("No traces survived preprocessing.")
    info = {"method": method, "origin": origin.isoformat(), "n_loaded": len(records),
            "n_used": len(traces), "n_skipped": len(skipped), "skipped": skipped}
    return traces, info


# =============================================================================
# Forward model per trace (reuses fit_parametric's log-normal mixture)
# =============================================================================

def solve_trace(gp: GroupParams, tr: Trace, Rh: np.ndarray, wx: np.ndarray, T_K: float) -> dict:
    """Per-trace inner solve: build the two columns at this trace's time, NNLS amplitudes."""
    rh_col = float(gp.colloid_rh(tr.time_h))
    pack = fp.ParamPack(
        ln_mu=np.log([gp.dye_mu_nm, rh_col]),
        ln_sigma=np.log([gp.dye_sigma, gp.colloid_sigma]),
        t0_s=tr.time0_s,
    )
    C = fp.design_matrix(pack, tr.t_fit, Rh, wx, T_K)
    a = fp.solve_amplitudes(C, tr.s_fit)
    s_model = C @ a
    return {"rh_col": rh_col, "a": a, "s_model": s_model, "resid": s_model - tr.s_fit, "pack": pack, "C": C}


def independent_colloid_rh(gp: GroupParams, tr: Trace, Rh: np.ndarray, wx: np.ndarray, T_K: float) -> float:
    """Colloid centre fit to this trace ALONE - the proper per-trace check.

    Holds the shared shapes (dye centre/width, colloid width) fixed at the global
    values and frees only the colloid Rh for this one trace (amplitudes via NNLS).
    Unlike apparent_rh, it uses the real dye+colloid model and deconvolves the
    dye, so it is directly comparable to the global kinetic curve: if the global
    line passes through these points, the shared-kinetics constraint is justified.
    """
    from scipy.optimize import minimize_scalar

    def objective(ln_rc: float) -> float:
        pack = fp.ParamPack(
            ln_mu=np.array([np.log(gp.dye_mu_nm), ln_rc]),
            ln_sigma=np.log([gp.dye_sigma, gp.colloid_sigma]),
            t0_s=tr.time0_s,
        )
        C = fp.design_matrix(pack, tr.t_fit, Rh, wx, T_K)
        a = fp.solve_amplitudes(C, tr.s_fit)
        return float(np.mean((C @ a - tr.s_fit) ** 2))

    res = minimize_scalar(objective, bounds=(np.log(RH_BOUNDS_NM[0]), np.log(RH_BOUNDS_NM[1])), method="bounded")
    return float(np.exp(res.x))


def trace_residual_at_rh(gp: GroupParams, tr: Trace, rh_col: float,
                         Rh: np.ndarray, wx: np.ndarray, T_K: float) -> np.ndarray:
    """Normalised residual (model - data) for this trace with the colloid at
    ``rh_col`` (dye/widths from ``gp``, amplitudes via NNLS).  Used for both the
    global residual (rh_col = gp.colloid_rh(t)) and the free residual (rh_col =
    the per-trace independent fit)."""
    pack = fp.ParamPack(
        ln_mu=np.log([gp.dye_mu_nm, rh_col]),
        ln_sigma=np.log([gp.dye_sigma, gp.colloid_sigma]),
        t0_s=tr.time0_s,
    )
    C = fp.design_matrix(pack, tr.t_fit, Rh, wx, T_K)
    a = fp.solve_amplitudes(C, tr.s_fit)
    return C @ a - tr.s_fit


def make_groups(traces: list[Trace]) -> dict[str, list[Trace]]:
    groups: dict[str, list[Trace]] = {}
    for tr in traces:
        groups.setdefault(tr.group, []).append(tr)
    return dict(sorted(groups.items()))


def seed_group(group_traces: list[Trace]) -> GroupParams:
    """Initial guess for a group from quick apparent-Rh estimates vs time."""
    times = np.array([t.time_h for t in group_traces])
    sizes = np.array([t.apparent_rh_nm if t.apparent_rh_nm else np.nan for t in group_traces])
    order = np.argsort(times)
    sizes = sizes[order]
    valid = sizes[np.isfinite(sizes)]
    if valid.size >= 2:
        n = max(1, valid.size // 3)
        Rh0 = float(np.clip(np.median(valid[:n]), *RH_BOUNDS_NM))
        Rhinf = float(np.clip(np.median(valid[-n:]), *RH_BOUNDS_NM))
    else:
        Rh0 = Rhinf = 20.0
    span = float(np.max(times) - np.min(times)) or 1.0
    tau = float(np.clip(span / 3.0, *TAU_BOUNDS_H))
    return GroupParams(Rh0, Rhinf, tau, np.exp(COLLOID_SIGMA_LN), DYE_RH_NM, np.exp(DYE_SIGMA_LN))


# =============================================================================
# Global fit
# =============================================================================

def fit_series(traces: list[Trace]) -> dict:
    T_K = 273.15 + crli.TEMPERATURE_C
    rh_min, rh_max, n_grid = RH_GRID
    Rh = np.geomspace(rh_min, rh_max, int(n_grid))
    wx = crli.trapezoid_weights(np.log(Rh))

    groups = make_groups(traces)
    names = list(groups.keys())
    seeds = {g: seed_group(groups[g]) for g in names}
    x0 = np.concatenate([_group_vector(seeds[g]) for g in names])
    lo_g, hi_g = _group_bounds()
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
                out.append(solve_trace(gp, tr, Rh, wx, T_K)["resid"])
        return np.concatenate(out)

    sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", x_scale="jac")

    fitted = {g: _group_from_vector(sol.x[slice_for(i)]) for i, g in enumerate(names)}
    per_trace = []
    for g in names:
        for tr in groups[g]:
            r = solve_trace(fitted[g], tr, Rh, wx, T_K)
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
                "indep_colloid_rh_nm": independent_colloid_rh(fitted[g], tr, Rh, wx, T_K),
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

    return {
        "names": names, "groups": groups, "fitted": fitted, "seeds": seeds,
        "Rh": Rh, "wx": wx, "T_K": T_K, "per_trace": per_trace,
        "success": bool(sol.success), "cost": float(sol.cost), "nfev": int(sol.nfev),
        "status": int(sol.status), "x0": x0, "x": sol.x,
    }


# =============================================================================
# Outputs
# =============================================================================

def kinetics_rows(res: dict) -> list[dict]:
    rows = []
    for g in res["names"]:
        gp, sd = res["fitted"][g], res["seeds"][g]
        rows.append({
            "group": g,
            "n_traces": len(res["groups"][g]),
            "Rh0_nm": gp.Rh0_nm, "Rh_plateau_nm": gp.Rhinf_nm, "tau_h": gp.tau_h,
            "delta_Rh_nm": gp.Rh0_nm - gp.Rhinf_nm,
            "colloid_sigma_ln": gp.colloid_sigma, "dye_mu_nm": gp.dye_mu_nm, "dye_sigma_ln": gp.dye_sigma,
            "seed_Rh0_nm": sd.Rh0_nm, "seed_Rh_plateau_nm": sd.Rhinf_nm, "seed_tau_h": sd.tau_h,
        })
    return rows


def save_outputs(res: dict, info: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    crli.write_rows(out_dir / "kinetics_summary.csv", kinetics_rows(res))
    crli.write_rows(out_dir / "per_trace.csv", res["per_trace"])
    settings = {k: globals()[k] for k in (
        "DATA_DIR", "FILE_PATTERN", "GROUP_MODE", "FIT_SIDE", "PEAK_FRACTION",
        "RH_GRID", "DYE_RH_NM", "DYE_SIGMA_LN", "COLLOID_SIGMA_LN", "RH_BOUNDS_NM",
        "DYE_RH_BOUNDS_NM", "SIGMA_LN_BOUNDS", "TAU_BOUNDS_H", "OUTPUT_DIR")}
    settings.update({"t0_mode": crli.T0_MODE, "t0_manual_s": crli.T0_MANUAL_S,
                     "t0_refine_bound_s": crli.T0_REFINE_BOUND_S})
    settings.update({"fit_success": res["success"], "fit_cost": res["cost"], "fit_nfev": res["nfev"], **info})
    (out_dir / "run_settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    if crli.SAVE_PLOTS:
        save_preprocessing(res, out_dir)
        plot_kinetics(res, out_dir)
        plot_residuals_free_vs_global(res, out_dir)
        plot_example_fits(res, out_dir)
        plot_trace_diagnostics(res, out_dir)


def save_preprocessing(res: dict, out_dir: Path) -> None:
    """Per-trace preprocessing plots (baseline, t0, flagged spikes) under
    ``<out_dir>/preprocessing/``, same format as the Laplace/CRLI scripts."""
    for g in res["names"]:
        for tr in res["groups"][g]:
            crli.save_preprocessing_plots(tr.prep, out_dir, crli.safe_filename(tr.record.path.stem))


def plot_kinetics(res: dict, out_dir: Path) -> None:
    """The key plot: colloid Rh vs time.

    Scatter = per-trace *independent* colloid fits (the real dye+colloid model fit
    to each trace alone); the global line should pass through these if the
    shared-kinetics constraint holds.  The y-axis is clamped to the data range.
    """
    plt.figure(figsize=(9, 5))
    colors = {g: f"C{i}" for i, g in enumerate(res["names"])}
    rows = res["per_trace"]
    yvals: list[np.ndarray] = []
    for g in res["names"]:
        gr = [r for r in rows if r["group"] == g]
        t = np.array([r["time_h"] for r in gr])
        indep = np.array([r["indep_colloid_rh_nm"] for r in gr])
        plt.plot(t, indep, "o", color=colors[g], markersize=5,
                 label=f"{g}: per-trace colloid fit (independent)")
        gp = res["fitted"][g]
        tt = np.linspace(0, max(t.max(), 1e-6), 200)
        curve = gp.colloid_rh(tt)
        plt.plot(tt, curve, "-", color=colors[g],
                 label=f"{g}: global fit (Rh0={gp.Rh0_nm:.1f}, plateau={gp.Rhinf_nm:.1f} nm, tau={gp.tau_h:.2f} h)")
        plt.axhline(gp.Rhinf_nm, color=colors[g], linestyle=":", linewidth=0.8)
        yvals += [indep, curve]
    ys = np.concatenate(yvals)
    ys = ys[np.isfinite(ys)]
    lo, hi = float(np.min(ys)), float(np.max(ys))
    pad = 0.15 * (hi - lo) if hi > lo else 0.5
    plt.ylim(lo - pad, hi + pad)
    plt.xlabel("time / h"); plt.ylabel(r"colloid $R_h$ / nm")
    plt.title("Global exponential fit: colloid size vs time")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / "kinetics_colloid_rh.svg")


def plot_residuals_free_vs_global(res: dict, out_dir: Path) -> None:
    """Per-trace residuals vs in-trace time: free (per-trace) fit on top, global
    fit below.  Each trace is one line, coloured by its measurement time (jet:
    early = blue, late = red), with a zero reference line in each panel.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    Rh, wx, T_K = res["Rh"], res["wx"], res["T_K"]
    all_t = [tr.time_h for g in res["names"] for tr in res["groups"][g]]
    norm = Normalize(vmin=min(all_t), vmax=max(all_t))
    cmap = plt.get_cmap("jet")

    fig, (ax_free, ax_glob) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True,
                                            constrained_layout=True)
    for g in res["names"]:
        gp = res["fitted"][g]
        for tr in res["groups"][g]:
            color = cmap(norm(tr.time_h))
            rh_indep = independent_colloid_rh(gp, tr, Rh, wx, T_K)
            free = trace_residual_at_rh(gp, tr, rh_indep, Rh, wx, T_K)
            glob = trace_residual_at_rh(gp, tr, float(gp.colloid_rh(tr.time_h)), Rh, wx, T_K)
            ax_free.plot(tr.t_fit, free, "-", color=color, linewidth=0.6, alpha=0.7)
            ax_glob.plot(tr.t_fit, glob, "-", color=color, linewidth=0.6, alpha=0.7)

    for ax, label in ((ax_free, "free (per-trace independent) fit"), (ax_glob, "global fit")):
        ax.axhline(0.0, color="k", linewidth=0.9)
        ax.set_ylabel("residual (norm. signal)")
        ax.set_title(f"Residuals - {label}", fontsize=10)
    ax_glob.set_xlabel("time in trace / s")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=(ax_free, ax_glob), label="measurement time / h")
    # Save directly (not via save_current_figure) to avoid tight_layout clashing
    # with the shared colorbar; constrained_layout already spaces the figure.
    out_path = out_dir / "residuals_free_vs_global.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.show() if crli.SHOW_PLOTS else plt.close(fig)


def plot_example_fits(res: dict, out_dir: Path) -> None:
    """First / middle / last trace of each group, one stacked panel each.

    Each panel shows the *whole* trace (all samples), the subset actually used in
    the fit (the rising-side window, signal in [CRLI_SIGNAL_MIN, CRLI_SIGNAL_MAX],
    strided), the model on that window, the baseline, t0 and any flagged spikes -
    so it is clear what is fit vs merely excluded (vs removed as a spike).
    """
    Rh, wx, T_K = res["Rh"], res["wx"], res["T_K"]
    for g in res["names"]:
        trs = sorted(res["groups"][g], key=lambda x: x.time_h)
        picks = sorted({0, len(trs) // 2, len(trs) - 1})
        fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3.0 * len(picks)), sharex=True, squeeze=False)
        for ax, j in zip(axes[:, 0], picks):
            tr = trs[j]
            r = solve_trace(res["fitted"][g], tr, Rh, wx, T_K)
            t_all, y_all, base_all = tr.prep.time_s, tr.prep.signal, tr.prep.baseline
            base_fit = np.interp(tr.t_fit, t_all, base_all)
            ph = tr.prep.peak_height
            ax.plot(t_all, y_all, ".", color="0.8", markersize=1.5, label="all data")
            ax.plot(t_all, base_all, "--", color="gray", linewidth=0.8, label="baseline")
            ax.plot(tr.t_fit, ph * tr.s_fit + base_fit, ".", color="C0", markersize=3.5, label="fit-window data")
            ax.plot(tr.t_fit, ph * r["s_model"] + base_fit, "-", color="C3", linewidth=1.5, label="model")
            sp = tr.prep.spike_mask
            if np.any(sp):
                ax.plot(t_all[sp], y_all[sp], "x", color="red", markersize=6, markeredgewidth=1.3,
                        label=f"{int(sp.sum())} spikes (ignored)")
            ax.axvline(tr.time0_s, linestyle=":", color="k", linewidth=0.8)
            ax.set_ylabel("raw signal")
            ax.set_title(f"t={tr.time_h:.1f} h | colloid Rh={r['rh_col']:.1f} nm | "
                         f"{tr.s_fit.size} fit pts | t0={tr.time0_s:.1f} s", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
        axes[-1, 0].set_xlabel("time / s")
        fig.suptitle(f"Example fits - group {g}")
        crli.save_current_figure(out_dir / f"example_fits_{crli.safe_filename(g)}.svg")


def plot_trace_diagnostics(res: dict, out_dir: Path) -> None:
    rows = res["per_trace"]
    t = np.array([r["time_h"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, [r["residual_rms_over_noise"] for r in rows], "o", markersize=4)
    ax1.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="noise level")
    ax1.set_ylabel("residual RMS / noise"); ax1.legend(fontsize=8)
    ax1.set_title("Per-trace fit quality and composition vs time")
    ax2.plot(t, [100 * r["dye_fraction"] for r in rows], "o", markersize=4, label="dye %")
    ax2.plot(t, [100 * r["colloid_fraction"] for r in rows], "s", markersize=4, label="colloid %")
    ax2.set_xlabel("time / h"); ax2.set_ylabel("amplitude fraction / %"); ax2.legend(fontsize=8)
    crli.save_current_figure(out_dir / "per_trace_diagnostics.svg")


# =============================================================================
# Runner
# =============================================================================

def run(data_dir: str, pattern: str, group_mode: str, out_dir: str) -> dict:
    traces, info = load_traces(data_dir, pattern, group_mode)
    print(f"Method (from first data point): {info['method']}")
    print(f"Traces: {info['n_used']} used / {info['n_loaded']} loaded "
          f"({info['n_skipped']} skipped), grouping={group_mode}")
    res = fit_series(traces)
    root = Path(out_dir)
    save_outputs(res, info, root)
    print(f"\nConverged={res['success']} cost={res['cost']:.4g} nfev={res['nfev']}")
    for row in kinetics_rows(res):
        print(f"  group {row['group']:12s} (n={row['n_traces']:2d}): "
              f"Rh0={row['Rh0_nm']:.2f} -> plateau={row['Rh_plateau_nm']:.2f} nm "
              f"(dRh={row['delta_Rh_nm']:+.2f}), tau={row['tau_h']:.2f} h, "
              f"colloid_sigma={row['colloid_sigma_ln']:.3f}")
    print(f"\nOutputs in {root.resolve()}")
    return res


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global exponential-kinetics fit over a FidaNeo series.")
    ap.add_argument("--data", dest="data_dir", default=DATA_DIR, help="Folder of FidaNeo files.")
    ap.add_argument("--pattern", default=FILE_PATTERN, help="Glob within the folder.")
    ap.add_argument("--group", dest="group_mode", choices=["pooled", "per_sample"], default=GROUP_MODE)
    ap.add_argument("--output", dest="out_dir", default=OUTPUT_DIR, help="Output folder.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_dir, args.pattern, args.group_mode, args.out_dir)


if __name__ == "__main__":
    main()
