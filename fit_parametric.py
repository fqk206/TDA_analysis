#!/usr/bin/env python3
"""Parametric (log-normal mixture) fit of a Taylorgram - the constrained
counterpart to the free-form CRLI / Laplace inversion.

Where ``fit_laplace.py`` solves for a free, smooth size distribution, this script
fits the same physics with a *small, fixed* number of log-normal components
(default two: a small "dye" species ~0.75 nm and a colloid ~23 nm). The point is
a constrained, interpretable result that can be checked against the free-form
inversion: if the two agree, the breadth in the inversion was regularisation, not
the sample.

The fit uses variable projection (VARPRO): for fixed component *shapes* (centers
``mu_k`` and log-widths ``sigma_k``) the model is linear in the amplitudes
``a_k``, so those are solved by NNLS at every step and the nonlinear optimiser
only handles the handful of shape parameters (+ an optional ``t0`` nuisance).

Reading/preprocessing, the Taylor kernel, grid weights, NNLS, distribution stats
and plotting helpers are all reused from ``fit_crli_simple`` (one source of truth
for data, physics and preprocessing); the Laplace overlay reuses
``fit_laplace.invert``.

See ``docs/parametric.md`` for the design. Global (multi-trace) fitting and
bootstrap uncertainty are deferred; the pure forward-model / residual-vector
interfaces below (``forward_signal`` / ``residual_vector``) are kept clean so
those layers can wrap this one without rework.

Run:
    python fit_parametric.py
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, nnls

import fit_crli_simple as crli
import fit_laplace


# =============================================================================
# CONFIG - edit these and run the script directly
# =============================================================================

N_COMPONENTS = 2
INIT_RH_NM = [0.75, 23.0]            # initial centers (one per component)
INIT_SIGMA_LN = [0.05, 0.25]         # initial log-widths
RH_BOUNDS_NM = (0.3, 200.0)          # per-center bounds
SIGMA_LN_BOUNDS = (0.02, 1.5)        # per-width bounds
FIT_T0 = True                        # refine t0 as a nuisance parameter
T0_BOUND_S = 10.0                    # t0 is bounded to prep.t0 +/- this (seconds)
RH_GRID = (0.2, 300.0, 400)          # (min_nm, max_nm, n) dense grid for the model integral

# Reuse the inversion's fit-window controls so the comparison is apples-to-apples.
PEAK_FRACTION = crli.PEAK_FRACTION   # which part of the peak to fit (see fit_crli_simple)
REFERENCE_FREE_RH_NM = 0.59          # drawn on the distribution plot for orientation
COST_SLICES = True                   # produce the (optional) cost-vs-parameter slice plot

OUTPUT_DIR = "output/parametric"


# =============================================================================
# Parameter pack/unpack (kept small so a later global layer can map a
# shared/trend/per-trace vector onto each trace - see docs/parametric.md sec 7)
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


def parameter_bounds(t0_fixed: float, fit_t0: bool, k: int) -> tuple[np.ndarray, np.ndarray]:
    lo_rh, hi_rh = np.log(RH_BOUNDS_NM[0]), np.log(RH_BOUNDS_NM[1])
    lo_sg, hi_sg = np.log(SIGMA_LN_BOUNDS[0]), np.log(SIGMA_LN_BOUNDS[1])
    lo = np.tile([lo_rh, lo_sg], k)
    hi = np.tile([hi_rh, hi_sg], k)
    if fit_t0:
        lo = np.r_[lo, t0_fixed - T0_BOUND_S]
        hi = np.r_[hi, t0_fixed + T0_BOUND_S]
    return lo, hi


# =============================================================================
# Forward model and residual (pure functions - the interface §7 builds on)
# =============================================================================

def lognormal_pdf_logRh(Rh_nm: np.ndarray, mu_nm: float, sigma: float) -> np.ndarray:
    """Log-normal in Rh, expressed as a density per ln(Rh), unit area in ln(Rh)."""
    z = (np.log(Rh_nm) - np.log(mu_nm)) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def design_matrix(pack: ParamPack, t: np.ndarray, Rh: np.ndarray, wx: np.ndarray, T_K: float) -> np.ndarray:
    """Per-component normalised-signal columns C[:,k] = integral g_k(Rh) K(t,Rh) dlnRh.

    The model is ``C @ a`` and is linear in the amplitudes ``a`` (VARPRO).
    """
    kernel = crli.rh_kernel(t, Rh, pack.t0_s, T_K)         # (n_t, n_grid)
    cols = [kernel @ (lognormal_pdf_logRh(Rh, mu, sg) * wx)
            for mu, sg in zip(pack.Rh_nm, pack.sigma)]
    return np.column_stack(cols)


def solve_amplitudes(C: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Non-negative amplitudes for fixed shapes (the linear VARPRO sub-problem)."""
    a, _ = nnls(C, s, maxiter=20000)
    return a


def forward_signal(pack: ParamPack, a: np.ndarray, t: np.ndarray, Rh: np.ndarray,
                   wx: np.ndarray, T_K: float) -> np.ndarray:
    """Pure forward model: params -> normalised signal(t). Used by fit and later layers."""
    return design_matrix(pack, t, Rh, wx, T_K) @ a


def residual_vector(x: np.ndarray, t: np.ndarray, s: np.ndarray, Rh: np.ndarray, wx: np.ndarray,
                    T_K: float, k: int, fit_t0: bool, t0_fixed: float) -> np.ndarray:
    """Residual (model - data) on the normalised signal, amplitudes via NNLS inside.

    Returning a vector (not a scalar cost) is what lets a later global fit just
    concatenate residuals over traces, and bootstrap refit on resampled data.
    """
    pack = vector_to_pack(x, k, fit_t0, t0_fixed)
    C = design_matrix(pack, t, Rh, wx, T_K)
    a = solve_amplitudes(C, s)
    return C @ a - s


# =============================================================================
# Single-trace fit
# =============================================================================

def fit_parametric(prep, side: str) -> dict:
    T_K = 273.15 + crli.TEMPERATURE_C
    t0_fixed = prep.t0_s
    k = int(N_COMPONENTS)
    if len(INIT_RH_NM) != k or len(INIT_SIGMA_LN) != k:
        raise ValueError("INIT_RH_NM and INIT_SIGMA_LN must each have N_COMPONENTS entries.")

    indices = np.flatnonzero(crli.crli_fit_mask(prep, side, PEAK_FRACTION))[:: max(1, int(crli.CRLI_FIT_STRIDE))]
    if indices.size < 25:
        raise ValueError("Too few fitting points for the parametric fit.")
    t = prep.time_s[indices]
    y = prep.signal[indices]
    s = prep.normalized[indices]

    rh_min, rh_max, n_grid = RH_GRID
    Rh = np.geomspace(rh_min, rh_max, int(n_grid))
    wx = crli.trapezoid_weights(np.log(Rh))

    init = ParamPack(ln_mu=np.log(INIT_RH_NM), ln_sigma=np.log(INIT_SIGMA_LN), t0_s=t0_fixed)
    x0 = pack_to_vector(init, FIT_T0)
    lo, hi = parameter_bounds(t0_fixed, FIT_T0, k)
    x0 = np.clip(x0, lo, hi)

    args = (t, s, Rh, wx, T_K, k, FIT_T0, t0_fixed)
    sol = least_squares(residual_vector, x0, bounds=(lo, hi), args=args, method="trf", x_scale="jac")

    pack = vector_to_pack(sol.x, k, FIT_T0, t0_fixed)
    C = design_matrix(pack, t, Rh, wx, T_K)
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

    # Combined parametric distribution, normalised to unit area in ln(Rh).
    component_density = np.column_stack([
        a_k * lognormal_pdf_logRh(Rh, mu, sg) for a_k, mu, sg in zip(a, pack.Rh_nm, pack.sigma)
    ])                                                    # (n_grid, K), each unit-area * a_k
    dens = component_density.sum(axis=1)
    area = float(np.sum(dens * wx))
    pdf_log = dens / area
    pdf_R = pdf_log / Rh
    cdf = np.cumsum(pdf_log * wx)
    cdf = cdf / cdf[-1]

    baseline = np.interp(t, prep.time_s, prep.baseline)
    model_raw = baseline + prep.peak_height * s_model
    residual_norm = s_model - s
    residual_raw = model_raw - y
    rms_norm = float(np.sqrt(np.mean(residual_norm ** 2)))
    rms_raw = float(np.sqrt(np.mean(residual_raw ** 2)))
    noise = prep.noise_raw

    stats = {
        "success": bool(sol.success),
        "fit_side": side,
        "peak_fraction": float(PEAK_FRACTION),
        "n_components": k,
        "fit_t0": bool(FIT_T0),
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
    stats.update(crli.distribution_stats(Rh, pdf_log))

    return {
        "t": t, "y": y, "s": s, "s_model": s_model, "model_raw": model_raw, "baseline": baseline,
        "comp_signal": comp_signal, "residual_norm": residual_norm, "peak_height": prep.peak_height,
        "Rh": Rh, "pdf_log": pdf_log, "pdf_R": pdf_R, "cdf": cdf,
        "component_density": component_density, "area": area,
        "pack": pack, "a": a, "fractions": fractions, "wx": wx, "T_K": T_K, "x0": x0,
        "lo": lo, "hi": hi, "side": side, "stats": stats,
    }


# =============================================================================
# Plots (docs/parametric.md sec 8)
# =============================================================================

def _component_label(i: int, pack: ParamPack) -> str:
    role = "dye" if i == 0 else ("colloid" if i == len(pack.Rh_nm) - 1 else f"comp {i}")
    return f"{role} ({pack.Rh_nm[i]:.2g} nm)"


def save_plots(res: dict, prep, out_dir: Path, stem: str) -> None:
    t, ph, base = res["t"], res["peak_height"], res["baseline"]
    pack, a = res["pack"], res["a"]
    noise = prep.noise_raw
    colors = [f"C{i}" for i in range(len(a))]

    # 1. Fit overlay (raw signal).
    plt.figure(figsize=(9, 5))
    plt.plot(t, res["y"], ".", markersize=3, label="fit data")
    plt.plot(t, res["model_raw"], "-", label="parametric model")
    plt.xlabel("time / s"); plt.ylabel("raw signal")
    plt.title(f"{stem}: parametric fit")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_parametric_fit.svg")

    # 2. Residual vs time, with a +/- noise band (raw signal units).
    plt.figure(figsize=(9, 4))
    plt.plot(t, res["model_raw"] - res["y"], ".", markersize=3)
    plt.axhspan(-noise, noise, color="gray", alpha=0.25, label=r"$\pm$ noise")
    plt.axhline(0, linestyle="--", linewidth=0.8, color="k")
    plt.xlabel("time / s"); plt.ylabel("residual (raw signal)")
    plt.title(f"{stem}: residual vs time "
              f"(RMS {res['stats']['residual_rms_raw']:.3g} = {res['stats']['residual_rms_over_noise']:.2f}x noise)")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_parametric_residual_time.svg")

    # 3. Residual vs signal level, binned (exposes apex-vs-wing structure).
    s, r = res["s"], res["residual_norm"]
    edges = np.linspace(float(np.min(s)), float(np.max(s)), 16)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(centers) - 1)
    mean = np.array([r[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(centers))])
    std = np.array([r[idx == b].std() if np.any(idx == b) else np.nan for b in range(len(centers))])
    plt.figure(figsize=(8, 4))
    plt.plot(s, r, ".", markersize=2, alpha=0.3, color="gray", label="points")
    plt.errorbar(centers, mean, yerr=std, fmt="o-", capsize=3, label="binned mean +/- sd")
    plt.axhspan(-noise / ph, noise / ph, color="gray", alpha=0.2, label=r"$\pm$ noise")
    plt.axhline(0, linestyle="--", linewidth=0.8, color="k")
    plt.xlabel("normalized signal level"); plt.ylabel("residual / peak height")
    plt.title(f"{stem}: residual vs signal level")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_parametric_residual_signal.svg")

    # 4. Signal-space decomposition: total model plus each component's signal.
    plt.figure(figsize=(9, 5))
    plt.plot(t, res["model_raw"], "-", color="k", label="total model")
    for i in range(len(a)):
        plt.plot(t, base + ph * res["comp_signal"][:, i], ":", color=colors[i], label=_component_label(i, pack))
    plt.plot(t, base, "--", color="gray", label="baseline")
    plt.xlabel("time / s"); plt.ylabel("raw signal")
    plt.title(f"{stem}: signal-space decomposition")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_parametric_decomposition.svg")

    # 5. Distribution overlay vs the free-form Laplace inversion (the key cross-check).
    plt.figure(figsize=(7, 5))
    plt.semilogx(res["Rh"], res["pdf_log"], "-", color="k", label="parametric (total)")
    for i in range(len(a)):
        plt.semilogx(res["Rh"], res["component_density"][:, i] / res["area"], ":", color=colors[i],
                     label=_component_label(i, pack))
    laplace = _laplace_overlay(prep, res["side"])
    if laplace is not None:
        plt.semilogx(laplace["Rh"], laplace["pdf_log"], "-", color="C3", alpha=0.7, label="Laplace (free-form)")
    plt.axvline(REFERENCE_FREE_RH_NM, linestyle="--", color="gray", linewidth=0.8,
                label=f"dye ref {REFERENCE_FREE_RH_NM:g} nm")
    plt.xlabel(r"$R_h$ / nm"); plt.ylabel(r"PDF per $\ln R_h$")
    plt.title(f"{stem}: parametric vs Laplace distribution")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_parametric_distribution.svg")

    # 6. (Optional) cost-vs-parameter slices around the optimum.
    if COST_SLICES:
        _save_cost_slices(res, out_dir, stem)


def _laplace_overlay(prep, side: str) -> dict | None:
    """Free-form Laplace distribution for the overlay; None if it can't be computed."""
    try:
        out = fit_laplace.invert(prep, side)
        return {"Rh": out["Rh"], "pdf_log": out["pdf_log"]}
    except Exception as exc:
        warnings.warn(f"Laplace overlay unavailable: {exc}")
        return None


def _save_cost_slices(res: dict, out_dir: Path, stem: str) -> None:
    """1-D profiles of the residual cost as each shape parameter is scanned.

    A cheap, pre-bootstrap look at how well each parameter is determined and which
    are flat/degenerate. Other parameters are held at the optimum; amplitudes are
    re-solved by NNLS at each scanned point (the same VARPRO sub-problem).
    """
    t, s, Rh, wx, T_K = res["t"], res["s"], res["Rh"], res["wx"], res["T_K"]
    k, x_opt, lo, hi = len(res["a"]), pack_to_vector(res["pack"], FIT_T0), res["lo"], res["hi"]
    args = (t, s, Rh, wx, T_K, k, FIT_T0, float(res["pack"].t0_s))

    labels = []
    for i in range(k):
        labels += [f"ln mu (comp {i}, {res['pack'].Rh_nm[i]:.2g} nm)", f"ln sigma (comp {i})"]
    if FIT_T0:
        labels.append("t0 / s")

    n = len(x_opt)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for j in range(n):
        ax = axes[j // ncol][j % ncol]
        span = max(abs(hi[j] - lo[j]) * 0.15, 1e-6)
        grid = np.clip(np.linspace(x_opt[j] - span, x_opt[j] + span, 31), lo[j], hi[j])
        cost = []
        for v in grid:
            x = x_opt.copy(); x[j] = v
            r = residual_vector(x, *args)
            cost.append(0.5 * float(np.dot(r, r)))
        ax.plot(grid, cost, "-")
        ax.axvline(x_opt[j], linestyle="--", color="C3", linewidth=0.8)
        ax.set_xlabel(labels[j]); ax.set_ylabel("0.5 * SSR")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{stem}: cost-vs-parameter slices")
    crli.save_current_figure(out_dir / f"{stem}_parametric_cost_slices.svg")


def write_distribution_csv(res: dict, out_dir: Path, stem: str) -> None:
    rows = [
        {"Rh_nm": float(rh), "pdf_per_ln_Rh": float(p), "pdf_per_nm": float(pr), "cdf": float(c)}
        for rh, p, pr, c in zip(res["Rh"], res["pdf_log"], res["pdf_R"], res["cdf"])
    ]
    crli.write_rows(out_dir / f"{stem}_parametric_distribution.csv", rows)


# =============================================================================
# Config dump and batch runner
# =============================================================================

def config_dump() -> dict:
    keys = [
        "N_COMPONENTS", "INIT_RH_NM", "INIT_SIGMA_LN", "RH_BOUNDS_NM", "SIGMA_LN_BOUNDS",
        "FIT_T0", "T0_BOUND_S", "RH_GRID", "PEAK_FRACTION", "REFERENCE_FREE_RH_NM",
        "COST_SLICES", "OUTPUT_DIR",
    ]
    out = {k: globals()[k] for k in keys}
    # Physics / preprocessing live in fit_crli_simple - record the ones that matter here.
    out["shared_from_fit_crli_simple"] = {
        "DATA_PATTERN": crli.DATA_PATTERN, "TIME_UNIT": crli.TIME_UNIT,
        "TEMPERATURE_C": crli.TEMPERATURE_C, "CAPILLARY_RADIUS_M": crli.CAPILLARY_RADIUS_M,
        "VISCOSITY_PA_S": crli.VISCOSITY_PA_S, "KERNEL_REFERENCE_RH_NM": crli.KERNEL_REFERENCE_RH_NM,
        "FIT_SIDE": crli.FIT_SIDE, "CRLI_FIT_STRIDE": crli.CRLI_FIT_STRIDE,
        "CRLI_SIGNAL_MIN": crli.CRLI_SIGNAL_MIN, "CRLI_SIGNAL_MAX": crli.CRLI_SIGNAL_MAX,
    }
    return out


def print_result(stem: str, stats: dict) -> None:
    comps = " | ".join(
        f"c{i}: Rh={stats[f'comp{i}_Rh_nm']:.2f}nm sig={stats[f'comp{i}_sigma_ln']:.3f} "
        f"f={100*stats[f'comp{i}_fraction']:.1f}%"
        for i in range(stats["n_components"])
    )
    print(f"  {stem:28s} | t0={stats['t0_s']:.1f}s (dt={stats['t0_shift_s']:+.2f}) | {comps} "
          f"| RMS/noise={stats['residual_rms_over_noise']:.2f}", flush=True)


def main() -> None:
    files = sorted(Path(p) for p in glob(crli.DATA_PATTERN, recursive=True) if Path(p).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {crli.DATA_PATTERN!r}")

    root = Path(OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_settings.json").write_text(json.dumps(config_dump(), indent=2), encoding="utf-8")
    print(f"Found {len(files)} file(s). Output folder: {root.resolve()}")

    rows = []
    for path in files:
        print(f"\nAnalysing {path}", flush=True)
        try:
            prep = crli.prepare(path)
            stem = crli.safe_filename(path.stem)
            if crli.SAVE_PLOTS:
                crli.save_preprocessing_plots(prep, root, stem)
            res = fit_parametric(prep, side=crli.FIT_SIDE)
            write_distribution_csv(res, root, stem)
            if crli.SAVE_PLOTS:
                save_plots(res, prep, root, stem)
            rows.append({"file": str(path), **res["stats"]})
            print_result(stem, res["stats"])
        except Exception as exc:
            warnings.warn(f"Skipped {path.name}: {exc}")
            rows.append({"file": str(path), "success": False, "error": str(exc)})

    crli.write_rows(root / "parametric_summary.csv", rows)
    print(f"\nSaved summary: {root / 'parametric_summary.csv'}\nDone.")


if __name__ == "__main__":
    main()
