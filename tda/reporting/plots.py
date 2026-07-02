#!/usr/bin/env python3
"""Plotting, ported from the four fit scripts' near-duplicate plot functions.

Where a plot only needs fields every :class:`~tda.models.result.FitResult`
carries (``time_fit_s``, ``signal_model_raw``, the open ``components`` dict,
``Rh_nm``/``pdf_logRh``, ...) it is written once here and shared by CRLI,
Laplace and parametric - most visibly the decomposition plot, which used to
be three separate functions (free/colloid, low/high, comp0/comp1/...) and is
now one, driven entirely by ``result.components``. Where a plot is genuinely
model-specific (CRLI's truncation test, the parametric residual-vs-signal-level
and cost-slice plots, every global-series plot) it stays its own function,
grouped below by model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tda.models.result import FitResult
from tda.preprocessing.pipeline import PreparedTrace
from tda.reporting.io import save_current_figure


# =============================================================================
# Preprocessing (shared by every model)
# =============================================================================

def plot_preprocessing(prep: PreparedTrace, out_dir: Path, stem: str) -> None:
    t, y = prep.time_s, prep.signal

    plt.figure(figsize=(8, 4))
    plt.plot(t, y, ".", markersize=2, label="raw data")
    plt.plot(t, prep.baseline, "-", label="baseline")
    spikes = prep.spike_mask
    if spikes is not None and np.any(spikes):
        plt.plot(t[spikes], y[spikes], "x", color="red", markersize=7, markeredgewidth=1.5,
                 label=f"{int(np.sum(spikes))} spikes (ignored)")
    plt.axvline(prep.t0_s, linestyle="--", label="t0")
    plt.xlabel("time / s")
    plt.ylabel("signal")
    plt.title(f"{stem}: preprocessing")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / "preprocessing" / f"{stem}_preprocessing.svg")

    plt.figure(figsize=(8, 4))
    plt.plot(t[prep.good_mask], prep.normalized[prep.good_mask], ".", markersize=2)
    plt.axvline(prep.t0_s, linestyle="--")
    plt.xlabel("time / s")
    plt.ylabel("normalized signal")
    plt.title(f"{stem}: normalized trace")
    save_current_figure(out_dir / "preprocessing" / f"{stem}_normalized.svg")


# =============================================================================
# Single-trace generic plots (shared by CRLI / Laplace / parametric)
# =============================================================================

def plot_fit_overlay(result: FitResult, out_path: Path, title: str, model_label: str = "model") -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(result.time_fit_s, result.signal_fit_raw, ".", markersize=3, label="fit data")
    plt.plot(result.time_fit_s, result.signal_model_raw, "-", label=model_label)
    plt.xlabel("time / s")
    plt.ylabel("raw signal")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(out_path)


def plot_residuals_normalized(result: FitResult, out_path: Path, title: str) -> None:
    """Residual / peak height vs time (CRLI / Laplace convention)."""
    plt.figure(figsize=(9, 4))
    plt.plot(result.time_fit_s, result.residuals_raw / result.peak_height, ".", markersize=3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("time / s")
    plt.ylabel("residual / peak height")
    plt.title(title)
    save_current_figure(out_path)


def plot_residuals_raw_with_noise_band(result: FitResult, noise_raw: float, out_path: Path, title: str) -> None:
    """Raw residual vs time with a +/- noise band (parametric convention)."""
    plt.figure(figsize=(9, 4))
    plt.plot(result.time_fit_s, result.residuals_raw, ".", markersize=3)
    plt.axhspan(-noise_raw, noise_raw, color="gray", alpha=0.25, label=r"$\pm$ noise")
    plt.axhline(0, linestyle="--", linewidth=0.8, color="k")
    plt.xlabel("time / s")
    plt.ylabel("residual (raw signal)")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(out_path)


def plot_decomposition(result: FitResult, out_path: Path, title: str, show_data: bool = False,
                        component_labels: dict[str, str] | None = None) -> None:
    """Total model + baseline + each named component (``result.components``).
    ``component_labels`` overrides the legend text for named components
    (default: ``"baseline + <name>"``)."""
    t, base = result.time_fit_s, result.baseline_model_raw
    labels = component_labels or {}
    plt.figure(figsize=(9, 5))
    if show_data:
        plt.plot(t, result.signal_fit_raw, ".", markersize=3, label="fit data")
    plt.plot(t, result.signal_model_raw, "-", color="k", label="total model")
    for name, comp in result.components.items():
        plt.plot(t, base + comp, ":", label=labels.get(name, f"baseline + {name}"))
    plt.plot(t, base, "--", color="gray", label="baseline")
    plt.xlabel("time / s")
    plt.ylabel("raw signal")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(out_path)


def plot_distribution(Rh_nm: np.ndarray, pdf_logRh: np.ndarray, out_path: Path, title: str,
                       label: str = "distribution", extra_curves: list[tuple] = (),
                       reference_lines: list[tuple] = ()) -> None:
    """Rh distribution (semilog-x). ``extra_curves``: ``(Rh, pdf, label, style)``
    overlay curves; ``reference_lines``: ``(x_nm, label, style)`` vertical lines.
    """
    plt.figure(figsize=(7, 5))
    plt.semilogx(Rh_nm, pdf_logRh, "-", label=label)
    for rh, pdf, curve_label, style in extra_curves:
        style = dict(style)
        linestyle = style.pop("linestyle", ":")
        plt.semilogx(rh, pdf, linestyle, label=curve_label, **style)
    for x, ref_label, style in reference_lines:
        plt.axvline(x, label=ref_label, **style)
    plt.xlabel(r"$R_h$ / nm")
    plt.ylabel(r"PDF per $\ln R_h$")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(out_path)


def plot_scan(x: np.ndarray, y: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str,
              chosen_x: float | None = None, chosen_label: str = "chosen") -> None:
    """Generic log-log scan plot (alpha-vs-residual, roughness-vs-residual, ...)."""
    plt.figure(figsize=(6, 4))
    plt.loglog(x, y, ".-")
    if chosen_x is not None and np.isfinite(chosen_x):
        plt.axvline(chosen_x, linestyle="--", label=chosen_label)
        plt.legend(fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_current_figure(out_path)


# =============================================================================
# CRLI-specific
# =============================================================================

def plot_crli_truncation_test(result: FitResult, noise_raw: float, out_path: Path, title: str) -> None:
    """Compare the full model with one where colloid mass below the cutoff is
    zeroed - the lower panel shows the signal that mass alone contributes,
    against the +/- noise band, so artefact (within noise) vs feature (well
    above noise) is read off directly."""
    cutoff = result.extra["trunc_rh_nm"]
    low_raw = result.extra["trunc_low_signal_raw"]
    t = result.time_fit_s
    trunc_model = result.signal_model_raw - low_raw
    rms = result.stats["trunc_signal_rms_raw"]
    peak = result.stats["trunc_signal_peak_raw"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, result.signal_fit_raw, ".", markersize=3, label="fit data")
    ax1.plot(t, result.signal_model_raw, "-", label="full model")
    ax1.plot(t, trunc_model, "--", label=f"model with $R_h<{cutoff:g}$ nm removed")
    ax1.set_ylabel("raw signal")
    ax1.set_title(title)
    ax1.legend(fontsize=8)

    ax2.plot(t, low_raw, "-", color="C3", label=f"signal from $R_h<{cutoff:g}$ nm mass")
    ax2.axhspan(-noise_raw, noise_raw, color="gray", alpha=0.25, label=r"$\pm$ noise")
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="k")
    ax2.set_xlabel("time / s")
    ax2.set_ylabel("signal difference")
    ax2.set_title(f"peak {peak:.3g} = {peak/noise_raw:.2f}x noise | RMS {rms:.3g} = {rms/noise_raw:.2f}x noise",
                  fontsize=9)
    ax2.legend(fontsize=8)
    save_current_figure(out_path)


# =============================================================================
# Parametric-specific
# =============================================================================

def plot_parametric_residual_vs_signal(result: FitResult, noise_raw: float, out_path: Path, title: str) -> None:
    """Residual vs signal level, binned - exposes apex-vs-wing structure."""
    s, r = result.signal_fit_normalized, result.extra["residual_norm"]
    ph = result.peak_height
    edges = np.linspace(float(np.min(s)), float(np.max(s)), 16)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(centers) - 1)
    mean = np.array([r[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(centers))])
    std = np.array([r[idx == b].std() if np.any(idx == b) else np.nan for b in range(len(centers))])
    plt.figure(figsize=(8, 4))
    plt.plot(s, r, ".", markersize=2, alpha=0.3, color="gray", label="points")
    plt.errorbar(centers, mean, yerr=std, fmt="o-", capsize=3, label="binned mean +/- sd")
    plt.axhspan(-noise_raw / ph, noise_raw / ph, color="gray", alpha=0.2, label=r"$\pm$ noise")
    plt.axhline(0, linestyle="--", linewidth=0.8, color="k")
    plt.xlabel("normalized signal level")
    plt.ylabel("residual / peak height")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(out_path)


def plot_parametric_cost_slices(result: FitResult, out_dir: Path, stem: str) -> None:
    """1-D profiles of the residual cost as each shape parameter is scanned
    around the optimum; other parameters held fixed, amplitudes re-solved by
    NNLS at each scanned point."""
    from tda.models.parametric import pack_to_vector, residual_vector

    pack = result.extra["pack"]
    x0, lo, hi = result.extra["x0"], result.extra["lo"], result.extra["hi"]
    t, s = result.time_fit_s, result.signal_fit_normalized
    Rh, wx, T_K = result.Rh_nm, result.extra["wx"], result.extra["T_K"]
    fit_t0 = bool(result.stats["fit_t0"])
    k = int(result.stats["n_components"])
    physics = result.extra["physics"]
    x_opt = pack_to_vector(pack, fit_t0)
    args = (t, s, Rh, wx, T_K, k, fit_t0, float(pack.t0_s), physics)

    labels = []
    for i in range(k):
        labels += [f"ln mu (comp {i}, {pack.Rh_nm[i]:.2g} nm)", f"ln sigma (comp {i})"]
    if fit_t0:
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
            x = x_opt.copy()
            x[j] = v
            r = residual_vector(x, *args)
            cost.append(0.5 * float(np.dot(r, r)))
        ax.plot(grid, cost, "-")
        ax.axvline(x_opt[j], linestyle="--", color="C3", linewidth=0.8)
        ax.set_xlabel(labels[j])
        ax.set_ylabel("0.5 * SSR")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{stem}: cost-vs-parameter slices")
    save_current_figure(out_dir / f"{stem}_parametric_cost_slices.svg")


# =============================================================================
# Per-model plot sets - what each root wrapper script calls per file
# =============================================================================

def save_crli_plots(result: FitResult, prep: PreparedTrace, out_dir: Path, stem: str) -> None:
    plot_fit_overlay(result, out_dir / f"{stem}_crli_fit.svg", f"{stem}: CRLI fit", model_label="fit")
    plot_residuals_normalized(result, out_dir / f"{stem}_crli_residuals.svg", f"{stem}: CRLI residuals")
    plot_decomposition(result, out_dir / f"{stem}_crli_decomposition.svg", f"{stem}: CRLI decomposition",
                        show_data=True, component_labels={"free": "baseline + free fluorophore",
                                                            "colloid": "baseline + colloid"})
    free_rh_nm = result.stats["free_Rh_nm"]
    plot_distribution(
        result.Rh_nm, result.pdf_logRh, out_dir / f"{stem}_crli_distribution.svg",
        f"{stem}: colloid distribution", label="crli",
        reference_lines=[(free_rh_nm, f"free fluorophore {free_rh_nm:g} nm", {"linestyle": ":"})],
    )
    chosen_alpha = result.stats.get("crli_alpha", np.nan)
    plot_scan(result.extra["alpha_scan"], result.extra["residual_rms_scan"],
              out_dir / f"{stem}_crli_alpha_residual_scan.svg", f"{stem}: CRLI alpha scan",
              r"regularization $\alpha$", "normalized residual RMS", chosen_x=chosen_alpha, chosen_label="chosen alpha")
    plot_scan(result.extra["roughness_scan"], result.extra["residual_rms_scan"],
              out_dir / f"{stem}_crli_roughness_scan.svg", f"{stem}: CRLI roughness scan",
              "roughness", "normalized residual RMS")
    plot_crli_truncation_test(result, prep.noise_raw, out_dir / f"{stem}_crli_truncation_test.svg",
                               f"{stem}: truncation test (mass below {result.extra['trunc_rh_nm']:g} nm)")


def save_laplace_plots(result: FitResult, out_dir: Path, stem: str, reference_free_rh_nm: float = 0.59) -> None:
    cutoff = result.stats["small_cutoff_nm"]
    plot_fit_overlay(result, out_dir / f"{stem}_laplace_fit.svg", f"{stem}: Laplace inversion fit",
                      model_label="inversion model")
    plot_residuals_normalized(result, out_dir / f"{stem}_laplace_residuals.svg", f"{stem}: residuals")
    plot_decomposition(
        result, out_dir / f"{stem}_laplace_decomposition.svg", f"{stem}: decomposition (emergent small species)",
        component_labels={"low": f"$R_h<{cutoff:g}$ nm component", "high": f"$R_h\\geq{cutoff:g}$ nm component"},
    )
    plot_distribution(
        result.Rh_nm, result.pdf_logRh, out_dir / f"{stem}_laplace_distribution.svg",
        f"{stem}: recovered size distribution",
        reference_lines=[
            (reference_free_rh_nm, f"dye ref {reference_free_rh_nm:g} nm", {"linestyle": ":", "color": "C3"}),
            (cutoff, f"cutoff {cutoff:g} nm", {"linestyle": "--", "color": "gray", "linewidth": 0.8}),
        ],
    )
    chosen_alpha = result.stats.get("crli_alpha", np.nan)
    plot_scan(result.extra["alpha_scan"], result.extra["residual_scan"],
              out_dir / f"{stem}_laplace_alpha_scan.svg", f"{stem}: alpha scan",
              r"regularization $\alpha$", "normalized residual RMS", chosen_x=chosen_alpha, chosen_label="chosen alpha")


def parametric_component_role(i: int, k: int) -> str:
    """Semantic role of parametric component ``i`` of ``k``: the smallest is
    "dye", the largest is "colloid", anything in between is "compN"."""
    return "dye" if i == 0 else ("colloid" if i == k - 1 else f"comp{i}")


def _parametric_component_label(i: int, k: int, Rh_nm: float) -> str:
    return f"{parametric_component_role(i, k)} ({Rh_nm:.2g} nm)"


def save_parametric_plots(result: FitResult, prep: PreparedTrace, out_dir: Path, stem: str,
                           laplace_overlay: FitResult | None = None, reference_free_rh_nm: float = 0.59,
                           cost_slices: bool = True) -> None:
    pack = result.extra["pack"]
    k = int(result.stats["n_components"])

    plot_fit_overlay(result, out_dir / f"{stem}_parametric_fit.svg", f"{stem}: parametric fit",
                      model_label="parametric model")

    noise = prep.noise_raw
    plot_residuals_raw_with_noise_band(
        result, noise, out_dir / f"{stem}_parametric_residual_time.svg",
        f"{stem}: residual vs time (RMS {result.stats['residual_rms_raw']:.3g} = "
        f"{result.stats['residual_rms_over_noise']:.2f}x noise)",
    )
    plot_parametric_residual_vs_signal(result, noise, out_dir / f"{stem}_parametric_residual_signal.svg",
                                        f"{stem}: residual vs signal level")

    comp_labels = {f"comp{i}": _parametric_component_label(i, k, pack.Rh_nm[i]) for i in range(k)}
    plot_decomposition(result, out_dir / f"{stem}_parametric_decomposition.svg",
                        f"{stem}: signal-space decomposition", component_labels=comp_labels)

    area, component_density = result.extra["area"], result.extra["component_density"]
    extra_curves = [
        (result.Rh_nm, component_density[:, i] / area, comp_labels[f"comp{i}"], {"linestyle": ":"})
        for i in range(k)
    ]
    if laplace_overlay is not None:
        extra_curves.append((laplace_overlay.Rh_nm, laplace_overlay.pdf_logRh, "Laplace (free-form)",
                              {"linestyle": "-", "color": "C3", "alpha": 0.7}))
    plot_distribution(
        result.Rh_nm, result.pdf_logRh, out_dir / f"{stem}_parametric_distribution.svg",
        f"{stem}: parametric vs Laplace distribution", label="parametric (total)",
        extra_curves=extra_curves,
        reference_lines=[(reference_free_rh_nm, f"dye ref {reference_free_rh_nm:g} nm",
                           {"linestyle": "--", "color": "gray", "linewidth": 0.8})],
    )

    if cost_slices:
        plot_parametric_cost_slices(result, out_dir, stem)


# =============================================================================
# Global-series-specific
# =============================================================================

def plot_kinetics(res, out_dir: Path) -> None:
    """Colloid Rh vs time: per-trace independent fits (scatter) vs the shared
    global curve (line) - the global line should pass through the scatter if
    the shared-kinetics constraint holds."""
    plt.figure(figsize=(9, 5))
    colors = {g: f"C{i}" for i, g in enumerate(res.names)}
    rows = res.per_trace
    yvals: list[np.ndarray] = []
    for g in res.names:
        gr = [r for r in rows if r["group"] == g]
        t = np.array([r["time_h"] for r in gr])
        indep = np.array([r["indep_colloid_rh_nm"] for r in gr])
        plt.plot(t, indep, "o", color=colors[g], markersize=5,
                 label=f"{g}: per-trace colloid fit (independent)")
        gp = res.fitted[g]
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
    plt.xlabel("time / h")
    plt.ylabel(r"colloid $R_h$ / nm")
    plt.title("Global exponential fit: colloid size vs time")
    plt.legend(fontsize=8)
    save_current_figure(out_dir / "kinetics_colloid_rh.svg")


def plot_residuals_free_vs_global(res, out_dir: Path, physics, show: bool = False) -> None:
    """Per-trace residuals vs in-trace time: free (per-trace) fit on top,
    global fit below, each trace one line coloured by measurement time."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from tda.models.global_series import independent_colloid_rh, trace_residual_at_rh

    Rh, wx, T_K = res.Rh, res.wx, res.T_K
    all_t = [tr.time_h for g in res.names for tr in res.groups[g]]
    norm = Normalize(vmin=min(all_t), vmax=max(all_t))
    cmap = plt.get_cmap("jet")

    fig, (ax_free, ax_glob) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True,
                                            constrained_layout=True)
    for g in res.names:
        gp = res.fitted[g]
        for tr in res.groups[g]:
            color = cmap(norm(tr.time_h))
            rh_indep = independent_colloid_rh(gp, tr, Rh, wx, T_K, (float(np.min(Rh)), float(np.max(Rh))), physics)
            free = trace_residual_at_rh(gp, tr, rh_indep, Rh, wx, T_K, physics)
            glob = trace_residual_at_rh(gp, tr, float(gp.colloid_rh(tr.time_h)), Rh, wx, T_K, physics)
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
    out_path = out_dir / "residuals_free_vs_global.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.show() if show else plt.close(fig)


def plot_example_fits(res, out_dir: Path, physics) -> None:
    """First / middle / last trace of each group, one stacked panel each."""
    from tda.models.global_series import solve_trace

    Rh, wx, T_K = res.Rh, res.wx, res.T_K
    for g in res.names:
        trs = sorted(res.groups[g], key=lambda x: x.time_h)
        picks = sorted({0, len(trs) // 2, len(trs) - 1})
        fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3.0 * len(picks)), sharex=True, squeeze=False)
        for ax, j in zip(axes[:, 0], picks):
            tr = trs[j]
            r = solve_trace(res.fitted[g], tr, Rh, wx, T_K, physics)
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
        save_current_figure(out_dir / f"example_fits_{g}.svg")


def plot_trace_diagnostics(res, out_dir: Path) -> None:
    rows = res.per_trace
    t = np.array([r["time_h"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, [r["residual_rms_over_noise"] for r in rows], "o", markersize=4)
    ax1.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="noise level")
    ax1.set_ylabel("residual RMS / noise")
    ax1.legend(fontsize=8)
    ax1.set_title("Per-trace fit quality and composition vs time")
    ax2.plot(t, [100 * r["dye_fraction"] for r in rows], "o", markersize=4, label="dye %")
    ax2.plot(t, [100 * r["colloid_fraction"] for r in rows], "s", markersize=4, label="colloid %")
    ax2.set_xlabel("time / h")
    ax2.set_ylabel("amplitude fraction / %")
    ax2.legend(fontsize=8)
    save_current_figure(out_dir / "per_trace_diagnostics.svg")
