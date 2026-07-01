#!/usr/bin/env python3
"""Unsupervised factorisation of a FidaNeo Taylorgram series.

Estimate **how many independent components / how much information** live in a
folder of Taylorgrams (e.g. ``data_series/series_01``) by stacking every trace
as a row of a matrix and factorising it:

    X  =  [ n_traces  x  n_tau ]     (one measurement per row, time per column)

Three lenses, in the order you actually want them:

1. **PCA / SVD (numpy)** answers *how many components*.  We mean-centre the
   matrix (PCA) so the huge shared peak + baseline don't eat the first modes,
   then look at the singular-value spectrum:
     - right singular vectors  V  = characteristic Taylorgram *shapes*
       ("eigen-Taylorgrams", deviations from the mean trace);
     - left  singular vectors  U  = how each shape's weight evolves *across the
       series* (plotted against experiment time -> the kinetic trend lives here);
     - singular values         s  = importance -> scree / cumulative variance.
   With ``CENTER = False`` you get the raw SVD instead, whose first mode is the
   common shape itself (more literal, less sensitive to variation).

2. **Parallel analysis (permutation)** turns the scree elbow into an *unsupervised
   rank*: shuffle each column independently many times, re-SVD, and keep only the
   components whose singular value beats the shuffled-noise spectrum.  This is the
   "how much is signal vs noise" number, with no eyeballing.

3. **NMF (scikit-learn)** at the chosen rank gives *physically interpretable*
   components: Taylorgrams are non-negative mixtures of non-negative species
   peaks, so NMF factors X ~= W @ H with W,H >= 0 -> H rows look like species
   shapes, W columns look like concentration-vs-time.  Run after step 2 fixes k.

The crucial preprocessing (shared with the fitters) is reused, not reinvented:
``fidaneo`` parses the files and gives the experiment-time axis, and
``fit_crli_simple.prepare_from_trace`` despikes (Hampel), subtracts the linear
baseline, and locates t0.  We then **align every trace at its own t0** onto a
common tau = t - t0 grid before stacking, because SVD is not shift-invariant: an
unaligned series spends components encoding peak drift instead of real physics.

Usage:
    python3 factor_series.py data_series/series_01
    python3 factor_series.py data_series/series_01 --output factor_out --rank 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

import fidaneo
import fit_crli_simple as crli

# =============================================================================
# Configuration
# =============================================================================

ALIGN_AT_T0 = True          # resample every trace onto a common tau = t - t0 grid
N_TAU = 1600                # samples on the common tau grid (<= typical good pts)
TAU_PAD_FRAC = 0.02         # shrink the common support by this fraction each end

NORMALIZE = "peak"          # per-trace scaling before SVD: "peak", "area", or "none"
CENTER = True               # True = PCA (modes of variation); False = raw SVD

N_PERMUTATIONS = 200        # parallel-analysis shuffles for the noise threshold
PERM_PERCENTILE = 95.0      # keep a component if s_i beats this pct of shuffled s_i
PERM_SEED = 0

VARIANCE_TARGET = 0.99      # also report the rank reaching this cumulative variance
MAX_PLOT_MODES = 6          # eigen-Taylorgrams / score curves to draw

NMF_MAX_ITER = 1000
NMF_SEED = 0


# =============================================================================
# Step 1 - load, preprocess, and align the series into a matrix
# =============================================================================

def build_matrix(folder: str | Path):
    """Return ``(X, tau, meta)`` for the aligned, scaled trace matrix.

    ``X`` is ``(n_traces, n_tau)``; ``tau`` is the common time-since-t0 grid;
    ``meta`` carries the per-trace bookkeeping the plots key off (experiment
    time in hours, vial/sample name, t0, peak height).
    """
    records = fidaneo.with_method_of_first(fidaneo.load_folder(folder))
    if len(records) < 3:
        raise ValueError(f"Need at least 3 traces to factorise; found {len(records)}.")
    hours = fidaneo.relative_hours(records)

    preps, kept_hours, names, t0s, peaks = [], [], [], [], []
    for rec, h in zip(records, hours):
        try:
            p = crli.prepare_from_trace(rec.time_s, rec.signal, rec.path)
        except Exception as exc:  # a single bad trace shouldn't sink the batch
            import warnings
            warnings.warn(f"Skipped {Path(rec.path).name}: {exc}")
            continue
        preps.append(p)
        kept_hours.append(h)
        names.append(rec.sample_name)
        t0s.append(p.t0_s)
        peaks.append(p.peak_height)

    if not ALIGN_AT_T0:
        # Stack on the raw sample index; assumes a shared acquisition grid.
        n = min(p.corrected.size for p in preps)
        tau = preps[0].time_s[:n] - np.median(t0s)
        rows = [p.corrected[:n] for p in preps]
    else:
        # Common tau support that needs no extrapolation for any trace.
        lo = max(float(p.time_s[p.good_mask].min() - p.t0_s) for p in preps)
        hi = min(float(p.time_s[p.good_mask].max() - p.t0_s) for p in preps)
        span = hi - lo
        lo, hi = lo + TAU_PAD_FRAC * span, hi - TAU_PAD_FRAC * span
        tau = np.linspace(lo, hi, N_TAU)
        rows = []
        for p in preps:
            g = p.good_mask                      # drops spikes + non-finite points
            tg = p.time_s[g] - p.t0_s
            rows.append(np.interp(tau, tg, p.corrected[g]))

    X = np.vstack(rows)

    # Per-trace scaling: isolate shape changes ("peak"/"area") or keep amplitude.
    if NORMALIZE == "peak":
        scale = np.array(peaks, dtype=float)
    elif NORMALIZE == "area":
        scale = np.trapz(np.clip(X, 0, None), tau, axis=1)
    elif NORMALIZE == "none":
        scale = np.ones(X.shape[0])
    else:
        raise ValueError("NORMALIZE must be 'peak', 'area', or 'none'.")
    scale[scale <= 0] = 1.0
    X = X / scale[:, None]

    meta = {
        "hours": np.array(kept_hours, dtype=float),
        "names": names,
        "t0s": np.array(t0s, dtype=float),
        "peaks": np.array(peaks, dtype=float),
        "scale": scale,
        "n_records": len(records),
    }
    return X, tau, meta


# =============================================================================
# Step 2 - SVD / PCA and the permutation rank test
# =============================================================================

def svd_decompose(X: np.ndarray, center: bool):
    """Thin SVD via numpy. Returns ``(U, s, Vt, mean_row)``."""
    mean_row = X.mean(axis=0) if center else np.zeros(X.shape[1])
    U, s, Vt = np.linalg.svd(X - mean_row, full_matrices=False)
    return U, s, Vt, mean_row


def parallel_analysis(X: np.ndarray, center: bool, n_perm: int, percentile: float, seed: int):
    """Horn-style parallel analysis: significant rank from shuffled-column noise.

    Each column is permuted independently to destroy between-column structure
    while preserving each column's marginal; the SVD of every shuffle gives a
    null spectrum.  A component is "real" if its singular value exceeds
    ``percentile`` of the shuffled singular values at the same index.
    """
    rng = np.random.default_rng(seed)
    Xc = X - (X.mean(axis=0) if center else 0.0)
    k = min(Xc.shape)
    null = np.empty((n_perm, k))
    for i in range(n_perm):
        Xp = np.column_stack([rng.permutation(col) for col in Xc.T])
        if center:
            Xp = Xp - Xp.mean(axis=0)
        null[i] = np.linalg.svd(Xp, compute_uv=False)
    threshold = np.percentile(null, percentile, axis=0)
    observed = np.linalg.svd(Xc, compute_uv=False)
    significant = observed > threshold
    # Rank = leading run of significant components (stop at the first failure).
    rank = int(np.argmax(~significant)) if not significant.all() else int(k)
    return rank, observed, threshold


# =============================================================================
# Step 3 - NMF (scikit-learn)
# =============================================================================

def nmf_decompose(X: np.ndarray, rank: int):
    """Non-negative factorisation X ~= W @ H. Returns ``(W, H, rel_err)``."""
    Xnn = np.clip(X, 0.0, None)          # noise can push baseline-subtracted dips <0
    model = NMF(n_components=rank, init="nndsvda", max_iter=NMF_MAX_ITER,
                random_state=NMF_SEED)
    W = model.fit_transform(Xnn)
    H = model.components_
    rel_err = float(np.linalg.norm(Xnn - W @ H) / np.linalg.norm(Xnn))
    return W, H, rel_err


# =============================================================================
# Plotting
# =============================================================================

def _vial_colors(names):
    uniq = sorted(set(names))
    cmap = plt.get_cmap("tab10")
    lut = {n: cmap(i % 10) for i, n in enumerate(uniq)}
    return [lut[n] for n in names], lut


def plot_scree(s, threshold, rank, out: Path):
    var = s**2 / np.sum(s**2)
    cum = np.cumsum(var)
    k = min(len(s), 20)
    idx = np.arange(1, k + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.semilogy(idx, s[:k], "o-", label="observed")
    ax1.semilogy(idx, threshold[:k], "x--", color="0.5",
                 label=f"perm. noise (p{PERM_PERCENTILE:g})")
    ax1.axvline(rank + 0.5, color="crimson", ls=":", label=f"rank = {rank}")
    ax1.set(xlabel="component", ylabel="singular value", title="Scree + noise floor")
    ax1.legend(fontsize=8)
    captured = cum[rank - 1] if rank >= 1 else 0.0
    ax2.plot(idx, cum[:k], "o-")
    ax2.axvline(rank + 0.5, color="crimson", ls=":",
                label=f"top {rank} = {captured:.1%} of variance")
    ax2.set(xlabel="component", ylabel="cumulative variance", ylim=(0, 1.02),
            title="Information captured (rest is noise tail)")
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_modes(tau, Vt, mean_row, n_modes, out: Path):
    n = min(n_modes, Vt.shape[0])
    fig, axes = plt.subplots(n + 1, 1, figsize=(8, 1.6 * (n + 1)), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(tau, mean_row, color="k")
    axes[0].set_ylabel("mean")
    axes[0].set_title("Eigen-Taylorgrams (right singular vectors)")
    for i in range(n):
        axes[i + 1].plot(tau, Vt[i], color=plt.get_cmap("viridis")(i / max(n - 1, 1)))
        axes[i + 1].axhline(0, color="0.8", lw=0.7)
        axes[i + 1].set_ylabel(f"V{i + 1}")
    axes[-1].set_xlabel("tau = t - t0  (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_scores(hours, names, U, s, n_modes, out: Path):
    n = min(n_modes, U.shape[1])
    scores = U * s  # principal-component scores: weight of each mode per trace
    colors, lut = _vial_colors(names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(n):
        for vial, c in lut.items():
            m = [nm == vial for nm in names]
            axes[i].scatter(hours[m], scores[m, i], s=18, color=c,
                            label=vial if i == 0 else None)
        axes[i].axhline(0, color="0.8", lw=0.7)
        axes[i].set_ylabel(f"PC{i + 1}")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_title("Mode weight vs experiment time (left singular vectors)")
    axes[-1].set_xlabel("experiment time (h)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_nmf(tau, hours, names, W, H, out: Path):
    rank = H.shape[0]
    uniq = sorted(set(names))
    markers = ["o", "s", "^", "D", "v", "P"]
    vial_marker = {v: markers[i % len(markers)] for i, v in enumerate(uniq)}
    comp_cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for i in range(rank):
        axes[0].plot(tau, H[i], color=comp_cmap(i), label=f"component {i + 1}")
    axes[0].set(xlabel="tau = t - t0  (s)", ylabel="shape (H)",
                title=f"NMF species shapes (rank {rank})")
    axes[0].legend(fontsize=8)
    # Colour encodes component; marker encodes vial.
    for i in range(rank):
        c = comp_cmap(i)
        for v in uniq:
            m = [nm == v for nm in names]
            axes[1].scatter(hours[m], W[m, i], s=20, color=c,
                            marker=vial_marker[v])
        axes[1].plot([], [], color=c, marker="o", ls="", label=f"comp {i + 1}")
    for v in uniq:
        axes[1].plot([], [], color="0.4", marker=vial_marker[v], ls="", label=v)
    axes[1].set(xlabel="experiment time (h)", ylabel="weight (W)",
                title="NMF component weight vs time  (colour=component, marker=vial)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

def run(folder: str, output: str, rank_override: int | None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    X, tau, meta = build_matrix(folder)
    print(f"Loaded {meta['n_records']} files -> matrix X = {X.shape} "
          f"(traces x tau)  | NORMALIZE={NORMALIZE}  CENTER={CENTER}  "
          f"ALIGN_AT_T0={ALIGN_AT_T0}")
    print(f"t0 drift across series: {np.ptp(meta['t0s']):.2f} s "
          f"({meta['t0s'].min():.2f}-{meta['t0s'].max():.2f})")

    U, s, Vt, mean_row = svd_decompose(X, CENTER)
    rank, observed, threshold = parallel_analysis(
        X, CENTER, N_PERMUTATIONS, PERM_PERCENTILE, PERM_SEED)
    var = s**2 / np.sum(s**2)
    cum = np.cumsum(var)
    captured = cum[rank - 1] if rank >= 1 else 0.0
    var_rank = int(np.searchsorted(cum, VARIANCE_TARGET) + 1)

    print("\nSingular values (top 8) and noise threshold:")
    for i in range(min(8, len(s))):
        flag = "  <- signal" if i < rank else ""
        print(f"  s[{i + 1}] = {s[i]:11.4g}   noise(p{PERM_PERCENTILE:g}) = "
              f"{threshold[i]:11.4g}   var = {var[i]:6.2%}{flag}")
    print(f"\n>>> Parallel-analysis rank (components above noise): {rank}")
    print(f">>> Top {rank} components capture {captured:.1%} of total variance.")
    print(f"    (Reaching {VARIANCE_TARGET:.0%} needs k={var_rank}, but that tail is "
          f"noise, not {var_rank} real components.)")

    plot_scree(s, threshold, rank, out / "scree.png")
    plot_modes(tau, Vt, mean_row, MAX_PLOT_MODES, out / "eigen_taylorgrams.png")
    plot_scores(meta["hours"], meta["names"], U, s, MAX_PLOT_MODES,
                out / "mode_scores_vs_time.png")

    nmf_rank = rank_override or max(rank, 1)
    W, H, rel_err = nmf_decompose(X, nmf_rank)
    print(f"\nNMF at rank {nmf_rank}: relative reconstruction error = {rel_err:.3%}")
    plot_nmf(tau, meta["hours"], meta["names"], W, H, out / "nmf_components.png")

    print(f"\nWrote plots to {out}/  (scree, eigen_taylorgrams, "
          f"mode_scores_vs_time, nmf_components)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="Folder of FidaNeo .txt traces (a series).")
    ap.add_argument("--output", default="factor_out", help="Output directory.")
    ap.add_argument("--rank", type=int, default=None,
                    help="Override NMF rank (default: parallel-analysis rank).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run(args.folder, args.output, args.rank)


if __name__ == "__main__":
    main()
