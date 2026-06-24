#!/usr/bin/env python3
"""Pure Laplace / CRLI inversion of a Taylorgram - no separate free-fluorophore term.

Where ``fit_crli_simple.py`` models the free fluorophore as a fixed-Rh component
bolted onto a colloid grid, this script does the opposite: it solves for a single
smooth, non-negative size distribution over a grid that reaches down to very
small Rh, and lets the inversion recover the small (free-fluorophore-like)
species on its own - because, as the truncation test and the two-Gaussian
simulator showed, that signal really is in the data.

It is the same constrained regularized linear inversion (NNLS + second-difference
smoothness + an alpha scan choosing the smoothest fit within a residual
tolerance) as the CRLI fitter, with two differences:

    1. No free-fluorophore column, no free-fraction logic or bounds.
    2. The grid starts near zero (RH_MIN_NM, default 0.3 nm) so the dye-sized
       species falls inside the distribution rather than being handled specially.

Note: the Taylor kernel contains 1/Rh and is singular at exactly Rh = 0, so the
grid must start at a small positive value, not literally zero.

The reading/preprocessing, kernel and helpers are reused from fit_crli_simple, so
the data pattern, time unit and physical constants all come from that one place.

Run:
    python fit_laplace.py
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

import fit_crli_simple as crli


# Fraction of the peak fit (rising side): 0.50 = rising half, 1.00 = whole peak.
PEAK_FRACTION = 0.51

# Inversion grid - starts near zero so the small species sits inside the grid.
RH_MIN_NM = 0.2
RH_MAX_NM = 200.0
N_RH_GRID = 160

# Cutoff used only for reporting/plotting the emergent "small species" component.
SMALL_CUTOFF_NM = 2.0
REFERENCE_FREE_RH_NM = 0.59          # drawn on the distribution plot for orientation

OUTPUT_DIR = "output/laplace"


def invert(prep, side: str) -> dict:
    T_K = 273.15 + crli.TEMPERATURE_C
    t0 = prep.t0_s
    indices = np.flatnonzero(crli.crli_fit_mask(prep, side, PEAK_FRACTION))[:: max(1, int(crli.CRLI_FIT_STRIDE))]
    if indices.size < 25:
        raise ValueError("Too few fitting points for the inversion.")

    t = prep.time_s[indices]
    y = prep.signal[indices]
    s = prep.normalized[indices]

    Rh = np.geomspace(RH_MIN_NM, RH_MAX_NM, N_RH_GRID)
    wx = crli.trapezoid_weights(np.log(Rh))
    A = crli.rh_kernel(t, Rh, t0, T_K) * wx[None, :]
    L = crli.second_difference_matrix(N_RH_GRID)

    solutions, residual_rms, roughness = [], [], []
    for alpha in crli.CRLI_ALPHAS:
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
    acceptable = np.flatnonzero(residual_rms <= crli.CRLI_RESIDUAL_FACTOR * best_rms)
    chosen = int(acceptable[-1]) if acceptable.size else int(np.argmin(residual_rms))

    q = solutions[chosen]
    area = float(np.sum(q * wx))
    if area <= 0:
        raise ValueError("Inversion recovered zero distribution area.")
    pdf_log = q / area
    pdf_R = pdf_log / Rh
    cdf = np.cumsum(pdf_log * wx)
    cdf = cdf / cdf[-1]

    s_model = A @ q
    baseline = np.interp(t, prep.time_s, prep.baseline)
    model_raw = baseline + prep.peak_height * s_model

    # Emergent small-species component (mass below the cutoff), for reporting/plots.
    low = Rh < SMALL_CUTOFF_NM
    s_low = A[:, low] @ q[low]
    s_high = A[:, ~low] @ q[~low]
    small_area_fraction = float(np.sum(q[low] * wx[low]) / area)

    stats = {
        "success": True,
        "fit_side": side,
        "peak_fraction": float(PEAK_FRACTION),
        "t0_s": float(t0),
        "noise_raw": float(prep.noise_raw),
        "n_fit_points": int(t.size),
        "rh_min_nm": float(RH_MIN_NM),
        "rh_max_nm": float(RH_MAX_NM),
        "small_cutoff_nm": float(SMALL_CUTOFF_NM),
        "small_area_fraction_below_cutoff": small_area_fraction,
        "residual_rms_normalized_to_peak": float(np.sqrt(np.mean((s_model - s) ** 2))),
        "residual_rms_raw": float(np.sqrt(np.mean((model_raw - y) ** 2))),
        "crli_alpha": float(crli.CRLI_ALPHAS[chosen]),
        "crli_alpha_at_scan_boundary": chosen in {0, len(crli.CRLI_ALPHAS) - 1},
    }
    stats.update({f"all_{k}": v for k, v in crli.distribution_stats(Rh, pdf_log).items()})
    stats.update({f"above_cutoff_{k}": v for k, v in crli.distribution_stats(Rh[~low], pdf_log[~low]).items()})

    return {
        "t": t, "y": y, "s": s, "s_model": s_model, "model_raw": model_raw, "baseline": baseline,
        "s_low": s_low, "s_high": s_high, "peak_height": prep.peak_height,
        "Rh": Rh, "pdf_log": pdf_log, "pdf_R": pdf_R, "cdf": cdf,
        "alpha_scan": crli.CRLI_ALPHAS, "residual_scan": residual_rms, "roughness_scan": roughness,
        "stats": stats,
    }


def save_plots(res: dict, out_dir: Path, stem: str) -> None:
    t, ph = res["t"], res["peak_height"]
    base = res["baseline"]

    plt.figure(figsize=(9, 5))
    plt.plot(t, res["y"], ".", markersize=3, label="fit data")
    plt.plot(t, res["model_raw"], "-", label="inversion model")
    plt.xlabel("time / s"); plt.ylabel("raw signal")
    plt.title(f"{stem}: Laplace inversion fit")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_laplace_fit.svg")

    plt.figure(figsize=(9, 4))
    plt.plot(t, (res["model_raw"] - res["y"]) / ph, ".", markersize=3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("time / s"); plt.ylabel("residual / peak height")
    plt.title(f"{stem}: residuals")
    crli.save_current_figure(out_dir / f"{stem}_laplace_residuals.svg")

    plt.figure(figsize=(9, 5))
    plt.plot(t, res["model_raw"], "-", color="k", label="total model")
    plt.plot(t, base + ph * res["s_low"], ":", color="C3", label=f"$R_h<{SMALL_CUTOFF_NM:g}$ nm component")
    plt.plot(t, base + ph * res["s_high"], "-.", color="C0", label=f"$R_h\\geq{SMALL_CUTOFF_NM:g}$ nm component")
    plt.plot(t, base, "--", color="gray", label="baseline")
    plt.xlabel("time / s"); plt.ylabel("raw signal")
    plt.title(f"{stem}: decomposition (emergent small species)")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_laplace_decomposition.svg")

    plt.figure(figsize=(7, 5))
    plt.semilogx(res["Rh"], res["pdf_log"], "-")
    plt.axvline(REFERENCE_FREE_RH_NM, linestyle=":", color="C3", label=f"dye ref {REFERENCE_FREE_RH_NM:g} nm")
    plt.axvline(SMALL_CUTOFF_NM, linestyle="--", color="gray", linewidth=0.8, label=f"cutoff {SMALL_CUTOFF_NM:g} nm")
    plt.xlabel(r"$R_h$ / nm"); plt.ylabel(r"PDF per $\ln R_h$")
    plt.title(f"{stem}: recovered size distribution")
    plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_laplace_distribution.svg")

    plt.figure(figsize=(6, 4))
    plt.loglog(res["alpha_scan"], res["residual_scan"], ".-")
    plt.axvline(res["stats"]["crli_alpha"], linestyle="--", label="chosen alpha")
    plt.xlabel(r"regularization $\alpha$"); plt.ylabel("normalized residual RMS")
    plt.title(f"{stem}: alpha scan"); plt.legend(fontsize=8)
    crli.save_current_figure(out_dir / f"{stem}_laplace_alpha_scan.svg")


def write_distribution_csv(res: dict, out_dir: Path, stem: str) -> None:
    rows = [
        {"Rh_nm": float(rh), "pdf_per_ln_Rh": float(p), "pdf_per_nm": float(pr), "cdf": float(c)}
        for rh, p, pr, c in zip(res["Rh"], res["pdf_log"], res["pdf_R"], res["cdf"])
    ]
    crli.write_rows(out_dir / f"{stem}_laplace_distribution.csv", rows)


def main() -> None:
    files = sorted(Path(p) for p in glob(crli.DATA_PATTERN, recursive=True) if Path(p).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {crli.DATA_PATTERN!r}")
    root = Path(OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} file(s). Output folder: {root.resolve()}")

    rows = []
    for path in files:
        print(f"\nAnalysing {path}", flush=True)
        try:
            prep = crli.prepare(path)
            stem = crli.safe_filename(path.stem)
            if crli.SAVE_PLOTS:
                crli.save_preprocessing_plots(prep, root, stem)
            res = invert(prep, side=crli.FIT_SIDE)
            write_distribution_csv(res, root, stem)
            if crli.SAVE_PLOTS:
                save_plots(res, root, stem)
            rows.append({"file": str(path), **res["stats"]})
            s = res["stats"]
            print(f"  t0={s['t0_s']:.1f}s | small(<{SMALL_CUTOFF_NM:g}nm)={100*s['small_area_fraction_below_cutoff']:.2f}% "
                  f"| colloid median Rh={s['above_cutoff_colloid_median_Rh_nm']:.2f}nm | RMS/peak={s['residual_rms_normalized_to_peak']:.4g}")
        except Exception as exc:
            warnings.warn(f"Skipped {path.name}: {exc}")
            rows.append({"file": str(path), "success": False, "error": str(exc)})

    crli.write_rows(root / "laplace_summary.csv", rows)
    print(f"\nSaved summary: {root / 'laplace_summary.csv'}\nDone.")


if __name__ == "__main__":
    main()
