#!/usr/bin/env python3
"""Free-fluorophore size diagnostic for the CRLI fit.

Purpose
-------
The truncation test in ``fit_crli_simple.py`` shows that the sub-cutoff (small
Rh) signal in this data is real - it sits well above the noise floor - but Taylor
dispersion cannot cleanly separate a small free fluorophore (fixed at
``FREE_RH_NM``) from a slightly larger 1-4 nm colloid mode: the two are
degenerate at the peak apex.  This script asks one focused question:

    Can a single, slightly larger free-fluorophore size explain that
    small-species signal on its own?

For a range of assumed free sizes it refits the trace and, for each, reports the
overall misfit, the residual concentrated at the peak apex, and how much signal
is still left in the colloid distribution below the truncation cutoff.

    - If one free size near the bare-dye value flattens the apex residual *and*
      drains the leftover sub-cutoff colloid signal, the mode is simply the dye
      appearing slightly larger than its nominal Rh.
    - If the apex misfit and leftover signal persist across all free sizes, that
      points to a distinct small population (or a peak-shape model limitation) -
      i.e. something to understand about the sample, not a fit setting to tweak.

The free fraction is fit *unbounded* here so each candidate size can claim as
much signal as the data support, independent of the bounds in
``fit_crli_simple.py``.  All other settings (data pattern, grid, alpha scan,
truncation cutoff) are reused from that module so this stays a thin diagnostic.

Run:
    python diagnose_free_size.py
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fit_crli_simple as crli


# Assumed free-fluorophore sizes to test (nm).  The first is the nominal value.
FREE_RH_SCAN_NM = [0.59, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
APEX_SIGNAL_MIN = 0.85               # "apex" = normalized signal at/above this level
SCAN_OUTPUT_DIR = "output/free_size_scan"


def apex_resid_over_noise(result, prep) -> float:
    """RMS fit residual in the apex band, expressed in units of the noise."""
    s_fit = result.extra["s_fit"]
    s_model = result.extra["s_model"]
    apex = s_fit >= APEX_SIGNAL_MIN
    if not np.any(apex):
        return float("nan")
    rms_norm = float(np.sqrt(np.mean((s_model[apex] - s_fit[apex]) ** 2)))
    noise_norm = prep.noise_raw / prep.peak_height
    return rms_norm / noise_norm if noise_norm > 0 else float("nan")


def scan_file(path: Path) -> None:
    prep = crli.prepare(path)
    stem = crli.safe_filename(path.stem)
    cutoff = crli.TRUNCATION_RH_NM
    noise_norm = prep.noise_raw / prep.peak_height

    rows = []
    dists = []
    for free_rh in FREE_RH_SCAN_NM:
        crli.FREE_RH_NM = free_rh
        result = crli.fit_crli(prep, side=crli.FIT_SIDE)
        s = result.stats
        dists.append((free_rh, result.Rh_nm, result.pdf_logRh))
        rows.append({
            "free_Rh_nm": free_rh,
            "free_pct": 100 * s["free_fraction_detector_weighted"],
            "overall_over_noise": s["residual_rms_normalized_to_peak"] / noise_norm,
            "apex_over_noise": apex_resid_over_noise(result, prep),
            "sub_mass_pct": 100 * s["trunc_mass_fraction_below"],
            "sub_sig_rms_over_noise": s["trunc_signal_rms_over_noise"],
        })

    print(f"\n{path.name}   (cutoff = {cutoff:g} nm, noise/peak = {noise_norm:.5f})")
    header = ("free_Rh", "free%", "overall/noise", "apex/noise", f"mass<{cutoff:g}nm%", f"sub-{cutoff:g}nm sig/noise")
    print("  {:>7} {:>6} {:>13} {:>10} {:>11} {:>17}".format(*header))
    for r in rows:
        print("  {free_Rh_nm:7.2f} {free_pct:6.2f} {overall_over_noise:13.2f} "
              "{apex_over_noise:10.2f} {sub_mass_pct:11.2f} {sub_sig_rms_over_noise:17.2f}".format(**r))

    best = min(rows, key=lambda r: r["apex_over_noise"])
    print(f"  -> lowest apex misfit at free_Rh = {best['free_Rh_nm']:.2f} nm "
          f"(apex {best['apex_over_noise']:.2f}x noise, leftover sub-{cutoff:g}nm signal "
          f"{best['sub_sig_rms_over_noise']:.2f}x noise)")

    _save_plot(stem, rows, cutoff)
    _save_distribution_plot(stem, dists, cutoff)


def _save_plot(stem: str, rows: list[dict], cutoff: float) -> None:
    x = [r["free_Rh_nm"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(x, [r["apex_over_noise"] for r in rows], "o-", label="apex residual / noise")
    plt.plot(x, [r["sub_sig_rms_over_noise"] for r in rows], "s-", label=f"leftover sub-{cutoff:g}nm signal / noise")
    plt.plot(x, [r["overall_over_noise"] for r in rows], "^-", label="overall residual / noise")
    plt.axhline(1.0, linestyle="--", linewidth=0.8, color="gray", label="noise level")
    plt.xlabel(r"assumed free fluorophore $R_h$ / nm")
    plt.ylabel("residual / leftover signal  (units of noise)")
    plt.title(f"{stem}: free-fluorophore size scan")
    plt.legend(fontsize=8)
    out = Path(SCAN_OUTPUT_DIR) / f"{stem}_free_size_scan.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def _save_distribution_plot(stem: str, dists: list[tuple], cutoff: float) -> None:
    """Overlay the recovered colloid distributions, coloured by assumed free size."""
    plt.figure(figsize=(8, 5))
    colors = plt.cm.jet(np.linspace(0, 1, len(dists)))
    for (free_rh, Rh, pdf), color in zip(dists, colors):
        plt.semilogx(Rh, pdf, "-", color=color, label=f"{free_rh:.2f} nm")
    plt.axvline(cutoff, linestyle=":", linewidth=0.8, color="gray", label=f"cutoff {cutoff:g} nm")
    plt.xlabel(r"$R_h$ / nm")
    plt.ylabel(r"colloid PDF per $\ln R_h$")
    plt.title(f"{stem}: colloid distribution vs assumed free size")
    plt.legend(title=r"free $R_h$", fontsize=8)
    out = Path(SCAN_OUTPUT_DIR) / f"{stem}_free_size_distributions.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    crli.FREE_FRACTION_MODE = "fit"   # unbounded free fraction for a clean size scan
    files = sorted(Path(p) for p in glob(crli.DATA_PATTERN, recursive=True) if Path(p).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {crli.DATA_PATTERN!r}")
    for path in files:
        scan_file(path)
    print(f"\nSaved scan plots under {Path(SCAN_OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
