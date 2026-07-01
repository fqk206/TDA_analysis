#!/usr/bin/env python3
"""Forward two-Gaussian Taylorgram simulator (an intuition-builder).

Purpose
-------
Taylor dispersion signals are *additive*: the detector sees the sum of every
species' contribution, and the per-species contribution is the species' amount
times the Taylor kernel for its size.  The catch is that the kernel is tall and
narrow for small Rh and short and broad for large Rh, so a tiny mass fraction of
a small species can contribute a surprisingly large signal near the peak apex -
which is exactly what makes a small free-fluorophore-like population easy to
confuse with a fit artefact.

This script makes that concrete.  You fix two Gaussian species - a small one and
a large one - each by its hydrodynamic radius (nm), width (nm), and fraction,
and it plots the total Taylorgram together with the individual component
signals.  Pass a list of small-species sizes (e.g. 0.59 and 0.80 nm) to see, at
the same small mass fraction, how differently they show up in the signal.

This is a *forward model*: nothing is fitted, you set the parameters and look at
the result.  The kernel and physical constants are reused from
``tda.physics`` so the signals are on the same footing as the fitters.

Definition of "fraction": each species' distribution is normalised to unit area
in Rh, then scaled by its fraction; the signal it produces is
``fraction * (kernel integrated over that distribution)``.  Equal fractions at
different Rh therefore give different signal heights - that is the whole point.

Run:
    python simulate_two_gaussian.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tda.physics import rh_kernel


# --- Time axis (a realistic Taylorgram window; absolute values just set widths) ---
T0_S = 426.0
TIME_HALF_WINDOW_S = 300.0
N_TIME = 1500
TEMPERATURE_C = 25.0                 # matches fit_crli_simple.py's default

# --- Large species (the main colloid peak), fixed ---
LARGE_RH_NM = 23.0
LARGE_SD_NM = 5.0
LARGE_FRACTION = 0.97

# --- Small species: sizes to compare, at a fixed fraction ---
SMALL_RH_NM = [0.59, 0.80]           # list of small sizes to compare
SMALL_SD_NM = 0.05
SMALL_FRACTION = 0.03                # fraction of the small species (try 0.02 - 0.04)

# --- Rh integration grid and output ---
RH_GRID_MIN_NM = 0.3
RH_GRID_MAX_NM = 300.0
N_RH_GRID = 600
OUTPUT_DIR = "output/two_gaussian_sim"


def normal_pdf_Rh(Rh: np.ndarray, center: float, sd: float) -> np.ndarray:
    """Gaussian in Rh (nm), normalised to unit area over the grid."""
    g = np.exp(-0.5 * ((Rh - center) / sd) ** 2)
    g[Rh <= 0] = 0.0
    return g / np.trapezoid(g, Rh)


def species_signal(t: np.ndarray, Rh: np.ndarray, center: float, sd: float,
                   fraction: float, T_K: float) -> np.ndarray:
    """Signal contributed by one Gaussian species = fraction * kernel-over-distribution."""
    pdf = normal_pdf_Rh(Rh, center, sd)
    K = rh_kernel(t, Rh, T0_S, T_K)             # (n_time, n_Rh)
    return fraction * np.trapezoid(K * pdf[None, :], Rh, axis=1)


def save_decomposition_plot(small_rh: float, t: np.ndarray, Rh: np.ndarray,
                            large_sig: np.ndarray, small_sig: np.ndarray, norm: float) -> None:
    total = large_sig + small_sig
    fig, (axd, axs) = plt.subplots(1, 2, figsize=(12, 5))

    axd.semilogx(Rh, LARGE_FRACTION * normal_pdf_Rh(Rh, LARGE_RH_NM, LARGE_SD_NM), "-",
                 color="C0", label=f"large {LARGE_RH_NM:g} nm ({100*LARGE_FRACTION:g}%)")
    axd.semilogx(Rh, SMALL_FRACTION * normal_pdf_Rh(Rh, small_rh, SMALL_SD_NM), "-",
                 color="C3", label=f"small {small_rh:g} nm ({100*SMALL_FRACTION:g}%)")
    axd.set_xlabel(r"$R_h$ / nm")
    axd.set_ylabel("scaled distribution (per nm)")
    axd.set_title("species distributions")
    axd.legend(fontsize=8)

    axs.plot(t, total / norm, "-", color="k", label="total")
    axs.plot(t, large_sig / norm, "--", color="C0", label="large component")
    axs.plot(t, small_sig / norm, "-", color="C3", label="small component")
    axs.set_xlabel("time / s")
    axs.set_ylabel("signal (large peak = 1)")
    peak_pct = 100 * np.max(small_sig) / np.max(total)
    axs.set_title(f"small = {small_rh:g} nm: {100*SMALL_FRACTION:g}% of mass -> "
                  f"{peak_pct:.1f}% of the peak signal")
    axs.legend(fontsize=8)

    out = Path(OUTPUT_DIR) / f"two_gaussian_small_{small_rh:g}nm.svg".replace(".", "p", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_comparison_plot(t: np.ndarray, results: list[tuple], norm: float) -> None:
    colors = plt.cm.jet(np.linspace(0, 1, len(results)))
    fig, (axt, axs) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for (small_rh, large_sig, small_sig), color in zip(results, colors):
        axt.plot(t, (large_sig + small_sig) / norm, "-", color=color, label=f"{small_rh:g} nm")
        axs.plot(t, small_sig / norm, "-", color=color, label=f"{small_rh:g} nm")
    axt.set_ylabel("total signal (large peak = 1)")
    axt.set_title(f"Two-Gaussian totals vs small-species size ({100*SMALL_FRACTION:g}% mass)")
    axt.legend(title=r"small $R_h$", fontsize=8)
    axs.set_xlabel("time / s")
    axs.set_ylabel("small component")
    axs.set_title("small-species signal alone")
    axs.legend(title=r"small $R_h$", fontsize=8)
    out = Path(OUTPUT_DIR) / "two_gaussian_small_size_comparison.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    T_K = 273.15 + TEMPERATURE_C
    t = np.linspace(T0_S - TIME_HALF_WINDOW_S, T0_S + TIME_HALF_WINDOW_S, N_TIME)
    Rh = np.geomspace(RH_GRID_MIN_NM, RH_GRID_MAX_NM, N_RH_GRID)

    large_sig = species_signal(t, Rh, LARGE_RH_NM, LARGE_SD_NM, LARGE_FRACTION, T_K)
    norm = float(np.max(large_sig))

    print(f"Large species: {LARGE_RH_NM:g} nm, {100*LARGE_FRACTION:g}% mass")
    print(f"{'small Rh':>9} {'mass %':>7} {'small peak':>11} {'% of total peak':>16}")
    results = []
    for small_rh in SMALL_RH_NM:
        small_sig = species_signal(t, Rh, small_rh, SMALL_SD_NM, SMALL_FRACTION, T_K)
        total = large_sig + small_sig
        peak_pct = 100 * np.max(small_sig) / np.max(total)
        print(f"{small_rh:9.2f} {100*SMALL_FRACTION:7.1f} {np.max(small_sig)/norm:11.3f} {peak_pct:16.1f}")
        save_decomposition_plot(small_rh, t, Rh, large_sig, small_sig, norm)
        results.append((small_rh, large_sig, small_sig))

    save_comparison_plot(t, results, norm)
    print(f"\nSaved plots under {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
