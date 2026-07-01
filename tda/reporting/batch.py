#!/usr/bin/env python3
"""Generic batch runner: glob -> prepare -> fit -> save -> summary CSV.

Replaces the four near-identical ``run()``/``main()`` loops that used to live
at the bottom of ``fit_crli_simple.py``, ``fit_laplace.py``,
``fit_parametric.py`` and ``fit_global_series.py``. ``docs/simpl.md`` (the
prior refactor) deliberately removed a callable-injection batch abstraction
because there was exactly one model in scope at the time ("with one model
these all collapse to straight-line code"). With four real models sharing
~90% of this loop and differing only in their plot set, the calculus flips:
the shared runner earns its keep, with the model-specific plot/CSV/print
functions from :mod:`tda.reporting.plots` injected in.
"""

from __future__ import annotations

from glob import glob
import json
from pathlib import Path
from typing import Callable
import warnings

from tda.models.base import SingleTraceModel
from tda.models.result import FitResult
from tda.preprocessing.pipeline import PreparedTrace
from tda.reporting.io import safe_filename, write_rows
from tda.reporting.plots import plot_preprocessing


def run_batch(
    model: SingleTraceModel,
    data_pattern: str,
    output_dir: str,
    summary_name: str,
    save_plots: Callable[[FitResult, PreparedTrace, Path, str], None],
    write_csv: Callable[[FitResult, Path, str], None],
    print_result: Callable[[str, FitResult], None] | None = None,
    save_plots_enabled: bool = True,
) -> None:
    files = sorted(Path(p) for p in glob(data_pattern, recursive=True) if Path(p).is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {data_pattern!r}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_settings.json").write_text(json.dumps(model.config_dict(), indent=2, default=str), encoding="utf-8")
    print(f"Found {len(files)} file(s). Output folder: {root.resolve()}")

    rows: list[dict] = []
    for path in files:
        print(f"\nAnalysing {path}", flush=True)
        try:
            prepared = model.prepare(path)
            stem = safe_filename(path.stem)
            if save_plots_enabled:
                plot_preprocessing(prepared, root, stem)
            result = model.fit(prepared)
            write_csv(result, root, stem)
            if save_plots_enabled:
                save_plots(result, prepared, root, stem)
            rows.append({"file": str(path), **result.stats})
            if print_result is not None:
                print_result(stem, result)
        except Exception as exc:
            warnings.warn(f"Skipped {path.name}: {exc}")
            rows.append({"file": str(path), "success": False, "error": str(exc)})

    summary_path = root / summary_name
    write_rows(summary_path, rows)
    print(f"\nSaved summary: {summary_path}\nDone.")


def print_crli_result(stem: str, result: FitResult) -> None:
    s = result.stats
    print(
        f"  {stem:30s} | "
        f"t0={s['t0_s']:.2f} s | "
        f"free={100 * s['free_fraction_detector_weighted']:.2f}% | "
        f"median Rh={s['colloid_median_Rh_nm']:.2f} nm | "
        f"mean Rh={s['colloid_mean_Rh_nm']:.2f} nm | "
        f"RMS/peak={s['residual_rms_normalized_to_peak']:.4g}",
        flush=True,
    )


def print_laplace_result(stem: str, result: FitResult) -> None:
    s = result.stats
    print(f"  t0={s['t0_s']:.1f}s | small(<{s['small_cutoff_nm']:g}nm)={100 * s['small_area_fraction_below_cutoff']:.2f}% "
          f"| colloid median Rh={s['above_cutoff_colloid_median_Rh_nm']:.2f}nm "
          f"| RMS/peak={s['residual_rms_normalized_to_peak']:.4g}")


def print_parametric_result(stem: str, result: FitResult) -> None:
    s = result.stats
    comps = " | ".join(
        f"c{i}: Rh={s[f'comp{i}_Rh_nm']:.2f}nm sig={s[f'comp{i}_sigma_ln']:.3f} f={100 * s[f'comp{i}_fraction']:.1f}%"
        for i in range(s["n_components"])
    )
    print(f"  {stem:28s} | t0={s['t0_s']:.1f}s (dt={s['t0_shift_s']:+.2f}) | {comps} "
          f"| RMS/noise={s['residual_rms_over_noise']:.2f}", flush=True)
