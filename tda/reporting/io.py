#!/usr/bin/env python3
"""Small, config-free output helpers shared by every model's batch runner.

Moved out of ``fit_crli_simple.py`` (``safe_filename``, ``write_rows``,
``save_current_figure``) plus two generic helpers, ``write_distribution_csv``
and ``write_decomposition_csv``, that work for any model because every
:class:`~tda.models.result.FitResult` carries the same
``Rh_nm``/``pdf_logRh``/``pdf_Rh_per_nm``/``cdf`` and
``time_fit_s``/``signal_fit_raw``/``signal_model_raw``/``baseline_model_raw``/
``components`` fields.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt

from tda.models.result import FitResult


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries to CSV without requiring pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_current_figure(path: Path, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close()


def write_distribution_csv(result: FitResult, out_dir: Path, stem: str, suffix: str) -> None:
    rows = [
        {"Rh_nm": float(rh), "pdf_per_ln_Rh": float(p), "pdf_per_nm": float(pr), "cdf": float(c)}
        for rh, p, pr, c in zip(result.Rh_nm, result.pdf_logRh, result.pdf_Rh_per_nm, result.cdf)
    ]
    write_rows(out_dir / f"{stem}_{suffix}_distribution.csv", rows)


def write_decomposition_csv(result: FitResult, out_dir: Path, stem: str, suffix: str,
                             component_labels: dict[str, str] | None = None) -> None:
    """Time, raw data, total model, baseline + each named component
    (``result.components``), and baseline - the signal-space counterpart to
    ``write_distribution_csv``. Column set follows whatever components the
    model actually produced (matches ``plot_decomposition``'s ``components``
    loop), so e.g. a model with no free-fluorophore term simply has no such
    column. ``component_labels`` renames a component's column (default: the
    component's key in ``result.components``, e.g. ``"colloid"``)."""
    labels = component_labels or {}
    t, y, model, base = result.time_fit_s, result.signal_fit_raw, result.signal_model_raw, result.baseline_model_raw
    component_columns = {
        f"baseline_{safe_filename(labels.get(name, name)).lower()}": base + comp
        for name, comp in result.components.items()
    }
    rows = [
        {
            "time_s": float(t[i]),
            "raw_data": float(y[i]),
            "total_model": float(model[i]),
            **{col: float(arr[i]) for col, arr in component_columns.items()},
            "baseline": float(base[i]),
        }
        for i in range(len(t))
    ]
    write_rows(out_dir / f"{stem}_{suffix}_decomposition.csv", rows)
