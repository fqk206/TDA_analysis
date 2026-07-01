#!/usr/bin/env python3
"""Small, config-free output helpers shared by every model's batch runner.

Moved out of ``fit_crli_simple.py`` (``safe_filename``, ``write_rows``,
``save_current_figure``) plus one new generic helper, ``write_distribution_csv``,
that works for any model because every :class:`~tda.models.result.FitResult`
carries the same ``Rh_nm``/``pdf_logRh``/``pdf_Rh_per_nm``/``cdf`` fields.
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
