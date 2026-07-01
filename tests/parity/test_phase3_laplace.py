#!/usr/bin/env python3
"""Phase-3 parity check: LaplaceModel vs fit_laplace.invert.

Run:
    python3 tests/parity/test_phase3_laplace.py
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fit_crli_simple as crli  # noqa: E402
import fit_laplace  # noqa: E402

from tda.models.laplace import LaplaceModel  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{(' - ' + detail) if (not ok and detail) else ''}")
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def close(a, b, rtol=1e-9, atol=1e-12) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        return False
    both_nan = np.isnan(a_arr) & np.isnan(b_arr)
    return bool(np.all(both_nan | np.isclose(a_arr, b_arr, rtol=rtol, atol=atol)))


def main() -> None:
    # t0_mode='fit' (default) nests a nontrivial Brent search around an
    # 80-point alpha scan (minutes per file on some traces), so this checks
    # one real file - t0-refinement plumbing is generic (not file-specific).
    paths = sorted(Path(p) for p in glob(str(REPO_ROOT / "data" / "*.txt")))[:1]
    assert paths, "expected files under data/*.txt"

    model = LaplaceModel()
    for path in paths:
        prep_old = crli.prepare(path)
        prep_new = model.prepare(path)
        old = fit_laplace.invert(prep_old, side=crli.FIT_SIDE)
        new = model.fit(prep_new)

        label = path.name
        check(f"{label}: t", close(old["t"], new.time_fit_s))
        check(f"{label}: model_raw", close(old["model_raw"], new.signal_model_raw))
        check(f"{label}: s_low vs components['low']", close(old["peak_height"] * old["s_low"], new.components["low"]))
        check(f"{label}: s_high vs components['high']", close(old["peak_height"] * old["s_high"], new.components["high"]))
        check(f"{label}: Rh", close(old["Rh"], new.Rh_nm))
        check(f"{label}: pdf_log", close(old["pdf_log"], new.pdf_logRh))
        check(f"{label}: cdf", close(old["cdf"], new.cdf))

        for key, old_val in old["stats"].items():
            new_val = new.stats.get(key)
            check(f"{label}: stats[{key}]", close(old_val, new_val, rtol=1e-6, atol=1e-9), f"old={old_val!r} new={new_val!r}")

    print("\nAll phase-3 LaplaceModel parity checks passed.")


if __name__ == "__main__":
    main()
