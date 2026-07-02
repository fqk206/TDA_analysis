#!/usr/bin/env python3
"""Phase-2 parity check: CRLIModel vs fit_crli_simple.fit_crli.

Run:
    python3 tests/parity/test_phase2_crli.py
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fit_crli_simple as crli  # noqa: E402

from tda.models.crli import CRLIModel, CRLIParams  # noqa: E402


STATS_ATOL = {"__default__": 0.0}


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


def compare_result(old, new, label: str) -> None:
    check(f"{label}: time_fit_s", close(old.time_fit_s, new.time_fit_s))
    check(f"{label}: signal_model_raw", close(old.signal_model_raw, new.signal_model_raw))
    check(f"{label}: residuals_raw", close(old.residuals_raw, new.residuals_raw))
    check(f"{label}: free_signal_raw vs components['free']", close(old.free_signal_raw, new.components["free"]))
    check(f"{label}: colloid_signal_raw vs components['colloid']", close(old.colloid_signal_raw, new.components["colloid"]))
    check(f"{label}: Rh_nm", close(old.Rh_nm, new.Rh_nm))
    check(f"{label}: pdf_logRh", close(old.pdf_logRh, new.pdf_logRh))
    check(f"{label}: cdf", close(old.cdf, new.cdf))

    for key, old_val in old.stats.items():
        new_val = new.stats.get(key)
        ok = close(old_val, new_val, rtol=1e-6, atol=1e-9)
        check(f"{label}: stats[{key}]", ok, f"old={old_val!r} new={new_val!r}")


def run_default(paths):
    """t0_mode='fit' (the real default): expensive (nested Brent search around
    an 80-point alpha scan, minutes per file) but the setting actually used in
    production, so it's tested on one real file rather than skipped entirely -
    t0-refinement plumbing is generic (not file-specific), so one confirmed
    instance validates the refine_t0/Brent integration."""
    model = CRLIModel()
    for path in paths[:1]:
        prep_old = crli.prepare(path)
        prep_new = model.prepare(path)
        old = crli.fit_crli(prep_old, side=crli.FIT_SIDE)
        new = model.fit(prep_new)
        compare_result(old, new, f"default[{path.name}]")


def run_free_fraction_bound_edge(paths):
    """Deliberately tight bounds so the fit_bounded discontinuous branch
    (un-toleranced float comparison in _crli_scan_at_t0) is exercised, and
    confirm old vs new pick the identical branch/solution. t0_mode='auto'
    here to isolate the branch-selection check from the (already-covered,
    expensive) t0 Brent refinement."""
    saved = {k: getattr(crli, k) for k in ("FREE_FRACTION_MODE", "FREE_FRACTION_BOUNDS", "T0_MODE")}
    try:
        crli.FREE_FRACTION_MODE = "fit_bounded"
        crli.FREE_FRACTION_BOUNDS = (0.045, 0.055)
        crli.T0_MODE = "auto"
        params = CRLIParams(free_fraction_mode="fit_bounded", free_fraction_bounds=(0.045, 0.055), t0_mode="auto")
        model = CRLIModel(params=params)
        for path in paths[:2]:
            prep_old = crli.prepare(path)
            prep_new = model.prepare(path)
            old = crli.fit_crli(prep_old, side=crli.FIT_SIDE)
            new = model.fit(prep_new)
            compare_result(old, new, f"tight_bounds[{path.name}]")
            active = new.stats["free_fraction_bound_active"]
            print(f"    bound_active={active} free_fraction={new.stats['free_fraction_detector_weighted']:.4f}")
    finally:
        for k, v in saved.items():
            setattr(crli, k, v)


def run_ridge_and_autonarrow(paths):
    """A second config combo (ridge on, autonarrow on) as a broader smoke
    test; t0_mode='auto' to keep this fast."""
    saved = {k: getattr(crli, k) for k in ("RIDGE_RHO", "RH_GRID_AUTONARROW", "T0_MODE")}
    try:
        crli.RIDGE_RHO = 0.05
        crli.RH_GRID_AUTONARROW = True
        crli.T0_MODE = "auto"
        params = CRLIParams(ridge_rho=0.05, rh_grid_autonarrow=True, t0_mode="auto")
        model = CRLIModel(params=params)
        for path in paths[:1]:
            prep_old = crli.prepare(path)
            prep_new = model.prepare(path)
            old = crli.fit_crli(prep_old, side=crli.FIT_SIDE)
            new = model.fit(prep_new)
            compare_result(old, new, f"ridge_autonarrow[{path.name}]")
    finally:
        for k, v in saved.items():
            setattr(crli, k, v)


def main() -> None:
    paths = sorted(Path(p) for p in glob(str(REPO_ROOT / "data" / "*.txt")))
    assert paths, "expected files under data/*.txt"

    print("run_default")
    run_default(paths)

    print("run_free_fraction_bound_edge")
    run_free_fraction_bound_edge(paths)

    print("run_ridge_and_autonarrow")
    run_ridge_and_autonarrow(paths)

    print("\nAll phase-2 CRLIModel parity checks passed.")


if __name__ == "__main__":
    main()
