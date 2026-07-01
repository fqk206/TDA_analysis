#!/usr/bin/env python3
"""Phase-4 parity check: ParametricModel vs fit_parametric.fit_parametric.

Also confirms design_matrix/solve_amplitudes/ParamPack work standalone
(without instantiating ParametricModel), since GlobalSeriesModel needs that.

Run:
    python3 tests/parity/test_phase4_parametric.py
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fit_crli_simple as crli  # noqa: E402
import fit_parametric  # noqa: E402

from tda.models.parametric import ParametricModel, ParamPack, design_matrix, solve_amplitudes  # noqa: E402
from tda.physics import PhysicsParams  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{(' - ' + detail) if (not ok and detail) else ''}")
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def close(a, b, rtol=1e-6, atol=1e-9) -> bool:
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
    paths = sorted(Path(p) for p in glob(str(REPO_ROOT / "data" / "*.txt")))[:2]
    assert paths, "expected files under data/*.txt"

    model = ParametricModel()
    for path in paths:
        prep_old = crli.prepare(path)
        prep_new = model.prepare(path)
        old = fit_parametric.fit_parametric(prep_old, side=crli.FIT_SIDE)
        new = model.fit(prep_new)

        label = path.name
        check(f"{label}: t", close(old["t"], new.time_fit_s, rtol=1e-9, atol=1e-12))
        check(f"{label}: model_raw", close(old["model_raw"], new.signal_model_raw))
        check(f"{label}: Rh", close(old["Rh"], new.Rh_nm, rtol=1e-9, atol=1e-12))
        check(f"{label}: pdf_log", close(old["pdf_log"], new.pdf_logRh))
        check(f"{label}: cdf", close(old["cdf"], new.cdf))
        for i in range(old["stats"]["n_components"]):
            old_comp = old["peak_height"] * old["comp_signal"][:, i]
            new_comp = new.components[f"comp{i}"]
            check(f"{label}: comp{i} signal", close(old_comp, new_comp))

        for key, old_val in old["stats"].items():
            new_val = new.stats.get(key)
            check(f"{label}: stats[{key}]", close(old_val, new_val), f"old={old_val!r} new={new_val!r}")

    print("standalone design_matrix/solve_amplitudes/ParamPack (no ParametricModel instance)")
    prep = crli.prepare(paths[0])
    t = prep.time_s[:50]
    Rh = np.geomspace(0.3, 200.0, 100)
    wx = crli.trapezoid_weights(np.log(Rh))
    T_K = 273.15 + crli.TEMPERATURE_C
    pack = ParamPack(ln_mu=np.log([0.75, 20.0]), ln_sigma=np.log([0.1, 0.3]), t0_s=prep.t0_s)
    C_old = fit_parametric.design_matrix(pack, t, Rh, wx, T_K)
    C_new = design_matrix(pack, t, Rh, wx, T_K, PhysicsParams())
    check("standalone design_matrix", close(C_old, C_new, rtol=1e-9, atol=1e-12))
    s = prep.normalized[:50]
    a_old = fit_parametric.solve_amplitudes(C_old, s)
    a_new = solve_amplitudes(C_new, s)
    check("standalone solve_amplitudes", close(a_old, a_new, rtol=1e-9, atol=1e-12))

    print("\nAll phase-4 ParametricModel parity checks passed.")


if __name__ == "__main__":
    main()
