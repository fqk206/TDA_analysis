#!/usr/bin/env python3
"""Phase-5 parity check: GlobalSeriesModel vs fit_global_series.py.

Run:
    python3 tests/parity/test_phase5_global_series.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fit_crli_simple as crli  # noqa: E402
import fit_global_series as fgs  # noqa: E402

from tda.models.global_series import GlobalSeriesModel  # noqa: E402


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
    data_dir = REPO_ROOT / "data_series" / "series_01"
    assert data_dir.exists()

    print("loading traces (old)")
    traces_old, info_old = fgs.load_traces(str(data_dir), "*.txt", fgs.GROUP_MODE)
    print(f"  n_used={info_old['n_used']} n_skipped={info_old['n_skipped']}")

    print("loading traces (new)")
    model = GlobalSeriesModel()
    traces_new, info_new = model.load_traces(str(data_dir), "*.txt")
    check("n_used matches", info_old["n_used"] == info_new["n_used"])
    check("n_skipped matches", info_old["n_skipped"] == info_new["n_skipped"])

    for i, (to, tn) in enumerate(zip(traces_old, traces_new)):
        check(f"trace[{i}] time0_s", close(to.time0_s, tn.time0_s), f"old={to.time0_s} new={tn.time0_s}")
        check(f"trace[{i}] t_fit", close(to.t_fit, tn.t_fit))
        check(f"trace[{i}] s_fit", close(to.s_fit, tn.s_fit))
        check(f"trace[{i}] apparent_rh_nm", close(to.apparent_rh_nm, tn.apparent_rh_nm)
              if to.apparent_rh_nm is not None else to.apparent_rh_nm == tn.apparent_rh_nm)

    print("fitting series (old)")
    res_old = fgs.fit_series(traces_old)
    print("fitting series (new)")
    res_new = model.fit(traces_new)

    check("success", res_old["success"] == res_new.success)
    check("cost", close(res_old["cost"], res_new.cost))
    check("nfev", res_old["nfev"] == res_new.nfev)
    check("x", close(res_old["x"], res_new.x))

    for g in res_old["names"]:
        go, gn = res_old["fitted"][g], res_new.fitted[g]
        check(f"group[{g}] Rh0_nm", close(go.Rh0_nm, gn.Rh0_nm))
        check(f"group[{g}] Rhinf_nm", close(go.Rhinf_nm, gn.Rhinf_nm))
        check(f"group[{g}] tau_h", close(go.tau_h, gn.tau_h))
        check(f"group[{g}] colloid_sigma", close(go.colloid_sigma, gn.colloid_sigma))
        check(f"group[{g}] dye_mu_nm", close(go.dye_mu_nm, gn.dye_mu_nm))
        check(f"group[{g}] dye_sigma", close(go.dye_sigma, gn.dye_sigma))

    for i, (ro, rn) in enumerate(zip(res_old["per_trace"], res_new.per_trace)):
        for key in ("time_h", "t0_s", "model_colloid_rh_nm", "indep_colloid_rh_nm", "dye_fraction",
                    "colloid_fraction", "residual_rms_raw", "residual_rms_over_noise"):
            check(f"per_trace[{i}][{key}]", close(ro[key], rn[key]), f"old={ro[key]!r} new={rn[key]!r}")

    print("\nAll phase-5 GlobalSeriesModel parity checks passed.")


if __name__ == "__main__":
    main()
