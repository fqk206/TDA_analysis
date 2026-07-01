#!/usr/bin/env python3
"""Phase-1 parity check: mechanical extraction must be bitwise-identical.

Compares the new ``tda.physics`` / ``tda.numerics`` / ``tda.io`` /
``tda.preprocessing`` modules against the still-untouched
``fit_crli_simple.py`` (imported as ``crli``) on real data files. No
tolerance: this phase is pure code motion, so mismatches are bugs, not
floating-point noise.

Run:
    python3 tests/parity/test_phase1_mechanical.py
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fit_crli_simple as crli  # noqa: E402
import fidaneo as fidaneo_old  # noqa: E402

from tda import physics, numerics  # noqa: E402
from tda.io import trace as tda_trace  # noqa: E402
from tda.io import fidaneo as fidaneo_new  # noqa: E402
from tda.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from tda.preprocessing.steps import Despike, Bin  # noqa: E402


def check(name: str, ok: bool) -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        raise AssertionError(name)


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def test_physics_rh_kernel():
    t = np.linspace(0, 120, 500)
    Rh = np.geomspace(0.5, 100.0, 40)
    T_K = 273.15 + crli.TEMPERATURE_C
    old = crli.rh_kernel(t, Rh, 45.0, T_K)
    new = physics.rh_kernel(t, Rh, 45.0, T_K, physics.PhysicsParams())
    check("rh_kernel matrix", same(old, new))

    old_scalar = crli.rh_kernel(t, 12.3, 45.0, T_K)
    new_scalar = physics.rh_kernel(t, 12.3, 45.0, T_K, physics.PhysicsParams())
    check("rh_kernel scalar Rh", same(old_scalar, new_scalar))


def test_physics_apparent_rh(prep_old):
    T_K = 273.15 + crli.TEMPERATURE_C
    old = crli.apparent_rh_from_slope(prep_old, "rising", T_K)
    new = physics.apparent_rh_from_slope(
        prep_old.time_s, prep_old.normalized, prep_old.good_mask, prep_old.t0_s,
        "rising", T_K, crli.RH_CENTER_BOUNDS_NM,
    )
    check("apparent_rh_from_slope", old == new or (old is None and new is None))


def test_numerics():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    check("robust_std", crli.robust_std(values) == numerics.robust_std(values))

    x = np.geomspace(0.3, 300.0, 57)
    check("trapezoid_weights", same(crli.trapezoid_weights(x), numerics.trapezoid_weights(x)))
    check("second_difference_matrix", same(crli.second_difference_matrix(30), numerics.second_difference_matrix(30)))

    t = np.linspace(0, 10, 41)
    y = -((t - 4.3) ** 2) + rng.normal(scale=1e-3, size=t.size)
    i_peak = int(np.argmax(y))
    check("quadratic_peak_time", crli.quadratic_peak_time(t, y, i_peak) == numerics.quadratic_peak_time(t, y, i_peak))

    Rh = np.geomspace(0.3, 300.0, 200)
    pdf = np.exp(-0.5 * ((np.log(Rh) - np.log(20.0)) / 0.3) ** 2)
    old_stats = crli.distribution_stats(Rh, pdf)
    new_stats = numerics.distribution_stats(Rh, pdf)
    for k in old_stats:
        ov, nv = old_stats[k], new_stats[k]
        ok = (ov == nv) or (np.isnan(ov) and np.isnan(nv))
        check(f"distribution_stats[{k}]", ok)


def test_io_read_trace(old_files):
    for path in old_files:
        old_t, old_y = crli.read_trace(path)
        new_t, new_y = tda_trace.read_trace(
            path, file_format="old", time_column=crli.TIME_COLUMN, signal_column=crli.SIGNAL_COLUMN,
            delimiter=crli.DELIMITER, skiprows=crli.SKIPROWS, comments=crli.COMMENTS, time_unit=crli.TIME_UNIT,
        )
        check(f"read_trace old-format {path.name}", same(old_t, new_t) and same(old_y, new_y))


def test_io_read_trace_fidaneo(fidaneo_file):
    saved = crli.FILE_FORMAT
    try:
        crli.FILE_FORMAT = "fidaneo"
        old_t, old_y = crli.read_trace(fidaneo_file)
    finally:
        crli.FILE_FORMAT = saved
    new_t, new_y = tda_trace.read_trace(fidaneo_file, file_format="fidaneo")
    check("read_trace fidaneo-format", same(old_t, new_t) and same(old_y, new_y))

    rec_old = fidaneo_old.read(fidaneo_file)
    rec_new = fidaneo_new.read(fidaneo_file)
    check("tda.io.fidaneo matches root fidaneo module", same(rec_old.time_s, rec_new.time_s)
          and same(rec_old.signal, rec_new.signal) and rec_old.method == rec_new.method)


def _to_pipeline_bin(width_s: float) -> PreprocessingPipeline:
    return PreprocessingPipeline(steps=(Despike(window=crli.DESPIKE_WINDOW, n_sigma=crli.DESPIKE_SIGMA), Bin(width_s=width_s)))


def test_preprocessing_pipeline(old_files, fidaneo_file):
    default_pipeline = PreprocessingPipeline(
        steps=(Despike(window=crli.DESPIKE_WINDOW, n_sigma=crli.DESPIKE_SIGMA),),
    )

    for path in list(old_files) + [fidaneo_file]:
        if path == fidaneo_file:
            t, y = fidaneo_new.read(path).as_trace()
        else:
            t, y = tda_trace.read_two_column(
                path, crli.TIME_COLUMN, crli.SIGNAL_COLUMN, crli.DELIMITER, crli.SKIPROWS, crli.COMMENTS, crli.TIME_UNIT,
            )
        old_prep = crli.prepare_from_trace(t, y, path)
        new_prep = default_pipeline.run(t, y, path)

        ok = (
            same(old_prep.time_s, new_prep.time_s)
            and same(old_prep.signal, new_prep.signal)
            and same(old_prep.baseline, new_prep.baseline)
            and same(old_prep.corrected, new_prep.corrected)
            and same(old_prep.normalized, new_prep.normalized)
            and same(old_prep.good_mask, new_prep.good_mask)
            and old_prep.baseline_window_s == new_prep.baseline_window_s
            and old_prep.noise_raw == new_prep.noise_raw
            and old_prep.t0_s == new_prep.t0_s
            and old_prep.peak_height == new_prep.peak_height
            and same(old_prep.spike_mask, new_prep.spike_mask)
        )
        check(f"prepare_from_trace parity (despike only) {Path(path).name}", ok)

    # Binning path: BIN=True on the old side, matching Bin() step on the new side.
    saved_bin, saved_width = crli.BIN, crli.BIN_WIDTH_S
    try:
        crli.BIN = True
        crli.BIN_WIDTH_S = 0.5
        bin_pipeline = _to_pipeline_bin(0.5)
        for path in old_files[:2]:
            t, y = tda_trace.read_two_column(
                path, crli.TIME_COLUMN, crli.SIGNAL_COLUMN, crli.DELIMITER, crli.SKIPROWS, crli.COMMENTS, crli.TIME_UNIT,
            )
            old_prep = crli.prepare_from_trace(t, y, path)
            new_prep = bin_pipeline.run(t, y, path)
            ok = (
                same(old_prep.time_s, new_prep.time_s)
                and same(old_prep.signal, new_prep.signal)
                and old_prep.t0_s == new_prep.t0_s
                and old_prep.peak_height == new_prep.peak_height
                and same(old_prep.good_mask, new_prep.good_mask)
                and same(old_prep.spike_mask, new_prep.spike_mask)
            )
            check(f"prepare_from_trace parity (binned) {Path(path).name}", ok)
    finally:
        crli.BIN, crli.BIN_WIDTH_S = saved_bin, saved_width


def main() -> None:
    old_files = sorted(Path(p) for p in glob(str(REPO_ROOT / "data" / "*.txt")))
    series_dir = REPO_ROOT / "data_series" / "series_01"
    fidaneo_file = sorted(series_dir.glob("*.txt"))[0]
    assert old_files, "expected files under data/*.txt"
    assert fidaneo_file.exists()

    print("test_physics_rh_kernel")
    test_physics_rh_kernel()

    print("test_physics_apparent_rh")
    prep_old = crli.prepare(old_files[0])
    test_physics_apparent_rh(prep_old)

    print("test_numerics")
    test_numerics()

    print("test_io_read_trace")
    test_io_read_trace(old_files)

    print("test_io_read_trace_fidaneo")
    test_io_read_trace_fidaneo(fidaneo_file)

    print("test_preprocessing_pipeline")
    test_preprocessing_pipeline(old_files, fidaneo_file)

    print("\nAll phase-1 parity checks passed.")


if __name__ == "__main__":
    main()
