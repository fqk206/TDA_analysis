#!/usr/bin/env python3
"""Trace readers: plain two-column files, plus the ``old``/``fidaneo`` dispatch.

``read_two_column`` and ``read_trace`` are the parametrized equivalents of
``fit_crli_simple.read_trace`` - the file-format knobs (time column, delimiter,
time unit, ...) are function arguments instead of module globals, so a caller
can read two files under different settings without mutating shared state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TIME_SCALES = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "min": 60.0, "minutes": 60.0, "ms": 1e-3}


@dataclass
class ReaderParams:
    """File-reading knobs, composed into every single-trace model.

    Defaults match the historical ``fit_crli_simple.py`` CONFIG block.
    """

    file_format: str = "old"           # "old" (plain two-column) or "fidaneo"
    time_column: int = 0               # "old" format only
    signal_column: int = 1             # "old" format only
    delimiter: str | None = None       # None splits on any whitespace; "old" format only
    skiprows: int = 0                  # "old" format only
    comments: str = "#"                # "old" format only
    time_unit: str = "min"             # "s", "min", or "ms"; "old" format only (fidaneo is always minutes)


def read_two_column(
    path: str | Path,
    time_column: int = 0,
    signal_column: int = 1,
    delimiter: str | None = None,
    skiprows: int = 0,
    comments: str = "#",
    time_unit: str = "min",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a plain columnar ``(time, signal)`` trace, time scaled to seconds."""
    data = np.genfromtxt(path, delimiter=delimiter, comments=comments, skip_header=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] <= max(time_column, signal_column):
        raise ValueError(f"Could not parse requested columns from {path}")

    unit = time_unit.lower()
    if unit not in _TIME_SCALES:
        raise ValueError("time_unit must be 's', 'min', or 'ms'.")

    time_raw = data[:, time_column]
    signal = data[:, signal_column]
    finite = np.isfinite(time_raw) & np.isfinite(signal)
    time_s = time_raw[finite] * _TIME_SCALES[unit]
    signal = signal[finite]
    order = np.argsort(time_s)
    return time_s[order], signal[order]


def read_trace(
    path: str | Path,
    file_format: str = "old",
    time_column: int = 0,
    signal_column: int = 1,
    delimiter: str | None = None,
    skiprows: int = 0,
    comments: str = "#",
    time_unit: str = "min",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column ``(time_s, signal)`` trace in ``file_format``.

    ``"old"`` reads a plain columnar file using the column/delimiter/... knobs.
    ``"fidaneo"`` delegates to :mod:`tda.io.fidaneo` (time column always in
    minutes, converted here).
    """
    fmt = file_format.lower()
    if fmt == "fidaneo":
        from tda.io import fidaneo
        return fidaneo.read(path).as_trace()
    if fmt != "old":
        raise ValueError("file_format must be 'old' or 'fidaneo'.")
    return read_two_column(path, time_column, signal_column, delimiter, skiprows, comments, time_unit)
