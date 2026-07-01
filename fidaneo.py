#!/usr/bin/env python3
"""Parser for FidaNeo ``.txt`` datafiles.

A FidaNeo export bundles one Taylor-dispersion trace together with a structured
header. This module reads that file into a :class:`FidaNeoRecord` exposing both
the two-column ``(time, signal)`` data and the metadata that downstream analysis
needs - above all the **measurement time point** and the **method**, which are
what a kinetic / time-series global fit keys off.

File layout (tagged blocks inside ``<header>`` ... ``</header>`` then ``<data>``):

    <--FidaNeo datafile-->
    <header>
      <sample>      Sample name / concentration
      <timestamps>  Sequence start / Measurement start / end / Report generation
      <step>        Step idx, Seq line idx   <- selects this trace's row in <seq>
      <seq>         the run table; one row per step, last column is "Method"
      <meth>        the pressure/time program
      <detector>    detector id/type/sensitivity
      ...
    </header>
    <data>
      t<TAB>signal   (time in minutes, FidaNeo convention)
    </data>

The **method for a given file** is not stored inline: ``<step>`` gives a
``Seq line idx`` that indexes the data rows of the ``<seq>`` table, whose last
column holds the method name. :meth:`FidaNeoRecord.method` resolves that.

Usage:
    import fidaneo
    rec = fidaneo.read("data_series/series_01/[0 nM] 0p05perc_1-....txt")
    t_s, y = rec.time_s, rec.signal
    print(rec.method, rec.measurement_start)

    records = fidaneo.load_folder("data_series/series_01")
    same = fidaneo.with_method_of_first(records)   # share the first trace's method
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
import re

import numpy as np

MAGIC = "<--FidaNeo datafile-->"
TIME_COLUMN_UNIT = "min"        # FidaNeo writes the data time column in minutes
_SECONDS_PER_MIN = 60.0

# Canonical keys for the <timestamps> block (raw label -> snake_case attribute).
_TIMESTAMP_KEYS = {
    "Sequence start": "sequence_start",
    "Measurement start": "measurement_start",
    "Measurement end": "measurement_end",
    "Report generation": "report_generation",
}


@dataclass
class FidaNeoRecord:
    """One parsed FidaNeo trace plus its metadata."""

    path: Path
    time_min: np.ndarray                 # data time column, as written (minutes)
    signal: np.ndarray
    sample_name: str
    concentration: str                   # as text, e.g. "0 nM"
    step_idx: int
    seq_line_idx: int
    timestamps: dict[str, datetime]      # snake_case keys, see _TIMESTAMP_KEYS
    seq_table: list[dict[str, str]]
    method_table: list[dict[str, str]]
    detector: dict[str, str]
    header_text: str

    # -- convenience accessors -------------------------------------------------

    @property
    def method(self) -> str:
        """Method name for this trace, resolved via ``Seq line idx``."""
        if not 0 <= self.seq_line_idx < len(self.seq_table):
            raise ValueError(
                f"Seq line idx {self.seq_line_idx} out of range for the "
                f"{len(self.seq_table)}-row <seq> table in {self.path.name}"
            )
        row = self.seq_table[self.seq_line_idx]
        if "Method" not in row:
            raise ValueError(f"<seq> table in {self.path.name} has no 'Method' column")
        return row["Method"]

    @property
    def time_s(self) -> np.ndarray:
        """Data time column in seconds (physics / preprocessing convention)."""
        return self.time_min * _SECONDS_PER_MIN

    @property
    def measurement_start(self) -> datetime:
        return self.timestamps["measurement_start"]

    def as_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """``(time_s, signal)`` - the two columns a fitter consumes."""
        return self.time_s, self.signal

    def metadata(self) -> dict:
        """Flat, serialisable metadata dict (no array data)."""
        return {
            "path": str(self.path),
            "sample_name": self.sample_name,
            "concentration": self.concentration,
            "method": self.method,
            "step_idx": self.step_idx,
            "seq_line_idx": self.seq_line_idx,
            "n_points": int(self.signal.size),
            **{k: (v.isoformat() if v is not None else None) for k, v in self.timestamps.items()},
        }


# =============================================================================
# Block-level parsing helpers
# =============================================================================

def _extract_block(text: str, tag: str, path: Path) -> str:
    match = re.search(rf"<{tag}>\s*\n(.*?)\n?\s*</{tag}>", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"<{tag}> block not found in {path.name}")
    return match.group(1)


def _key_value_block(block: str) -> dict[str, str]:
    """Parse 'Label:<TAB or spaces>value' lines into a dict."""
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _table_block(block: str) -> list[dict[str, str]]:
    """Parse a tab-separated table (first non-empty line is the header)."""
    rows = [ln for ln in block.splitlines() if ln.strip()]
    if not rows:
        return []
    header = rows[0].split("\t")
    table: list[dict[str, str]] = []
    for line in rows[1:]:
        cells = line.split("\t")
        table.append({h: (cells[i].strip() if i < len(cells) else "") for i, h in enumerate(header)})
    return table


def _parse_timestamp(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Fall back: drop a trailing colon-less or unusual zone gracefully.
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


# =============================================================================
# Public API
# =============================================================================

def read(path: str | Path) -> FidaNeoRecord:
    """Parse one FidaNeo datafile into a :class:`FidaNeoRecord`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if MAGIC not in text.splitlines()[0]:
        raise ValueError(f"{path.name} does not start with the FidaNeo magic line {MAGIC!r}")

    header_text = _extract_block(text, "header", path)

    sample = _key_value_block(_extract_block(header_text, "sample", path))
    step = _key_value_block(_extract_block(header_text, "step", path))
    raw_ts = _key_value_block(_extract_block(header_text, "timestamps", path))
    seq_table = _table_block(_extract_block(header_text, "seq", path))
    method_table = _table_block(_extract_block(header_text, "meth", path))
    detector = _key_value_block(_extract_block(header_text, "detector", path))

    timestamps = {attr: _parse_timestamp(raw_ts.get(label, "")) for label, attr in _TIMESTAMP_KEYS.items()}

    def _int(block: dict, *labels: str) -> int:
        for label in labels:
            if label in block:
                return int(float(block[label]))
        raise ValueError(f"None of {labels} found in <step> of {path.name}")

    step_idx = _int(step, "Step idx")
    seq_line_idx = _int(step, "Seq line idx")

    data_block = _extract_block(text, "data", path)
    data = np.genfromtxt(data_block.splitlines())
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Could not parse a two-column <data> block in {path.name}")
    time_min, signal = data[:, 0], data[:, 1]
    finite = np.isfinite(time_min) & np.isfinite(signal)
    time_min, signal = time_min[finite], signal[finite]
    order = np.argsort(time_min)

    return FidaNeoRecord(
        path=path,
        time_min=time_min[order],
        signal=signal[order],
        sample_name=sample.get("Sample name", path.stem),
        concentration=sample.get("Sample concentration", ""),
        step_idx=step_idx,
        seq_line_idx=seq_line_idx,
        timestamps=timestamps,
        seq_table=seq_table,
        method_table=method_table,
        detector=detector,
        header_text=header_text,
    )


def load_folder(folder: str | Path, pattern: str = "*.txt", sort_by_time: bool = True) -> list[FidaNeoRecord]:
    """Read every FidaNeo file matching ``pattern`` under ``folder``.

    By default the records are returned sorted by measurement start time, so the
    first element is the earliest data point.
    """
    paths = sorted(glob(str(Path(folder) / pattern)))
    records: list[FidaNeoRecord] = []
    for p in paths:
        try:
            records.append(read(p))
        except Exception as exc:  # keep going; a bad file shouldn't sink the batch
            import warnings
            warnings.warn(f"Skipped {Path(p).name}: {exc}")
    if sort_by_time:
        records = sort_records_by_time(records)
    return records


def sort_records_by_time(records: list[FidaNeoRecord]) -> list[FidaNeoRecord]:
    """Sort by measurement start (records lacking a timestamp go last)."""
    far_future = datetime.max
    return sorted(records, key=lambda r: (r.measurement_start or far_future))


def filter_by_method(records: list[FidaNeoRecord], method: str) -> list[FidaNeoRecord]:
    return [r for r in records if r.method == method]


def with_method_of_first(records: list[FidaNeoRecord]) -> list[FidaNeoRecord]:
    """Keep only records whose method matches the first (earliest) data point.

    This is the selection a time-series global fit uses: it fixes the method to
    that of the first trace so the kinetic series is measured consistently.
    """
    if not records:
        return []
    ordered = sort_records_by_time(records)
    return filter_by_method(ordered, ordered[0].method)


def relative_hours(records: list[FidaNeoRecord], origin: datetime | None = None) -> np.ndarray:
    """Measurement start of each record in hours relative to ``origin``.

    ``origin`` defaults to the earliest measurement start in ``records`` - the
    natural time axis for the kinetic fit (t = 0 at the first data point).
    """
    starts = [r.measurement_start for r in records]
    if origin is None:
        origin = min(starts)
    return np.array([(s - origin).total_seconds() / 3600.0 for s in starts], dtype=float)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect FidaNeo datafiles.")
    ap.add_argument("path", help="A FidaNeo .txt file or a folder of them.")
    args = ap.parse_args()

    target = Path(args.path)
    recs = load_folder(target) if target.is_dir() else [read(target)]
    origin = min(r.measurement_start for r in recs) if recs else None
    print(f"{'measurement_start':25s} {'method':28s} {'sample':14s} {'t/h':>7s}  n")
    for r in recs:
        dt_h = (r.measurement_start - origin).total_seconds() / 3600.0 if origin else 0.0
        print(f"{r.measurement_start.isoformat():25s} {r.method:28s} {r.sample_name:14s} {dt_h:7.3f}  {r.signal.size}")
