"""
timsa.ingest.prescribed
=======================
Read prescribed gauge water-level records directly, at their native temporal
resolution, for use as TiMSA forcing.

This is the Python equivalent of the C reference's prescribed-waterdepth mode
(iterateday_prescribewd_NADV88.c), where

    CSV2array2d_double(config.surveytimes_filename, gaugewdepths, nrows, tidegauges)

loads an (nrows x n_gauges) table and each row advances the simulation by
`config.simtimestep` minutes. Feeding 6-minute CO-OPS observations with
simtimestep=6 runs the model on the observations themselves, with no
interpolation step in between.

Why this matters: interpolating 6-minute observations up to 1-minute and then
integrating at 1-minute does not add information. It inflates the record
sixfold and buries the observed values inside an interpolation choice. For a
present-day simulation driven by real gauge data, stepping natively at 6
minutes is both faster and more honest about the input resolution.

Accepted layouts
----------------
1. Headerless numeric  - pure C parity. One column per gauge, in gauge-index
                         order (column 0 -> gauge 1).
2. Header with station IDs - columns named by CO-OPS station ID. Reordered to
                         match the caller's `station_order`.
3. With a time column  - datetime column plus one column per gauge. The record
                         interval is inferred and regularity is verified.

Gap handling
------------
The C reference holds the water surface constant across a no-data record
(`depths_sim->data[bb] += 0`). `fill_gaps="hold"` reproduces that and is the
default. `"interpolate"` fills linearly across short gaps; `"nan"` leaves gaps
as NaN, which propagates to the depth comparison and simply contributes no
time to any accumulator for the affected steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
VALID_INTERVALS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36,
                   40, 45, 48, 60, 72, 80, 90, 96, 120, 144, 160, 180, 240,
                   288, 360, 480, 720, 1440)


class PrescribedWLError(ValueError):
    """Raised when a prescribed water-level table cannot be used as forcing."""


@dataclass
class PrescribedWL:
    """
    A gauge water-level table ready to hand to TimsaInputs.

    Attributes
    ----------
    values : np.ndarray
        (n_records, n_gauges) float64, metres, on the DEM's vertical datum
        once offsets have been applied.
    record_interval_min : int
        Minutes between consecutive rows.
    station_ids : list[str]
        Column order; index i corresponds to gauge index i+1 in the zone raster.
    start_time : pd.Timestamp or None
        Timestamp of row 0, when a time column was present.
    n_gaps_filled : int
        Count of missing VALUES filled by `fill_gaps` (cells, not rows: a row
        missing at two gauges counts as two).
    source : str
        Path the table was read from, for provenance.
    """

    values: np.ndarray
    record_interval_min: int
    station_ids: list = field(default_factory=list)
    start_time: object | None = None
    n_gaps_filled: int = 0
    source: str = ""

    @property
    def n_records(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_gauges(self) -> int:
        return int(self.values.shape[1])

    @property
    def records_per_day(self) -> int:
        return MINUTES_PER_DAY // self.record_interval_min

    @property
    def n_whole_days(self) -> int:
        return self.n_records // self.records_per_day

    def truncate_to_whole_days(self) -> "PrescribedWL":
        """
        Trim trailing partial-day records. TiMSA iterates whole days (the
        daylight table is indexed per day), so a ragged tail would be silently
        ignored; trimming makes that explicit.
        """
        keep = self.n_whole_days * self.records_per_day
        if keep == self.n_records:
            return self
        self.values = self.values[:keep]
        return self

    def describe(self) -> str:
        span_days = self.n_records * self.record_interval_min / MINUTES_PER_DAY
        pct_nan = 100.0 * float(np.isnan(self.values).sum()) / max(self.values.size, 1)
        return (
            f"{self.n_records} records x {self.n_gauges} gauge(s) "
            f"@ {self.record_interval_min} min "
            f"({span_days:.2f} days, {self.n_gaps_filled} gaps filled, "
            f"{pct_nan:.3f}% still NaN)"
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def load_prescribed_wl(
    path: str | Path,
    station_order: list | None = None,
    time_col: str | None = None,
    record_interval_min: int | None = None,
    value_scale: float = 1.0,
    datum_offsets: dict | None = None,
    fill_gaps: str = "hold",
    max_gap_records: int | None = None,
    has_header: bool | None = None,
) -> PrescribedWL:
    """
    Load a prescribed water-level table.

    Parameters
    ----------
    path : str or Path
        CSV (or whitespace-delimited .txt) file.
    station_order : list of str, optional
        Desired column order, index i -> gauge i+1. Required when the file has
        a header and the caller needs a specific gauge ordering (it must match
        the gauge-zone raster). Ignored for headerless files.
    time_col : str, optional
        Name of a datetime column. If given, the interval is inferred from it
        and regularity is checked. If omitted, `record_interval_min` is required.
    record_interval_min : int, optional
        Explicit record spacing in minutes. Required when there is no time
        column. When both are supplied, the inferred value must agree.
    value_scale : float
        Multiplier applied to all values (e.g. 0.3048 for feet -> metres).
    datum_offsets : dict, optional
        {station_id: offset_m} added per column after scaling, to convert the
        record onto the DEM's vertical datum (e.g. MLLW -> NAVD88). Stations
        absent from the dict raise, rather than silently defaulting to zero.
    fill_gaps : {'hold', 'interpolate', 'nan'}
        Missing-value policy. 'hold' forward-fills (C reference behaviour).
    max_gap_records : int, optional
        Refuse to fill any run of missing values longer than this. Guards
        against silently inventing a week of tide.
    has_header : bool, optional
        Force header detection instead of sniffing.

    Returns
    -------
    PrescribedWL
    """
    path = Path(path)
    if not path.exists():
        raise PrescribedWLError(f"prescribed water-level file not found: {path}")

    if fill_gaps not in ("hold", "interpolate", "nan"):
        raise PrescribedWLError(
            f"fill_gaps must be 'hold', 'interpolate', or 'nan'; got {fill_gaps!r}"
        )

    sep = r"\s+" if path.suffix.lower() in (".txt", ".dat", ".tsv") else ","

    if has_header is None:
        has_header = _sniff_header(path, sep)

    df = pd.read_csv(path, sep=sep, header=0 if has_header else None,
                     engine="python" if sep == r"\s+" else "c")

    if not has_header:
        df.columns = [f"gauge_{i + 1}" for i in range(df.shape[1])]

    # ---- time column -------------------------------------------------
    start_time = None
    inferred_interval = None
    if time_col is not None:
        if time_col not in df.columns:
            raise PrescribedWLError(
                f"time_col={time_col!r} not in file columns: {list(df.columns)}"
            )
        times = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if times.isna().any():
            n_bad = int(times.isna().sum())
            raise PrescribedWLError(
                f"{n_bad} unparseable timestamp(s) in column {time_col!r}"
            )
        inferred_interval, start_time = _infer_interval(times)
        df = df.drop(columns=[time_col])

    if record_interval_min is None:
        if inferred_interval is None:
            raise PrescribedWLError(
                "record_interval_min is required when the file has no time column. "
                "For native CO-OPS observations this is 6."
            )
        record_interval_min = inferred_interval
    elif inferred_interval is not None and inferred_interval != record_interval_min:
        raise PrescribedWLError(
            f"record_interval_min={record_interval_min} contradicts the interval "
            f"inferred from {time_col!r} ({inferred_interval} min)."
        )

    record_interval_min = int(record_interval_min)
    if MINUTES_PER_DAY % record_interval_min != 0:
        raise PrescribedWLError(
            f"record_interval_min={record_interval_min} does not divide "
            f"{MINUTES_PER_DAY} evenly. Valid values: {VALID_INTERVALS}"
        )

    # ---- column ordering ---------------------------------------------
    if station_order is not None:
        wanted = [str(s) for s in station_order]
        available = {str(c): c for c in df.columns}
        missing = [s for s in wanted if s not in available]
        if missing:
            raise PrescribedWLError(
                f"station_order lists column(s) absent from {path.name}: {missing}. "
                f"Available: {list(df.columns)}"
            )
        df = df[[available[s] for s in wanted]]
        station_ids = wanted
    else:
        station_ids = [str(c) for c in df.columns]

    # ---- numeric conversion ------------------------------------------
    # copy=True: pandas may hand back a read-only view, and the scaling and
    # offset steps below write in place.
    values = np.array(
        df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
    )
    if values.ndim != 2 or values.shape[1] == 0:
        raise PrescribedWLError(f"no gauge columns parsed from {path}")

    if value_scale != 1.0:
        values *= float(value_scale)

    # ---- datum offsets ------------------------------------------------
    if datum_offsets is not None:
        offsets = np.empty(values.shape[1], dtype=np.float64)
        missing = []
        for j, sid in enumerate(station_ids):
            if str(sid) in datum_offsets:
                offsets[j] = float(datum_offsets[str(sid)])
            else:
                missing.append(sid)
        if missing:
            raise PrescribedWLError(
                f"no datum offset for station(s) {missing}. Supply them or pass "
                f"datum_offsets=None if the record is already on the DEM datum."
            )
        values += offsets[np.newaxis, :]

    # ---- gaps ----------------------------------------------------------
    n_missing = int(np.isnan(values).sum())
    if n_missing and max_gap_records is not None:
        longest = _longest_nan_run(values)
        if longest > max_gap_records:
            raise PrescribedWLError(
                f"longest missing run is {longest} records "
                f"({longest * record_interval_min} min), exceeding "
                f"max_gap_records={max_gap_records}."
            )

    n_filled = 0
    if n_missing and fill_gaps != "nan":
        filled = pd.DataFrame(values)
        if fill_gaps == "hold":
            filled = filled.ffill().bfill()
        else:
            filled = filled.interpolate(limit_direction="both")
        values = filled.to_numpy(dtype=np.float64)
        n_filled = n_missing - int(np.isnan(values).sum())

    out = PrescribedWL(
        values=values,
        record_interval_min=record_interval_min,
        station_ids=station_ids,
        start_time=start_time,
        n_gaps_filled=n_filled,
        source=str(path),
    )
    return out.truncate_to_whole_days()


# ---------------------------------------------------------------------------
# Writer (for round-tripping a fetched record to a reusable forcing file)
# ---------------------------------------------------------------------------

def write_prescribed_wl(
    wl: PrescribedWL,
    path: str | Path,
    include_time: bool = True,
) -> Path:
    """
    Write a PrescribedWL back out as CSV, so a fetched-and-corrected record can
    be reused as a stable simulation input without re-hitting the CO-OPS API.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(wl.values, columns=wl.station_ids)
    if include_time and wl.start_time is not None:
        idx = pd.date_range(
            start=wl.start_time,
            periods=wl.n_records,
            freq=f"{wl.record_interval_min}min",
            tz="UTC",
        )
        df.insert(0, "time", idx)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _sniff_header(path: Path, sep: str) -> bool:
    """
    True if the first row looks like column names rather than data.

    Rule: ANY non-numeric token in row 0 means a header. This is deliberately
    strict, because CO-OPS station IDs are numeric strings ("8723970"), so a
    majority-vote heuristic misreads `time,8723970,8724580` as data.

    Genuinely ambiguous case: a header consisting only of numeric station IDs,
    with no time column. Row 0 is then indistinguishable from a data row, and
    the first record would be silently consumed as column names. That case
    raises, so the caller must pass has_header explicitly.
    """
    import re

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        second = f.readline().strip()
    if not first:
        return False

    def _tokens(line: str) -> list:
        return re.split(sep, line) if sep != "," else line.split(",")

    def _all_numeric(line: str) -> bool:
        toks = [t.strip() for t in _tokens(line) if t.strip() != ""]
        if not toks:
            return False
        for t in toks:
            try:
                float(t)
            except ValueError:
                return False
        return True

    if not _all_numeric(first):
        return True

    # Row 0 is all numeric. If row 1 is too, we cannot tell a numeric header
    # from data. Flag the ambiguity rather than guessing.
    if second and _all_numeric(second):
        row0 = [float(t) for t in _tokens(first) if t.strip() != ""]
        row1 = [float(t) for t in _tokens(second) if t.strip() != ""]
        # Station IDs are large integers; water levels are small and fractional.
        looks_like_ids = all(v.is_integer() and abs(v) > 1000 for v in row0)
        looks_like_data = any(not v.is_integer() or abs(v) < 100 for v in row1)
        if looks_like_ids and looks_like_data:
            raise PrescribedWLError(
                f"{path.name}: row 0 looks like numeric station IDs but cannot be "
                f"distinguished from data with certainty. Pass has_header=True "
                f"(or has_header=False if that row really is water levels)."
            )
    return False


def _infer_interval(times: pd.Series) -> tuple[int, object]:
    """Infer the record interval in minutes; verify the series is regular."""
    if len(times) < 2:
        raise PrescribedWLError("need at least two timestamps to infer an interval")
    deltas = times.diff().dropna()
    minutes = (deltas.dt.total_seconds() / 60.0).round().astype(int)
    counts = minutes.value_counts()
    modal = int(counts.index[0])

    if modal <= 0:
        raise PrescribedWLError(
            "timestamps are not strictly increasing; sort or de-duplicate the file"
        )

    irregular = int((minutes != modal).sum())
    if irregular:
        frac = 100.0 * irregular / len(minutes)
        offenders = counts.drop(index=modal).head(3).to_dict()
        raise PrescribedWLError(
            f"irregular time spacing: {irregular} of {len(minutes)} intervals "
            f"({frac:.2f}%) differ from the modal {modal} min. Other spacings "
            f"seen: {offenders}. Reindex the record onto a regular grid before "
            f"use — TiMSA assumes one row per fixed timestep."
        )
    return modal, times.iloc[0]


def _longest_nan_run(values: np.ndarray) -> int:
    """Longest consecutive run of rows where ANY gauge is missing."""
    row_missing = np.isnan(values).any(axis=1)
    if not row_missing.any():
        return 0
    longest = current = 0
    for m in row_missing:
        current = current + 1 if m else 0
        longest = max(longest, current)
    return longest
