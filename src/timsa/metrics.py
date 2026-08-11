"""
metrics.py
==========
Per-cell metric rasters and domain-aggregated summaries for a TiMSA run.

Metrics, computed for every depth window and refugia threshold declared in
the simulation config:

  1. Area availability            - binary; cell fell inside the depth window
                                    at least once during the run.
  2. Time-integrated availability - minutes inside the depth window, summed
                                    over the run.
  3. Refugia time                 - minutes below each refugia threshold.

Generalized from the SLR analysis pipeline:
  - A run is identified by an opaque `run_id` string rather than by
    (scenario, horizon, slr_m).
  - All named depth windows are handled, not only a hard-coded
    'shallow_band'.
  - Cell area is derived from the raster transform when available instead
    of assuming a 10 m grid.
  - RunResult.metadata is populated from TimsaSimulation.describe() when the
    caller does not supply its own, so record interval and timestep travel
    with the summary.
  - Baseline-relative loss trajectories are gone; summaries are absolute.
    Comparison between runs is the caller's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from timsa.core import TimsaSimulation

NODATA = -9999.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_area_m2_from_profile(profile: dict, default: float | None = None) -> float:
    """
    Derive cell area (m^2) from a rasterio profile's affine transform.

    Assumes a projected CRS in metres. For a geographic CRS (degrees) the
    result is meaningless, so an explicit `cell_area_m2` should be supplied
    in that case.

    Raises ValueError if the transform is absent and no default is given.
    """
    transform = profile.get("transform")
    if transform is None:
        if default is not None:
            return float(default)
        raise ValueError(
            "profile has no transform; supply cell_area_m2 explicitly"
        )
    # transform.a = pixel width, transform.e = pixel height (negative)
    return abs(float(transform.a)) * abs(float(transform.e))


def _thr_slug(thr: float) -> str:
    """Filename-safe threshold token: 0.2 -> '0p20', 1.0 -> '1p00'."""
    return f"{float(thr):.2f}".replace(".", "p")


def _slug(name: str) -> str:
    """Filename-safe metric/window token."""
    return str(name).strip().replace(" ", "_").replace("/", "-")


# ---------------------------------------------------------------------------
# Run result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Result container for a single TiMSA run.

    Parameters
    ----------
    run_id : str
        Opaque identifier used for output filenames and summary rows.
        Callers building multi-run experiments encode whatever they need
        here (e.g. "present_day", "2019_daylight", "offset_0p30").
    sim : TimsaSimulation
        Completed simulation (``sim.run()`` already called), or a stub with
        the same attribute surface (see ``load_run_from_rasters``).
    profile : dict
        rasterio profile for raster I/O.
    cell_area_m2 : float or None
        Cell area. If None, derived from ``profile['transform']``.
    metadata : dict
        Free-form key/value pairs propagated into summary rows. Use this
        for anything run-specific the caller wants to carry through
        (year, offset applied, site name, ...).
    """

    run_id: str
    sim: TimsaSimulation
    profile: dict
    cell_area_m2: float | None = None
    metadata: dict = field(default_factory=dict)

    # Computed by compute_summaries()
    area_available_ha: dict = field(default_factory=dict)      # window -> ha
    time_integrated_ha_h: dict = field(default_factory=dict)   # window -> ha-hours
    refugia_time_ha_h: dict = field(default_factory=dict)      # threshold -> ha-hours

    # ---------------------------------------------------------------

    def resolve_cell_area(self) -> float:
        if self.cell_area_m2 is not None:
            return float(self.cell_area_m2)
        return cell_area_m2_from_profile(self.profile)

    def compute_summaries(self, cell_area_m2: float | None = None) -> None:
        """
        Aggregate per-cell metric rasters to domain totals.

        Accumulators are in cell-minutes; outputs are hectare-hours (and
        plain hectares for the binary area metric).
        """
        if cell_area_m2 is not None:
            self.cell_area_m2 = cell_area_m2
        ha_per_cell = self.resolve_cell_area() / 1e4

        self.area_available_ha = {}
        self.time_integrated_ha_h = {}
        for window, arr in self.sim.time_in_band.items():
            self.area_available_ha[window] = float((arr > 0).sum()) * ha_per_cell
            self.time_integrated_ha_h[window] = float(arr.sum()) * ha_per_cell / 60.0

        self.refugia_time_ha_h = {
            float(thr): float(arr.sum()) * ha_per_cell / 60.0
            for thr, arr in self.sim.time_below_threshold.items()
        }

    # ---------------------------------------------------------------

    def to_rows(self) -> list[dict]:
        """
        Long-format summary rows for this run.

        Columns: run_id, metric, window_or_threshold, value, units,
        plus any keys in ``metadata``.
        """
        rows: list[dict] = []

        def _row(metric: str, key, value: float, units: str) -> dict:
            r = {
                "run_id": self.run_id,
                "metric": metric,
                "window_or_threshold": key,
                "value": value,
                "units": units,
            }
            r.update(self.metadata)
            return r

        for window, val in self.area_available_ha.items():
            rows.append(_row("area_available", window, val, "ha"))
        for window, val in self.time_integrated_ha_h.items():
            rows.append(_row("time_integrated", window, val, "ha-h"))
        for thr, val in self.refugia_time_ha_h.items():
            rows.append(_row("refugia_time", f"{thr:g}", val, "ha-h"))

        return rows


# ---------------------------------------------------------------------------
# Raster output
# ---------------------------------------------------------------------------

def metric_raster_paths(
    run_id: str,
    out_dir: Path,
    depth_windows,
    refugia_thresholds,
) -> dict:
    """
    Canonical output paths for a run. Single source of truth shared by
    the writer and the loader so the two can never drift apart.

    Returns a dict of {logical_key: Path}, where logical_key is one of:
      ("time_integrated", window)        raw total, cell-minutes over the run
      ("time_average", window)           total / n_days, minutes per day
      ("area_available", window)
      ("refugia_time", threshold_float)         raw total
      ("refugia_average", threshold_float)      total / n_days

    The "*_average" outputs are the day-averaged rasters, matching the C
    reference's save_average_raster (minutes per day). The "time_integrated"
    and "refugia_time" totals match its save_sum_raster (minutes over the run).
    """
    out_dir = Path(out_dir)
    paths: dict = {}
    for window in depth_windows:
        w = _slug(window)
        paths[("time_integrated", window)] = \
            out_dir / f"{run_id}__time_integrated_min__{w}.tif"
        paths[("time_average", window)] = \
            out_dir / f"{run_id}__time_average_min_per_day__{w}.tif"
        paths[("area_available", window)] = \
            out_dir / f"{run_id}__area_available_bin__{w}.tif"
    for thr in refugia_thresholds:
        thr_f = float(thr)
        paths[("refugia_time", thr_f)] = \
            out_dir / f"{run_id}__refugia_time_min__thr_{_thr_slug(thr_f)}.tif"
        paths[("refugia_average", thr_f)] = \
            out_dir / f"{run_id}__refugia_average_min_per_day__thr_{_thr_slug(thr_f)}.tif"
    return paths


def write_metric_rasters(
    run: RunResult,
    out_dir: Path,
    dtype: str = "float32",
    compress: str = "lzw",
    write_area_binary: bool = True,
    write_total: bool = True,
    write_average: bool = False,
) -> list[Path]:
    """
    Write one GeoTIFF per metric for a single run.

    Parameters
    ----------
    write_total : bool
        Write the raw accumulator, in cell-minutes over the run. This is the
        analog of the C reference's save_sum_raster.
    write_average : bool
        Write total / n_days, in minutes per day. This is the analog of the
        C reference's save_average_raster and is the raster to compare against
        a C day-averaged run. Off by default so existing behaviour is
        unchanged; enable it (or both) for parity testing.

    Returns the list of paths written. Returns an empty list (silently) if
    the profile lacks a CRS or transform, which is the synthetic/dry-run
    case; a real run over a projected DEM will have both.
    """
    if not write_total and not write_average:
        raise ValueError("write_total and write_average are both False; "
                         "nothing would be written")

    profile = dict(run.profile)
    if profile.get("crs") is None or profile.get("transform") is None:
        return []

    import rasterio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update(dtype=dtype, count=1, compress=compress, nodata=NODATA)

    valid = getattr(run.sim, "base_valid_mask", run.sim.valid_mask)
    n_days = max(int(getattr(run.sim, "n_days", 1)), 1)
    written: list[Path] = []

    def _write(arr: np.ndarray, path: Path) -> None:
        out = arr.astype(dtype, copy=True)
        out[~valid] = NODATA
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(out, 1)
        written.append(path)

    paths = metric_raster_paths(
        run.run_id, out_dir,
        list(run.sim.time_in_band.keys()),
        list(run.sim.time_below_threshold.keys()),
    )

    for window, arr in run.sim.time_in_band.items():
        if write_total:
            _write(arr, paths[("time_integrated", window)])
        if write_average:
            _write(arr / n_days, paths[("time_average", window)])
        if write_area_binary:
            _write((arr > 0).astype(np.float32), paths[("area_available", window)])

    for thr, arr in run.sim.time_below_threshold.items():
        if write_total:
            _write(arr, paths[("refugia_time", float(thr))])
        if write_average:
            _write(arr / n_days, paths[("refugia_average", float(thr))])

    return written


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def build_summary_table(runs: list[RunResult]) -> pd.DataFrame:
    """Long-format DataFrame of absolute domain totals across runs."""
    rows: list[dict] = []
    for r in runs:
        rows.extend(r.to_rows())
    return pd.DataFrame(rows)


def write_summary(runs: list[RunResult], out_csv: Path) -> pd.DataFrame:
    """Build the summary table and write it to CSV."""
    df = build_summary_table(runs)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Reload from cached rasters
# ---------------------------------------------------------------------------

def load_run_from_rasters(
    run_id: str,
    out_dir: Path,
    depth_windows,
    refugia_thresholds,
    profile: dict,
    metadata: dict | None = None,
    cell_area_m2: float | None = None,
) -> RunResult:
    """
    Reconstruct a RunResult from rasters previously written by
    ``write_metric_rasters``. Supports --skip-existing style reruns where
    summaries are rebuilt without re-simulating.

    NoData is converted to 0.0 so downstream sums behave; the valid mask is
    recovered from the first raster read.
    """
    import rasterio
    from types import SimpleNamespace

    out_dir = Path(out_dir)
    paths = metric_raster_paths(run_id, out_dir, depth_windows, refugia_thresholds)

    valid_mask = None

    def _read(path: Path) -> np.ndarray:
        nonlocal valid_mask
        if not path.exists():
            raise FileNotFoundError(f"Missing cached raster: {path}")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nodata = src.nodata if src.nodata is not None else NODATA
        v = arr != nodata
        arr[~v] = 0.0
        if valid_mask is None:
            valid_mask = v
        return arr

    time_in_band = {
        window: _read(paths[("time_integrated", window)])
        for window in depth_windows
    }
    time_below_threshold = {
        float(thr): _read(paths[("refugia_time", float(thr))])
        for thr in refugia_thresholds
    }

    if valid_mask is None:
        raise ValueError(f"No rasters found for run_id={run_id!r} in {out_dir}")

    sim_stub = SimpleNamespace(
        time_in_band=time_in_band,
        time_below_threshold=time_below_threshold,
        valid_mask=valid_mask,
        domain_time_integrated=lambda name, _t=time_in_band: float(_t[name].sum()),
        domain_refugia_time=lambda thr, _t=time_below_threshold: float(_t[float(thr)].sum()),
        area_available=lambda name, _t=time_in_band: float((_t[name] > 0).sum()),
    )

    return RunResult(
        run_id=run_id,
        sim=sim_stub,
        profile=profile,
        cell_area_m2=cell_area_m2,
        metadata=metadata or {},
    )
