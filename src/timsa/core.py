"""
core.py
=======
Vectorized Python port of the Tidal Inundation Model of Shallow-water
Availability (TiMSA), originally implemented in C by L. Calle (lcalle/timsa).

The model is a bathtub inundation simulation. A water surface is set from the
tide gauge, water is added or removed numerically at each timestep according to
the gauge's rate of change, and per-cell time is accumulated whenever the cell's
elevation relative to that surface falls inside a depth window.

Faithful to the C reference:
  - Gauge zones via reference raster (each cell assigned to a tide gauge).
  - Day-1 height adjustment sets the water surface from the gauge.
  - Water surface then evolves by the gauge-prescribed increment per step
    (see `surface_update`).
  - Depth-window indicator integration, accumulated over the simulation.
  - Optional daylight constraint via a sunrise/sunset table.
  - Sinusoidal interpolation between sparse gauge records (tide_wdchange.c),
    so the simulation can step FINER than the water-level record.
  - Prescribed water-level records used at their native interval
    (iterateday_prescribewd_NADV88.c), so the simulation can also step at
    exactly the record spacing, or coarser.

Differences from the C reference:
  - Vectorized over all grid cells simultaneously using NumPy.
  - Multi-threshold metrics computed in a single time-loop pass.
  - A uniform vertical offset may be applied to the gauge record
    (`water_level_offset_m`) for scenario work; defaults to 0.0.
  - Optional per-day accumulator emission (see `daily_callback`), so daily
    rasters stream to disk from a preallocated buffer rather than being
    reallocated each day (the C reference leaks a raster per day via repeated
    rastercopy with no matching free).

Sign convention (C reference -- POSITIVE IS DRY)
-----------------------------------------------
    depth = dem_elevation - water_surface

  depth  > 0  ->  cell is DRY, standing that far above the water surface
  depth  = 0  ->  cell is exactly at the waterline
  depth  < 0  ->  cell is SUBMERGED by that much water

  Depth windows are therefore expressed with NEGATIVE bounds. A shallow band
  of 0 to 1.5 m of water is written:

      depth_windows: {"shallow_band": (-1.5, 0.0)}

  Refugia thresholds remain POSITIVE water depths, because they name a depth of
  water rather than an elevation: a threshold of 0.5 selects cells under more
  than 0 and at most 0.5 m of water, i.e. -0.5 <= depth < 0.

Time base
---------
  `record_interval_min` is the spacing, in minutes, between consecutive ROWS of
  `gauge_wdepths`. `timestep_min` is the simulation step. They are independent:

    record 360 min (hi/lo), step 1 min    -> interpolated up (tide_wdchange.c)
    record 6 min (CO-OPS obs), step 1 min -> interpolated up
    record 6 min, step 6 min              -> used directly (prescribed mode)
    record 6 min, step 30 min             -> decimated

  The gauge record is resampled onto the timestep grid once, at construction.
  The array is small (n_gauges is O(10)), so this costs little. Both
  `record_interval_min` and `timestep_min` must divide 1440 evenly, so that a
  day contains a whole number of records and a whole number of steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np

MINUTES_PER_DAY = 1440

GAUGE_INTERP_METHODS = ("linear", "sinusoidal", "hold")
SURFACE_UPDATE_MODES = ("direct", "incremental")
DAYLIGHT_BOUNDS = ("inclusive", "exclusive")


# ---------------------------------------------------------------------------
# Gauge record resampling
# ---------------------------------------------------------------------------

def resample_gauge_records(
    values: np.ndarray,
    record_interval_min: int,
    timestep_min: int,
    method: str = "linear",
) -> np.ndarray:
    """
    Resample a gauge water-level table onto the simulation timestep grid.

    Handles all three cases with one code path:
      timestep < record interval  -> interpolate up
      timestep = record interval  -> returned unchanged
      timestep > record interval  -> decimate (target times coincide with
                                     source times, so every method agrees)

    Parameters
    ----------
    values : (n_records, n_gauges) float array
        Water levels on the DEM's vertical datum.
    record_interval_min, timestep_min : int
        Minutes between source rows, and between output rows.
    method : {'linear', 'sinusoidal', 'hold'}
        'linear'     - straight line between consecutive records.
        'sinusoidal' - half-cycle sine between consecutive records, with zero
                       rate of change at each record and maximum at the
                       midpoint. This is tide_wdchange.c, and is the right
                       choice when the records are tidal extrema (hi/lo).
        'hold'       - step function; the surface holds each record's value
                       until the next. Matches the C reference's no-data
                       behaviour of adding zero change.

    Returns
    -------
    (n_steps, n_gauges) float array, where
        n_steps = n_records * record_interval_min // timestep_min

    Notes
    -----
    Output times beyond the final record (up to one record interval past it)
    hold the last observed value rather than extrapolating.
    """
    if method not in GAUGE_INTERP_METHODS:
        raise ValueError(
            f"gauge_interp must be one of {GAUGE_INTERP_METHODS}, got {method!r}"
        )

    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("values must be 1D or 2D")
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]

    n_rec = values.shape[0]
    ri, ts = int(record_interval_min), int(timestep_min)

    if ts == ri:
        out = values.copy()
        return out[:, 0] if squeeze else out
    if n_rec < 2:
        raise ValueError("at least two gauge records are required to resample")

    span_min = n_rec * ri
    n_steps = span_min // ts
    if n_steps < 1:
        raise ValueError(
            f"timestep_min={ts} exceeds the {span_min}-minute record span"
        )

    t_src = np.arange(n_rec, dtype=np.float64) * ri
    t_dst = np.arange(n_steps, dtype=np.float64) * ts

    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, n_rec - 2)
    f = (t_dst - t_src[idx]) / float(ri)          # fractional position in [0, 1)

    h0 = values[idx]                               # (n_steps, n_gauges)
    h1 = values[idx + 1]

    if method == "hold":
        # Use the unclipped bracket index: at exactly t_src[-1], `idx` above is
        # clamped to n_rec-2 with f == 1.0, which the interpolating methods
        # resolve to values[-1] but a step function would wrongly hold
        # values[-2].
        hold_idx = np.clip(
            np.searchsorted(t_src, t_dst, side="right") - 1, 0, n_rec - 1
        )
        out = values[hold_idx].copy()
    elif method == "linear":
        out = h0 + f[:, None] * (h1 - h0)
    else:  # sinusoidal
        out = 0.5 * (h0 + h1) + 0.5 * (h1 - h0) * np.sin(np.pi * (f[:, None] - 0.5))

    # Tail beyond the last record: hold, do not extrapolate.
    tail = t_dst > t_src[-1]
    if tail.any():
        out[tail] = values[-1]

    return out[:, 0] if squeeze else out


def apply_surface_update(
    ws: np.ndarray,
    mode: str = "direct",
    gap_policy: str = "hold",
) -> np.ndarray:
    """
    Produce the water-surface series the simulation will use.

    Parameters
    ----------
    ws : (n_steps, n_gauges)
        Resampled gauge levels.
    mode : {'direct', 'incremental'}
        'direct'      - the water surface IS the gauge level at each step.
        'incremental' - the C reference's bathtub accumulation: the surface is
                        set once from the first record, then each step adds the
                        gauge's change since the previous step. With complete
                        data the two agree to floating-point rounding, because
                        the increments telescope. They diverge only where the
                        record has gaps, since the C reference adds zero change
                        across a no-data record, holding the surface in place.
                        Use 'incremental' for C parity checks.
    gap_policy : {'hold', 'nan'}
        'hold' carries the last good level forward across gaps (C behaviour).
        'nan' leaves them, so those steps contribute no time to any accumulator.

    Returns
    -------
    (n_steps, n_gauges) float array.
    """
    if mode not in SURFACE_UPDATE_MODES:
        raise ValueError(
            f"surface_update must be one of {SURFACE_UPDATE_MODES}, got {mode!r}"
        )
    if gap_policy not in ("hold", "nan"):
        raise ValueError("gap_policy must be 'hold' or 'nan'")

    ws = np.array(np.asarray(ws, dtype=np.float64), copy=True)
    if ws.ndim == 1:
        ws = ws[:, None]

    if gap_policy == "hold" and np.isnan(ws).any():
        n_rows = ws.shape[0]
        idx = np.where(~np.isnan(ws), np.arange(n_rows)[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        ws = np.take_along_axis(ws, idx, axis=0)
        # Back-fill a leading gap.
        for j in range(ws.shape[1]):
            if np.isnan(ws[0, j]):
                good = np.flatnonzero(~np.isnan(ws[:, j]))
                if good.size:
                    ws[: good[0], j] = ws[good[0], j]

    if mode == "direct":
        return ws

    # Incremental: start at the first record, accumulate the per-step gauge
    # change. Missing changes contribute zero, matching the C `+= 0`.
    deltas = np.nan_to_num(np.diff(ws, axis=0), nan=0.0)
    out = np.empty_like(ws)
    out[0] = ws[0]
    np.cumsum(deltas, axis=0, out=out[1:])
    out[1:] += ws[0]
    return out


# ---------------------------------------------------------------------------
# Inputs / config
# ---------------------------------------------------------------------------

@dataclass
class TimsaInputs:
    """Container for TiMSA simulation inputs."""

    dem: np.ndarray               # 2D bed elevation (m, NAVD88-referenced)
    gauge_zones: np.ndarray       # 2D int raster, gauge index per cell (1..N), 0 = no data
    gauge_wdepths: np.ndarray     # 2D gauge water levels (n_records x n_gauges), NAVD88
    nodata_mask: np.ndarray       # 2D bool, True where cell is no-data (excluded)

    # Spacing between consecutive rows of gauge_wdepths, in minutes.
    # 1 for a per-minute series, 6 for native CO-OPS records, 360 for hi/lo.
    record_interval_min: int = 1

    # Uniform vertical offset added to all gauge levels. Scenario work sets
    # this; present-day runs leave it at 0.
    water_level_offset_m: float = 0.0

    # Deprecated alias for water_level_offset_m, retained for compatibility
    # with the SLR pipeline. Do not use in new code.
    slr_offset_m: float | None = None

    def __post_init__(self):
        if self.slr_offset_m is not None:
            warnings.warn(
                "TimsaInputs.slr_offset_m is deprecated; "
                "use water_level_offset_m instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.water_level_offset_m:
                raise ValueError(
                    "Set either slr_offset_m or water_level_offset_m, not both."
                )
            self.water_level_offset_m = float(self.slr_offset_m)

        assert self.dem.shape == self.gauge_zones.shape == self.nodata_mask.shape, \
            "DEM, gauge_zones, and nodata_mask must have identical shape"
        assert self.gauge_wdepths.ndim == 2, \
            "gauge_wdepths must be 2D: (n_records, n_gauges)"

        ri = int(self.record_interval_min)
        if ri < 1:
            raise ValueError(f"record_interval_min must be >= 1, got {ri}")
        if MINUTES_PER_DAY % ri != 0:
            raise ValueError(
                f"record_interval_min={ri} does not divide {MINUTES_PER_DAY} "
                f"minutes evenly; a day must hold a whole number of records. "
                f"Valid values include 1, 2, 3, 5, 6, 10, 15, 20, 30, 60, 360."
            )
        self.record_interval_min = ri

    @property
    def n_records(self) -> int:
        return int(self.gauge_wdepths.shape[0])

    @property
    def n_gauges(self) -> int:
        return int(self.gauge_wdepths.shape[1])


@dataclass
class TimsaConfig:
    """
    Container for TiMSA simulation parameters.

    depth_windows use the C sign convention: POSITIVE IS DRY. A band covering
    0 to 1.5 m of water is (-1.5, 0.0).
    """

    depth_windows: dict           # e.g. {'shallow_band': (-1.5, 0.0)}
    refugia_thresholds: list      # positive water depths, e.g. [0.2, 0.5, 1.0]
    timestep_min: int = 1         # simulation step; independent of record interval

    # Interpolation used when the timestep is finer than the record interval.
    gauge_interp: str = "linear"

    # Bathtub surface construction; see apply_surface_update().
    surface_update: str = "direct"
    gap_policy: str = "hold"

    constrain_daylight: bool = False
    sunrise_sunset_min: np.ndarray | None = None   # (n_days, 2), minutes-of-day

    # User-selectable daylight boundary handling.
    #   'inclusive' : sunrise <= t <= sunset
    #   'exclusive' : sunrise <  t <  sunset   (matches the C reference, which
    #                 skips a step when sunrise - t >= 0 or sunset - t <= 0)
    # Affects at most the two boundary steps per day.
    daylight_bounds: str = "inclusive"

    def __post_init__(self):
        self.timestep_min = int(self.timestep_min)
        if self.timestep_min < 1:
            raise ValueError("timestep_min must be >= 1")
        if MINUTES_PER_DAY % self.timestep_min != 0:
            raise ValueError(
                f"timestep_min={self.timestep_min} does not divide "
                f"{MINUTES_PER_DAY} minutes evenly; a day must hold a whole "
                f"number of steps."
            )
        if self.gauge_interp not in GAUGE_INTERP_METHODS:
            raise ValueError(f"gauge_interp must be one of {GAUGE_INTERP_METHODS}")
        if self.surface_update not in SURFACE_UPDATE_MODES:
            raise ValueError(f"surface_update must be one of {SURFACE_UPDATE_MODES}")
        if self.daylight_bounds not in DAYLIGHT_BOUNDS:
            raise ValueError(f"daylight_bounds must be one of {DAYLIGHT_BOUNDS}")

        self.refugia_thresholds = [float(t) for t in self.refugia_thresholds]
        if any(t <= 0 for t in self.refugia_thresholds):
            raise ValueError(
                "refugia_thresholds are positive water depths (e.g. 0.2, 0.5); "
                "they name a depth of water, not an elevation."
            )

        self.depth_windows = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in self.depth_windows.items()
        }
        for name, (lo, hi) in self.depth_windows.items():
            if lo > hi:
                raise ValueError(
                    f"depth window {name!r}: lower bound {lo} exceeds upper {hi}."
                )
            if lo >= 0.0 and hi > 0.0:
                warnings.warn(
                    f"depth window {name!r} = ({lo}, {hi}) lies entirely above "
                    f"the waterline and will only ever match dry cells. This "
                    f"port uses the C convention where POSITIVE IS DRY, so a "
                    f"shallow band of 0-{hi} m of water is written "
                    f"({-hi}, {-lo}).",
                    stacklevel=3,
                )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TimsaSimulation:
    """
    Runs a TiMSA simulation and accumulates depth-window indicator counts.

    Output (per cell, per metric):
      - time_in_band[name]        : minutes with elevation inside a depth window.
      - time_below_threshold[thr] : minutes under at most `thr` metres of water.

    All accumulators are 2D rasters with the same shape as the DEM.
    """

    def __init__(self, inputs: TimsaInputs, config: TimsaConfig):
        self.inputs = inputs
        self.config = config

        self.record_interval_min = inputs.record_interval_min
        self.steps_per_day = MINUTES_PER_DAY // config.timestep_min

        # ---- build the water-surface series ------------------------------
        levels = inputs.gauge_wdepths + inputs.water_level_offset_m
        ws = resample_gauge_records(
            levels,
            record_interval_min=self.record_interval_min,
            timestep_min=config.timestep_min,
            method=config.gauge_interp,
        )
        self.water_surface = apply_surface_update(
            ws, mode=config.surface_update, gap_policy=config.gap_policy
        )

        n_steps, n_gauges = self.water_surface.shape
        self.n_steps = n_steps
        self.n_gauges = n_gauges
        self.n_days = n_steps // self.steps_per_day
        if self.n_days < 1:
            raise ValueError(
                f"the record spans {n_steps} step(s) of {config.timestep_min} min, "
                f"less than one whole day ({self.steps_per_day} steps)."
            )

        if config.constrain_daylight:
            if config.sunrise_sunset_min is None:
                raise ValueError(
                    "Daylight constraint requires a sunrise_sunset_min table"
                )
            ss = np.asarray(config.sunrise_sunset_min)
            if ss.ndim != 2 or ss.shape[1] != 2:
                raise ValueError("sunrise_sunset_min must have shape (n_days, 2)")
            if ss.shape[0] < self.n_days:
                raise ValueError(
                    f"sunrise_sunset_min has {ss.shape[0]} days but the record "
                    f"spans {self.n_days} days."
                )

        # ---- cell masks ---------------------------------------------------
        gz = inputs.gauge_zones.astype(np.int32)
        base_valid_mask = (~inputs.nodata_mask) & (gz > 0)

        if int(gz.max(initial=0)) > n_gauges:
            raise ValueError(
                f"gauge_zones references gauge index {int(gz.max())} but the "
                f"water-level table has only {n_gauges} column(s). The zone "
                f"raster and the gauge record are out of sync."
            )

        # Deepest water depth any metric cares about. A cell whose bed sits
        # below (lowest water surface at its gauge - that depth) is submerged
        # deeper than anything of interest at every step, so all accumulators
        # are guaranteed zero. Skip those cells.
        deepest_of_interest = max(
            max((-lo for (lo, _) in config.depth_windows.values()), default=0.0),
            max(config.refugia_thresholds, default=0.0),
            0.0,
        )

        ws_min_per_gauge = np.nanmin(self.water_surface, axis=0)   # (n_gauges,)

        too_deep_mask = np.zeros_like(base_valid_mask)
        valid_gz = (gz - 1)[base_valid_mask]
        cell_ws_min = ws_min_per_gauge[valid_gz]
        cell_dem = inputs.dem[base_valid_mask]
        too_deep_mask[base_valid_mask] = cell_dem < (cell_ws_min - deepest_of_interest)

        self.valid_mask = base_valid_mask & (~too_deep_mask)
        # Mask used for OUTPUT no-data: genuine no-data only (land / DEM-nodata
        # / no gauge zone). Excludes the 'too deep' cells, which are inside the
        # domain and provably contribute 0 to every metric, so their rasters
        # should serialize 0, not no-data. (valid_mask above is the narrower
        # COMPUTE mask that also drops the too-deep cells for speed.)
        self.base_valid_mask = base_valid_mask
        self.gauge_idx_flat = gz[self.valid_mask] - 1          # 0-based
        self.dem_flat = inputs.dem[self.valid_mask]
        self.n_valid = int(self.dem_flat.size)
        self.n_skipped_too_deep = int(too_deep_mask.sum())

        # ---- accumulators --------------------------------------------------
        self.shape = inputs.dem.shape
        self.time_in_band = {
            name: np.zeros(self.shape, dtype=np.float64)
            for name in config.depth_windows
        }
        self.time_below_threshold = {
            float(t): np.zeros(self.shape, dtype=np.float64)
            for t in config.refugia_thresholds
        }
        self.completed = False

    # ------------------------------------------------------------------

    def unflatten(self, flat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        Scatter a 1D per-valid-cell array back onto the 2D grid.

        Pass `out` to reuse a preallocated buffer, so the daily raster writer
        allocates nothing inside the day loop.
        """
        if out is None:
            out = np.zeros(self.shape, dtype=np.float64)
        else:
            out.fill(0)
        out[self.valid_mask] = flat
        return out

    def _daylight_mask(self, day: int) -> np.ndarray:
        """
        Boolean (steps_per_day, 1) mask of steps falling within daylight.

        Minute-of-day for step j is (j + 1) * timestep_min, matching the C
        reference's `dayminute = simtimestep * (i+1) - 1440 * trackday`.
        """
        sunrise = float(self.config.sunrise_sunset_min[day, 0])
        sunset = float(self.config.sunrise_sunset_min[day, 1])
        minutes_of_day = (np.arange(self.steps_per_day) + 1) * self.config.timestep_min
        if self.config.daylight_bounds == "exclusive":
            active = (minutes_of_day > sunrise) & (minutes_of_day < sunset)
        else:
            active = (minutes_of_day >= sunrise) & (minutes_of_day <= sunset)
        return active[:, np.newaxis]

    # ------------------------------------------------------------------

    def run(
        self,
        verbose: bool = True,
        cell_chunk: int = 100_000,
        daily_callback: Callable[[int, dict, dict, "TimsaSimulation"], None] | None = None,
        progress_every: int = 30,
    ) -> None:
        """
        Execute the simulation.

        Iterates one day at a time and processes cells in blocks to cap peak
        memory. Peak working arrays are (steps_per_day, cell_chunk).

        Parameters
        ----------
        verbose : bool
            Print periodic progress.
        cell_chunk : int
            Valid cells per inner block. Caps peak memory at roughly
            steps_per_day * cell_chunk * 8 bytes per working array.
        daily_callback : callable or None
            Called at the end of each day as
                daily_callback(day_index, flat_band, flat_below, sim)
            where the dicts map metric key -> 1D array over valid cells holding
            THAT DAY's minutes. The arrays are reused between days, so a
            callback needing to retain them must copy. This is the streaming
            hook for daily raster output; leaving it None costs nothing.
        progress_every : int
            Progress print interval, in days.
        """
        timestep = self.config.timestep_min
        n_days = self.n_days
        want_daily = daily_callback is not None

        accum_band = {
            name: np.zeros(self.n_valid, dtype=np.float64)
            for name in self.config.depth_windows
        }
        accum_below = {
            float(t): np.zeros(self.n_valid, dtype=np.float64)
            for t in self.config.refugia_thresholds
        }

        # Per-day buffers, allocated once and refilled with zeros each day.
        if want_daily:
            day_band = {k: np.zeros(self.n_valid) for k in accum_band}
            day_below = {k: np.zeros(self.n_valid) for k in accum_below}
        else:
            day_band, day_below = accum_band, accum_below

        for day in range(n_days):
            if want_daily:
                for arr in day_band.values():
                    arr.fill(0)
                for arr in day_below.values():
                    arr.fill(0)

            s0 = day * self.steps_per_day
            day_block = self.water_surface[s0:s0 + self.steps_per_day, :]

            day_active = (
                self._daylight_mask(day) if self.config.constrain_daylight else None
            )

            for c0 in range(0, self.n_valid, cell_chunk):
                c1 = min(c0 + cell_chunk, self.n_valid)
                water_surface_t = day_block[:, self.gauge_idx_flat[c0:c1]]

                # C convention: positive = dry, negative = submerged.
                depth = self.dem_flat[np.newaxis, c0:c1] - water_surface_t

                for name, (d_low, d_high) in self.config.depth_windows.items():
                    in_band = (depth >= d_low) & (depth <= d_high)
                    if day_active is not None:
                        in_band &= day_active
                    day_band[name][c0:c1] += in_band.sum(axis=0) * timestep

                for thr in self.config.refugia_thresholds:
                    # Submerged, by at most `thr` metres of water.
                    below = (depth < 0.0) & (depth >= -float(thr))
                    if day_active is not None:
                        below &= day_active
                    day_below[float(thr)][c0:c1] += below.sum(axis=0) * timestep

            if want_daily:
                for name in accum_band:
                    accum_band[name] += day_band[name]
                for thr in accum_below:
                    accum_below[thr] += day_below[thr]
                daily_callback(day, day_band, day_below, self)

            if verbose and (day % progress_every == 0 or day == n_days - 1):
                pct = 100.0 * (day + 1) / n_days
                print(f"  day {day + 1:>3}/{n_days} ({pct:5.1f}%)")

        for name in self.config.depth_windows:
            self.time_in_band[name] = self.unflatten(accum_band[name])
        for thr in self.config.refugia_thresholds:
            self.time_below_threshold[float(thr)] = self.unflatten(accum_below[float(thr)])

        self.completed = True

    # ------------------------------------------------------------------

    def area_available(self, band_name: str = "shallow_band") -> float:
        """Cells recording at least one step inside the depth band."""
        return float((self.time_in_band[band_name] > 0).sum())

    def domain_time_integrated(self, band_name: str = "shallow_band") -> float:
        """Domain-aggregated time-integrated availability (cell-minutes)."""
        return float(self.time_in_band[band_name].sum())

    def domain_refugia_time(self, threshold: float) -> float:
        """Domain-aggregated refugia-time at a threshold (cell-minutes)."""
        return float(self.time_below_threshold[float(threshold)].sum())

    def describe(self) -> dict:
        """Run metadata, for RunResult.metadata or a log line."""
        return {
            "n_days": self.n_days,
            "record_interval_min": self.record_interval_min,
            "timestep_min": self.config.timestep_min,
            "steps_per_day": self.steps_per_day,
            "gauge_interp": self.config.gauge_interp,
            "surface_update": self.config.surface_update,
            "n_gauges": self.n_gauges,
            "n_valid_cells": self.n_valid,
            "n_skipped_too_deep": self.n_skipped_too_deep,
            "water_level_offset_m": self.inputs.water_level_offset_m,
            "constrain_daylight": self.config.constrain_daylight,
            "daylight_bounds": self.config.daylight_bounds,
        }


# ---------------------------------------------------------------------------
# Tidal series synthesis (kept for irregular hi/lo event lists)
# ---------------------------------------------------------------------------

def synthesize_tidal_series_sinusoidal(
    high_low_events_min: np.ndarray,
    high_low_levels_m: np.ndarray,
    n_minutes: int,
) -> np.ndarray:
    """
    Sinusoidal interpolation of water level between observed high/low events,
    on an IRREGULAR event grid.

    Mirrors tide_wdchange.c: a half-cycle sine between consecutive extrema,
    with zero rate of change at each extremum and maximum at the midpoint.

    For a REGULAR record grid use `resample_gauge_records(..., method=
    'sinusoidal')`, which is vectorized and handles many gauges at once. This
    function remains for irregular event lists such as raw hi/lo tables.
    """
    out = np.zeros(n_minutes, dtype=np.float64)
    ev = np.asarray(high_low_events_min)
    lv = np.asarray(high_low_levels_m, dtype=np.float64)
    for i in range(len(ev) - 1):
        t0, t1 = int(ev[i]), int(ev[i + 1])
        h0, h1 = lv[i], lv[i + 1]
        if t1 <= t0:
            continue
        n = t1 - t0
        phase = np.linspace(-np.pi / 2, np.pi / 2, n, endpoint=False)
        seg = 0.5 * (h1 + h0) + 0.5 * (h1 - h0) * np.sin(phase)
        # Events may bracket the window: the hi/lo adapter keeps a couple of
        # extrema on each side so boundary half-cycles have the correct phase.
        # Clamp the write window to [0, n_minutes] and slice the segment to
        # match, so an out-of-range pair contributes only its in-window part
        # (t0 >= n_minutes -> nothing) instead of raising a broadcast error.
        a = max(t0, 0)
        b = min(t1, n_minutes)
        if b <= a:
            continue
        out[a:b] = seg[a - t0:b - t0]
    last = int(ev[-1])
    if 0 <= last < n_minutes:
        out[last:] = lv[-1]
    return out
