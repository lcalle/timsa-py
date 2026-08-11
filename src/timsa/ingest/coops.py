"""
timsa.ingest.coops
==================
Multi-gauge CO-OPS water level acquisition.

Provides:
  - fetch_station_record               : 6-min obs with hi/lo predictions fallback
  - sinusoidal_minute_series_from_hilo : half-cycle cosine interpolation
  - build_multigauge_minute_array      : assembles (n_min, n_gauges) array
  - apply_m2_amplitude_scaling         : M2 harmonic sensitivity
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from timsa.core import synthesize_tidal_series_sinusoidal


# ---------------------------------------------------------------------------
# CO-OPS API helpers
# ---------------------------------------------------------------------------

COOPS_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# CO-OPS limits: water_level requests are capped at 31 days per call.
# Predictions/hilo are capped at 1 year per call, but we chunk monthly to
# keep memory steady and simplify retry logic.
_CHUNK_DAYS = 30


def _coops_request(params: dict, retries: int = 3, timeout: int = 60) -> dict:
    """Single CO-OPS API call with simple retry."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(COOPS_BASE, params=params, timeout=(10, timeout))
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            import time as _t; _t.sleep(2 ** attempt)  # 1, 2, 4 sec backoff
            continue
    raise RuntimeError(f"CO-OPS request failed after {retries} retries: {last_err}")


def _fetch_chunked(station_id: str, start: dt.date, end: dt.date,
                   product: str, datum: str, interval: str | None = None,
                   chunk_days: int = _CHUNK_DAYS) -> pd.DataFrame:
    """
    Generic chunked CO-OPS fetch. Returns concatenated DataFrame with
    columns ['t', 'v'] where t is UTC datetime and v is value in meters.
    Empty DataFrame if all chunks return no data.
    """
    rows = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), end)
        params = {
            "product": product,
            "station": station_id,
            "begin_date": cursor.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "datum": datum,
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
            "application": "timsa",
        }
        if interval is not None:
            params["interval"] = interval

        try:
            payload = _coops_request(params)
        except RuntimeError:
            cursor = chunk_end + dt.timedelta(days=1)
            continue

        # CO-OPS returns 'data' for water_level and 'predictions' for predictions
        records = payload.get("data") or payload.get("predictions") or []
        for rec in records:
            t = rec.get("t")
            v = rec.get("v")
            if t is None or v in (None, ""):
                continue
            try:
                rows.append((pd.to_datetime(t, utc=True), float(v)))
            except (ValueError, TypeError):
                continue

        cursor = chunk_end + dt.timedelta(days=1)
        import time as _t; _t.sleep(0.3)

    if not rows:
        return pd.DataFrame(columns=["t", "v"])
    df = pd.DataFrame(rows, columns=["t", "v"]).drop_duplicates("t").sort_values("t")
    return df.reset_index(drop=True)


def fetch_station_record(
    station_id: str,
    year: int,
    datum: str = "MLLW",
    cache_dir: Path | None = None,
    min_obs_coverage_pct: float = 80.0,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch a station's record for a given tidal year.

    Strategy
    --------
    1. Try `water_level` (6-min observations) for the full year.
    2. If empty or coverage < min_obs_coverage_pct of expected 6-min records,
       fall back to `predictions?interval=hilo`.

    Cache layer
    -----------
    If `cache_dir` provided, writes/reads parquet at
    `{cache_dir}/{station_id}_{year}_{record_type}.parquet`.

    Returns
    -------
    df : DataFrame with columns ['t' (UTC datetime), 'v' (m, datum-referenced)]
    record_type : 'obs_6min' or 'pred_hilo'
    """
    start = dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for rt in ("obs_6min", "pred_hilo"):
            cached = cache_dir / f"{station_id}_{year}_{rt}.parquet"
            if cached.exists():
                return pd.read_parquet(cached), rt

    # Attempt 6-min observations
    df_obs = _fetch_chunked(station_id, start, end,
                            product="water_level", datum=datum)

    # Expected count for full year at 6-min intervals
    days_in_year = (dt.date(year + 1, 1, 1) - start).days
    expected_6min = days_in_year * 24 * 10  # 10 obs per hour
    coverage = 100.0 * len(df_obs) / expected_6min if expected_6min else 0.0

    if coverage >= min_obs_coverage_pct:
        if cache_dir is not None:
            df_obs.to_parquet(cache_dir / f"{station_id}_{year}_obs_6min.parquet")
        return df_obs, "obs_6min"

    # Fall back to high/low predictions
    df_hilo = _fetch_chunked(station_id, start, end,
                             product="predictions", datum=datum,
                             interval="hilo")
    if cache_dir is not None and not df_hilo.empty:
        df_hilo.to_parquet(cache_dir / f"{station_id}_{year}_pred_hilo.parquet")
    return df_hilo, "pred_hilo"


# ---------------------------------------------------------------------------
# Per-minute synthesis from station record
# ---------------------------------------------------------------------------

def _resample_6min_to_minutes(df_obs: pd.DataFrame,
                              sim_start: dt.datetime,
                              sim_end: dt.datetime) -> np.ndarray:
    """Linear interpolation of 6-min obs onto 1-min grid spanning [sim_start, sim_end)."""
    t_minutes = pd.date_range(sim_start, sim_end, freq="1min", inclusive="left")
    s = df_obs.set_index("t")["v"].astype(float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s.reindex(s.index.union(t_minutes)).interpolate(method="time")
    return s.reindex(t_minutes).to_numpy()


def sinusoidal_minute_series_from_hilo(
    df_hilo: pd.DataFrame,
    sim_start: dt.datetime,
    sim_end: dt.datetime,
) -> np.ndarray:
    """
    Build per-minute water surface from high/low events using half-cycle
    cosine interpolation (see timsa.core.synthesize_tidal_series_sinusoidal).

    Events outside [sim_start, sim_end] are clipped; the function pads with
    the nearest event level at either end.
    """
    n_minutes = int((sim_end - sim_start).total_seconds() // 60)

    # Convert event timestamps to minutes-since-sim_start
    times = pd.to_datetime(df_hilo["t"], utc=True)
    rel_min = ((times - sim_start).dt.total_seconds() / 60.0).astype(int).to_numpy()
    levels = df_hilo["v"].astype(float).to_numpy()

    # Keep events that fall within or bracket the simulation window
    mask = (rel_min >= -2 * 1440) & (rel_min <= n_minutes + 2 * 1440)
    rel_min = rel_min[mask]
    levels = levels[mask]

    if rel_min.size < 2:
        # Degenerate: not enough events to interpolate
        return np.full(n_minutes, np.nan, dtype=np.float64)

    # Clip events to valid index range expected by timsa.core helper
    # The helper indexes out[t0:t1] so events must lie within [0, n_minutes].
    # Sort and deduplicate
    order = np.argsort(rel_min)
    rel_min = rel_min[order]
    levels = levels[order]
    uniq = np.concatenate(([True], np.diff(rel_min) > 0))
    rel_min = rel_min[uniq]
    levels = levels[uniq]

    # Pad: ensure first event <= 0 and last event >= n_minutes
    if rel_min[0] > 0:
        rel_min = np.concatenate(([0], rel_min))
        levels = np.concatenate(([levels[0]], levels))
    if rel_min[-1] < n_minutes:
        rel_min = np.concatenate((rel_min, [n_minutes]))
        levels = np.concatenate((levels, [levels[-1]]))

    # Clip negative starts to 0 (preserve first level)
    if rel_min[0] < 0:
        rel_min[0] = 0

    return synthesize_tidal_series_sinusoidal(
        rel_min.astype(int), levels.astype(np.float64), n_minutes
    )


# ---------------------------------------------------------------------------
# Multi-gauge array assembly
# ---------------------------------------------------------------------------

def build_multigauge_minute_array(
    stations_df: pd.DataFrame,
    year: int,
    sim_start: dt.datetime,
    sim_end: dt.datetime,
    datum: str = "MLLW",
    cache_dir: Path | None = None,
    drop_empty: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Assemble the (n_minutes, n_gauges) array TiMSA expects, plus a manifest.

    Parameters
    ----------
    stations_df : DataFrame
        Must have columns ['StationID', 'Name', 'Latitude', 'Longitude'].
    year : int
        Reference tidal year (e.g., 2010 or 2025).
    sim_start, sim_end : datetime
        Simulation window (typically Jan 1 to Jan 1 next year, UTC).
    datum : str
        CO-OPS datum (default 'MLLW').
    cache_dir : Path
        If provided, station records are cached as parquet.
    drop_empty : bool
        If True, stations with no usable record are excluded from the output.
        The returned manifest still lists them with record_type='empty'.
    verbose : bool
        Print per-station progress.

    Returns
    -------
    gauge_wdepths : np.ndarray (n_minutes, n_kept_gauges)
    manifest : DataFrame indexed by gauge_idx (0..n_kept-1) with columns
        [station_id, name, lat, lon, record_type, n_records, gap_pct]
        Note: gauge_idx is 0-based here; gauge_zones raster will use
        (gauge_idx + 1) to match the C reference 1-based convention.
    """
    n_minutes = int((sim_end - sim_start).total_seconds() // 60)
    series_list = []
    manifest_rows = []

    for _, row in stations_df.iterrows():
        sid = str(row["StationID"])
        name = row["Name"]
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        try:
            df, rec_type = fetch_station_record(
                sid, year, datum=datum, cache_dir=cache_dir
            )
        except Exception as e:
            if verbose:
                print(f"  [{sid}] {name}: fetch failed ({e}); skipping")
            manifest_rows.append({
                "station_id": sid, "name": name, "lat": lat, "lon": lon,
                "record_type": "error", "n_records": 0, "gap_pct": 100.0,
            })
            continue

        if df.empty:
            if verbose:
                print(f"  [{sid}] {name}: no data; skipping")
            manifest_rows.append({
                "station_id": sid, "name": name, "lat": lat, "lon": lon,
                "record_type": "empty", "n_records": 0, "gap_pct": 100.0,
            })
            continue

        if rec_type == "obs_6min":
            series = _resample_6min_to_minutes(df, sim_start, sim_end)
        else:  # pred_hilo
            series = sinusoidal_minute_series_from_hilo(df, sim_start, sim_end)

        gap_pct = 100.0 * np.isnan(series).sum() / n_minutes

        if verbose:
            print(f"  [{sid}] {name}: {rec_type}, n={len(df)}, gap={gap_pct:.1f}%")

        if drop_empty and gap_pct >= 100.0:
            manifest_rows.append({
                "station_id": sid, "name": name, "lat": lat, "lon": lon,
                "record_type": rec_type, "n_records": len(df), "gap_pct": gap_pct,
            })
            continue

        series_list.append(series)
        manifest_rows.append({
            "station_id": sid, "name": name, "lat": lat, "lon": lon,
            "record_type": rec_type, "n_records": len(df), "gap_pct": gap_pct,
        })

    if not series_list:
        raise RuntimeError("No usable station records assembled.")

    # Stack to (n_minutes, n_kept)
    gauge_wdepths = np.column_stack(series_list)

    # Build manifest: only kept rows get a gauge_idx, in order
    manifest = pd.DataFrame(manifest_rows)
    kept_mask = ~manifest["record_type"].isin(["error", "empty"])
    if drop_empty:
        manifest_kept = manifest[kept_mask].reset_index(drop=True).copy()
        manifest_kept.insert(0, "gauge_idx", np.arange(len(manifest_kept)))
        manifest_full = manifest.copy()
        manifest_full["gauge_idx"] = np.nan
        manifest_full.loc[kept_mask.values, "gauge_idx"] = manifest_kept["gauge_idx"].values
    else:
        manifest_full = manifest.copy()
        manifest_full.insert(0, "gauge_idx", np.arange(len(manifest_full)))

    return gauge_wdepths, manifest_full


# ---------------------------------------------------------------------------
# M2 amplitude scaling sensitivity
# ---------------------------------------------------------------------------

M2_PERIOD_HOURS = 12.4206012  # Principal lunar semidiurnal


def apply_m2_amplitude_scaling(
    wl_minutes: np.ndarray,
    pct: float,
    sample_rate_min: float = 1.0,
) -> np.ndarray:
    """
    Scale the M2 tidal component of a water-level series by (1 + pct/100).

    Uses a least-squares harmonic fit at the M2 frequency to extract the
    M2 sine/cosine coefficients, scales them, and reconstructs the series
    with the scaled M2 contribution. All other frequency content (S2, K1,
    O1, residuals, mean) is preserved exactly.

    Appropriate as a first-order sensitivity test for the Lower Keys, which
    are strongly M2-dominated (M2 form-number << 0.25). For mixed or
    diurnal-dominated regimes, use a full harmonic decomposition instead.

    Parameters
    ----------
    wl_minutes : array of float
        Per-minute water-level series (NaNs allowed; handled by mask).
    pct : float
        Percent change to apply to M2 amplitude. e.g. +10.0 for +10%, -10.0 for -10%.
    sample_rate_min : float
        Sample rate in minutes (default 1.0).

    Returns
    -------
    np.ndarray
        Water-level series with M2 amplitude scaled.
    """
    if pct == 0.0:
        return wl_minutes.copy()

    n = wl_minutes.size
    t_hours = np.arange(n) * sample_rate_min / 60.0
    omega = 2.0 * np.pi / M2_PERIOD_HOURS  # rad / hour

    # Fit M2: wl = a*cos(omega*t) + b*sin(omega*t) + residual
    valid = ~np.isnan(wl_minutes)
    if valid.sum() < 4:
        return wl_minutes.copy()

    c = np.cos(omega * t_hours[valid])
    s = np.sin(omega * t_hours[valid])
    # Design matrix; mean handled implicitly via residual (don't subtract here
    # so we preserve any DC offset in the output)
    A = np.column_stack([c, s])
    coef, *_ = np.linalg.lstsq(A, wl_minutes[valid], rcond=None)
    a, b = coef

    # Original M2 contribution (full series, including NaN slots)
    m2_full = a * np.cos(omega * t_hours) + b * np.sin(omega * t_hours)

    # Scaled M2 contribution
    scale = 1.0 + pct / 100.0
    m2_scaled = scale * m2_full

    # Output: original − original_M2 + scaled_M2  = original + (scale-1)*M2
    out = wl_minutes + (scale - 1.0) * m2_full
    return out


# ---------------------------------------------------------------------------
# Native-interval assembly
# ---------------------------------------------------------------------------

def build_multigauge_array(
    stations_df: "pd.DataFrame",
    year: int,
    sim_start: "dt.datetime",
    sim_end: "dt.datetime",
    record_interval_min: int = 6,
    datum: str = "MLLW",
    cache_dir: "Path | None" = None,
    hilo_interp: str = "sinusoidal",
    drop_empty: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Assemble an (n_records, n_gauges) array at a chosen record interval.

    `build_multigauge_minute_array` above always produces a 1-minute series,
    which forces an interpolation step even when the underlying record is
    6-minute observations. This variant keeps the record at its native
    resolution by default, so TiMSA can step on the observations themselves
    (see timsa.core, prescribed mode). Interpolating up to 1 minute remains
    available by passing record_interval_min=1.

    Stations returning high/low predictions rather than observations are
    always interpolated, since hi/lo events are sparse and irregular by
    nature; `hilo_interp` selects 'sinusoidal' (tide_wdchange.c) or 'linear'.

    Returns
    -------
    (values, manifest, record_interval_min)
        values : (n_records, n_kept_gauges) float64
        manifest : DataFrame as in build_multigauge_minute_array
    """
    import datetime as _dt

    if 1440 % int(record_interval_min) != 0:
        raise ValueError(
            f"record_interval_min={record_interval_min} must divide 1440 evenly"
        )
    ri = int(record_interval_min)

    grid = pd.date_range(sim_start, sim_end, freq=f"{ri}min", inclusive="left")
    n_rec = len(grid)

    series_list, manifest_rows = [], []

    for _, row in stations_df.iterrows():
        sid = str(row["StationID"])
        name = row.get("Name", sid)
        lat, lon = float(row["Latitude"]), float(row["Longitude"])

        try:
            df, rec_type = fetch_station_record(
                sid, year, datum=datum, cache_dir=cache_dir
            )
        except Exception as e:
            if verbose:
                print(f"  [{sid}] {name}: fetch failed ({e}); skipping")
            manifest_rows.append({"station_id": sid, "name": name, "lat": lat,
                                  "lon": lon, "record_type": "error",
                                  "n_records": 0, "gap_pct": 100.0})
            continue

        if df.empty:
            manifest_rows.append({"station_id": sid, "name": name, "lat": lat,
                                  "lon": lon, "record_type": "empty",
                                  "n_records": 0, "gap_pct": 100.0})
            if verbose:
                print(f"  [{sid}] {name}: no data; skipping")
            continue

        if rec_type == "obs_6min":
            # Snap observations onto the target grid. When ri == 6 and the
            # record is complete this is a straight reindex, no interpolation.
            s = df.set_index("t")["v"].astype(float)
            s = s[~s.index.duplicated(keep="first")].sort_index()
            if ri == 6:
                series = s.reindex(grid).to_numpy(dtype=np.float64)
            else:
                s = s.reindex(s.index.union(grid)).interpolate(method="time")
                series = s.reindex(grid).to_numpy(dtype=np.float64)
        else:  # pred_hilo — sparse events, always interpolated
            minutes = sinusoidal_minute_series_from_hilo(df, sim_start, sim_end) \
                if hilo_interp == "sinusoidal" else None
            if minutes is None:
                s = df.set_index("t")["v"].astype(float).sort_index()
                s = s.reindex(s.index.union(grid)).interpolate(method="time")
                series = s.reindex(grid).to_numpy(dtype=np.float64)
            else:
                series = minutes[::ri][:n_rec]
                if series.size < n_rec:
                    series = np.pad(series, (0, n_rec - series.size),
                                    constant_values=np.nan)

        gap_pct = 100.0 * float(np.isnan(series).sum()) / max(n_rec, 1)
        if verbose:
            print(f"  [{sid}] {name}: {rec_type}, n={len(df)}, gap={gap_pct:.1f}%")

        manifest_rows.append({"station_id": sid, "name": name, "lat": lat,
                              "lon": lon, "record_type": rec_type,
                              "n_records": len(df), "gap_pct": gap_pct})
        if drop_empty and gap_pct >= 100.0:
            continue
        series_list.append(series)

    if not series_list:
        raise RuntimeError(
            "No usable station records assembled. Check the station list, the "
            "year, and network access to the CO-OPS API."
        )

    values = np.column_stack(series_list)

    manifest = pd.DataFrame(manifest_rows)
    kept = ~manifest["record_type"].isin(["error", "empty"]) & (manifest["gap_pct"] < 100.0)
    manifest["gauge_idx"] = np.nan
    manifest.loc[kept, "gauge_idx"] = np.arange(int(kept.sum()))

    return values, manifest, ri
