"""
timsa.ingest.vdatum
===================
MLLW -> NAVD88 vertical datum offsets for tide stations.

Merges the SLR pipeline's `data_ingest/vdatum.py` with the standalone
`data/vdatum_batch.py` script, and removes three site-specific assumptions:

  1. `vdatum_batch.py` hardcoded INPUT_FILE / OUTPUT_FILE. Now a function over
     a station DataFrame.
  2. It sent `region="contiguous"` unconditionally, which is wrong for Alaska,
     Hawaii, Puerto Rico, Guam, and American Samoa. The region is now derived
     from the station coordinate.
  3. `resolve_offset()` had `try_coops_first=False` and ignored its own
     `fetch_coops_datum_offset`. That was a defensible Lower Keys choice —
     subordinate stations there mostly lack published datums — but as a
     general default it discards the most authoritative source available.
     The resolution order is now CO-OPS -> VDatum API -> manual CSV, each
     step recorded per station so the provenance is visible.

Offset convention
-----------------
    offset_m = NAVD88_reading - MLLW_reading

so a value reported in MLLW converts as `val_navd88 = val_mllw + offset_m`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COOPS_DATUMS_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
    "{sid}/datums.json"
)
VDATUM_API_URL = "https://vdatum.noaa.gov/vdatumweb/api/convert"

OFFSET_COLUMN = "vdatum_navd88_offset_m"
SOURCE_COLUMN = "vdatum_offset_source"

RESOLUTION_ORDER = ("coops", "vdatum_api", "manual_csv")

# Source label for offsets filled by spatial interpolation from resolved
# neighbours (see idw_fill_offsets). Kept distinct so provenance stays visible.
IDW_SOURCE = "idw"


class VdatumError(RuntimeError):
    """Raised when datum offsets cannot be resolved for required stations."""


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

# VDatum region codes, with approximate lon/lat extents. Ordered so that the
# first match wins; `contiguous` is last because its box overlaps others.
_VDATUM_REGIONS = (
    ("ak",         (-180.0, 50.0, -129.0, 72.0)),
    ("as",         (-172.0, -16.0, -168.0, -13.0)),
    ("gcnmi",      (144.0, 13.0, 146.5, 21.0)),
    ("prvi",       (-68.0, 17.0, -64.0, 19.0)),
    ("westcoast",  (-180.0, 18.0, -154.0, 23.0)),   # Hawaii
    ("contiguous", (-128.0, 23.0, -65.0, 50.0)),
)


def vdatum_region(lon: float, lat: float) -> str:
    """
    VDatum region code containing a point.

    Falls back to 'contiguous' when nothing matches, which is what the old
    script always used — but now it is a fallback rather than an assumption,
    and a warning is emitted so a mis-located domain is visible.
    """
    for code, (xmin, ymin, xmax, ymax) in _VDATUM_REGIONS:
        if xmin <= lon <= xmax and ymin <= lat <= ymax:
            return code
    import warnings
    warnings.warn(
        f"({lon:.4f}, {lat:.4f}) falls outside every known VDatum region; "
        f"defaulting to 'contiguous'. Offsets for this station may be wrong.",
        stacklevel=2,
    )
    return "contiguous"


def region_for_domain(domain) -> str:
    """VDatum region for a Domain, taken at its centroid."""
    lon, lat = domain.centroid
    return vdatum_region(lon, lat)


# ---------------------------------------------------------------------------
# Source 1: CO-OPS published datums
# ---------------------------------------------------------------------------

def fetch_coops_datum_offset(
    station_id: str,
    cache_dir: str | Path | None = None,
    timeout: int = 30,
) -> float | None:
    """
    Published NAVD88 - MLLW offset from the CO-OPS metadata API.

    Returns None when the station has no published datums, which is common for
    subordinate stations that exist only as predictions.
    """
    data = None
    cached = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{station_id}_datums.json"
        if cached.exists():
            try:
                data = json.loads(cached.read_text())
            except json.JSONDecodeError:
                cached.unlink(missing_ok=True)

    if data is None:
        try:
            r = requests.get(COOPS_DATUMS_URL.format(sid=station_id), timeout=timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            if cached is not None:
                cached.write_text(json.dumps(data, indent=2))
        except (requests.RequestException, ValueError):
            return None

    datums = {str(d.get("name", "")).upper(): d.get("value")
              for d in (data.get("datums") or [])}
    units = str(data.get("units") or "").lower()
    to_m = 0.3048 if units.startswith("f") else 1.0

    navd = datums.get("NAVD88") or datums.get("NAVD")
    mllw = datums.get("MLLW")
    if navd is None or mllw is None:
        return None
    try:
        return (float(navd) - float(mllw)) * to_m
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Source 2: VDatum transformation API
# ---------------------------------------------------------------------------

def fetch_vdatum_offset(
    latitude: float,
    longitude: float,
    region: str | None = None,
    timeout: int = 30,
) -> float | None:
    """
    MLLW -> NAVD88 offset at a coordinate, from the VDatum API.

    Converts a zero-height MLLW value, so the returned target height IS the
    offset. Region is derived from the coordinate unless given.
    """
    params = {
        "s_x": longitude,
        "s_y": latitude,
        "s_z": 0.0,
        "s_h_frame": "NAD83_2011",
        "s_coor": "geo",
        "s_v_frame": "MLLW",
        "s_v_unit": "m",
        "t_h_frame": "NAD83_2011",
        "t_coor": "geo",
        "t_v_frame": "NAVD88",
        "t_v_unit": "m",
        "region": region or vdatum_region(longitude, latitude),
    }
    try:
        r = requests.get(VDATUM_API_URL, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        val = r.json().get("t_z")
        return None if val is None else float(val)
    except (requests.RequestException, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Source 3: manual CSV
# ---------------------------------------------------------------------------

def load_manual_offsets(csv_path: str | Path | None) -> dict:
    """
    Load offsets from a CSV carrying StationID and `vdatum_navd88_offset_m`.

    Returns an empty dict when the path is None or absent, so this can always
    be used as a last-resort source without a branch at the call site.
    """
    if csv_path is None:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path, dtype={"StationID": str})
    if OFFSET_COLUMN not in df.columns:
        raise VdatumError(f"{csv_path}: expected column {OFFSET_COLUMN!r}")

    out = {}
    for sid, val in zip(df["StationID"].astype(str), df[OFFSET_COLUMN]):
        try:
            out[sid] = float(val)
        except (TypeError, ValueError):
            continue          # 'ERROR' rows written by the old batch script
    return out


# ---------------------------------------------------------------------------
# Source 4 (last resort): inverse-distance interpolation from resolved neighbours
# ---------------------------------------------------------------------------

def _haversine_km(lat0: float, lon0: float,
                  lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays of points."""
    r = 6371.0088
    p0 = np.radians(lat0)
    p = np.radians(lat)
    dphi = np.radians(lat - lat0)
    dlmb = np.radians(lon - lon0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def idw_fill_offsets(
    stations_df: pd.DataFrame,
    power: float = 2.0,
    k: int | None = None,
    coincident_tol_m: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fill missing MLLW->NAVD88 offsets by inverse-distance weighting from the
    stations that ARE resolved.

    This is a last resort for the occasional station that no authoritative
    source can supply (CO-OPS has no published datums, VDatum returns nothing,
    no manual value). It assumes the offset varies smoothly over the domain --
    true for a compact tidal cluster, questionable across a large or
    hydrodynamically complex extent, so filled rows are tagged IDW_SOURCE and
    should be spot-checked against VDatum for any critical gauge.

    Parameters
    ----------
    power : float
        IDW exponent. 2.0 gives the usual inverse-square weighting.
    k : int, optional
        Use only the k nearest resolved stations. None uses all of them.
    coincident_tol_m : float
        If a target sits within this distance of a resolved station, copy that
        station's value outright instead of weighting (avoids 1/0).

    Returns a copy with NaN offsets filled and their source set to IDW_SOURCE.
    Rows already resolved are untouched.
    """
    if OFFSET_COLUMN not in stations_df.columns:
        raise VdatumError(
            f"{OFFSET_COLUMN!r} missing; run batch_resolve_offsets first"
        )
    df = stations_df.copy()
    lat = df["Latitude"].astype(float).to_numpy()
    lon = df["Longitude"].astype(float).to_numpy()
    off = df[OFFSET_COLUMN].astype(float).to_numpy(copy=True)
    if SOURCE_COLUMN in df.columns:
        src = df[SOURCE_COLUMN].astype(object).to_numpy(copy=True)
    else:
        src = np.array(["none"] * len(df), dtype=object)

    known = ~np.isnan(off)
    miss = np.where(~known)[0]
    if miss.size == 0:
        return df
    if known.sum() == 0:
        raise VdatumError(
            "cannot IDW-fill offsets: no station has a resolved value to "
            "interpolate from."
        )

    klat, klon, kval = lat[known], lon[known], off[known]
    for i in miss:
        d = _haversine_km(lat[i], lon[i], klat, klon)
        hit = np.where(d <= coincident_tol_m / 1000.0)[0]
        if hit.size:
            off[i] = float(kval[hit[0]])
        else:
            order = np.argsort(d)
            if k is not None:
                order = order[:k]
            w = 1.0 / np.power(d[order], power)
            off[i] = float(np.sum(w * kval[order]) / np.sum(w))
        src[i] = IDW_SOURCE
        if verbose:
            sid = str(df.iloc[i].get("StationID", i))
            print(f"  [{sid}] {off[i]:+.3f} m  ({IDW_SOURCE}, from "
                  f"{k or known.sum()} neighbour(s))")

    df[OFFSET_COLUMN] = off
    df[SOURCE_COLUMN] = src
    return df


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_offset(
    station_id: str,
    latitude: float,
    longitude: float,
    manual_offsets: dict | None = None,
    region: str | None = None,
    order: tuple = RESOLUTION_ORDER,
    coops_cache_dir: str | Path | None = None,
) -> tuple:
    """
    Resolve one station's MLLW -> NAVD88 offset.

    Returns (offset_m or None, source_label). `order` controls which sources
    are tried and in what sequence; pass ('manual_csv',) to reproduce the
    Lower Keys behaviour of trusting a curated CSV exclusively.
    """
    manual_offsets = manual_offsets or {}

    for source in order:
        if source == "coops":
            v = fetch_coops_datum_offset(station_id, cache_dir=coops_cache_dir)
        elif source == "vdatum_api":
            v = fetch_vdatum_offset(latitude, longitude, region=region)
        elif source == "manual_csv":
            v = manual_offsets.get(str(station_id))
        else:
            raise ValueError(f"unknown offset source {source!r}")
        if v is not None:
            return float(v), source

    return None, "none"


def batch_resolve_offsets(
    stations_df: pd.DataFrame,
    domain=None,
    manual_csv: str | Path | None = None,
    order: tuple = RESOLUTION_ORDER,
    coops_cache_dir: str | Path | None = None,
    out_csv: str | Path | None = None,
    polite_sec: float = 0.5,
    require_all: bool = True,
    idw_fill: bool = False,
    idw_power: float = 2.0,
    idw_k: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Resolve offsets for every station in a table.

    This is `vdatum_batch.main()` as a function: no hardcoded filenames, the
    region derived from the domain, and the source recorded per station.

    Parameters
    ----------
    stations_df : DataFrame
        Requires StationID, Latitude, Longitude.
    domain : Domain, optional
        Supplies the VDatum region. Falls back to per-station derivation.
    manual_csv : path, optional
        Curated offsets used as the last resort.
    require_all : bool
        Raise if any station is left unresolved. A missing offset silently
        defaulting to zero would put that gauge on the wrong datum, which
        shifts every depth in its zone — worth failing loudly for.

    Returns
    -------
    A copy of `stations_df` with `vdatum_navd88_offset_m` and
    `vdatum_offset_source` appended.
    """
    manual = load_manual_offsets(manual_csv)
    region = region_for_domain(domain) if domain is not None else None

    offsets, sources = [], []
    for _, row in stations_df.iterrows():
        sid = str(row["StationID"])
        lat, lon = float(row["Latitude"]), float(row["Longitude"])

        val, src = resolve_offset(
            sid, lat, lon,
            manual_offsets=manual,
            region=region,
            order=order,
            coops_cache_dir=coops_cache_dir,
        )
        offsets.append(np.nan if val is None else val)
        sources.append(src)

        if verbose:
            shown = "unresolved" if val is None else f"{val:+.3f} m"
            print(f"  [{sid}] {shown}  ({src})")
        if polite_sec and src in ("coops", "vdatum_api"):
            time.sleep(polite_sec)

    out = stations_df.copy()
    out[OFFSET_COLUMN] = offsets
    out[SOURCE_COLUMN] = sources

    if idw_fill:
        out = idw_fill_offsets(out, power=idw_power, k=idw_k, verbose=verbose)

    unresolved = out[out[OFFSET_COLUMN].isna()]
    if len(unresolved) and require_all:
        ids = ", ".join(unresolved["StationID"].astype(str).tolist())
        raise VdatumError(
            f"no MLLW->NAVD88 offset for station(s): {ids}. Supply them via "
            f"manual_csv (columns StationID, {OFFSET_COLUMN}), enable IDW fill "
            f"(idw_fill=True), or drop those stations. Defaulting them to zero "
            f"would place those gauges on the wrong vertical datum."
        )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    if verbose:
        counts = out[SOURCE_COLUMN].value_counts().to_dict()
        print(f"  offset sources: {counts}")

    return out


def offsets_as_dict(stations_df: pd.DataFrame) -> dict:
    """{StationID: offset_m} from a table produced by batch_resolve_offsets."""
    if OFFSET_COLUMN not in stations_df.columns:
        raise VdatumError(
            f"station table has no {OFFSET_COLUMN!r}; run batch_resolve_offsets first"
        )
    return {
        str(s): float(v)
        for s, v in zip(stations_df["StationID"], stations_df[OFFSET_COLUMN])
        if pd.notna(v)
    }


def apply_offset_to_series(wl: np.ndarray, offset_m: float) -> np.ndarray:
    """Convert a water-level series from MLLW to NAVD88."""
    return wl + offset_m
