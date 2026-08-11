"""
timsa.ingest.stations
=====================
Discover NOAA CO-OPS tide stations for a simulation domain.

Replaces `data/get_noaa_bouy_list.py`, which hardcoded both the state filter
(`station.get("state") == "FL"`) and the bounding box:

    LAT_MIN, LAT_MAX = 24.3, 24.9
    LNG_MIN, LNG_MAX = -82.2, -81.0

Those two lines were the whole reason the SLR pipeline could not be pointed at
another site. Here the extent comes from a `Domain`, and the station list is a
generated cache artifact rather than a file tracked in the repo.

The station table is the join point for the rest of the pipeline: its row order
defines gauge index order, which must agree with the gauge-zone raster and the
columns of the water-level array.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

MDAPI_STATIONS = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
)

# Station types worth pulling. 'tidepredictions' includes subordinate stations
# that exist only as predictions; 'waterlevels' is the subset with physical
# gauges reporting observations.
STATION_TYPES = ("tidepredictions", "waterlevels")

COLUMNS = ["StationID", "Name", "Latitude", "Longitude", "State", "StationType"]


class StationDiscoveryError(RuntimeError):
    """Raised when no usable stations can be found for a domain."""


def fetch_station_catalog(
    station_type: str = "tidepredictions",
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch the full national CO-OPS station catalog for one station type.

    Returns a DataFrame with COLUMNS. The catalog is national and modest in
    size, so it is fetched whole and filtered locally rather than queried
    per-region — that keeps the domain filter under our control instead of
    NOAA's region definitions.
    """
    if station_type not in STATION_TYPES:
        raise ValueError(f"station_type must be one of {STATION_TYPES}")

    r = requests.get(MDAPI_STATIONS, params={"type": station_type}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    stations = payload.get("stations", payload.get("stationList", []))
    rows = []
    for st in stations:
        lat, lng = st.get("lat"), st.get("lng")
        if lat is None or lng is None:
            continue
        rows.append({
            "StationID": str(st.get("id")),
            "Name": st.get("name"),
            "Latitude": float(lat),
            "Longitude": float(lng),
            "State": st.get("state"),
            "StationType": station_type,
        })

    if not rows:
        raise StationDiscoveryError(
            f"CO-OPS returned no stations for type={station_type!r}"
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def discover_stations(
    domain,
    station_types: tuple = STATION_TYPES,
    buffer_km: float | None = None,
    min_stations: int = 1,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Find the CO-OPS stations that force a domain.

    Parameters
    ----------
    domain : timsa.domain.Domain
        Supplies the extent and the search buffer.
    station_types : tuple
        Catalogs to search. Results are de-duplicated by StationID, keeping
        the first occurrence, so order the tuple by preference.
    buffer_km : float, optional
        Overrides `domain.gauge_search_buffer_km`.
    min_stations : int
        Fail if fewer survive the filter. Tide gauges are sparse; an empty
        result here otherwise surfaces much later as an opaque error.
    cache_dir : path, optional
        Writes `stations_{domain.cache_key()}.csv` and reads it back unless
        `refresh` is set.
    refresh : bool
        Ignore any cached table and re-query.

    Returns
    -------
    DataFrame with COLUMNS plus 'gauge_idx' (0-based), sorted by StationID so
    the ordering is stable across runs.
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f"stations_{domain.cache_key()}.csv"
        if cache_path.exists() and not refresh:
            df = pd.read_csv(cache_path, dtype={"StationID": str})
            if verbose:
                print(f"  stations: {len(df)} from cache ({cache_path.name})")
            return df

    frames = []
    for st in station_types:
        try:
            frames.append(fetch_station_catalog(st))
        except Exception as e:
            if verbose:
                print(f"  station catalog {st!r} failed: {e}")
    if not frames:
        raise StationDiscoveryError(
            "could not fetch any CO-OPS station catalog; check network access"
        )

    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.drop_duplicates(subset="StationID", keep="first")

    # Domain.filter_stations raises with actionable guidance when the result
    # is too small (wrong lon/lat order, too tight a buffer).
    out = domain.filter_stations(
        catalog,
        lon_col="Longitude",
        lat_col="Latitude",
        buffer_km=buffer_km,
        min_stations=min_stations,
    )

    out = out.sort_values("StationID").reset_index(drop=True)
    out.insert(0, "gauge_idx", range(len(out)))

    if verbose:
        km = buffer_km if buffer_km is not None else domain.gauge_search_buffer_km
        print(f"  stations: {len(out)} within {km} km of {domain.name}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)

    return out


def load_stations(path: str | Path) -> pd.DataFrame:
    """
    Read a station table, tolerating the column spellings used across the
    older scripts ('lat'/'lon' as well as 'Latitude'/'Longitude').
    """
    df = pd.read_csv(path, dtype={"StationID": str})
    renames = {"lat": "Latitude", "lon": "Longitude", "lng": "Longitude",
               "id": "StationID", "name": "Name"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    missing = [c for c in ("StationID", "Latitude", "Longitude")
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: station table is missing {missing}. Expected columns "
            f"StationID, Name, Latitude, Longitude."
        )
    if "gauge_idx" not in df.columns:
        df.insert(0, "gauge_idx", range(len(df)))
    return df
