"""
timsa.solar
===========
Sunrise/sunset tables for the daylight constraint.

Lifted from preprocess_data.py with one change: the location now comes from a
Domain rather than being passed as loose lat/lon, so a run's daylight table is
derived from the same geometry as everything else.

Returns minutes-of-day, matching the C reference's sun[day][0..1] table.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np


def days_in_year(year: int) -> int:
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return 366 if leap else 365


def sunrise_sunset_table(
    lat: float,
    lon: float,
    year: int,
    out_csv: str | Path | None = None,
) -> np.ndarray:
    """
    Sunrise/sunset for each day of the year, as minutes from midnight UTC.

    Uses `astral` when available and falls back to an analytic approximation
    otherwise. Returns shape (n_days, 2): [sunrise_min, sunset_min].
    """
    try:
        from astral import LocationInfo
        from astral.sun import sun as astral_sun
    except ImportError:
        out = _fallback_sunrise_sunset(lat, lon, year)
    else:
        import datetime as _dt

        loc = LocationInfo(latitude=lat, longitude=lon, timezone="UTC")
        n_days = days_in_year(year)
        out = np.zeros((n_days, 2), dtype=np.int32)
        for d in range(n_days):
            date = dt.date(year, 1, 1) + dt.timedelta(days=d)
            s = astral_sun(loc.observer, date=date, tzinfo=_dt.timezone.utc)
            out[d, 0] = s["sunrise"].hour * 60 + s["sunrise"].minute
            out[d, 1] = s["sunset"].hour * 60 + s["sunset"].minute

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_csv, out, delimiter=",", fmt="%d",
                   header="sunrise_min,sunset_min", comments="")
    return out


def sunrise_sunset_for_domain(domain, year: int, out_csv=None) -> np.ndarray:
    """Sunrise/sunset table taken at the domain centroid."""
    lon, lat = domain.centroid
    return sunrise_sunset_table(lat, lon, year, out_csv=out_csv)


def load_sunrise_sunset(path: str | Path) -> np.ndarray:
    """Read a two-column sunrise/sunset CSV back into an (n_days, 2) array."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns (sunrise_min, sunset_min)")
    return arr


def _fallback_sunrise_sunset(lat: float, lon: float, year: int) -> np.ndarray:
    """
    Analytic approximation used when `astral` is unavailable.

    Accurate to a few minutes at mid latitudes; adequate for a daylight mask
    at a 1-6 minute timestep, but install `astral` for anything more exacting.
    """
    n_days = days_in_year(year)
    out = np.zeros((n_days, 2), dtype=np.int32)
    lat_rad = np.deg2rad(lat)
    for d in range(n_days):
        gamma = 2 * np.pi * d / n_days
        decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma))
        cos_h = np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1)
        day_hours = 2 * np.rad2deg(np.arccos(cos_h)) / 15
        noon_min = 12 * 60 - 4 * lon
        out[d, 0] = int(noon_min - day_hours * 30)
        out[d, 1] = int(noon_min + day_hours * 30)
    return out
