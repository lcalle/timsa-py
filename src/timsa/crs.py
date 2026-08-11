"""
timsa.crs
=========
Coordinate-reference-system helpers.

Replaces the hardcoded "EPSG:32617" (UTM 17N) that the SLR pipeline carried in
config.yaml and in cudem.mosaic_and_reproject_cudem's default. That value is
correct for the Florida Keys and wrong everywhere else, which is the first
thing a new domain would hit.
"""

from __future__ import annotations


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """
    EPSG code of the WGS84 UTM zone containing a point.

    Northern hemisphere -> 326xx, southern -> 327xx.
    """
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude out of range: {lon}")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude out of range: {lat}")

    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def utm_epsg_from_domain(domain) -> str:
    """UTM EPSG for a Domain, taken at its centroid."""
    lon, lat = domain.centroid
    return utm_epsg_from_lonlat(lon, lat)


def resolve_target_crs(domain, configured: str | None = None) -> str:
    """
    Target projected CRS for a run.

    An explicit config value wins; otherwise the UTM zone is derived from the
    domain centroid. A domain spanning more than about two UTM zones will
    distort at its edges, so warn rather than silently proceed.
    """
    if configured:
        return str(configured)

    import warnings

    xmin, _, xmax, _ = domain.bbox
    span_zones = int((xmax + 180.0) / 6.0) - int((xmin + 180.0) / 6.0)
    if span_zones >= 2:
        warnings.warn(
            f"domain {domain.name!r} spans {span_zones + 1} UTM zones; a single "
            f"UTM projection will distort near the edges. Consider an equal-area "
            f"projection set explicitly via domain.crs.",
            stacklevel=2,
        )
    return utm_epsg_from_domain(domain)
