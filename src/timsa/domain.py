"""
domain.py
=========
Resolve a simulation domain from config into a single object that every
ingest step consumes.

Replaces the hard-coded Florida Keys extent that the SLR pipeline carried
in fixed CSVs and scripts. A domain is declared one of two ways:

    domain:
      name: lower_keys
      bbox: [-81.90, 24.50, -81.20, 24.80]     # xmin ymin xmax ymax, EPSG:4326

or:

    domain:
      name: my_site
      boundary_file: aoi/my_site.geojson       # .geojson / .shp / .gpkg
      boundary_layer: aoi                      # optional, for multi-layer gpkg

or, as a point and a radius (a square bbox is derived; pass shape: circle
to also attach a circular AOI polygon for geometry-based clipping):

    domain:
      name: my_site
      center: [-82.40, 27.85]                  # [lon, lat], EPSG:4326
      radius_km: 8                             # half-width of the derived bbox
      shape: box                               # box (default) | circle

Common optional keys:

      gauge_search_buffer_km: 40   # how far beyond the AOI to look for gauges
      resolution_m: 30             # target DEM resolution
      crs: EPSG:4326               # CRS of `bbox` (boundary files carry theirs)

Downstream consumers:
    stations  -> Domain.bbox_buffered() filters the CO-OPS station list
    vdatum    -> only stations surviving that filter are batched
    cudem     -> Domain.bbox selects tiles; Domain.geometry clips
    gauge_ref -> cKDTree over the filtered stations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

WGS84 = "EPSG:4326"
DEFAULT_GAUGE_BUFFER_KM = 40.0


class DomainError(ValueError):
    """Raised when a domain spec is missing, ambiguous, or unusable."""


# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """
    A simulation domain in geographic coordinates (EPSG:4326).

    Attributes
    ----------
    name : str
        Short slug used in cache filenames and output paths.
    bbox : tuple
        (xmin, ymin, xmax, ymax) in EPSG:4326.
    geometry : shapely geometry or None
        Full AOI polygon when a boundary file was supplied; None when the
        domain was declared as a bare bbox (in which case the bbox polygon
        is the AOI).
    gauge_search_buffer_km : float
        Radius beyond the AOI within which tide gauges are still considered
        to force the domain.
    resolution_m : float or None
        Target DEM resolution.
    source : str
        'bbox' or the boundary file path, for provenance in logs/metadata.
    """

    name: str
    bbox: tuple
    geometry: object | None = None
    gauge_search_buffer_km: float = DEFAULT_GAUGE_BUFFER_KM
    resolution_m: float | None = None
    source: str = "bbox"
    extra: dict = field(default_factory=dict)

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "Domain":
        """
        Build a Domain from the ``domain:`` block of a config dict.

        Exactly one of `bbox`, `boundary_file`, or `center` (+ `radius_km`)
        must be present.
        """
        if "domain" in cfg:
            cfg = cfg["domain"]

        name = str(cfg.get("name") or "domain")
        has_bbox = cfg.get("bbox") is not None
        has_file = cfg.get("boundary_file") is not None
        has_center = cfg.get("center") is not None

        n_specs = sum((has_bbox, has_file, has_center))
        if n_specs > 1:
            raise DomainError(
                "domain: specify exactly one of 'bbox', 'boundary_file', or "
                "'center' (+ 'radius_km'); got more than one"
            )
        if n_specs == 0:
            raise DomainError(
                "domain: one of 'bbox', 'boundary_file', or 'center' "
                "(+ 'radius_km') is required"
            )

        buffer_km = float(
            cfg.get("gauge_search_buffer_km", DEFAULT_GAUGE_BUFFER_KM)
        )
        resolution_m = cfg.get("resolution_m")
        resolution_m = float(resolution_m) if resolution_m is not None else None

        if has_center:
            center = cfg["center"]
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise DomainError(
                    "domain.center must be [lon, lat] in EPSG:4326"
                )
            lon, lat = float(center[0]), float(center[1])
            radius_km = cfg.get("radius_km")
            if radius_km is None:
                raise DomainError(
                    "domain.center requires domain.radius_km (km)"
                )
            radius_km = float(radius_km)
            if radius_km <= 0:
                raise DomainError(
                    f"domain.radius_km must be positive; got {radius_km}"
                )
            shape = str(cfg.get("shape", "box")).lower()
            if shape not in ("box", "circle"):
                raise DomainError(
                    f"domain.shape must be 'box' or 'circle'; got {shape!r}"
                )

            bbox = _bbox_from_center(lon, lat, radius_km)
            _validate_bbox(bbox)
            geom = _circle_polygon(lon, lat, radius_km) if shape == "circle" \
                else None
            return cls(
                name=name,
                bbox=bbox,
                geometry=geom,
                gauge_search_buffer_km=buffer_km,
                resolution_m=resolution_m,
                source=f"center+radius ({shape})",
                extra=dict(cfg),
            )

        if has_bbox:
            bbox = tuple(float(v) for v in cfg["bbox"])
            if len(bbox) != 4:
                raise DomainError(
                    "domain.bbox must be [xmin, ymin, xmax, ymax]"
                )
            crs = cfg.get("crs") or WGS84
            crs = str(crs)
            if crs.upper() not in (WGS84.upper(), "WGS84", "EPSG:4326", "NONE"):
                bbox = _reproject_bbox(bbox, crs, WGS84)
            _validate_bbox(bbox)
            return cls(
                name=name,
                bbox=bbox,
                geometry=None,
                gauge_search_buffer_km=buffer_km,
                resolution_m=resolution_m,
                source="bbox",
                extra=dict(cfg),
            )

        path = Path(cfg["boundary_file"])
        layer = cfg.get("boundary_layer")
        geom, bbox = _read_boundary(path, layer)
        _validate_bbox(bbox)
        return cls(
            name=name,
            bbox=bbox,
            geometry=geom,
            gauge_search_buffer_km=buffer_km,
            resolution_m=resolution_m,
            source=str(path),
            extra=dict(cfg),
        )

    # -- geometry accessors ---------------------------------------------

    @property
    def centroid(self) -> tuple:
        xmin, ymin, xmax, ymax = self.bbox
        return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

    def bbox_buffered(self, buffer_km: float | None = None) -> tuple:
        """
        The bbox expanded by `buffer_km` (default: gauge_search_buffer_km),
        converted from km to degrees with a latitude correction on longitude.
        """
        km = self.gauge_search_buffer_km if buffer_km is None else float(buffer_km)
        if km <= 0:
            return self.bbox

        xmin, ymin, xmax, ymax = self.bbox
        dlat = km / 110.574
        mid_lat = 0.5 * (ymin + ymax)
        cos_lat = max(math.cos(math.radians(mid_lat)), 1e-6)
        dlon = km / (111.320 * cos_lat)

        return (
            max(xmin - dlon, -180.0),
            max(ymin - dlat, -90.0),
            min(xmax + dlon, 180.0),
            min(ymax + dlat, 90.0),
        )

    def aoi_geometry(self):
        """
        The AOI as a shapely geometry: the boundary polygon if one was
        supplied, otherwise the bbox as a rectangle.
        """
        if self.geometry is not None:
            return self.geometry
        from shapely.geometry import box
        return box(*self.bbox)

    # -- filtering helpers ----------------------------------------------

    def filter_points(
        self,
        lon,
        lat,
        buffer_km: float | None = None,
        use_geometry: bool = False,
    ) -> np.ndarray:
        """
        Boolean mask of which (lon, lat) points fall inside the domain.

        Parameters
        ----------
        buffer_km : float or None
            Expansion applied before testing. Defaults to
            gauge_search_buffer_km. Pass 0 for a strict AOI test.
        use_geometry : bool
            If True and a boundary polygon exists, test against the polygon
            rather than its bbox. Slower; rarely what you want for gauges,
            since a gauge just outside the shoreline still forces the domain.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)

        if use_geometry and self.geometry is not None:
            from shapely.geometry import Point
            km = self.gauge_search_buffer_km if buffer_km is None else float(buffer_km)
            geom = self.geometry
            if km > 0:
                dlat = km / 110.574
                geom = geom.buffer(dlat)  # approximate, degrees
            return np.array([geom.contains(Point(x, y)) for x, y in zip(lon, lat)])

        xmin, ymin, xmax, ymax = self.bbox_buffered(buffer_km)
        return (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)

    def filter_stations(
        self,
        stations_df,
        lon_col: str = "lon",
        lat_col: str = "lat",
        buffer_km: float | None = None,
        min_stations: int = 1,
    ):
        """
        Subset a station DataFrame to those within the buffered domain.

        Raises DomainError if fewer than `min_stations` survive. Tide gauges
        are sparse; a silent empty result here produces a confusing failure
        much further down the pipeline, so fail loudly and early.
        """
        for col in (lon_col, lat_col):
            if col not in stations_df.columns:
                raise DomainError(
                    f"station table has no column {col!r}; "
                    f"available: {list(stations_df.columns)}"
                )

        mask = self.filter_points(
            stations_df[lon_col].to_numpy(),
            stations_df[lat_col].to_numpy(),
            buffer_km=buffer_km,
        )
        out = stations_df.loc[mask].copy()

        if len(out) < min_stations:
            km = self.gauge_search_buffer_km if buffer_km is None else buffer_km
            raise DomainError(
                f"domain {self.name!r}: only {len(out)} station(s) found within "
                f"{km} km of the AOI (need >= {min_stations}). "
                f"Increase domain.gauge_search_buffer_km, or check that the "
                f"bbox/boundary is in EPSG:4326 with lon/lat in that order."
            )
        return out

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "gauge_search_buffer_km": self.gauge_search_buffer_km,
            "resolution_m": self.resolution_m,
            "source": self.source,
        }

    def cache_key(self) -> str:
        """Stable slug for cache filenames: 'lower_keys_30m'."""
        res = f"_{int(self.resolution_m)}m" if self.resolution_m else ""
        return f"{self.name}{res}"

    def __repr__(self) -> str:
        xmin, ymin, xmax, ymax = self.bbox
        return (
            f"Domain(name={self.name!r}, "
            f"bbox=({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}), "
            f"buffer={self.gauge_search_buffer_km}km, source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _km_to_degrees(lat_deg: float, radius_km: float) -> tuple:
    """
    Convert a radius in km to (dlon, dlat) in degrees at a given latitude,
    using the same latitude-corrected constants as Domain.bbox_buffered so
    a center+radius domain and a buffered bbox agree.
    """
    dlat = radius_km / 110.574
    cos_lat = max(math.cos(math.radians(lat_deg)), 1e-6)
    dlon = radius_km / (111.320 * cos_lat)
    return dlon, dlat


def _bbox_from_center(lon: float, lat: float, radius_km: float) -> tuple:
    """
    Square bbox (xmin, ymin, xmax, ymax) enclosing a circle of `radius_km`
    around (lon, lat). `radius_km` is the half-width, so the bbox side is
    2 * radius_km. Clamped to valid lon/lat ranges.
    """
    dlon, dlat = _km_to_degrees(lat, radius_km)
    return (
        max(lon - dlon, -180.0),
        max(lat - dlat, -90.0),
        min(lon + dlon, 180.0),
        min(lat + dlat, 90.0),
    )


def _circle_polygon(lon: float, lat: float, radius_km: float, n: int = 128):
    """
    A closed circular AOI polygon (shapely) of `radius_km` around (lon, lat),
    approximated in degree space with a separate lon/lat scaling so it stays
    close to a true geodesic circle for the small radii typical of coastal
    domains. Used only when domain.shape == 'circle'; the enclosing bbox is
    derived independently and remains the extent for tiling and discovery.
    """
    from shapely.geometry import Polygon

    dlon, dlat = _km_to_degrees(lat, radius_km)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = [(lon + dlon * math.cos(a), lat + dlat * math.sin(a))
            for a in angles]
    ring.append(ring[0])
    return Polygon(ring)


def _validate_bbox(bbox: tuple) -> None:
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        raise DomainError(
            f"invalid bbox {bbox}: expected (xmin, ymin, xmax, ymax) with "
            f"xmin < xmax and ymin < ymax"
        )
    if not (-180.0 <= xmin <= 180.0 and -180.0 <= xmax <= 180.0):
        raise DomainError(f"bbox longitudes out of range: {bbox}")
    if not (-90.0 <= ymin <= 90.0 and -90.0 <= ymax <= 90.0):
        raise DomainError(
            f"bbox latitudes out of range: {bbox}. "
            f"Note the order is (xmin, ymin, xmax, ymax) = (lon, lat, lon, lat)."
        )
    # A lon/lat swap cannot be detected in general -- a swapped bbox is often
    # still a valid bbox somewhere else on the globe. Latitudes beyond 80
    # degrees are the one reliable tell for coastal work, so warn there rather
    # than pretending to catch every case. `Domain.filter_stations` carries the
    # same hint for the more common symptom, an empty station set.
    import warnings
    if max(abs(ymin), abs(ymax)) > 80.0:
        warnings.warn(
            f"bbox {bbox} reaches beyond 80 degrees latitude. If this is not "
            f"deliberate, check the ordering: (lon_min, lat_min, lon_max, "
            f"lat_max).",
            stacklevel=3,
        )


def _reproject_bbox(bbox: tuple, src_crs: str, dst_crs: str) -> tuple:
    from pyproj import Transformer
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin, xmax, ymax = bbox
    xs, ys = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymax, ymax, ymin])
    return (min(xs), min(ys), max(xs), max(ys))


def _read_boundary(path: Path, layer: str | None = None):
    """
    Read a vector boundary file, dissolve to a single geometry, and
    reproject to EPSG:4326. Returns (geometry, bbox).
    """
    if not path.exists():
        raise DomainError(f"domain.boundary_file not found: {path}")

    import geopandas as gpd

    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if len(gdf) == 0:
        raise DomainError(f"boundary file is empty: {path}")

    if gdf.crs is None:
        raise DomainError(
            f"boundary file {path} has no CRS; assign one before use"
        )
    if gdf.crs.to_string() != WGS84:
        gdf = gdf.to_crs(WGS84)

    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
        else gdf.geometry.unary_union

    return geom, tuple(float(v) for v in geom.bounds)
