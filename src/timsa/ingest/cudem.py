"""
timsa.ingest.cudem
==================
CUDEM (NOAA NCEI 1/9 arc-second topobathy) tile acquisition and mosaicking.

Generalized from the SLR pipeline, which hardcoded two things:

    CUDEM_FL_INDEX_URL = ".../NCEI_ninth_Topobathy_2014_8483/FL/index.html"
    target_crs: str = "EPSG:32617"

The tile-name regex was already national; only the index URL was state-scoped.
This module resolves the region directory (or directories) from the domain, and
takes the target CRS from `timsa.crs` so a domain outside Florida projects
correctly.

Host / layout note (2026)
-------------------------
NOAA moved the ninth-arc-second bulk download from
`chs.coast.noaa.gov/htdata/raster2/elevation/...` to the public S3 mirror
`noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/...`. Two consequences the
code now accounts for:

  * The top-level listing writes subdirectory links as RELATIVE hrefs
    (`<a href="FL/index.html">`), not the old trailing-slash form
    (`<a href="FL/">`). `_DIR_PATTERN` matches the new form.
  * The server groups some states into combined directories
    (`AL_nwFL`, `LA_MS`, `MA_NH_ME`) and adds feature-named ones
    (`chesapeake_bay`, `columbia_river`, `southeast`, `northeast_sandy`,
    `rima`, `wash_*`). `_REGION_EXTENTS` is keyed to those exact directory
    names so auto-discovery resolves them from a bbox without a flag.

Puerto Rico, US Virgin Islands, and Hawaii are SEPARATE datasets (their own
NCEI IDs), not subdirectories of 8483, so they are not resolvable here.

Region resolution, in order:
  1. Explicit `regions=[...]` argument or config key. Always wins.
  2. Coarse coastal-extent lookup, intersected against the directories the
     server actually lists.
The extent table is approximate and only used to shortlist directories; the
authoritative check is whether a listed tile's footprint intersects the bbox.
A domain straddling a state line resolves to several directories, whose indices
are fetched and concatenated.
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import numpy as np
import requests

CUDEM_BASE_URL = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
    "NCEI_ninth_Topobathy_2014_8483"
)
CUDEM_TILE_DEG = 0.25

# Tile naming, verified against the FL listing and national in form:
#   ncei19_n{LL}x{ll}_w{WWW}x{ww}_{YYYY}v{V}.tif
# Case mixes ('n25X75' and 'n25x75') occur within a single listing.
_TILE_PATTERN = re.compile(
    r"ncei19_n(\d+)[xX](\d+)_w(\d+)[xX](\d+)_(\d{4})v(\d+)\.tif",
    re.IGNORECASE,
)

_HREF_PATTERN = re.compile(
    r'<a href="([^"]+\.tif)"[^>]*>([^<]+\.tif)</a>\s*\(([\d.]+)\s*MB\)',
    re.IGNORECASE,
)

# Subdirectory links on the top-level S3 listing are RELATIVE: the whole href
# value is "<CODE>/index.html". Anchoring the code immediately after the
# opening quote excludes the "Related Datasets" table, whose links are
# absolute URLs to other NCEI dataset IDs.
_DIR_PATTERN = re.compile(
    r'<a href="([A-Za-z0-9_]+)/index\.html"', re.IGNORECASE
)

# Approximate coastal extents (lon_min, lat_min, lon_max, lat_max), used only
# to shortlist region directories. Keyed to the S3 directory names EXACTLY
# (including combined and feature-named dirs). Deliberately generous, and
# overlaps at seams are intentional -- the per-tile bbox test downstream is
# authoritative, so a shortlist that returns one region too many is harmless.
_REGION_EXTENTS = {
    "AK":              (-180.0, 50.0, -129.0, 72.0),
    "AL_nwFL":         (-88.6, 29.5, -84.0, 31.1),
    "CA":              (-125.0, 32.4, -117.0, 42.1),
    "FL":              (-84.5, 24.2, -79.8, 31.2),
    "LA_MS":           (-94.1, 28.6, -88.0, 31.1),
    "MA_NH_ME":        (-73.6, 41.0, -66.8, 45.3),
    "NC":              (-78.6, 33.7, -75.4, 36.6),
    "OR":              (-124.7, 41.9, -123.2, 46.3),
    "TX":              (-97.4, 25.8, -93.5, 30.1),
    "chesapeake_bay":  (-77.5, 36.8, -75.5, 39.7),
    "columbia_river":  (-124.2, 45.5, -122.0, 46.3),
    "northeast_sandy": (-75.9, 38.3, -71.5, 41.6),
    "rima":            (-71.9, 41.0, -69.8, 42.9),
    "southeast":       (-82.0, 30.3, -78.5, 34.0),
    "wash_bellingham": (-123.0, 48.5, -122.2, 49.0),
    "wash_juandefuca": (-124.9, 47.9, -122.5, 48.6),
    "wash_outercoast": (-124.9, 46.2, -123.8, 48.5),
    "wash_pugetsound": (-123.3, 47.0, -122.0, 48.6),
}


class CudemError(RuntimeError):
    """Raised when CUDEM tiles cannot be resolved or downloaded."""


# ---------------------------------------------------------------------------
# Region resolution
# ---------------------------------------------------------------------------

def list_available_regions(timeout: int = 60) -> list:
    """Region directory codes the CUDEM server currently lists."""
    r = requests.get(f"{CUDEM_BASE_URL}/index.html", timeout=timeout)
    r.raise_for_status()
    seen, out = set(), []
    for code in _DIR_PATTERN.findall(r.text):
        if code.lower() in ("..", "parent") or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def regions_for_bbox(
    bbox: tuple,
    available: list | None = None,
    verbose: bool = True,
) -> list:
    """
    Shortlist CUDEM region directories whose coastal extent intersects a bbox.

    `available` restricts the result to directories the server actually lists;
    pass None to skip that check (useful offline). Matching against `available`
    is case-insensitive, and the server's own casing is returned so the URL
    path is correct regardless of how `_REGION_EXTENTS` is keyed.
    """
    xmin, ymin, xmax, ymax = bbox
    hits = []
    for code, (rxmin, rymin, rxmax, rymax) in _REGION_EXTENTS.items():
        if rxmax <= xmin or rxmin >= xmax or rymax <= ymin or rymin >= ymax:
            continue
        hits.append(code)

    if available is not None:
        by_upper = {a.upper(): a for a in available}
        hits = [by_upper[h.upper()] for h in hits if h.upper() in by_upper]

    if not hits:
        raise CudemError(
            f"no CUDEM region directory matches bbox {bbox}. The built-in "
            f"extent table covers the ninth-arc-second (dataset 8483) coastal "
            f"directories only; Puerto Rico, USVI, and Hawaii are separate "
            f"datasets. Pass regions=[...] explicitly if needed. "
            f"Available directories: {available if available else 'not checked'}"
        )
    if verbose and len(hits) > 1:
        print(f"  domain spans {len(hits)} CUDEM regions: {hits}")
    return hits


def regions_for_domain(domain, available: list | None = None, **kw) -> list:
    return regions_for_bbox(domain.bbox, available=available, **kw)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def fetch_cudem_index(
    regions: list | str,
    index_html: str | Path | None = None,
    timeout: int = 60,
    verbose: bool = True,
) -> list:
    """
    Parse the CUDEM tile index for one or more region directories.

    Pass `index_html` to parse a saved local copy instead (single region only).

    Returns a list of dicts:
        {name, url, region, sw_lat, sw_lon, year, version, size_mb}
    De-duplicated by tile name across regions, keeping the newest year/version.
    """
    if isinstance(regions, str):
        regions = [regions]

    pages = []
    if index_html is not None:
        pages.append(("local", Path(index_html).read_text()))
    else:
        for region in regions:
            url = f"{CUDEM_BASE_URL}/{region}/index.html"
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                warnings.warn(f"CUDEM index for {region!r} returned {r.status_code}")
                continue
            pages.append((region, r.text))

    if not pages:
        raise CudemError(f"no CUDEM index pages retrieved for regions={regions}")

    best: dict = {}
    for region, html in pages:
        for url, _label, size_mb in _HREF_PATTERN.findall(html):
            canonical = url.rsplit("/", 1)[-1]
            m = _TILE_PATTERN.match(canonical)
            if not m:
                continue
            lat_deg, lat_frac, lon_deg, lon_frac, year, version = m.groups()
            tile = {
                "name": canonical,
                "url": url if url.startswith("http")
                       else f"{CUDEM_BASE_URL}/{region}/{url.lstrip('./')}",
                "region": region,
                # The tile name encodes its NORTH-WEST corner: the latitude is
                # the tile's NORTH edge and the longitude its WEST edge. The SW
                # corner latitude is therefore one tile-height SOUTH of the
                # parsed value. Longitude needs no shift (west edge == SW lon).
                "sw_lat": float(lat_deg) + float(lat_frac) / 100.0 - CUDEM_TILE_DEG,
                "sw_lon": -(float(lon_deg) + float(lon_frac) / 100.0),
                "year": int(year),
                "version": int(version),
                "size_mb": float(size_mb),
            }
            key = canonical.lower()
            prior = best.get(key)
            if prior is None or (tile["year"], tile["version"]) > (prior["year"], prior["version"]):
                best[key] = tile

    tiles = sorted(best.values(), key=lambda t: (t["sw_lat"], t["sw_lon"]))
    if verbose:
        print(f"  CUDEM index: {len(tiles)} tiles across {len(pages)} region(s)")
    return tiles


def filter_tiles_by_bbox(tiles: list, bbox: tuple) -> list:
    """Tiles whose 0.25-degree footprint intersects (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bbox
    keep = []
    for t in tiles:
        tlat_min, tlat_max = t["sw_lat"], t["sw_lat"] + CUDEM_TILE_DEG
        tlon_min, tlon_max = t["sw_lon"], t["sw_lon"] + CUDEM_TILE_DEG
        if (tlon_max <= xmin or tlon_min >= xmax
                or tlat_max <= ymin or tlat_min >= ymax):
            continue
        keep.append(t)
    return keep


def tiles_for_domain(
    domain,
    regions: list | None = None,
    index_html: str | Path | None = None,
    verbose: bool = True,
) -> list:
    """Resolve regions, fetch indices, and filter to the domain bbox."""
    if regions is None and index_html is None:
        try:
            available = list_available_regions()
        except Exception as e:
            if verbose:
                print(f"  region listing unavailable ({e}); using extent table only")
            available = None
        regions = regions_for_domain(domain, available=available, verbose=verbose)

    tiles = fetch_cudem_index(regions or [], index_html=index_html, verbose=verbose)
    hits = filter_tiles_by_bbox(tiles, domain.bbox)
    if not hits:
        raise CudemError(
            f"no CUDEM tiles intersect {domain.name} bbox {domain.bbox}. "
            f"CUDEM covers US coastal areas only; confirm the bbox is in "
            f"EPSG:4326 as (lon_min, lat_min, lon_max, lat_max)."
        )
    if verbose:
        print(f"  {len(hits)} tile(s) intersect {domain.name}, "
              f"~{sum(t['size_mb'] for t in hits):.0f} MB")
    return hits


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_cudem_tiles(
    tiles: list,
    out_dir: str | Path,
    skip_existing: bool = True,
    polite_sec: float = 1.0,
    verbose: bool = True,
) -> list:
    """Stream-download tiles, with a size-match cache check."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    if verbose:
        print(f"  downloading {len(tiles)} tiles "
              f"(~{sum(t['size_mb'] for t in tiles):.0f} MB)")

    for i, t in enumerate(tiles, 1):
        dest = out_dir / t["name"]
        expected_b = int(t["size_mb"] * 1024 * 1024)

        if skip_existing and dest.exists():
            actual = dest.stat().st_size
            if abs(actual - expected_b) / max(expected_b, 1) < 0.05:
                if verbose:
                    print(f"  [{i}/{len(tiles)}] cache: {t['name']}")
                downloaded.append(dest)
                continue
            if verbose:
                print(f"  [{i}/{len(tiles)}] size mismatch, re-downloading")

        try:
            if verbose:
                print(f"  [{i}/{len(tiles)}] fetch: {t['name']} ({t['size_mb']:.0f} MB)")
            with requests.get(t["url"], stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            downloaded.append(dest)
            time.sleep(polite_sec)
        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(tiles)}] FAILED: {e}")
            dest.unlink(missing_ok=True)

    if not downloaded:
        raise CudemError("no CUDEM tiles were downloaded successfully")
    return downloaded


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------

def mosaic_and_reproject_cudem(
    tile_paths: list,
    out_path: str | Path,
    domain=None,
    target_crs: str | None = None,
    target_res_m: float | None = None,
    bbox_lonlat: tuple | None = None,
    resampling: str = "bilinear",
    nodata: float = -9999.0,
    verbose: bool = True,
) -> dict:
    """
    Mosaic tiles, reproject, and optionally crop to a bbox.

    `target_crs` defaults to the UTM zone of the domain centroid rather than a
    fixed EPSG. `target_res_m` and `bbox_lonlat` likewise default from the
    domain when one is supplied.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.windows import from_bounds

    if not tile_paths:
        raise CudemError("no tiles to mosaic")

    if domain is not None:
        from timsa.crs import resolve_target_crs
        target_crs = resolve_target_crs(domain, target_crs)
        if target_res_m is None:
            target_res_m = domain.resolution_m
        if bbox_lonlat is None:
            bbox_lonlat = domain.bbox
    if target_crs is None:
        raise CudemError("target_crs is required when no domain is supplied")
    if target_res_m is None:
        raise CudemError("target_res_m is required when no domain is supplied")

    if verbose:
        print(f"  mosaicking {len(tile_paths)} tiles -> {target_crs} "
              f"@ {target_res_m} m")

    resampling_enum = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }[resampling]

    src_files = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, src_transform = merge(src_files, nodata=nodata)
        src_crs = src_files[0].crs
        src_meta = src_files[0].meta.copy()
    finally:
        for sf in src_files:
            sf.close()

    src_h, src_w = mosaic.shape[1], mosaic.shape[2]
    src_bounds = rasterio.transform.array_bounds(src_h, src_w, src_transform)

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, target_crs, src_w, src_h, *src_bounds, resolution=target_res_m,
    )

    dst_arr = np.full((dst_h, dst_w), nodata, dtype=np.float32)
    reproject(
        source=mosaic[0], destination=dst_arr,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=target_crs,
        resampling=resampling_enum,
        src_nodata=nodata, dst_nodata=nodata,
    )

    if bbox_lonlat is not None:
        from rasterio.warp import transform_bounds
        dst_bounds = transform_bounds("EPSG:4326", target_crs, *bbox_lonlat)
        window = from_bounds(*dst_bounds, transform=dst_transform)
        col_off = max(0, int(window.col_off))
        row_off = max(0, int(window.row_off))
        ncols = min(int(window.width), dst_w - col_off)
        nrows = min(int(window.height), dst_h - row_off)
        if ncols <= 0 or nrows <= 0:
            raise CudemError(
                "the crop bbox does not overlap the mosaic; check that the "
                "domain bbox and the downloaded tiles refer to the same area"
            )
        dst_arr = dst_arr[row_off:row_off + nrows, col_off:col_off + ncols]
        dst_transform = rasterio.transform.Affine(
            dst_transform.a, dst_transform.b,
            dst_transform.c + col_off * dst_transform.a,
            dst_transform.d, dst_transform.e,
            dst_transform.f + row_off * dst_transform.e,
        )
        dst_h, dst_w = dst_arr.shape

    dst_meta = src_meta.copy()
    dst_meta.update({
        "crs": target_crs, "transform": dst_transform,
        "width": dst_w, "height": dst_h, "nodata": nodata,
        "dtype": "float32", "compress": "lzw", "tiled": True, "count": 1,
    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **dst_meta) as dst:
        dst.write(dst_arr, 1)

    if verbose:
        print(f"  wrote {out_path} ({dst_h} x {dst_w})")

    return {
        "profile": dst_meta, "shape": (dst_h, dst_w),
        "transform": dst_transform, "crs": target_crs,
        "n_tiles": len(tile_paths), "path": out_path,
    }
