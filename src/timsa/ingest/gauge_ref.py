"""
timsa.ingest.gauge_ref
======================
Builder for the gauge_reference raster used by TiMSA.

Ported unchanged from the SLR pipeline: this module derives everything from
`dem_profile` and `stations_df`, so it carries no domain assumptions.

Produces an integer raster matched 1:1 to the DEM grid where each pixel
holds the 1-based index of its nearest tide gauge (1..n_gauges; 0 = no data).

Uses scipy.spatial.cKDTree for nearest-neighbor lookup — mathematically
equivalent to a Voronoi/Thiessen partition but ~100x faster than
rasterizing polygons and avoids GDAL rasterize dependencies.

The output can be written as ASCII grid (.asc, original C-reference
format) or as a GeoTIFF (recommended for the Python pipeline; matches
the gauge_zones_path config key).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def build_gauge_reference_kdtree(
    stations_df: pd.DataFrame,
    dem_profile: dict,
    dem_array: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    nodata_value: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Build the gauge_reference integer raster.

    Parameters
    ----------
    stations_df : DataFrame
        Must include columns ['gauge_idx', 'Latitude', 'Longitude'] (or
        ['gauge_idx', 'lat', 'lon']). gauge_idx is 0-based; output raster
        uses gauge_idx + 1 to match TiMSA's 1-based convention.
        IMPORTANT: pass the manifest from build_multigauge_minute_array
        filtered to kept gauges only — the gauge_idx values must align
        with the columns of the gauge_wdepths array.
    dem_profile : dict
        rasterio-style profile with 'transform', 'crs', 'width', 'height'.
    dem_array : np.ndarray, optional
        If provided, pixels where dem_array == dem nodata or NaN are set to
        nodata_value in the output.
    land_mask : np.ndarray, optional
        2D boolean array (same shape as DEM). True = land/dry/excluded.
        Pixels where True are set to nodata_value.
    nodata_value : int
        Integer code for excluded pixels (default 0; matches TiMSA convention).

    Returns
    -------
    gauge_raster : np.ndarray (int32)
        Integer gauge index per pixel; 0 = excluded.
    profile : dict
        rasterio profile suitable for writing (dtype int32, nodata=0).
    """
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp_transform

    # Resolve lat/lon column names
    if "Latitude" in stations_df.columns:
        lat_col, lon_col = "Latitude", "Longitude"
    elif "lat" in stations_df.columns:
        lat_col, lon_col = "lat", "lon"
    else:
        raise ValueError("stations_df must have Latitude/Longitude or lat/lon columns")

    if "gauge_idx" not in stations_df.columns:
        raise ValueError("stations_df must include 'gauge_idx' column "
                         "(use the kept-gauges manifest from "
                         "build_multigauge_minute_array)")

    gauge_idx_0 = stations_df["gauge_idx"].astype(int).to_numpy()
    lats = stations_df[lat_col].astype(float).to_numpy()
    lons = stations_df[lon_col].astype(float).to_numpy()

    # Project station lat/lon to target CRS (DEM CRS, typically UTM 17N)
    dst_crs = dem_profile["crs"]
    if str(dst_crs).upper() in ("EPSG:4326", "OGC:CRS84"):
        xs, ys = lons, lats
    else:
        xs, ys = warp_transform("EPSG:4326", dst_crs, lons.tolist(), lats.tolist())
        xs = np.asarray(xs); ys = np.asarray(ys)

    # Build cKDTree on station coordinates in target CRS
    station_coords = np.column_stack([xs, ys])
    tree = cKDTree(station_coords)

    # Generate pixel-center coordinates
    transform = dem_profile["transform"]
    if not isinstance(transform, Affine):
        transform = Affine(*transform[:6]) if hasattr(transform, "__len__") else transform
    width = int(dem_profile["width"])
    height = int(dem_profile["height"])

    cols = np.arange(width)
    rows = np.arange(height)
    col_grid, row_grid = np.meshgrid(cols, rows)
    # Pixel centers: shift by 0.5
    px = transform.a * (col_grid + 0.5) + transform.b * (row_grid + 0.5) + transform.c
    py = transform.d * (col_grid + 0.5) + transform.e * (row_grid + 0.5) + transform.f

    # Nearest-neighbor lookup
    pixel_coords = np.column_stack([px.ravel(), py.ravel()])
    _, nn_idx = tree.query(pixel_coords, k=1)

    # gauge_raster = 1-based station index (gauge_idx_0 + 1)
    gauge_raster = (gauge_idx_0[nn_idx] + 1).reshape(height, width).astype(np.int32)

    # Apply nodata masks
    if dem_array is not None:
        dem_nodata = dem_profile.get("nodata", None)
        nodata_mask = np.isnan(dem_array)
        if dem_nodata is not None and not np.isnan(dem_nodata):
            nodata_mask = nodata_mask | (dem_array == dem_nodata)
        gauge_raster[nodata_mask] = nodata_value

    if land_mask is not None:
        gauge_raster[land_mask] = nodata_value

    # Build output profile
    out_profile = dict(dem_profile)
    out_profile.update({
        "dtype": "int32",
        "nodata": nodata_value,
        "count": 1,
        "compress": "lzw",
    })

    return gauge_raster, out_profile


def write_gauge_reference_geotiff(
    gauge_raster: np.ndarray,
    profile: dict,
    out_path: Path,
) -> None:
    """Write gauge_reference raster as GeoTIFF."""
    import rasterio
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(gauge_raster, 1)


def write_gauge_reference_asc(
    gauge_raster: np.ndarray,
    profile: dict,
    out_path: Path,
) -> None:
    """
    Write gauge_reference raster as Esri ASCII Grid (.asc), the original
    C-reference TiMSA input format.

    Note: ASCII grid headers expect a square pixel and lower-left origin.
    For non-square pixels or rotated rasters, write GeoTIFF instead.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transform = profile["transform"]
    height, width = gauge_raster.shape
    cellsize = abs(transform.a)
    if abs(abs(transform.e) - cellsize) > 1e-6:
        raise ValueError("ASCII grid requires square pixels; got "
                         f"x={transform.a}, y={transform.e}. Write GeoTIFF instead.")
    # Lower-left corner
    xll = transform.c
    yll = transform.f + transform.e * height  # transform.e is negative

    nodata = int(profile.get("nodata", 0))

    with open(out_path, "w") as f:
        f.write(f"ncols {width}\n")
        f.write(f"nrows {height}\n")
        f.write(f"xllcorner {xll}\n")
        f.write(f"yllcorner {yll}\n")
        f.write(f"cellsize {cellsize}\n")
        f.write(f"NODATA_value {nodata}\n")
        np.savetxt(f, gauge_raster, fmt="%d", delimiter=" ")
