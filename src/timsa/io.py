"""
timsa.io
========
Raster input and output.

Contains:
  - load_dem, load_gauge_zones   (lifted from preprocess_data.py)
  - DailyRasterWriter            (new: streaming per-day output)

Why DailyRasterWriter exists
----------------------------
The C reference writes one raster per day, and reallocates the day
accumulator every day via `dayFHA = rastercopy(defaultRaster)` with no
matching free. Over a 365-day run that leaks a raster per day, and the
written files themselves accumulate to fill the disk.

This writer fixes both halves:
  - One preallocated 2D buffer, refilled with zeros per day. No allocation
    inside the day loop.
  - float32 with LZW compression by default, which is roughly a 6-8x
    reduction over uncompressed float64 for the same content.
  - Daily output is opt-in. Annual accumulation costs nothing extra when it
    is off, and remains the default.

Disk cost is worth stating plainly before enabling it: one year of daily
rasters is 365 files per metric. With one depth window and three refugia
thresholds that is 1,460 files per run. `estimate_daily_output_bytes` below
reports the figure before a run starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_dem(dem_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a DEM raster.

    Returns
    -------
    (array, profile)
        Array is 2D float64 with NaN where no data. Profile is the rasterio
        metadata dict (CRS, transform, ...).
    """
    import rasterio

    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            arr[arr == nodata] = np.nan
        profile = src.profile.copy()
    return arr, profile


def load_gauge_zones(gauge_zones_path: str | Path, dem_shape: tuple) -> np.ndarray:
    """
    Load the gauge-zone integer raster.

    If the file is absent, returns a uniform array of 1s: the single-gauge
    case, where one gauge forces the whole domain.
    """
    import rasterio

    p = Path(gauge_zones_path)
    if not p.exists():
        return np.ones(dem_shape, dtype=np.int32)

    with rasterio.open(p) as src:
        arr = src.read(1).astype(np.int32)

    if arr.shape != tuple(dem_shape):
        raise ValueError(
            f"gauge_zones shape {arr.shape} does not match DEM shape "
            f"{tuple(dem_shape)}. Rebuild the zone raster against this DEM."
        )
    return arr


# ---------------------------------------------------------------------------
# Daily output
# ---------------------------------------------------------------------------

def estimate_daily_output_bytes(
    n_days: int,
    shape: tuple,
    n_metrics: int,
    dtype: str = "float32",
    compression_factor: float = 0.25,
) -> int:
    """
    Rough size estimate for a daily-raster run, before committing to it.

    `compression_factor` is the assumed post-LZW fraction of raw size.
    0.25 is conservative for time-accumulator rasters, which contain large
    uniform regions and compress well.
    """
    itemsize = np.dtype(dtype).itemsize
    per_raster = shape[0] * shape[1] * itemsize * compression_factor
    return int(per_raster * n_days * n_metrics)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


class DailyRasterWriter:
    """
    Streams per-day metric rasters to disk from a preallocated buffer.

    Intended as the `daily_callback` for TimsaSimulation.run():

        writer = DailyRasterWriter(out_dir, run_id, profile)
        sim.run(daily_callback=writer)
        writer.close()

    Parameters
    ----------
    out_dir : Path
        Directory for daily rasters. A `daily/` subdirectory is created.
    run_id : str
        Prefix for filenames.
    profile : dict
        rasterio profile from the DEM. Must carry crs and transform; if either
        is missing (synthetic/dry-run inputs) the writer becomes a no-op.
    dtype, compress : str
        Output dtype and compression.
    nodata : float
        Value written outside the simulation's valid mask.
    metrics : iterable of str, optional
        Restrict output to a subset of metric keys. None writes all.
    day_offset : int
        Added to the zero-based day index for filenames, so day numbering can
        match a calendar day-of-year if wanted.
    """

    def __init__(
        self,
        out_dir: str | Path,
        run_id: str,
        profile: dict,
        dtype: str = "float32",
        compress: str = "lzw",
        nodata: float = -9999.0,
        metrics: list | None = None,
        day_offset: int = 0,
        verbose: bool = False,
    ):
        self.out_dir = Path(out_dir) / "daily"
        self.run_id = run_id
        self.profile = dict(profile)
        self.dtype = dtype
        self.compress = compress
        self.nodata = float(nodata)
        self.metrics = set(metrics) if metrics else None
        self.day_offset = int(day_offset)
        self.verbose = verbose

        self.enabled = (
            self.profile.get("crs") is not None
            and self.profile.get("transform") is not None
        )
        self._buffer: np.ndarray | None = None
        self.n_written = 0
        self.paths: list[Path] = []

        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.profile.update(
                dtype=self.dtype, count=1, compress=self.compress, nodata=self.nodata
            )

    # -- callback interface ------------------------------------------------

    def __call__(self, day: int, flat_band: dict, flat_below: dict, sim) -> None:
        if not self.enabled:
            return

        if self._buffer is None:
            # Allocated once, on first use, at the simulation's grid shape.
            self._buffer = np.empty(sim.shape, dtype=np.float64)

        d = day + self.day_offset

        for name, flat in flat_band.items():
            key = f"time_integrated__{_slug(name)}"
            if self.metrics and key not in self.metrics:
                continue
            self._write(flat, sim, f"{self.run_id}__day_{d:03d}__{key}.tif")

        for thr, flat in flat_below.items():
            key = f"refugia_time__thr_{_thr_slug(thr)}"
            if self.metrics and key not in self.metrics:
                continue
            self._write(flat, sim, f"{self.run_id}__day_{d:03d}__{key}.tif")

    def _write(self, flat: np.ndarray, sim, filename: str) -> None:
        import rasterio

        # Reuse the buffer; unflatten zeroes it before scattering.
        arr = sim.unflatten(flat, out=self._buffer)
        out = arr.astype(self.dtype, copy=True)
        out_valid = getattr(sim, "base_valid_mask", sim.valid_mask)
        out[~out_valid] = self.nodata

        path = self.out_dir / filename
        with rasterio.open(path, "w", **self.profile) as dst:
            dst.write(out, 1)

        self.n_written += 1
        self.paths.append(path)
        if self.verbose:
            print(f"    wrote {path.name}")

    def close(self) -> None:
        """Release the buffer and report."""
        self._buffer = None
        if self.verbose and self.n_written:
            total = sum(p.stat().st_size for p in self.paths if p.exists())
            print(f"  daily rasters: {self.n_written} files, {human_bytes(total)}")


# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "-")


def _thr_slug(thr: float) -> str:
    return f"{float(thr):.2f}".replace(".", "p")
