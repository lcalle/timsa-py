"""
timsa.estimate
==============
Size a run before committing to it: grid cells, timesteps, cell-steps, and
rough memory and output-disk figures.

Runtime scales with cell-steps (cells x timesteps), so that is the number to
cap when runs share a machine. The figures here are upper bounds: the
simulation also skips cells that are provably always too deep or always too
dry, which this estimate cannot know without the DEM and the tide record.

    from timsa.config import RunConfig
    from timsa.estimate import estimate_run
    est = estimate_run(RunConfig.from_yaml("configs/example_present_day.yaml"))
    print(est.report())

`timsa run --dry-run` prints the same report.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, asdict
from pathlib import Path

MINUTES_PER_DAY = 1440


@dataclass
class RunEstimate:
    nrows: int | None
    ncols: int | None
    cells: int | None
    n_days: int | None
    steps_per_day: int
    steps: int | None
    cell_steps: int | None
    block_mem_mb: float | None      # one inner-loop block (cell_chunk x steps_per_day)
    grid_mem_mb: float | None       # full-grid arrays held for the whole run
    output_raw_mb: float | None     # annual rasters before LZW
    source: str                     # 'dem' or 'bbox'

    @property
    def peak_mem_mb(self) -> float | None:
        if self.block_mem_mb is None or self.grid_mem_mb is None:
            return None
        return self.block_mem_mb + self.grid_mem_mb

    def to_dict(self) -> dict:
        d = asdict(self)
        d["peak_mem_mb"] = self.peak_mem_mb
        return d

    def report(self) -> str:
        def f(v, fmt="{:,}"):
            return "unknown" if v is None else fmt.format(v)
        lines = [
            f"grid        : {f(self.nrows)} x {f(self.ncols)} = {f(self.cells)} cells "
            f"(from {self.source})",
            f"time        : {f(self.n_days)} days x {self.steps_per_day} steps/day "
            f"= {f(self.steps)} steps",
            f"cell-steps  : {f(self.cell_steps, '{:.3g}')} (upper bound; always-deep "
            f"and always-dry cells are skipped at run time)",
            f"memory      : ~{f(self.peak_mem_mb, '{:,.0f}')} MB peak "
            f"(block ~{f(self.block_mem_mb, '{:,.0f}')} MB + grid "
            f"~{f(self.grid_mem_mb, '{:,.0f}')} MB)",
            f"output      : ~{f(self.output_raw_mb, '{:,.1f}')} MB of annual rasters "
            f"before compression",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------

def grid_shape_from_bbox(bbox, resolution_m: float, target_crs: str | None = None,
                         domain=None) -> tuple[int, int]:
    """
    (nrows, ncols) of a grid covering a lon/lat bbox at `resolution_m`.

    Projects the bbox into the target CRS when pyproj is available, matching
    what the DEM mosaic will produce; otherwise uses the same latitude-
    corrected km/degree constants as timsa.domain.
    """
    xmin, ymin, xmax, ymax = bbox
    res = float(resolution_m)
    try:
        from pyproj import Transformer
        from timsa.crs import resolve_target_crs, utm_epsg_from_lonlat
        if domain is not None:
            crs = resolve_target_crs(domain, target_crs)
        else:
            crs = target_crs or utm_epsg_from_lonlat(0.5 * (xmin + xmax),
                                                    0.5 * (ymin + ymax))
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        left, bottom, right, top = tr.transform_bounds(xmin, ymin, xmax, ymax)
        width_m, height_m = right - left, top - bottom
    except ImportError:
        mid_lat = 0.5 * (ymin + ymax)
        width_m = (xmax - xmin) * 111_320.0 * math.cos(math.radians(mid_lat))
        height_m = (ymax - ymin) * 110_574.0
    return max(1, math.ceil(height_m / res)), max(1, math.ceil(width_m / res))


def n_days_for_config(cfg) -> int | None:
    """Simulated days from the gauge window; None when it can't be known up front."""
    g = cfg.gauges
    if g.source not in ("coops", "hybrid") or g.year is None:
        return None
    start = g.start or dt.date(g.year, 1, 1)
    end = g.end or dt.date(g.year, 12, 31)
    return (end - start).days + 1


def estimate_run(cfg) -> RunEstimate:
    """Estimate grid size, cell-steps, memory and output size for a RunConfig."""
    s = cfg.simulation
    spd = MINUTES_PER_DAY // int(s.timestep_min)

    nrows = ncols = None
    source = "bbox"
    if cfg.dem_path is not None and Path(cfg.dem_path).exists():
        try:
            import rasterio
            with rasterio.open(cfg.dem_path) as src:
                nrows, ncols = src.height, src.width
            source = "dem"
        except ImportError:
            pass
    if nrows is None and cfg.domain.resolution_m:
        nrows, ncols = grid_shape_from_bbox(
            cfg.domain.bbox, cfg.domain.resolution_m,
            target_crs=cfg.target_crs, domain=cfg.domain,
        )
    cells = nrows * ncols if nrows is not None else None

    n_days = n_days_for_config(cfg)
    steps = n_days * spd if n_days is not None else None
    cell_steps = cells * steps if (cells is not None and steps is not None) else None

    n_win = len(cfg.depth_windows)
    n_thr = len(cfg.refugia_thresholds)
    n_metrics = n_win + n_thr

    # Inner block: auto mode targets chunk_mem_mb; a fixed chunk is what it is.
    from timsa.core import BYTES_PER_CELL_STEP
    if s.cell_chunk == "auto":
        block_mb = float(s.chunk_mem_mb)
        if cells is not None:
            block_mb = min(block_mb, cells * spd * BYTES_PER_CELL_STEP / 2**20)
    else:
        chunk = int(s.cell_chunk) if cells is None else min(int(s.cell_chunk), cells)
        block_mb = chunk * spd * BYTES_PER_CELL_STEP / 2**20

    grid_mb = out_mb = None
    if cells is not None:
        # dem float64 (+ flat copy), zones int32, three bool masks, and per
        # metric a flat float64 accumulator plus the 2D float64 result.
        per_cell = 8 + 8 + 4 + 3 + n_metrics * (8 + 8)
        grid_mb = cells * per_cell / 2**20
        o = cfg.output
        per_cell_out = 0
        if o.write_annual:
            per_window = (4 if o.write_total else 0) + (4 if o.write_average else 0) + 1
            per_thr = (4 if o.write_total else 0) + (4 if o.write_average else 0)
            per_cell_out = n_win * per_window + n_thr * per_thr + 1   # + land mask
        out_mb = cells * per_cell_out / 2**20
        if o.write_daily and n_days is not None:
            out_mb += cells * 4 * n_metrics * n_days / 2**20

    return RunEstimate(
        nrows=nrows, ncols=ncols, cells=cells, n_days=n_days,
        steps_per_day=spd, steps=steps, cell_steps=cell_steps,
        block_mem_mb=block_mb, grid_mem_mb=grid_mb, output_raw_mb=out_mb,
        source=source,
    )
