"""
examples/run_prescribed.py
==========================
Minimal end-to-end run driven by a prescribed 6-minute water-level record.

Demonstrates the feature ported from iterateday_prescribewd_NADV88.c: the
simulation steps on the observations themselves, with no interpolation up to
1-minute in between.

    python examples/run_prescribed.py \
        --wl cache/coops/wl_2025_6min.csv \
        --dem cache/dem/dem_30m.tif \
        --zones cache/dem/gauge_zones_30m.tif

With no arguments it fabricates a small synthetic domain and record, so the
code path can be exercised before any real data is in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from timsa.core import TimsaInputs, TimsaConfig, TimsaSimulation
from timsa.ingest.prescribed import load_prescribed_wl
from timsa.metrics import RunResult, write_metric_rasters, write_summary
from timsa.io import (
    load_dem,
    load_gauge_zones,
    DailyRasterWriter,
    estimate_daily_output_bytes,
    human_bytes,
)


def synthetic(n_days: int = 3, interval: int = 6):
    """Small stand-in domain and 6-minute record."""
    n_rec = n_days * (1440 // interval)
    t_hr = np.arange(n_rec) * interval / 60.0
    levels = (0.45 * np.sin(2 * np.pi * t_hr / 12.42)
              + 0.20 * np.sin(2 * np.pi * t_hr / 24.07))
    wl = levels[:, None]

    nr, nc = 60, 90
    dem = (np.linspace(-1.0, 1.0, nc)[None, :].repeat(nr, 0)
           + 0.05 * np.random.RandomState(0).randn(nr, nc))
    dem[:5, :5] = np.nan
    zones = np.ones((nr, nc), dtype=np.int32)
    zones[np.isnan(dem)] = 0
    profile = {
        "driver": "GTiff", "dtype": "float64", "nodata": np.nan,
        "width": nc, "height": nr, "count": 1,
        "crs": None, "transform": None,
    }
    return dem, zones, np.isnan(dem), wl, interval, profile


def main() -> None:
    ap = argparse.ArgumentParser(description="TiMSA prescribed-record example")
    ap.add_argument("--wl", type=Path, help="prescribed water-level CSV")
    ap.add_argument("--dem", type=Path)
    ap.add_argument("--zones", type=Path)
    ap.add_argument("--time-col", default="time",
                    help="'none' if the file has no time column")
    ap.add_argument("--record-interval-min", type=int, default=6)
    ap.add_argument("--timestep-min", type=int, default=6,
                    help="independent of --record-interval-min; finer values "
                         "interpolate the gauge record up")
    ap.add_argument("--gauge-interp", default="linear",
                    choices=("linear", "sinusoidal", "hold"))
    ap.add_argument("--cell-chunk", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--run-id", default="present_day")
    ap.add_argument("--daily-rasters", action="store_true",
                    help="also stream one raster per day per metric")
    args = ap.parse_args()

    if args.wl and args.dem:
        dem, profile = load_dem(args.dem)
        zones = (load_gauge_zones(args.zones, dem.shape) if args.zones
                 else np.ones(dem.shape, dtype=np.int32))
        wl_tab = load_prescribed_wl(
            args.wl,
            time_col=None if args.time_col.lower() == "none" else args.time_col,
            record_interval_min=args.record_interval_min,
        )
        print(f"water levels: {wl_tab.describe()}")
        wl, interval = wl_tab.values, wl_tab.record_interval_min
        nodata = np.isnan(dem)
        cell_area = None            # derived from the transform
    else:
        print("no --wl/--dem given; using synthetic inputs")
        dem, zones, nodata, wl, interval, profile = synthetic()
        cell_area = 900.0           # no transform on synthetic profile

    inputs = TimsaInputs(
        dem=dem, gauge_zones=zones, gauge_wdepths=wl,
        nodata_mask=nodata, record_interval_min=interval,
    )
    config = TimsaConfig(
        depth_windows={"shallow_band": (-1.5, 0.0)},   # POSITIVE IS DRY
        refugia_thresholds=[0.2, 0.5, 1.0],
        timestep_min=args.timestep_min,
        gauge_interp=args.gauge_interp,
    )
    sim = TimsaSimulation(inputs, config)
    print(f"simulation: {sim.describe()}")

    writer = None
    if args.daily_rasters:
        n_metrics = len(config.depth_windows) + len(config.refugia_thresholds)
        est = estimate_daily_output_bytes(sim.n_days, sim.shape, n_metrics)
        print(f"daily rasters: ~{sim.n_days * n_metrics} files, ~{human_bytes(est)}")
        writer = DailyRasterWriter(args.out, args.run_id, profile, verbose=True)

    sim.run(verbose=True, cell_chunk=args.cell_chunk, daily_callback=writer)
    if writer is not None:
        writer.close()

    run = RunResult(run_id=args.run_id, sim=sim, profile=profile,
                    cell_area_m2=cell_area, metadata=sim.describe())
    run.compute_summaries()
    write_metric_rasters(run, args.out / "rasters")
    df = write_summary([run], args.out / "summaries" / f"{args.run_id}_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
