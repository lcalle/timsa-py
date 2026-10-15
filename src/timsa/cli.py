"""
timsa.cli
=========
Command-line entry point.

    timsa run   configs/example_present_day.yaml [--daily-rasters]
    timsa fetch configs/example_present_day.yaml

Trimmed from the SLR pipeline's `run_pipeline.py`. Removed: `select_forcing_year`,
the scenario x horizon loop, `--sensitivity` and the M2 perturbation block, the
loss-trajectory CSV, and the interaction-rate and figure steps. What remains
runs ONE simulation from ONE config.

Kept from the original: `--dry-run`, `--quick-test`, `--skip-existing`, and the
synthetic-input path, all of which are useful independent of the manuscript.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    from timsa.config import RunConfig
    from timsa.core import TimsaInputs, TimsaSimulation
    from timsa.io import (load_dem, load_gauge_zones, DailyRasterWriter,
                          estimate_daily_output_bytes, human_bytes)
    from timsa.metrics import (RunResult, write_metric_rasters, write_summary,
                               metric_raster_paths, load_run_from_rasters)

    cfg = RunConfig.from_yaml(args.config)
    _apply_overrides(cfg, args)

    print(f"TiMSA run: {cfg.output.run_id}")
    print(cfg.report())

    if args.dry_run:
        from timsa.estimate import estimate_run
        print("\nestimate")
        for line in estimate_run(cfg).report().splitlines():
            print(f"      {line}")
        print("\ndry run: configuration is valid; nothing executed")
        return 0

    # ---- inputs ---------------------------------------------------------
    if args.synthetic or cfg.dem_path is None:
        print("\n[1/4] synthetic inputs (no dem_path configured)")
        dem, zones, nodata, wl, interval, profile = _synthetic_inputs()
        cell_area = 900.0
    else:
        print(f"\n[1/4] loading DEM {cfg.dem_path}")
        dem, profile = load_dem(cfg.dem_path)
        zones = (load_gauge_zones(cfg.gauge_zones_path, dem.shape)
                 if cfg.gauge_zones_path else np.ones(dem.shape, dtype=np.int32))
        nodata = np.isnan(dem)
        if cfg.land_mask_path and Path(cfg.land_mask_path).exists():
            import rasterio
            with rasterio.open(cfg.land_mask_path) as src:
                nodata |= src.read(1).astype(bool)
        wl, interval, gauges = _load_water_levels(cfg)
        if gauges is not None and cfg.gauge_zones_path:
            zones = _zones_matching_gauges(cfg, zones, gauges, dem, profile)
        cell_area = None
        print(f"      DEM {dem.shape}, {int((~nodata).sum())} data cells")

    if args.quick_test:
        n_keep = min(wl.shape[0], (1440 // interval) * args.quick_test)
        wl = wl[:n_keep]
        print(f"      quick test: {args.quick_test} day(s)")

    # ---- resume ---------------------------------------------------------
    if args.skip_existing:
        paths = metric_raster_paths(cfg.output.run_id, cfg.output.rasters,
                                    list(cfg.depth_windows), cfg.refugia_thresholds)
        if all(p.exists() for p in paths.values()):
            print("\n[2/4] all metric rasters present; rebuilding summary only")
            run = load_run_from_rasters(
                cfg.output.run_id, cfg.output.rasters,
                list(cfg.depth_windows), cfg.refugia_thresholds,
                profile, cell_area_m2=cell_area,
            )
            run.compute_summaries()
            _write_summary(cfg, run)
            return 0

    # ---- simulation -----------------------------------------------------
    sunrise_sunset = _daylight_table(cfg) if cfg.simulation.constrain_daylight else None

    inputs = TimsaInputs(
        dem=dem, gauge_zones=zones, gauge_wdepths=wl, nodata_mask=nodata,
        record_interval_min=interval,
        water_level_offset_m=cfg.simulation.water_level_offset_m,
    )
    sim = TimsaSimulation(inputs, cfg.timsa_config(sunrise_sunset))

    print(f"\n[2/4] simulating")
    for k, v in sim.describe().items():
        print(f"      {k}: {v}")

    writer = None
    if cfg.output.write_daily:
        n_metrics = len(cfg.depth_windows) + len(cfg.refugia_thresholds)
        est = estimate_daily_output_bytes(sim.n_days, sim.shape, n_metrics,
                                          dtype=cfg.output.daily_dtype)
        print(f"      daily rasters: ~{sim.n_days * n_metrics} files, "
              f"~{human_bytes(est)}")
        writer = DailyRasterWriter(
            cfg.output.root, cfg.output.run_id, profile,
            dtype=cfg.output.daily_dtype, compress=cfg.output.daily_compress,
            metrics=cfg.output.daily_metrics,
        )

    sim.run(verbose=True, cell_chunk=cfg.simulation.cell_chunk,
            chunk_mem_mb=cfg.simulation.chunk_mem_mb,
            daily_callback=writer)
    if writer is not None:
        writer.close()
        print(f"      wrote {writer.n_written} daily raster(s)")

    # ---- metrics --------------------------------------------------------
    print(f"\n[3/4] metrics")
    run = RunResult(run_id=cfg.output.run_id, sim=sim, profile=profile,
                    cell_area_m2=cell_area, metadata=sim.describe())
    run.compute_summaries()

    if cfg.output.write_annual:
        written = write_metric_rasters(
            run, cfg.output.rasters,
            write_total=cfg.output.write_total,
            write_average=cfg.output.write_average,
        )
        kinds = []
        if cfg.output.write_total:
            kinds.append("total")
        if cfg.output.write_average:
            kinds.append("day-average")
        print(f"      wrote {len(written)} annual raster(s) [{', '.join(kinds)}]")

    print(f"\n[4/4] summary")
    _write_summary(cfg, run)
    return 0


def _write_summary(cfg, run) -> None:
    from timsa.metrics import write_summary
    out = Path(cfg.output.summaries) / f"{cfg.output.run_id}_summary.csv"
    df = write_summary([run], out)
    cols = ["metric", "window_or_threshold", "value", "units"]
    print(df[cols].to_string(index=False))
    print(f"      -> {out}")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def cmd_fetch(args) -> int:
    """
    Acquire the inputs a run needs: stations, datum offsets, DEM, gauge zones.

    This is `fetch_data.py` as a subcommand, with the domain driving every step
    instead of fixed CSVs.
    """
    from timsa.config import RunConfig
    from timsa.ingest import stations as st
    from timsa.ingest import vdatum as vd
    from timsa.ingest import cudem, gauge_ref

    cfg = RunConfig.from_yaml(args.config)
    dom = cfg.domain
    print(f"TiMSA fetch: {dom}")

    cache = Path(cfg.gauges.cache_dir).parent

    print("\n[1/4] stations")
    if cfg.gauges.stations_csv and Path(cfg.gauges.stations_csv).exists():
        table = st.load_stations(cfg.gauges.stations_csv)
        print(f"      {len(table)} from {cfg.gauges.stations_csv}")
    else:
        table = st.discover_stations(dom, cache_dir=cache, refresh=args.refresh)

    print("\n[2/4] vertical datum offsets")
    table = vd.batch_resolve_offsets(
        table, domain=dom, manual_csv=cfg.gauges.vdatum_csv,
        coops_cache_dir=cache / "coops_meta",
        out_csv=cache / f"stations_vdatum_{dom.cache_key()}.csv",
        require_all=not args.allow_missing_offsets,
        idw_fill=cfg.gauges.vdatum_idw_fill or args.idw_fill_offsets,
        idw_power=cfg.gauges.vdatum_idw_power,
        idw_k=cfg.gauges.vdatum_idw_k,
    )

    print("\n[3/4] CUDEM bathymetry")
    if cfg.dem_path and Path(cfg.dem_path).exists() and not args.refresh:
        print(f"      {cfg.dem_path} exists; skipping (use --refresh to rebuild)")
    else:
        tiles = cudem.tiles_for_domain(dom, regions=args.cudem_regions)
        if args.dry_run:
            print(f"      dry run: would download {len(tiles)} tile(s)")
            return 0
        paths = cudem.download_cudem_tiles(tiles, cache / "cudem_tiles")
        cudem.mosaic_and_reproject_cudem(
            paths, cfg.dem_path, domain=dom, target_crs=cfg.target_crs,
        )

    print("\n[4/4] gauge zones")
    if cfg.gauge_zones_path:
        from timsa.io import load_dem
        dem, profile = load_dem(cfg.dem_path)
        if cfg.gauges.source == "hybrid":
            # Zones must span both sources in the same order the run assembles
            # its columns, so build them from the combined hybrid table.
            from timsa.ingest.hybrid import load_hybrid_wl
            h = load_hybrid_wl(cfg, dom, verbose=False)
            ztable = h.station_table
            print(f"      hybrid: {h.n_insitu} in-situ + {h.n_coops} "
                  f"CO-OPS gauge(s)")
        else:
            # Download the records now, so gauges with no usable data are
            # dropped before the zones are drawn; the run then finds the same
            # records in the cache and the same columns in the same order.
            print("      checking water-level records")
            table = _usable_offsets(table)
            _, _, ztable, dropped = _coops_records(cfg, table)
            _report_dropped(dropped)
            write_zone_stations(cfg, ztable)
            print(f"      {len(ztable)} gauge(s) kept -> {zone_stations_path(cfg)}")
        raster, zprof = gauge_ref.build_gauge_reference_kdtree(
            ztable, profile, dem_array=dem,
        )
        gauge_ref.write_gauge_reference_geotiff(raster, zprof, cfg.gauge_zones_path)
        print(f"      wrote {cfg.gauge_zones_path}")

    print("\nfetch complete")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_water_levels(cfg):
    """
    Return (values, record_interval_min, gauges). `gauges` is None for the
    prescribed and hybrid sources; for CO-OPS it is a dict with the kept
    station table ('kept') and the IDs that were tried ('tried').
    """
    g = cfg.gauges

    if g.source == "prescribed":
        from timsa.ingest.prescribed import load_prescribed_wl
        wl = load_prescribed_wl(
            g.prescribed_file,
            station_order=g.station_order,
            time_col=g.time_col,
            record_interval_min=g.record_interval_min,
            value_scale=g.value_scale,
            fill_gaps=g.fill_gaps,
            max_gap_records=g.max_gap_records,
        )
        print(f"      water levels: {wl.describe()}")
        return wl.values, wl.record_interval_min, None

    if g.source == "hybrid":
        from timsa.ingest.hybrid import load_hybrid_wl
        h = load_hybrid_wl(cfg, cfg.domain)
        print(f"      water levels: {h.describe()}")
        return h.values, h.record_interval_min, None

    from timsa.ingest import stations as st, vdatum as vd

    sidecar = zone_stations_path(cfg)
    if sidecar is not None and sidecar.exists():
        # Written by `timsa fetch`: the gauges the zone raster was drawn from.
        table = st.load_stations(sidecar)
    elif g.stations_csv and Path(g.stations_csv).exists():
        table = st.load_stations(g.stations_csv)
    else:
        table = st.discover_stations(cfg.domain, cache_dir=Path(g.cache_dir).parent)

    if vd.OFFSET_COLUMN not in table.columns:
        table = vd.batch_resolve_offsets(
            table, domain=cfg.domain, manual_csv=g.vdatum_csv, verbose=False,
            idw_fill=g.vdatum_idw_fill, idw_power=g.vdatum_idw_power,
            idw_k=g.vdatum_idw_k,
        )
    tried = table["StationID"].astype(str).tolist()
    table = _usable_offsets(table)

    values, ri, kept, dropped = _coops_records(cfg, table)
    _report_dropped(dropped)

    # MLLW -> NAVD88, per gauge, in the kept order.
    offsets = vd.offsets_as_dict(kept)
    shift = np.array([offsets[str(s)] for s in kept["StationID"]], dtype=np.float64)
    return values + shift[np.newaxis, :], ri, {"kept": kept, "tried": tried}


# ---------------------------------------------------------------------------
# Gauges that drop out
#
# A station can be found near the domain yet have no usable record for the
# year (fetch error, empty, all gaps) or no datum offset. Such gauges are
# dropped, and the gauge-zone raster must be drawn from the gauges that remain
# so that zone k always means column k-1 of the water-level table. `fetch`
# writes the kept list beside the zone raster; `run` reads it, and if the
# gauges still differ (an older fetch, or a record that vanished since),
# redraws the zones in memory from the gauges it actually has.
# ---------------------------------------------------------------------------

def zone_stations_path(cfg) -> "Path | None":
    """The kept-gauge table written next to the gauge-zone raster."""
    if not cfg.gauge_zones_path:
        return None
    z = Path(cfg.gauge_zones_path)
    return z.with_name(z.stem + "_stations.csv")


def write_zone_stations(cfg, kept) -> None:
    path = zone_stations_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(path, index=False)


def _usable_offsets(table):
    """Drop stations whose MLLW -> NAVD88 offset could not be resolved."""
    from timsa.ingest import vdatum as vd
    if vd.OFFSET_COLUMN not in table.columns:
        return table
    bad = table[vd.OFFSET_COLUMN].isna()
    for _, r in table[bad].iterrows():
        print(f"      dropping gauge {r['StationID']} {r.get('Name', '')}: "
              f"no datum offset")
    return _renumber(table[~bad])


def _renumber(table):
    out = table.reset_index(drop=True).copy()
    if "gauge_idx" in out.columns:
        out = out.drop(columns="gauge_idx")
    out.insert(0, "gauge_idx", range(len(out)))
    return out


def _sim_window(g):
    """(start, exclusive end) in UTC; both default to the full calendar year."""
    if g.start is not None:
        start = dt.datetime(g.start.year, g.start.month, g.start.day,
                            tzinfo=dt.timezone.utc)
    else:
        start = dt.datetime(g.year, 1, 1, tzinfo=dt.timezone.utc)
    if g.end is not None:
        # g.end is inclusive; build_multigauge_array's grid is left-inclusive.
        e = g.end + dt.timedelta(days=1)
        end = dt.datetime(e.year, e.month, e.day, tzinfo=dt.timezone.utc)
    else:
        end = dt.datetime(g.year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return start, end


def _coops_records(cfg, table, verbose: bool = True):
    """
    Water levels for the stations in `table`, plus the stations that kept a
    column. Returns (values, record_interval_min, kept_table, dropped_manifest)
    with kept_table renumbered so gauge_idx matches the value columns.
    """
    from timsa.ingest.coops import build_multigauge_array

    g = cfg.gauges
    start, end = _sim_window(g)
    values, manifest, ri = build_multigauge_array(
        table, g.year, start, end,
        record_interval_min=g.record_interval_min or 6, datum=g.datum,
        cache_dir=g.cache_dir, hilo_interp=g.hilo_interp, verbose=verbose,
    )
    ok = manifest["gauge_idx"].notna()
    kept_ids = (manifest[ok].sort_values("gauge_idx")["station_id"]
                .astype(str).tolist())
    by_id = table.assign(_sid=table["StationID"].astype(str)).set_index("_sid")
    kept = _renumber(by_id.loc[kept_ids])
    return values, ri, kept, manifest[~ok]


def _report_dropped(dropped) -> None:
    why = {"error": "download failed", "empty": "no data for the period"}
    for _, r in dropped.iterrows():
        print(f"      dropping gauge {r['station_id']} {r.get('name', '')}: "
              f"{why.get(r['record_type'], 'no usable records')}")


def _zones_matching_gauges(cfg, zones, gauges, dem, profile):
    """
    Return a zone raster whose indices match `kept`, redrawing it if the
    raster on disk was built from a different set of gauges.
    """
    from timsa.ingest import gauge_ref, stations as st

    kept = gauges["kept"]
    kept_ids = kept["StationID"].astype(str).tolist()
    sidecar = zone_stations_path(cfg)
    if sidecar.exists():
        drawn = st.load_stations(sidecar)["StationID"].astype(str).tolist()
    else:
        # Fetched by an earlier release, which drew zones for every station
        # it tried. Trust the raster when no gauge dropped out.
        drawn = gauges["tried"]
    if drawn == kept_ids:
        return zones
    print(f"      gauge zones were drawn for a different set of gauges; "
          f"redrawing them for the {len(kept_ids)} with data")
    raster, _ = gauge_ref.build_gauge_reference_kdtree(kept, profile, dem_array=dem)
    return raster.astype(np.int32)


def _daylight_table(cfg):
    from timsa.solar import sunrise_sunset_for_domain, load_sunrise_sunset

    if cfg.simulation.daylight_table and Path(cfg.simulation.daylight_table).exists():
        return load_sunrise_sunset(cfg.simulation.daylight_table)
    year = cfg.gauges.year or dt.date.today().year
    return sunrise_sunset_for_domain(cfg.domain, year)


def _synthetic_inputs(n_days: int = 3, interval: int = 6):
    """Small stand-in domain and record, for smoke-testing the wiring."""
    n_rec = n_days * (1440 // interval)
    t_hr = np.arange(n_rec) * interval / 60.0
    wl = (0.45 * np.sin(2 * np.pi * t_hr / 12.42)
          + 0.20 * np.sin(2 * np.pi * t_hr / 24.07))[:, None]

    nr, nc = 60, 90
    dem = (np.linspace(-1.0, 1.0, nc)[None, :].repeat(nr, 0)
           + 0.05 * np.random.RandomState(0).randn(nr, nc))
    dem[:5, :5] = np.nan
    zones = np.ones((nr, nc), dtype=np.int32)
    zones[np.isnan(dem)] = 0
    profile = {"driver": "GTiff", "dtype": "float64", "nodata": np.nan,
               "width": nc, "height": nr, "count": 1,
               "crs": None, "transform": None}
    return dem, zones, np.isnan(dem), wl, interval, profile


def _apply_overrides(cfg, args) -> None:
    if args.run_id:
        cfg.output.run_id = args.run_id
    if args.out:
        cfg.output.root = Path(args.out)
        cfg.output.rasters = Path(args.out) / "rasters"
        cfg.output.summaries = Path(args.out) / "summaries"
    if args.daily_rasters is not None:
        cfg.output.write_daily = args.daily_rasters
    if getattr(args, "write_average", None) is not None:
        cfg.output.write_average = args.write_average
    if getattr(args, "write_total", None) is not None:
        cfg.output.write_total = args.write_total
        if not args.write_total:
            cfg.output.write_average = True     # something must be written
    if args.timestep_min:
        cfg.simulation.timestep_min = args.timestep_min
    if args.gauge_interp:
        cfg.simulation.gauge_interp = args.gauge_interp
    cfg.validate()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="timsa",
        description="Tidal Inundation Model of Shallow-water Availability",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one simulation from a config")
    r.add_argument("config", type=Path)
    r.add_argument("--run-id")
    r.add_argument("--out", type=Path)
    r.add_argument("--timestep-min", type=int,
                   help="override simulation.timestep_min")
    r.add_argument("--gauge-interp", choices=("linear", "sinusoidal", "hold"))
    r.add_argument("--daily-rasters", dest="daily_rasters",
                   action="store_true", default=None,
                   help="stream one raster per day per metric")
    r.add_argument("--no-daily-rasters", dest="daily_rasters",
                   action="store_false")
    r.add_argument("--average", dest="write_average", action="store_true",
                   default=None,
                   help="also write day-averaged rasters (minutes/day); "
                        "matches the C save_average_raster for parity")
    r.add_argument("--no-total", dest="write_total", action="store_false",
                   default=None,
                   help="skip the raw-total rasters (implies --average)")
    r.add_argument("--dry-run", action="store_true",
                   help="validate the config and exit")
    r.add_argument("--quick-test", type=int, metavar="DAYS",
                   help="truncate the record to N days")
    r.add_argument("--skip-existing", action="store_true",
                   help="reuse metric rasters already on disk")
    r.add_argument("--synthetic", action="store_true",
                   help="ignore dem_path and use synthetic inputs")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("fetch", help="acquire inputs for a config")
    f.add_argument("config", type=Path)
    f.add_argument("--refresh", action="store_true",
                   help="ignore caches and re-acquire")
    f.add_argument("--cudem-regions", nargs="*",
                   help="override CUDEM region directories (e.g. FL AL)")
    f.add_argument("--allow-missing-offsets", action="store_true",
                   help="continue when a station has no datum offset")
    f.add_argument("--idw-fill-offsets", action="store_true",
                   help="fill unresolved datum offsets by inverse-distance "
                        "weighting from resolved neighbours (overrides config)")
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(func=cmd_fetch)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
