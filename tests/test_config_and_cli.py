"""Config parsing, CLI wiring, and the generalized ingest helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from timsa.config import RunConfig, ConfigError
from timsa.crs import utm_epsg_from_lonlat
from timsa.domain import Domain, DomainError
from timsa.ingest import vdatum as vd
from timsa.ingest import cudem


BASE = {
    "domain": {"name": "test", "bbox": [-81.95, 24.45, -80.95, 24.83],
               "resolution_m": 30},
    "gauges": {"source": "prescribed", "prescribed_file": "wl.csv",
               "time_col": "time"},
    "simulation": {"timestep_min": 6},
    "depth_windows": {"shallow_band": [-1.5, 0.0]},
    "refugia_thresholds": [0.2, 0.5],
    "output": {"root": "outputs", "run_id": "t"},
}


def _cfg(**over):
    import copy
    raw = copy.deepcopy(BASE)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k].update(v)
        else:
            raw[k] = v
    return raw


# --- config -----------------------------------------------------------

def test_parses_a_minimal_config():
    cfg = RunConfig.from_dict(_cfg())
    assert cfg.domain.name == "test"
    assert cfg.simulation.timestep_min == 6
    assert cfg.depth_windows["shallow_band"] == (-1.5, 0.0)
    assert cfg.refugia_thresholds == [0.2, 0.5]


def test_positive_window_is_rejected_with_the_sign_rule():
    with pytest.raises(ConfigError, match="POSITIVE IS DRY"):
        RunConfig.from_dict(_cfg(depth_windows={"shallow_band": [0.0, 1.5]}))


def test_manuscript_blocks_are_reported_not_silently_dropped():
    raw = _cfg()
    raw["scenarios"] = {"ar6_ssp245": {}}
    raw["figures"] = {"colormap": {}}
    cfg = RunConfig.from_dict(raw)
    joined = " ".join(cfg.unknown_keys)
    assert "scenarios" in joined and "figures" in joined
    assert "manuscript" in joined


def test_thresholds_nested_under_depth_windows_still_parse():
    """The SLR config nested them; accept that spelling."""
    raw = _cfg()
    raw["depth_windows"]["refugia_thresholds"] = [0.2, 1.0]
    del raw["refugia_thresholds"]
    cfg = RunConfig.from_dict(raw)
    assert cfg.refugia_thresholds == [0.2, 1.0]
    assert "refugia_thresholds" not in cfg.depth_windows


def test_patch001_daylight_boolean_still_understood():
    cfg = RunConfig.from_dict(
        _cfg(simulation={"daylight_bounds_exclusive": True}))
    assert cfg.simulation.daylight_bounds == "exclusive"


def test_prescribed_without_file_raises():
    raw = _cfg()
    raw["gauges"]["prescribed_file"] = None
    with pytest.raises(ConfigError, match="prescribed_file"):
        RunConfig.from_dict(raw)


def test_prescribed_without_time_col_or_interval_raises():
    raw = _cfg()
    raw["gauges"]["time_col"] = None
    with pytest.raises(ConfigError, match="record_interval_min"):
        RunConfig.from_dict(raw)


def test_coops_source_requires_year():
    with pytest.raises(ConfigError, match="year"):
        RunConfig.from_dict(_cfg(gauges={"source": "coops",
                                         "prescribed_file": None}))


def test_non_divisor_timestep_rejected():
    with pytest.raises(ConfigError, match="divide"):
        RunConfig.from_dict(_cfg(simulation={"timestep_min": 7}))


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_cfg()))
    cfg = RunConfig.from_yaml(p)
    assert cfg.source_path == p
    assert "domain" in cfg.report()


def test_shipped_example_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    cfg = RunConfig.from_yaml(root / "configs" / "example_present_day.yaml")
    assert cfg.depth_windows["shallow_band"] == (-1.5, 0.0)
    assert cfg.unknown_keys == []


# --- crs --------------------------------------------------------------

def test_utm_zone_from_lonlat():
    assert utm_epsg_from_lonlat(-81.5, 24.6) == "EPSG:32617"   # Florida Keys
    assert utm_epsg_from_lonlat(-122.4, 37.8) == "EPSG:32610"  # San Francisco
    assert utm_epsg_from_lonlat(151.2, -33.9) == "EPSG:32756"  # Sydney, south


def test_target_crs_defaults_from_domain_not_florida():
    from timsa.crs import resolve_target_crs
    d = Domain.from_config({"name": "wa", "bbox": [-124.0, 47.0, -123.0, 48.0]})
    assert resolve_target_crs(d) == "EPSG:32610"


# --- domain -----------------------------------------------------------

def test_bbox_or_boundary_but_not_both():
    with pytest.raises(DomainError, match="exactly one"):   # was: "not both"
        Domain.from_config({"name": "x", "bbox": [0, 0, 1, 1],
                            "boundary_file": "a.geojson"})

def test_impossible_latitude_raises():
    with pytest.raises(DomainError, match="latitudes out of range"):
        Domain.from_config({"name": "x", "bbox": [-81.9, -95.0, -81.2, -90.5]})


def test_probable_latlon_swap_warns():
    """A swapped bbox is often still a valid bbox elsewhere, so this warns
    rather than raising; only the polar tell is reliable."""
    with pytest.warns(UserWarning, match="80 degrees latitude"):
        Domain.from_config({"name": "x",
                            "bbox": [24.45, -81.95, 24.83, -80.95]})


def test_station_filter_fails_loudly_when_empty():
    d = Domain.from_config({"name": "x", "bbox": [-81.9, 24.5, -81.2, 24.8],
                            "gauge_search_buffer_km": 5})
    df = pd.DataFrame({"StationID": ["1"], "Latitude": [47.0],
                       "Longitude": [-122.0]})
    with pytest.raises(DomainError, match="gauge_search_buffer_km"):
        d.filter_stations(df, lon_col="Longitude", lat_col="Latitude")

def test_center_radius_derives_enclosing_bbox():
    d = Domain.from_config({"name": "pt", "center": [-82.40, 27.85],
                            "radius_km": 8})
    xmin, ymin, xmax, ymax = d.bbox
    # centroid returns to the center
    cx, cy = d.centroid
    assert abs(cx - (-82.40)) < 1e-6 and abs(cy - 27.85) < 1e-6
    # square-ish in km: latitude half-width is radius / 110.574 deg
    assert abs((ymax - ymin) / 2 - 8 / 110.574) < 1e-6
    assert d.geometry is None            # box shape attaches no polygon
    assert d.source.startswith("center+radius")


def test_center_radius_circle_attaches_polygon_within_bbox():
    d = Domain.from_config({"name": "pt", "center": [-82.40, 27.85],
                            "radius_km": 8, "shape": "circle"})
    assert d.geometry is not None
    bxmin, bymin, bxmax, bymax = d.bbox
    gxmin, gymin, gxmax, gymax = d.geometry.bounds
    assert bxmin <= gxmin + 1e-9 and bxmax >= gxmax - 1e-9
    assert bymin <= gymin + 1e-9 and bymax >= gymax - 1e-9


def test_center_is_exclusive_with_bbox():
    with pytest.raises(DomainError, match="exactly one"):
        Domain.from_config({"name": "x", "center": [-82, 27],
                            "radius_km": 5, "bbox": [0, 0, 1, 1]})


def test_center_requires_positive_radius():
    with pytest.raises(DomainError, match="radius_km"):
        Domain.from_config({"name": "x", "center": [-82, 27]})
    with pytest.raises(DomainError, match="positive"):
        Domain.from_config({"name": "x", "center": [-82, 27], "radius_km": 0})


def test_center_bad_shape_rejected():
    with pytest.raises(DomainError, match="shape"):
        Domain.from_config({"name": "x", "center": [-82, 27],
                            "radius_km": 5, "shape": "blob"})

# --- vdatum -----------------------------------------------------------

def test_vdatum_region_is_derived_not_assumed():
    assert vd.vdatum_region(-81.5, 24.6) == "contiguous"
    assert vd.vdatum_region(-149.9, 61.2) == "ak"
    assert vd.vdatum_region(-66.1, 18.4) == "prvi"


def test_missing_offset_fails_loudly_by_default():
    df = pd.DataFrame({"StationID": ["999"], "Latitude": [24.6],
                       "Longitude": [-81.5]})
    with pytest.raises(vd.VdatumError, match="wrong vertical datum"):
        vd.batch_resolve_offsets(df, manual_csv=None, order=("manual_csv",),
                                 polite_sec=0, verbose=False)


def test_manual_csv_resolution(tmp_path):
    p = tmp_path / "v.csv"
    pd.DataFrame({"StationID": ["999"],
                  vd.OFFSET_COLUMN: [0.123]}).to_csv(p, index=False)
    df = pd.DataFrame({"StationID": ["999"], "Latitude": [24.6],
                       "Longitude": [-81.5]})
    out = vd.batch_resolve_offsets(df, manual_csv=p, order=("manual_csv",),
                                   polite_sec=0, verbose=False)
    assert out[vd.OFFSET_COLUMN].iloc[0] == 0.123
    assert out[vd.SOURCE_COLUMN].iloc[0] == "manual_csv"
    assert vd.offsets_as_dict(out) == {"999": 0.123}


def test_error_rows_from_the_old_batch_script_are_skipped(tmp_path):
    p = tmp_path / "v.csv"
    pd.DataFrame({"StationID": ["1", "2"],
                  vd.OFFSET_COLUMN: [0.1, "ERROR"]}).to_csv(p, index=False)
    assert vd.load_manual_offsets(p) == {"1": 0.1}


# --- cudem ------------------------------------------------------------

def test_cudem_regions_resolve_beyond_florida():
    assert "FL" in cudem.regions_for_bbox((-81.95, 24.45, -80.95, 24.83))
    assert "TX" in cudem.regions_for_bbox((-97.0, 27.0, -96.0, 28.0))
    # Pacific NW isn't a "WA" directory on NOAA's S3 mirror — it's split into
    # feature-named dirs (wash_juandefuca / wash_outercoast / wash_pugetsound).
    wa = cudem.regions_for_bbox((-124.0, 47.0, -123.0, 48.0))
    assert any(r.startswith("wash_") for r in wa)


def test_cudem_domain_spanning_a_state_line_gets_both():
    hits = cudem.regions_for_bbox((-88.2, 30.0, -87.2, 30.6))   # MS/AL/FL
    assert len(hits) >= 2


def test_cudem_inland_bbox_raises():
    with pytest.raises(cudem.CudemError, match="no CUDEM region"):
        cudem.regions_for_bbox((-100.0, 40.0, -99.0, 41.0))


def test_tile_bbox_filter():
    tiles = [{"name": "a", "sw_lat": 24.5, "sw_lon": -81.5},
             {"name": "b", "sw_lat": 30.0, "sw_lon": -88.0}]
    keep = cudem.filter_tiles_by_bbox(tiles, (-81.6, 24.4, -81.3, 24.7))
    assert [t["name"] for t in keep] == ["a"]


# --- cli --------------------------------------------------------------

def test_cli_dry_run_validates_config(tmp_path, capsys):
    from timsa.cli import main
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_cfg()))
    assert main(["run", str(p), "--dry-run"]) == 0
    assert "configuration is valid" in capsys.readouterr().out


def test_cli_bad_config_returns_nonzero(tmp_path, capsys):
    from timsa.cli import main
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_cfg(depth_windows={"b": [0.0, 1.5]})))
    assert main(["run", str(p), "--dry-run"]) == 1
    assert "POSITIVE IS DRY" in capsys.readouterr().err


def test_cli_synthetic_run_end_to_end(tmp_path):
    from timsa.cli import main
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_cfg(output={"root": str(tmp_path / "out"),
                                             "run_id": "syn"})))
    assert main(["run", str(p), "--synthetic"]) == 0
    summary = tmp_path / "out" / "summaries" / "syn_summary.csv"
    assert summary.exists()
    df = pd.read_csv(summary)
    assert set(df["metric"]) == {"area_available", "time_integrated",
                                 "refugia_time"}


def test_cli_overrides_reach_the_simulation(tmp_path, capsys):
    from timsa.cli import main
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_cfg(output={"root": str(tmp_path / "o")})))
    main(["run", str(p), "--synthetic", "--timestep-min", "1",
          "--gauge-interp", "sinusoidal", "--quick-test", "1"])
    out = capsys.readouterr().out
    assert "timestep_min: 1" in out and "gauge_interp: sinusoidal" in out


# --- day-averaged output (C parity) -----------------------------------

def test_average_equals_total_over_ndays(tmp_path):
    """The averaged raster must be exactly total / n_days per cell."""
    import numpy as np
    from timsa.core import TimsaInputs, TimsaConfig, TimsaSimulation
    from timsa.metrics import RunResult, write_metric_rasters, metric_raster_paths
    import rasterio

    # A real (tiny) projected profile so rasters actually get written.
    n_days = 3
    n_rec = n_days * 240
    wl = (0.3 * np.sin(2 * np.pi * np.arange(n_rec) * 6 / 60.0 / 12.42))[:, None]
    dem = np.array([[-0.3, 0.4], [-0.1, -0.6]])
    zones = np.ones((2, 2), dtype=np.int32)
    inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=wl,
                      nodata_mask=np.zeros((2, 2), dtype=bool),
                      record_interval_min=6)
    cfg = TimsaConfig(depth_windows={"shallow_band": (-1.5, 0.0)},
                      refugia_thresholds=[0.5], timestep_min=6)
    sim = TimsaSimulation(inp, cfg); sim.run(verbose=False)
    assert sim.n_days == 3

    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "width": 2, "height": 2, "nodata": -9999.0,
               "crs": "EPSG:32617",
               "transform": rasterio.transform.from_origin(0, 0, 30, 30)}
    run = RunResult(run_id="p", sim=sim, profile=profile, cell_area_m2=900.0)
    run.compute_summaries()
    write_metric_rasters(run, tmp_path, write_total=True, write_average=True)

    paths = metric_raster_paths("p", tmp_path, ["shallow_band"], [0.5])
    with rasterio.open(paths[("time_integrated", "shallow_band")]) as src:
        total = src.read(1)
    with rasterio.open(paths[("time_average", "shallow_band")]) as src:
        avg = src.read(1)
    valid = total != -9999.0
    np.testing.assert_allclose(avg[valid], total[valid] / 3, rtol=1e-5)


def test_output_flags_parse():
    cfg = RunConfig.from_dict(_cfg(output={"root": "o", "run_id": "t",
                                           "write_average": True,
                                           "write_total": False}))
    assert cfg.output.write_average is True
    assert cfg.output.write_total is False


def test_write_metric_rasters_rejects_both_off():
    import numpy as np, pytest as _pt
    from timsa.core import TimsaInputs, TimsaConfig, TimsaSimulation
    from timsa.metrics import RunResult, write_metric_rasters
    dem = np.array([[-0.3]])
    inp = TimsaInputs(dem=dem, gauge_zones=np.ones((1, 1), dtype=np.int32),
                      gauge_wdepths=np.zeros((240, 1)),
                      nodata_mask=np.zeros((1, 1), dtype=bool),
                      record_interval_min=6)
    cfg = TimsaConfig(depth_windows={"b": (-1.5, 0.0)}, refugia_thresholds=[],
                      timestep_min=6)
    sim = TimsaSimulation(inp, cfg); sim.run(verbose=False)
    run = RunResult(run_id="x", sim=sim, profile={"crs": None, "transform": None})
    with _pt.raises(ValueError, match="nothing would be written"):
        write_metric_rasters(run, "/tmp/x", write_total=False, write_average=False)


def test_cli_no_total_implies_average(tmp_path, capsys):
    from timsa.cli import main
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_cfg(output={"root": str(tmp_path / "o"),
                                             "run_id": "s"})))
    # synthetic profile writes nothing, but the flag wiring still exercises
    assert main(["run", str(p), "--synthetic", "--no-total"]) == 0
