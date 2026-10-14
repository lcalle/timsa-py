"""
Cell skipping, automatic chunk sizing, run estimates, the per-tile CUDEM warp,
and the uint8 area raster.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from timsa.config import RunConfig
from timsa.core import TimsaInputs, TimsaConfig, TimsaSimulation
from timsa.estimate import estimate_run, grid_shape_from_bbox

RI = 6


def _tide(n_days=2, interval=RI, n_gauges=2):
    n_rec = n_days * (1440 // interval)
    t_hr = np.arange(n_rec) * interval / 60.0
    base = 0.45 * np.sin(2 * np.pi * t_hr / 12.42)
    return np.column_stack([base * (1.0 - 0.2 * g) for g in range(n_gauges)])


def _sim(dem, wl, windows, thresholds, timestep=RI):
    zones = np.ones(dem.shape, dtype=np.int32)
    zones[:, dem.shape[1] // 2:] = 2
    inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=wl,
                      nodata_mask=np.zeros(dem.shape, dtype=bool),
                      record_interval_min=RI)
    cfg = TimsaConfig(depth_windows=windows, refugia_thresholds=thresholds,
                      timestep_min=timestep)
    return TimsaSimulation(inp, cfg), zones


def _brute_force(dem, zones, ws, windows, thresholds, timestep):
    """Reference accumulators: every cell, every step, no skipping."""
    band = {k: np.zeros(dem.shape) for k in windows}
    below = {float(t): np.zeros(dem.shape) for t in thresholds}
    for step in range(ws.shape[0]):
        depth = dem - ws[step][zones - 1]
        for k, (lo, hi) in windows.items():
            band[k] += ((depth >= lo) & (depth <= hi)) * timestep
        for t in thresholds:
            below[float(t)] += ((depth < 0) & (depth >= -t)) * timestep
    return band, below


@pytest.mark.parametrize("windows,thresholds", [
    ({"shallow_band": (-1.5, 0.0)}, [0.2, 0.5]),
    ({"wet_and_edge": (-0.3, 0.4)}, []),       # window reaching above waterline
    ({}, [0.3]),                               # thresholds only
    ({"near_flood": (0.0, 0.5)}, []),          # ground 0-0.5 m above the water
])
@pytest.mark.filterwarnings("ignore:depth window")
def test_skipped_cells_match_brute_force(windows, thresholds):
    # Grid from deep water (-6 m) to well above the highest tide (+3 m).
    dem = np.linspace(-6.0, 3.0, 60)[None, :].repeat(20, 0)
    sim, zones = _sim(dem, _tide(), windows, thresholds)
    assert sim.n_skipped_too_dry > 0
    assert sim.n_skipped_too_deep > 0
    sim.run(verbose=False)

    band, below = _brute_force(dem, zones, sim.water_surface, windows,
                               thresholds, RI)
    for k in windows:
        np.testing.assert_array_equal(sim.time_in_band[k], band[k])
    for t in thresholds:
        np.testing.assert_array_equal(sim.time_below_threshold[float(t)],
                                      below[float(t)])


def test_window_above_waterline_counts_dry_time_and_warns():
    dem = np.linspace(-1.0, 2.0, 31)[None, :].repeat(4, 0)
    with pytest.warns(UserWarning, match="above the water surface"):
        sim, _ = _sim(dem, _tide(), {"near_flood": (0.0, 0.5)}, [])
    sim.run(verbose=False)
    t = sim.time_in_band["near_flood"]
    assert t[:, np.isclose(dem[0], 0.6)].min() > 0          # just above high tide: counted
    assert (t[:, dem[0] >= 1.0] == 0).all()        # far above the window: never


def test_skipped_cells_serialize_as_zero_not_nodata():
    dem = np.linspace(-6.0, 3.0, 60)[None, :].repeat(4, 0)
    sim, _ = _sim(dem, _tide(), {"shallow_band": (-1.5, 0.0)}, [0.5])
    sim.run(verbose=False)
    skipped = sim.base_valid_mask & ~sim.valid_mask
    assert skipped.any()
    assert (sim.time_in_band["shallow_band"][skipped] == 0).all()


def test_auto_chunk_gives_identical_results():
    dem = np.linspace(-2.0, 1.0, 300)[None, :].repeat(30, 0)
    windows, thresholds = {"shallow_band": (-1.5, 0.0)}, [0.2, 0.5]
    a, _ = _sim(dem, _tide(), windows, thresholds)
    b, _ = _sim(dem, _tide(), windows, thresholds)
    a.run(verbose=False, cell_chunk="auto")
    b.run(verbose=False, cell_chunk=13)
    for k in windows:
        np.testing.assert_array_equal(a.time_in_band[k], b.time_in_band[k])
    for t in thresholds:
        np.testing.assert_array_equal(a.time_below_threshold[t],
                                      b.time_below_threshold[t])


def test_auto_chunk_scales_inversely_with_steps_per_day():
    dem = np.zeros((4, 4))
    six, _ = _sim(dem, _tide(), {"b": (-1.0, 0.0)}, [], timestep=6)
    one, _ = _sim(dem, _tide(), {"b": (-1.0, 0.0)}, [], timestep=1)
    assert six.auto_cell_chunk(256) == pytest.approx(6 * one.auto_cell_chunk(256), rel=1e-3)
    # The budget is honoured: chunk * steps_per_day * bytes stays within it.
    from timsa.core import BYTES_PER_CELL_STEP
    assert one.auto_cell_chunk(256) * 1440 * BYTES_PER_CELL_STEP <= 256 * 2**20


def test_auto_chunk_has_a_floor():
    dem = np.zeros((4, 4))
    sim, _ = _sim(dem, _tide(), {"b": (-1.0, 0.0)}, [], timestep=1)
    assert sim.auto_cell_chunk(0.001) == 1000


# --- config -----------------------------------------------------------------

BASE = {
    "domain": {"name": "est", "center": [-97.1, 27.75], "radius_km": 15,
               "resolution_m": 30},
    "gauges": {"source": "coops", "year": 2025},
    "simulation": {"timestep_min": 6},
    "depth_windows": {"shallow_band": [-1.5, 0.0]},
    "refugia_thresholds": [0.2, 0.5, 1.0],
    "output": {"root": "outputs", "run_id": "t"},
}


def _cfg(**sim):
    raw = copy.deepcopy(BASE)
    raw["simulation"].update(sim)
    return RunConfig.from_dict(raw)


def test_cell_chunk_defaults_to_auto_and_parses_ints():
    assert _cfg().simulation.cell_chunk == "auto"
    assert _cfg(cell_chunk=5000).simulation.cell_chunk == 5000
    assert _cfg(cell_chunk="auto", chunk_mem_mb=64).simulation.chunk_mem_mb == 64


# --- estimate ---------------------------------------------------------------

def test_estimate_30km_box_at_30m():
    est = estimate_run(_cfg())
    # 30 km box at 30 m -> about 1000 x 1000 cells.
    assert 0.9e6 < est.cells < 1.15e6
    assert est.n_days == 365
    assert est.steps == 365 * 240
    assert est.cell_steps == est.cells * est.steps
    assert est.source == "bbox"


def test_estimate_scales_with_timestep_and_window():
    six = estimate_run(_cfg(timestep_min=6))
    one = estimate_run(_cfg(timestep_min=1))
    assert one.cell_steps == 6 * six.cell_steps

    raw = copy.deepcopy(BASE)
    raw["gauges"].update(start="2025-06-01", end="2025-06-30")
    est = estimate_run(RunConfig.from_dict(raw))
    assert est.n_days == 30


def test_estimate_block_memory_follows_chunk_budget():
    est = estimate_run(_cfg(chunk_mem_mb=128))
    assert est.block_mem_mb == pytest.approx(128)
    assert "cell-steps" in est.report()


def test_estimate_prescribed_days_unknown():
    raw = copy.deepcopy(BASE)
    raw["gauges"] = {"source": "prescribed", "prescribed_file": "wl.csv",
                     "time_col": "time"}
    est = estimate_run(RunConfig.from_dict(raw))
    assert est.n_days is None and est.cell_steps is None
    assert "unknown" in est.report()


def test_grid_shape_resolution_scaling():
    bbox = (-97.2, 27.7, -97.0, 27.8)
    r30 = grid_shape_from_bbox(bbox, 30)
    r3 = grid_shape_from_bbox(bbox, 3)
    assert abs(r3[0] - 10 * r30[0]) <= 10 and abs(r3[1] - 10 * r30[1]) <= 10


# --- CUDEM per-tile warp ----------------------------------------------------

def _write_tile(path, west, north, deg, value_fn, nodata=-999999.0):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    res = 1 / 9 / 3600
    n = int(round(deg / res))
    lon = west + (np.arange(n) + 0.5) * res
    lat = north - (np.arange(n) + 0.5) * res
    z = value_fn(lon[None, :], lat[:, None]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1,
                       dtype="float32", crs="EPSG:4269", nodata=nodata,
                       transform=from_origin(west, north, res, res)) as d:
        d.write(z, 1)


def test_cudem_warp_matches_analytic_surface(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from pyproj import Transformer
    from timsa.ingest import cudem

    def f(lon, lat):
        return 10 * (lon + 97.0) + 5 * (lat - 27.8)

    deg = 0.01
    t1, t2 = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_tile(t1, -97.02, 27.81, deg, f)
    _write_tile(t2, -97.02 + deg, 27.81, deg, f)
    out = tmp_path / "dem.tif"
    bbox = (-97.018, 27.802, -97.002, 27.808)
    info = cudem.mosaic_and_reproject_cudem(
        [t1, t2], out, target_crs="EPSG:32614", target_res_m=30,
        bbox_lonlat=bbox, verbose=False)
    assert info["n_tiles"] == 2

    with rasterio.open(out) as s:
        a = s.read(1)
        assert s.dtypes[0] == "float32"
        rows, cols = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        x, y = rasterio.transform.xy(s.transform, rows.ravel(), cols.ravel())
    lon, lat = Transformer.from_crs("EPSG:32614", "EPSG:4269",
                                    always_xy=True).transform(x, y)
    truth = f(np.asarray(lon), np.asarray(lat)).reshape(a.shape)
    valid = a != -9999.0
    assert valid.mean() > 0.9
    assert np.abs(a[valid] - truth[valid]).max() < 5e-3


def test_cudem_warp_grid_is_snapped_and_repeatable():
    pytest.importorskip("rasterio")
    from timsa.ingest import cudem
    bbox = (-97.018, 27.802, -97.002, 27.808)
    t1, w1, h1 = cudem.target_grid(bbox, "EPSG:32614", 30)
    t2, w2, h2 = cudem.target_grid(bbox, "EPSG:32614", 30)
    assert (t1, w1, h1) == (t2, w2, h2)
    assert t1.c % 30 == 0 and t1.f % 30 == 0


def test_cudem_warp_rejects_tiles_outside_grid(tmp_path):
    pytest.importorskip("rasterio")
    from timsa.ingest import cudem
    t = tmp_path / "far.tif"
    _write_tile(t, -90.0, 30.0, 0.01, lambda lon, lat: lon * 0 + 1.0)
    with pytest.raises(cudem.CudemError, match="overlap"):
        cudem.mosaic_and_reproject_cudem(
            [t], tmp_path / "o.tif", target_crs="EPSG:32614",
            target_res_m=30, bbox_lonlat=(-97.02, 27.80, -97.0, 27.81),
            verbose=False)


# --- area raster dtype ------------------------------------------------------

def test_area_available_raster_is_uint8(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from timsa.metrics import RunResult, write_metric_rasters, AREA_NODATA

    dem = np.linspace(-2.0, 1.0, 40)[None, :].repeat(10, 0)
    sim, _ = _sim(dem, _tide(), {"shallow_band": (-1.5, 0.0)}, [0.5])
    sim.run(verbose=False)
    profile = {"driver": "GTiff", "width": 40, "height": 10, "count": 1,
               "dtype": "float64", "crs": "EPSG:32614",
               "transform": from_origin(0, 300, 30, 30)}
    run = RunResult(run_id="u8", sim=sim, profile=profile)
    paths = write_metric_rasters(run, tmp_path)
    area = [p for p in paths if "area_available" in p.name][0]
    with rasterio.open(area) as s:
        assert s.dtypes[0] == "uint8"
        assert s.nodata == AREA_NODATA
        assert set(np.unique(s.read(1))) <= {0, 1, AREA_NODATA}
    total = [p for p in paths if "time_integrated" in p.name][0]
    with rasterio.open(total) as s:
        assert s.dtypes[0] == "float32"


# --- land (never-submerged) mask ------------------------------------------------

def test_never_submerged_mask_matches_water_levels():
    dem = np.linspace(-3.0, 2.0, 50)[None, :].repeat(6, 0)
    sim, zones = _sim(dem, _tide(), {"shallow_band": (-1.5, 0.0)}, [0.5])
    ws_max = np.nanmax(sim.water_surface, axis=0)[zones - 1]
    expected = (dem - ws_max) >= 0
    np.testing.assert_array_equal(sim.never_submerged_mask, expected)
    assert 0 < sim.never_submerged_mask.sum() < dem.size
    assert sim.describe()["n_never_submerged"] == int(expected.sum())


def test_never_submerged_raster_is_written(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from timsa.metrics import RunResult, write_metric_rasters, never_submerged_path

    dem = np.linspace(-3.0, 2.0, 40)[None, :].repeat(5, 0)
    sim, _ = _sim(dem, _tide(), {"shallow_band": (-1.5, 0.0)}, [])
    sim.run(verbose=False)
    profile = {"driver": "GTiff", "width": 40, "height": 5, "count": 1,
               "dtype": "float64", "crs": "EPSG:32614",
               "transform": from_origin(0, 150, 30, 30)}
    write_metric_rasters(RunResult(run_id="land", sim=sim, profile=profile), tmp_path)
    with rasterio.open(never_submerged_path("land", tmp_path)) as s:
        a = s.read(1)
        assert s.dtypes[0] == "uint8"
    np.testing.assert_array_equal(a == 1, sim.never_submerged_mask)
