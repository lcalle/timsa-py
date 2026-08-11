"""
Tests for the prescribed-record reader and the record-interval time base.
Run with:  pytest tests/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timsa.core import TimsaInputs, TimsaConfig, TimsaSimulation
from timsa.ingest.hybrid import assemble_hybrid, HybridError
from timsa.ingest.prescribed import load_prescribed_wl, PrescribedWLError

RI = 6


def _wl_csv(tmp_path, n_days=2, interval=RI, with_time=True, gap=False):
    n_rec = n_days * (1440 // interval)
    t_hr = np.arange(n_rec) * interval / 60.0
    v = 0.45 * np.sin(2 * np.pi * t_hr / 12.42)
    if gap:
        v[10:15] = np.nan
    d = {"8723970": v, "8724580": v * 0.8}
    if with_time:
        d = {"time": pd.date_range("2025-01-01", periods=n_rec,
                                   freq=f"{interval}min", tz="UTC"), **d}
    p = tmp_path / "wl.csv"
    pd.DataFrame(d).to_csv(p, index=False)
    return p


def _grid(nr=30, nc=40):
    dem = np.linspace(-1.0, 1.0, nc)[None, :].repeat(nr, 0)
    zones = np.ones((nr, nc), dtype=np.int32)
    zones[:, nc // 2:] = 2
    return dem, zones, np.zeros((nr, nc), dtype=bool)


def _sim(wl, timestep=RI, interval=RI, **kw):
    dem, zones, nod = _grid()
    inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=wl,
                      nodata_mask=nod, record_interval_min=interval)
    cfg = TimsaConfig(depth_windows={"shallow_band": (-1.5, 0.0)},
                      refugia_thresholds=[0.2, 0.5], timestep_min=timestep, **kw)
    return TimsaSimulation(inp, cfg)


# --- reader ------------------------------------------------------------

def test_infers_six_minute_interval(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    assert wl.record_interval_min == 6
    assert wl.n_gauges == 2
    assert wl.n_records == 2 * 240


def test_numeric_station_ids_are_not_eaten_as_data(tmp_path):
    """Regression: 'time,8723970,8724580' must be detected as a header row."""
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    assert wl.station_ids == ["8723970", "8724580"]


def test_station_order_is_respected(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time",
                            station_order=["8724580", "8723970"])
    assert wl.station_ids == ["8724580", "8723970"]


def test_missing_station_raises(tmp_path):
    with pytest.raises(PrescribedWLError, match="absent"):
        load_prescribed_wl(_wl_csv(tmp_path), time_col="time",
                           station_order=["9999999"])


def test_irregular_spacing_raises(tmp_path):
    p = tmp_path / "bad.csv"
    t = list(pd.date_range("2025-01-01", periods=10, freq="6min", tz="UTC"))
    t[5] = t[5] + pd.Timedelta(minutes=3)
    pd.DataFrame({"time": t, "a": np.arange(10.0)}).to_csv(p, index=False)
    with pytest.raises(PrescribedWLError, match="irregular"):
        load_prescribed_wl(p, time_col="time")


def test_hold_matches_c_reference_gap_behaviour(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path, gap=True), time_col="time",
                            fill_gaps="hold")
    # 5 missing rows x 2 gauge columns; n_gaps_filled counts VALUES, not rows.
    assert wl.n_gaps_filled == 10
    assert not np.isnan(wl.values).any()


def test_max_gap_records_guard(tmp_path):
    with pytest.raises(PrescribedWLError, match="longest missing run"):
        load_prescribed_wl(_wl_csv(tmp_path, gap=True), time_col="time",
                           max_gap_records=2)


def test_partial_day_is_trimmed(tmp_path):
    p = tmp_path / "ragged.csv"
    n = 240 + 7
    pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="6min", tz="UTC"),
        "a": np.zeros(n),
    }).to_csv(p, index=False)
    wl = load_prescribed_wl(p, time_col="time")
    assert wl.n_records == 240


def test_headerless_c_parity_layout(tmp_path):
    p = tmp_path / "plain.csv"
    np.savetxt(p, np.random.rand(240, 2) - 0.5, delimiter=",")
    wl = load_prescribed_wl(p, record_interval_min=6)
    assert wl.n_gauges == 2 and wl.n_records == 240
    assert wl.station_ids == ["gauge_1", "gauge_2"]


# --- time base ---------------------------------------------------------

def test_accumulator_capped_by_elapsed_minutes(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path, n_days=2), time_col="time")
    s = _sim(wl.values)
    s.run(verbose=False)
    assert s.n_days == 2
    assert s.steps_per_day == 240
    assert s.time_in_band["shallow_band"].max() <= 2 * 1440


def test_timestep_finer_than_record_interpolates(tmp_path):
    """The whole point of tide_wdchange.c: step finer than the record."""
    wl = load_prescribed_wl(_wl_csv(tmp_path, n_days=1), time_col="time")
    s = _sim(wl.values, timestep=1)
    assert s.steps_per_day == 1440
    assert s.n_steps == 1440
    assert s.n_days == 1
    s.run(verbose=False)
    assert s.time_in_band["shallow_band"].max() <= 1440


def test_timestep_coarser_than_record_decimates(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path, n_days=1), time_col="time")
    s = _sim(wl.values, timestep=30)
    assert s.steps_per_day == 48
    s.run(verbose=False)


def test_non_divisor_timestep_rejected(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    with pytest.raises(ValueError, match="does not divide"):
        _sim(wl.values, timestep=7)


def test_decimation_matches_source_values(tmp_path):
    """Target times coincide with source times, so no interpolation error."""
    from timsa.core import resample_gauge_records
    src = np.random.RandomState(1).rand(240, 2)
    out = resample_gauge_records(src, 6, 30, method="linear")
    np.testing.assert_allclose(out, src[::5])


def test_sinusoidal_hits_the_endpoints(tmp_path):
    from timsa.core import resample_gauge_records
    src = np.array([[0.0], [1.0], [0.0]])
    out = resample_gauge_records(src, 360, 1, method="sinusoidal")
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[360, 0], 1.0)
    assert np.isclose(out[180, 0], 0.5, atol=1e-9)     # midpoint
    # zero slope at the extrema, maximum at the midpoint
    d = np.diff(out[:360, 0])
    assert d[0] < d[len(d) // 2] and d[-1] < d[len(d) // 2]


def test_interp_methods_agree_at_record_times(tmp_path):
    from timsa.core import resample_gauge_records
    src = np.random.RandomState(2).rand(48, 3)
    outs = [resample_gauge_records(src, 30, 5, method=m)
            for m in ("linear", "sinusoidal", "hold")]
    for o in outs[1:]:
        np.testing.assert_allclose(o[::6], outs[0][::6])


def test_incremental_matches_direct_on_clean_data(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    a = _sim(wl.values); a.run(verbose=False)
    b = _sim(wl.values, surface_update="incremental"); b.run(verbose=False)
    np.testing.assert_allclose(a.water_surface, b.water_surface, atol=1e-9)
    np.testing.assert_allclose(a.time_in_band["shallow_band"],
                               b.time_in_band["shallow_band"])


def test_daylight_bounds_is_user_selectable(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path, n_days=1), time_col="time")
    ss = np.array([[360, 1080]])
    inc = _sim(wl.values, constrain_daylight=True, sunrise_sunset_min=ss,
               daylight_bounds="inclusive")
    exc = _sim(wl.values, constrain_daylight=True, sunrise_sunset_min=ss,
               daylight_bounds="exclusive")
    inc.run(verbose=False); exc.run(verbose=False)
    # Exclusive drops at most the two boundary steps per day.
    diff = (inc.time_in_band["shallow_band"] - exc.time_in_band["shallow_band"])
    assert diff.min() >= 0
    assert diff.max() <= 2 * 6


def test_positive_window_warns_about_sign_convention():
    with pytest.warns(UserWarning, match="POSITIVE IS DRY"):
        TimsaConfig(depth_windows={"oops": (0.0, 1.5)},
                    refugia_thresholds=[0.2], timestep_min=6)


def test_negative_refugia_threshold_rejected():
    with pytest.raises(ValueError, match="positive water depths"):
        TimsaConfig(depth_windows={"b": (-1.5, 0.0)},
                    refugia_thresholds=[-0.2], timestep_min=6)


def test_record_interval_must_divide_a_day():
    dem, zones, nod = _grid()
    with pytest.raises(ValueError, match="does not divide"):
        TimsaInputs(dem=dem, gauge_zones=zones,
                    gauge_wdepths=np.zeros((240, 2)), nodata_mask=nod,
                    record_interval_min=7)


def test_chunking_does_not_change_results(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    a, b = _sim(wl.values), _sim(wl.values)
    a.run(verbose=False, cell_chunk=1_000_000)
    b.run(verbose=False, cell_chunk=7)
    np.testing.assert_allclose(a.time_in_band["shallow_band"],
                               b.time_in_band["shallow_band"])


def test_daily_callback_sums_to_annual(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path, n_days=3), time_col="time")
    s = _sim(wl.values)
    total = []
    s.run(verbose=False,
          daily_callback=lambda d, fb, fbl, sim: total.append(fb["shallow_band"].sum()))
    assert np.isclose(sum(total), s.domain_time_integrated("shallow_band"))


def test_zone_index_beyond_gauge_columns_raises(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    dem, zones, nod = _grid()
    zones[0, 0] = 9
    inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=wl.values,
                      nodata_mask=nod, record_interval_min=RI)
    cfg = TimsaConfig(depth_windows={"b": (-1.5, 0.0)}, refugia_thresholds=[],
                      timestep_min=RI)
    with pytest.raises(ValueError, match="out of sync"):
        TimsaSimulation(inp, cfg)


def test_deprecated_slr_offset_alias(tmp_path):
    wl = load_prescribed_wl(_wl_csv(tmp_path), time_col="time")
    dem, zones, nod = _grid()
    with pytest.deprecated_call():
        inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=wl.values,
                          nodata_mask=nod, record_interval_min=RI,
                          slr_offset_m=0.25)
    assert inp.water_level_offset_m == 0.25


def test_submerged_cell_counts_as_refugia():
    """Sanity check on the restored sign convention."""
    dem = np.array([[-0.3]])                      # bed 0.3 m below datum
    zones = np.ones((1, 1), dtype=np.int32)
    ws = np.zeros((240, 1))                       # water surface at datum
    inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=ws,
                      nodata_mask=np.zeros((1, 1), dtype=bool),
                      record_interval_min=6)
    cfg = TimsaConfig(depth_windows={"shallow_band": (-1.5, 0.0)},
                      refugia_thresholds=[0.2, 0.5], timestep_min=6)
    s = TimsaSimulation(inp, cfg); s.run(verbose=False)
    assert s.time_in_band["shallow_band"][0, 0] == 1440   # 0.3 m of water
    assert s.time_below_threshold[0.5][0, 0] == 1440      # within 0.5 m
    assert s.time_below_threshold[0.2][0, 0] == 0         # deeper than 0.2 m


def test_dry_cell_counts_as_nothing():
    dem = np.array([[0.4]])                       # bed 0.4 m ABOVE datum
    zones = np.ones((1, 1), dtype=np.int32)
    ws = np.zeros((240, 1))
    inp = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=ws,
                      nodata_mask=np.zeros((1, 1), dtype=bool),
                      record_interval_min=6)
    cfg = TimsaConfig(depth_windows={"shallow_band": (-1.5, 0.0)},
                      refugia_thresholds=[0.5], timestep_min=6)
    s = TimsaSimulation(inp, cfg); s.run(verbose=False)
    assert s.time_in_band["shallow_band"][0, 0] == 0
    assert s.time_below_threshold[0.5][0, 0] == 0


# --- hybrid mode, use-case 4  ------------------------------------------------------------

def _kept(ids, lats, lons):
    return pd.DataFrame({"station_id": ids, "lat": lats, "lon": lons})


def test_hybrid_orders_insitu_then_coops():
    n = 5
    iv = np.arange(2 * n, dtype=float).reshape(n, 2)
    cv = 100 + np.arange(2 * n, dtype=float).reshape(n, 2)
    h = assemble_hybrid(["S1", "S2"], iv,
                        np.array([[27.80, -82.40], [27.85, -82.42]]),
                        _kept(["8726520", "8726607"], [27.90, 27.95],
                              [-82.50, -82.55]),
                        cv, 6)
    assert h.values.shape == (5, 4)
    np.testing.assert_allclose(h.values[:, :2], iv)   # in-situ first
    np.testing.assert_allclose(h.values[:, 2:], cv)   # then CO-OPS
    t = h.station_table
    assert list(t["gauge_idx"]) == [0, 1, 2, 3]
    assert list(t["StationID"]) == ["S1", "S2", "8726520", "8726607"]
    assert list(t["gauge_source"]) == ["insitu", "insitu", "coops", "coops"]
    assert h.n_insitu == 2 and h.n_coops == 2


def test_hybrid_length_mismatch_raises():
    iv = np.zeros((5, 1)); cv = np.zeros((4, 1))
    with pytest.raises(HybridError, match="record-length mismatch"):
        assemble_hybrid(["S1"], iv, np.array([[27.8, -82.4]]),
                        _kept(["8726520"], [27.9], [-82.5]), cv, 6)


def test_hybrid_column_id_guard():
    iv = np.zeros((5, 2))
    with pytest.raises(HybridError):
        assemble_hybrid(["S1"], iv, np.array([[27.8, -82.4]]),
                        _kept(["8726520"], [27.9], [-82.5]),
                        np.zeros((5, 1)), 6)
