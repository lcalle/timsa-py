"""
timsa.config
============
Parse and validate a TiMSA run configuration.

Replaces the inline config handling that lived in `run_pipeline.run()` and
`fetch_data._load_config()`. Scenario, horizon, sensitivity, interaction-rate,
and figure blocks are deliberately absent: those belong to the manuscript
pipeline, not to the general tool. A config carrying them is not rejected, but
the unknown keys are reported so a copied SLR config does not fail silently in
a way that looks like it worked.

The sign convention is validated here rather than being discovered at run time:
depth windows use the C reference's POSITIVE IS DRY convention, so a shallow
band of 0-1.5 m of water is written [-1.5, 0.0].
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

MINUTES_PER_DAY = 1440

KNOWN_TOP_LEVEL = {
    "domain", "gauges", "simulation", "depth_windows",
    "refugia_thresholds", "output",
}

# Blocks that belong to the manuscript pipeline. Named explicitly so the
# warning can say where they went rather than just "unknown key".
MANUSCRIPT_KEYS = {
    "scenarios": "SLR scenario sweep",
    "sensitivity": "M2 amplitude sensitivity",
    "interaction_thresholds": "interaction-rate model",
    "figures": "manuscript figure generation",
}


class ConfigError(ValueError):
    """Raised when a configuration is missing, contradictory, or unusable."""


@dataclass
class OutputConfig:
    root: Path = Path("outputs")
    rasters: Path = Path("outputs/rasters")
    summaries: Path = Path("outputs/summaries")
    run_id: str = "run"
    write_annual: bool = True
    write_total: bool = True         # raw cell-minutes (C save_sum_raster)
    write_average: bool = False      # minutes per day (C save_average_raster)
    write_daily: bool = False
    daily_dtype: str = "float32"
    daily_compress: str = "lzw"
    daily_metrics: list | None = None


@dataclass
class GaugeConfig:
    source: str = "coops"                 # 'coops' | 'prescribed'

    # prescribed
    prescribed_file: Path | None = None
    time_col: str | None = "time"
    record_interval_min: int | None = None
    fill_gaps: str = "hold"
    max_gap_records: int | None = None
    value_scale: float = 1.0
    station_order: list | None = None
    sensor_meta_csv: Path | None = None   # hybrid: in-situ sensor lon/lat for zones

    # coops
    stations_csv: Path | None = None
    datum: str = "MLLW"
    year: int | None = None
    # Optional sub-year simulation window (coops only). Both default to the
    # full calendar year of `year`. `start` is inclusive; `end` is inclusive
    # too (a day is added internally because the build grid is left-inclusive),
    # so end="2025-07-31" includes all of July 31. Must fall within `year`;
    # cross-year windows aren't supported because the fetch is keyed to a
    # single year.
    start: dt.date | None = None
    end: dt.date | None = None
    min_obs_coverage_pct: float = 80.0
    cache_dir: Path = Path("cache/coops")
    vdatum_csv: Path | None = None
    # Last-resort spatial fill for stations no source can resolve: inverse-
    # distance weighting from the resolved stations. Off by default.
    vdatum_idw_fill: bool = True
    vdatum_idw_power: float = 2.0
    vdatum_idw_k: int | None = None
    hilo_interp: str = "sinusoidal"


@dataclass
class SimulationConfig:
    timestep_min: int = 1
    gauge_interp: str = "linear"
    surface_update: str = "direct"
    gap_policy: str = "hold"
    constrain_daylight: bool = False
    daylight_bounds: str = "inclusive"
    daylight_table: Path | None = None
    water_level_offset_m: float = 0.0
    cell_chunk: int = 100_000


@dataclass
class RunConfig:
    """A fully resolved run configuration."""

    domain: object                        # timsa.domain.Domain
    gauges: GaugeConfig
    simulation: SimulationConfig
    output: OutputConfig
    depth_windows: dict
    refugia_thresholds: list
    dem_path: Path | None = None
    gauge_zones_path: Path | None = None
    land_mask_path: Path | None = None
    target_crs: str | None = None
    source_path: Path | None = None
    unknown_keys: list = field(default_factory=list)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        import yaml

        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls.from_dict(raw)
        cfg.source_path = path
        return cfg

    @classmethod
    def from_dict(cls, raw: dict) -> "RunConfig":
        from timsa.domain import Domain

        unknown = []
        for key in raw:
            if key in MANUSCRIPT_KEYS:
                unknown.append(
                    f"{key!r} ({MANUSCRIPT_KEYS[key]}) — belongs to the "
                    f"manuscript pipeline and is ignored here"
                )
            elif key not in KNOWN_TOP_LEVEL:
                unknown.append(f"{key!r} — unrecognized")

        if "domain" not in raw:
            raise ConfigError("config has no 'domain' block")
        dom_raw = dict(raw["domain"])
        domain = Domain.from_config(dom_raw)

        dem_path = _opt_path(dom_raw.get("dem_path"))
        zones_path = _opt_path(dom_raw.get("gauge_zones_path"))
        mask_path = _opt_path(dom_raw.get("land_mask_path"))
        target_crs = dom_raw.get("crs") or None

        gauges = _parse_gauges(raw.get("gauges", {}) or {})
        sim = _parse_simulation(raw.get("simulation", {}) or {})
        out = _parse_output(raw.get("output", {}) or {})

        windows = _parse_depth_windows(raw.get("depth_windows"))
        thresholds = _parse_thresholds(raw.get("refugia_thresholds"),
                                       raw.get("depth_windows"))

        cfg = cls(
            domain=domain, gauges=gauges, simulation=sim, output=out,
            depth_windows=windows, refugia_thresholds=thresholds,
            dem_path=dem_path, gauge_zones_path=zones_path,
            land_mask_path=mask_path, target_crs=target_crs,
            unknown_keys=unknown,
        )
        cfg.validate()
        return cfg

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        s = self.simulation
        if MINUTES_PER_DAY % s.timestep_min != 0:
            raise ConfigError(
                f"simulation.timestep_min={s.timestep_min} must divide "
                f"{MINUTES_PER_DAY} evenly."
            )

        g = self.gauges
        if g.source not in ("coops", "prescribed", "hybrid"):
            raise ConfigError(
                f"gauges.source must be 'coops', 'prescribed', or 'hybrid', "
                f"got {g.source!r}"
            )

        # The prescribed (in-situ) side is required by 'prescribed' and 'hybrid'.
        if g.source in ("prescribed", "hybrid"):
            if g.prescribed_file is None:
                raise ConfigError(
                    f"gauges.source={g.source!r} requires gauges.prescribed_file"
                )
            if g.time_col is None and g.record_interval_min is None:
                raise ConfigError(
                    "gauges.record_interval_min is required when "
                    "gauges.time_col is null (nothing to infer the spacing from)"
                )

        # The CO-OPS side is required by 'coops' and 'hybrid'.
        if g.source in ("coops", "hybrid"):
            if g.year is None:
                raise ConfigError(
                    f"gauges.year is required when source={g.source!r} "
                    f"(the CO-OPS fetch is keyed to a single year)"
                )
            for label, d in (("start", g.start), ("end", g.end)):
                if d is not None and d.year != g.year:
                    raise ConfigError(
                        f"gauges.{label}={d.isoformat()} must fall within "
                        f"gauges.year={g.year}; cross-year windows aren't "
                        f"supported (the CO-OPS fetch is keyed to a single year)."
                    )
            if g.start is not None and g.end is not None and g.end < g.start:
                raise ConfigError(
                    f"gauges.end ({g.end.isoformat()}) precedes gauges.start "
                    f"({g.start.isoformat()})."
                )

        # Hybrid needs sensor locations to place in-situ gauges in the zones.
        if g.source == "hybrid" and g.sensor_meta_csv is None:
            raise ConfigError(
                "gauges.source='hybrid' requires gauges.sensor_meta_csv: the "
                "lon/lat of each in-situ sensor, with StationID matching the "
                "prescribed file's data columns, so they can be placed in the "
                "gauge-zone raster."
            )

        if not self.depth_windows and not self.refugia_thresholds:
            raise ConfigError(
                "no metrics requested: define depth_windows, refugia_thresholds, "
                "or both."
            )

    def report(self) -> str:
        lines = [
            f"domain      : {self.domain}",
            f"gauges      : source={self.gauges.source}",
            f"simulation  : timestep={self.simulation.timestep_min} min, "
            f"interp={self.simulation.gauge_interp}, "
            f"surface={self.simulation.surface_update}",
            f"windows     : {self.depth_windows}",
            f"thresholds  : {self.refugia_thresholds}",
            f"output      : {self.output.root} (run_id={self.output.run_id}, "
            f"daily={self.output.write_daily})",
        ]
        for u in self.unknown_keys:
            lines.append(f"  note: config key {u}")
        return "\n".join(lines)

    # -- adapters --------------------------------------------------------

    def timsa_config(self, sunrise_sunset_min=None):
        """Build the core TimsaConfig from this run configuration."""
        from timsa.core import TimsaConfig

        s = self.simulation
        return TimsaConfig(
            depth_windows=self.depth_windows,
            refugia_thresholds=self.refugia_thresholds,
            timestep_min=s.timestep_min,
            gauge_interp=s.gauge_interp,
            surface_update=s.surface_update,
            gap_policy=s.gap_policy,
            constrain_daylight=s.constrain_daylight,
            sunrise_sunset_min=sunrise_sunset_min,
            daylight_bounds=s.daylight_bounds,
        )


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _opt_path(v):
    return None if v in (None, "", "null") else Path(v)


def _opt_date(v):
    """Parse an optional ISO date. PyYAML may hand us a date/datetime already."""
    if v in (None, "", "null"):
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return dt.date.fromisoformat(str(v))


def _parse_gauges(raw: dict) -> GaugeConfig:
    g = GaugeConfig()
    g.source = str(raw.get("source", g.source))
    g.prescribed_file = _opt_path(raw.get("prescribed_file"))
    g.time_col = raw.get("time_col", g.time_col)
    if g.time_col in ("", "null", "none", "None"):
        g.time_col = None
    ri = raw.get("record_interval_min")
    g.record_interval_min = None if ri in (None, "", "null") else int(ri)
    g.fill_gaps = str(raw.get("fill_gaps", g.fill_gaps))
    mg = raw.get("max_gap_records")
    g.max_gap_records = None if mg in (None, "", "null") else int(mg)
    g.value_scale = float(raw.get("value_scale", g.value_scale))
    so = raw.get("station_order")
    g.station_order = list(so) if so else None
    g.sensor_meta_csv = _opt_path(raw.get("sensor_meta_csv"))

    g.stations_csv = _opt_path(raw.get("stations_csv"))
    g.datum = str(raw.get("datum", g.datum))
    yr = raw.get("year")
    g.year = None if yr in (None, "", "null") else int(yr)
    g.start = _opt_date(raw.get("start"))
    g.end = _opt_date(raw.get("end"))
    g.min_obs_coverage_pct = float(
        raw.get("min_obs_coverage_pct", g.min_obs_coverage_pct)
    )
    g.cache_dir = Path(raw.get("cache_dir", g.cache_dir))
    g.vdatum_csv = _opt_path(raw.get("vdatum_csv"))
    g.vdatum_idw_fill = bool(raw.get("vdatum_idw_fill", g.vdatum_idw_fill))
    g.vdatum_idw_power = float(raw.get("vdatum_idw_power", g.vdatum_idw_power))
    idk = raw.get("vdatum_idw_k")
    g.vdatum_idw_k = None if idk in (None, "", "null") else int(idk)
    g.hilo_interp = str(raw.get("hilo_interp", g.hilo_interp))
    return g


def _parse_simulation(raw: dict) -> SimulationConfig:
    s = SimulationConfig()
    s.timestep_min = int(raw.get("timestep_min", s.timestep_min))
    s.gauge_interp = str(raw.get("gauge_interp", s.gauge_interp))
    s.surface_update = str(raw.get("surface_update", s.surface_update))
    s.gap_policy = str(raw.get("gap_policy", s.gap_policy))
    s.constrain_daylight = bool(raw.get("constrain_daylight", s.constrain_daylight))

    # Accept the patch-001 boolean spelling so older configs keep working.
    if "daylight_bounds_exclusive" in raw:
        s.daylight_bounds = ("exclusive" if raw["daylight_bounds_exclusive"]
                             else "inclusive")
    s.daylight_bounds = str(raw.get("daylight_bounds", s.daylight_bounds))

    s.daylight_table = _opt_path(raw.get("daylight_table"))
    s.water_level_offset_m = float(
        raw.get("water_level_offset_m", s.water_level_offset_m)
    )
    s.cell_chunk = int(raw.get("cell_chunk", s.cell_chunk))
    return s


def _parse_output(raw: dict) -> OutputConfig:
    o = OutputConfig()
    o.root = Path(raw.get("root", o.root))
    o.rasters = Path(raw.get("rasters", o.root / "rasters"))
    o.summaries = Path(raw.get("summaries", o.root / "summaries"))
    o.run_id = str(raw.get("run_id", o.run_id))
    o.write_annual = bool(raw.get("write_annual", o.write_annual))
    o.write_total = bool(raw.get("write_total", o.write_total))
    o.write_average = bool(raw.get("write_average", o.write_average))
    o.write_daily = bool(raw.get("write_daily", o.write_daily))
    o.daily_dtype = str(raw.get("daily_dtype", o.daily_dtype))
    o.daily_compress = str(raw.get("daily_compress", o.daily_compress))
    dm = raw.get("daily_metrics")
    o.daily_metrics = list(dm) if dm else None
    return o


def _parse_depth_windows(raw) -> dict:
    """
    Parse depth windows, tolerating the SLR config's habit of nesting
    `refugia_thresholds` inside the `depth_windows` block.
    """
    if not raw:
        return {}
    out = {}
    for name, bounds in raw.items():
        if name == "refugia_thresholds":
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ConfigError(
                f"depth_windows.{name} must be a two-element [low, high] list"
            )
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo > hi:
            raise ConfigError(
                f"depth_windows.{name} = [{lo}, {hi}]: low exceeds high"
            )
        if lo >= 0.0 and hi > 0.0:
            raise ConfigError(
                f"depth_windows.{name} = [{lo}, {hi}] lies entirely above the "
                f"waterline and would only ever match dry cells. This tool uses "
                f"the C convention where POSITIVE IS DRY: depth = elevation - "
                f"water_surface. A band covering 0-{hi} m of water is "
                f"[{-hi}, {-lo}]."
            )
        out[str(name)] = (lo, hi)
    return out


def _parse_thresholds(raw, depth_windows_raw) -> list:
    """Accept thresholds at top level or nested under depth_windows."""
    if raw is None and isinstance(depth_windows_raw, dict):
        raw = depth_windows_raw.get("refugia_thresholds")
    if not raw:
        return []
    out = [float(t) for t in raw]
    bad = [t for t in out if t <= 0]
    if bad:
        raise ConfigError(
            f"refugia_thresholds must be positive water depths; got {bad}. "
            f"They name a depth of water, not an elevation."
        )
    return out
