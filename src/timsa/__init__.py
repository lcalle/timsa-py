"""TiMSA — Tidal Inundation Model of Shallow-water Availability."""

__version__ = "0.1.0"

from timsa.core import (
    TimsaInputs, TimsaConfig, TimsaSimulation,
    resample_gauge_records, apply_surface_update,
    synthesize_tidal_series_sinusoidal,
)
from timsa.domain import Domain, DomainError
from timsa.config import RunConfig, ConfigError
from timsa.metrics import RunResult, write_metric_rasters, write_summary
from timsa.io import load_dem, load_gauge_zones, DailyRasterWriter
from timsa.crs import utm_epsg_from_lonlat, resolve_target_crs
from timsa.solar import sunrise_sunset_table, sunrise_sunset_for_domain

__all__ = [
    "TimsaInputs", "TimsaConfig", "TimsaSimulation",
    "resample_gauge_records", "apply_surface_update",
    "synthesize_tidal_series_sinusoidal",
    "Domain", "DomainError", "RunConfig", "ConfigError",
    "RunResult", "write_metric_rasters", "write_summary",
    "load_dem", "load_gauge_zones", "DailyRasterWriter",
    "utm_epsg_from_lonlat", "resolve_target_crs",
    "sunrise_sunset_table", "sunrise_sunset_for_domain",
]
