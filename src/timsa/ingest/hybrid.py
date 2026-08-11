"""
timsa.ingest.hybrid
===================
Merge an in-situ (prescribed) water-level record with fetched CO-OPS records
into a single gauge array, for the case where a user has their own sensors
*and* wants to supplement coverage with nearby NOAA tide gauges.

Why this is a module and not a config flag
------------------------------------------
The rest of the pipeline rests on one contract: the column order of the
water-level array is the gauge order, and the gauge-zone raster stores each
cell's gauge as ``column index + 1``. Merging two sources means producing one
array and one station table that both honour that contract, deterministically,
so that the zones built by ``fetch`` line up with the columns assembled by
``run``.

The order is fixed and documented: **in-situ gauges first** (in
``station_order``, matching the prescribed file's columns), **then the kept
CO-OPS gauges** (in the manifest's ``gauge_idx`` order, i.e. sorted by
StationID). Both ``fetch`` (for zones) and ``run`` (for values) call
``load_hybrid_wl`` and get the same ordering, because both derive it from the
same cached data.

Vertical datums
---------------
CO-OPS records are shifted MLLW->NAVD88 exactly as in the pure-CO-OPS path.
In-situ values are used as-is: it is the user's responsibility to supply sensor
data already referenced to the DEM's vertical datum (typically NAVD88). No
offset is applied to the in-situ columns.

Alignment
---------
The two sources must cover the same period at the same record interval so the
arrays concatenate. The target interval is the in-situ record's interval; the
CO-OPS side is fetched at that interval. If the two record lengths differ,
that is raised loudly rather than silently trimmed — align ``gauges.year`` /
``start`` / ``end`` with the span of the sensor file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd


class HybridError(ValueError):
    """Raised when in-situ and CO-OPS records cannot be merged."""


@dataclass
class HybridWL:
    """Assembled hybrid water-level record and its gauge table."""

    values: np.ndarray                 # (n_records, n_insitu + n_coops)
    record_interval_min: int
    station_table: "pd.DataFrame"      # gauge_idx, StationID, Latitude, Longitude, gauge_source
    n_insitu: int
    n_coops: int

    @property
    def n_records(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_gauges(self) -> int:
        return int(self.values.shape[1])

    def describe(self) -> str:
        return (f"{self.n_insitu} in-situ + {self.n_coops} CO-OPS gauge(s), "
                f"{self.n_records} records @ {self.record_interval_min} min")


# ---------------------------------------------------------------------------
# Pure assembly (no I/O) — the orderable core, unit-tested directly.
# ---------------------------------------------------------------------------

def assemble_hybrid(
    insitu_ids: list,
    insitu_values: np.ndarray,
    insitu_latlon: np.ndarray,
    kept: "pd.DataFrame",
    coops_values: np.ndarray,
    record_interval_min: int,
) -> HybridWL:
    """
    Concatenate in-situ and (already datum-shifted) CO-OPS arrays into one
    record, and build the matching combined station table.

    Parameters
    ----------
    insitu_ids : list[str]
        Sensor IDs, in the prescribed file's column order.
    insitu_values : (n_records, n_insitu) array
    insitu_latlon : (n_insitu, 2) array of [lat, lon], aligned to insitu_ids.
    kept : DataFrame
        The kept CO-OPS gauges, sorted by gauge_idx, with columns
        ['station_id', 'lat', 'lon']; row j corresponds to column j of
        ``coops_values``.
    coops_values : (n_records, n_coops) array
        CO-OPS records, MLLW->NAVD88 already applied.
    record_interval_min : int
        Shared record interval of both sources.

    Returns
    -------
    HybridWL
    """
    insitu_values = np.asarray(insitu_values, dtype=np.float64)
    coops_values = np.asarray(coops_values, dtype=np.float64)

    if insitu_values.ndim != 2 or coops_values.ndim != 2:
        raise HybridError("both record arrays must be 2-D (n_records, n_gauges)")

    n_i = insitu_values.shape[0]
    n_c = coops_values.shape[0]
    if n_i != n_c:
        raise HybridError(
            f"record-length mismatch: in-situ has {n_i} records, CO-OPS has "
            f"{n_c}, at {record_interval_min}-min spacing. The two sources must "
            f"cover the same period. Align gauges.year / start / end with the "
            f"span of the sensor file (or trim the sensor file to match)."
        )

    if insitu_values.shape[1] != len(insitu_ids):
        raise HybridError(
            f"insitu_values has {insitu_values.shape[1]} columns but "
            f"{len(insitu_ids)} ids were given"
        )
    if coops_values.shape[1] != len(kept):
        raise HybridError(
            f"coops_values has {coops_values.shape[1]} columns but the kept "
            f"table has {len(kept)} rows"
        )

    insitu_latlon = np.asarray(insitu_latlon, dtype=float).reshape(-1, 2)

    values = np.hstack([insitu_values, coops_values])

    insitu_tab = pd.DataFrame({
        "StationID": [str(s) for s in insitu_ids],
        "Latitude": insitu_latlon[:, 0],
        "Longitude": insitu_latlon[:, 1],
        "gauge_source": "insitu",
    })
    coops_tab = pd.DataFrame({
        "StationID": kept["station_id"].astype(str).to_numpy(),
        "Latitude": kept["lat"].astype(float).to_numpy(),
        "Longitude": kept["lon"].astype(float).to_numpy(),
        "gauge_source": "coops",
    })
    combined = pd.concat([insitu_tab, coops_tab], ignore_index=True)
    combined.insert(0, "gauge_idx", range(len(combined)))

    return HybridWL(
        values=values,
        record_interval_min=int(record_interval_min),
        station_table=combined,
        n_insitu=len(insitu_ids),
        n_coops=len(coops_tab),
    )


# ---------------------------------------------------------------------------
# Orchestration (I/O) — loads both sources, then calls assemble_hybrid.
# ---------------------------------------------------------------------------

def load_hybrid_wl(cfg, domain, verbose: bool = True) -> HybridWL:
    """
    Load the in-situ record, fetch the CO-OPS records, and merge them.

    `cfg` is a RunConfig; `domain` is cfg.domain. Reuses the prescribed reader,
    the CO-OPS multi-gauge builder, and the vdatum resolver, so behaviour on
    each side is identical to the single-source paths.
    """
    from timsa.ingest.prescribed import load_prescribed_wl
    from timsa.ingest.coops import build_multigauge_array
    from timsa.ingest import stations as st
    from timsa.ingest import vdatum as vd

    g = cfg.gauges

    # -- in-situ side ----------------------------------------------------
    wl = load_prescribed_wl(
        g.prescribed_file,
        station_order=g.station_order,
        time_col=g.time_col,
        record_interval_min=g.record_interval_min,
        value_scale=g.value_scale,
        fill_gaps=g.fill_gaps,
        max_gap_records=g.max_gap_records,
    )
    ri = int(wl.record_interval_min)
    insitu_ids = list(wl.station_ids)
    if verbose:
        print(f"      in-situ: {wl.describe()}")

    # sensor locations, needed to place in-situ gauges in the zone raster
    if g.sensor_meta_csv is None:
        raise HybridError(
            "gauges.sensor_meta_csv is required for a hybrid run: it supplies "
            "the lon/lat of each in-situ sensor so it can be placed in the "
            "gauge-zone raster. Its StationID column must match the prescribed "
            "file's data columns."
        )
    meta = st.load_stations(g.sensor_meta_csv)
    meta["StationID"] = meta["StationID"].astype(str)
    meta = meta.set_index("StationID")
    missing = [s for s in insitu_ids if s not in meta.index]
    if missing:
        raise HybridError(
            f"sensor_meta_csv is missing rows for in-situ gauge(s) {missing}. "
            f"Every data column in the prescribed file needs a matching "
            f"StationID with Latitude/Longitude."
        )
    insitu_meta = meta.loc[insitu_ids]
    insitu_latlon = np.column_stack([
        insitu_meta["Latitude"].astype(float).to_numpy(),
        insitu_meta["Longitude"].astype(float).to_numpy(),
    ])

    # -- CO-OPS side -----------------------------------------------------
    if g.stations_csv and Path(g.stations_csv).exists():
        table = st.load_stations(g.stations_csv)
    else:
        table = st.discover_stations(domain, cache_dir=Path(g.cache_dir).parent,
                                     verbose=verbose)

    if vd.OFFSET_COLUMN not in table.columns:
        table = vd.batch_resolve_offsets(
            table, domain=domain, manual_csv=g.vdatum_csv, verbose=False,
            idw_fill=g.vdatum_idw_fill, idw_power=g.vdatum_idw_power,
            idw_k=g.vdatum_idw_k,
        )

    sim_start, sim_end = _coops_window(g)
    coops_values, manifest, ri_c = build_multigauge_array(
        table, g.year, sim_start, sim_end,
        record_interval_min=ri, datum=g.datum,
        cache_dir=g.cache_dir, hilo_interp=g.hilo_interp, verbose=verbose,
    )

    # MLLW -> NAVD88, per kept gauge, in manifest order (mirrors cli coops path)
    offsets = vd.offsets_as_dict(table)
    kept = manifest[manifest["gauge_idx"].notna()].sort_values("gauge_idx")
    shift = np.array([offsets[str(s)] for s in kept["station_id"]],
                     dtype=np.float64)
    coops_values = coops_values + shift[np.newaxis, :]

    if verbose:
        print(f"      CO-OPS: {len(kept)} kept gauge(s) @ {ri_c} min")

    return assemble_hybrid(
        insitu_ids=insitu_ids,
        insitu_values=wl.values,
        insitu_latlon=insitu_latlon,
        kept=kept.rename(columns={"station_id": "station_id"})[
            ["station_id", "lat", "lon"]
        ].reset_index(drop=True),
        coops_values=coops_values,
        record_interval_min=ri,
    )


def _coops_window(g):
    """Simulation window for the CO-OPS fetch (mirrors cli._load_water_levels)."""
    if g.start is not None:
        sim_start = dt.datetime(g.start.year, g.start.month, g.start.day,
                                tzinfo=dt.timezone.utc)
    else:
        sim_start = dt.datetime(g.year, 1, 1, tzinfo=dt.timezone.utc)
    if g.end is not None:
        end_excl = g.end + dt.timedelta(days=1)
        sim_end = dt.datetime(end_excl.year, end_excl.month, end_excl.day,
                              tzinfo=dt.timezone.utc)
    else:
        sim_end = dt.datetime(g.year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return sim_start, sim_end
