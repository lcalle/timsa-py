# Configuration guide

This covers how a TiMSA config selects **where** the terrain and the
water-level record come from. For the sign convention and the
timestep/record-interval relationship, see the README; for every individual
key, see the comments in `configs/example_alafiaRiverMouth_2025.yaml`.

## How `fetch` and `run` divide the work

Two facts explain every combination below.

1. **The DEM is acquired by `fetch`, not by `run`.** `run` loads
   `domain.dem_path` if it points at an existing file. If `dem_path` is null,
   `run` falls back to **synthetic inputs** — a stand-in grid for smoke-testing
   the wiring, not a real result. So a real run always needs a DEM on disk
   first; "I have no DEM" means "run `fetch` to get one," never "omit it."

2. **`fetch` skips work you've already provided.** If `dem_path` exists, `fetch`
   does not re-download bathymetry (pass `--refresh` to force it). If
   `gauges.stations_csv` exists, `fetch` and `run` use that list instead of
   discovering stations from the domain.

Water levels are resolved at `run` time: `source: prescribed` reads your file;
`source: coops` fetches each station's record from the CO-OPS API (cached to
parquet) and shifts it MLLW→NAVD88; `source: hybrid` does both and merges them
(use case 4).

A **domain extent is always required** — even when you supply your own DEM —
because station discovery is driven by it. The one exception is when you also
supply an explicit `stations_csv`, though the `domain:` block itself is still
mandatory.

## Gauge zones, in one paragraph

When more than one gauge forces a domain, each grid cell is assigned to exactly
one gauge by an integer **gauge-zone raster** (a nearest-gauge / Voronoi
partition). The zone value is the gauge's **1-based index**, and that index is
the **column order** of the water-level array. `fetch` builds this raster from
the *station table* and writes it to `gauge_zones_path`. With a single gauge you
can omit `gauge_zones_path` entirely — the loader treats the whole domain as
zone 1.

---

## Use case 1 — you have a DEM, no water data (fetch CO-OPS)

Point `dem_path` at your GeoTIFF and let CO-OPS supply the tide record. Because
the DEM already exists, `fetch` skips the download and only resolves datums and
builds zones; `run` fetches the water levels.

```yaml
domain:
  name: mysite
  bbox: [-82.50, 27.77, -82.30, 27.95]        # still needed: drives discovery
  resolution_m: 30
  dem_path: "cache/dem/mysite_30m.tif"        # your existing DEM
  gauge_zones_path: "cache/dem/zones_mysite.tif"

gauges:
  source: coops
  year: 2025
  datum: MLLW
  stations_csv: null                          # null -> discover from domain
```

```bash
timsa fetch configs/mysite.yaml   # sees the DEM, skips it; builds zones
timsa run   configs/mysite.yaml   # fetches the CO-OPS record and simulates
```

## Use case 2 — you have water data, no DEM (multi-sensor)

Your columns are gauges. Two things need care:

- **You still need a DEM.** Set `dem_path` to a target path and run `fetch` to
  download CUDEM there (or point `dem_path` at your own tif and skip fetching
  bathymetry).
- **Mapping sensors to zones.** For more than one sensor, the zone raster's
  indices must line up with your CSV's columns. `fetch` builds zones from the
  *station table*, so supply a **`stations_csv` whose `StationID`s equal your
  CSV column headers, in the same order as `station_order`**. Datum offsets are
  irrelevant for prescribed data (your values are used directly), so build the
  zones with `--allow-missing-offsets`.

```yaml
domain:
  name: mysite
  bbox: [-82.50, 27.77, -82.30, 27.95]
  resolution_m: 30
  dem_path: "cache/dem/mysite_30m.tif"        # fetch will download here
  gauge_zones_path: "cache/dem/zones_mysite.tif"

gauges:
  source: prescribed
  prescribed_file: "data/sensors_2025.csv"    # columns: time, S1, S2, S3
  time_col: "time"                            # or set record_interval_min
  # record_interval_min: 12                   # e.g. 12-min sensors, no time col
  station_order: ["S1", "S2", "S3"]           # fixes column -> gauge index
  stations_csv: "data/sensors_meta.csv"       # StationID S1,S2,S3 + Lat/Lon
  value_scale: 1.0                            # 0.3048 to convert feet -> m
```

```bash
timsa fetch configs/mysite.yaml --allow-missing-offsets   # DEM + zones
timsa run   configs/mysite.yaml
```

**Single sensor:** drop `station_order`, `stations_csv`, and `gauge_zones_path`
— the whole domain becomes zone 1 automatically.

Mixed intervals across sensors aren't supported in one file: the reader expects
a single regular spacing. Resample your sensors onto a common interval first
(e.g. all to 6-min), then supply that.

## Use case 3 — you have a DEM and want only a subset of gauges

Discovery is bypassed whenever an explicit list is present, so *curate the
list*:

- **CO-OPS gauges** — put only the stations you want in `stations_csv`
  (`StationID, Name, Latitude, Longitude`). `fetch` and `run` use exactly those.

  ```yaml
  gauges:
    source: coops
    year: 2025
    stations_csv: "data/my_three_stations.csv"   # the subset, nothing else
  ```

- **Prescribed columns** — use `station_order` to pick and order a subset of the
  file's columns. Naming a column that isn't in the file raises; unlisted
  columns are dropped.

  ```yaml
  gauges:
    source: prescribed
    prescribed_file: "data/all_sensors.csv"      # S1..S6 present
    time_col: "time"
    station_order: ["S2", "S5"]                  # simulate with just these two
  ```

## Use case 4 — DEM + your gauge data, supplemented with CO-OPS

Set `source: hybrid`. TiMSA loads your in-situ sensors *and* fetches nearby
CO-OPS stations, then merges them into one gauge array whose columns are
partitioned across the domain by a single combined gauge-zone raster. This is
the union of use cases 1–3: your sensors provide dense local coverage, CO-OPS
fills in the rest of the extent.

A hybrid config carries **both** sides:

```yaml
domain:
  name: mysite
  bbox: [-82.50, 27.77, -82.30, 27.95]
  resolution_m: 30
  dem_path: "cache/dem/mysite_30m.tif"        # your DEM, or a fetch target
  gauge_zones_path: "cache/dem/zones_mysite.tif"

gauges:
  source: hybrid

  # --- in-situ side ---
  prescribed_file: "data/sensors_2025.csv"    # columns: time, S1, S2
  time_col: "time"                            # or record_interval_min
  station_order: ["S1", "S2"]                 # optional order/subset of columns
  sensor_meta_csv: "data/sensors_meta.csv"    # StationID S1,S2 + Latitude/Longitude
  value_scale: 1.0

  # --- CO-OPS side ---
  year: 2025
  datum: MLLW
  stations_csv: "data/coops_subset.csv"       # or null -> discover from domain
```

```bash
timsa fetch configs/mysite.yaml   # DEM (if needed) + combined zone raster
timsa run   configs/mysite.yaml   # merges both sources and simulates
```

**How the merge works — and the rules it enforces:**

- **Column order is fixed:** in-situ gauges first (in `station_order` /
  file-column order), then the kept CO-OPS gauges (sorted by StationID). The
  gauge-zone raster is built in exactly this order, so `fetch` and `run` always
  agree on which column is which cell's gauge.
- **Sensor locations are required.** `sensor_meta_csv` gives each in-situ
  sensor a lon/lat so it can be placed in the zone raster. Its `StationID`
  values must match the prescribed file's data-column names. (In pure
  `prescribed` mode this metadata goes in `stations_csv`; in hybrid,
  `stations_csv` is taken by the CO-OPS list, so sensor locations move to their
  own key.)
- **Vertical datums:** CO-OPS records are shifted MLLW→NAVD88 as usual;
  in-situ values are used as-is, so supply sensor data already in the DEM's
  datum (typically NAVD88).
- **Same period, same interval.** Both sources must cover the same span at the
  same record interval (taken from the in-situ record). If the two record
  lengths don't match, the run stops with a clear message — align
  `year` / `start` / `end` with the span of your sensor file rather than having
  data silently trimmed.
- **`--refresh` caveat:** the set of "kept" CO-OPS gauges is whichever stations
  returned usable data. If that set changes between `fetch` and `run` (e.g. you
  `--refresh` only one of them), the zone raster and the columns can drift.
  Fetch and run against the same cache.

**Single in-situ sensor + CO-OPS** still needs `sensor_meta_csv` (one row) and
`gauge_zones_path`, because the moment CO-OPS adds a second gauge the domain is
multi-gauge and needs a zone partition.

---

## Quick reference: which keys select what

| You have | `source` | `dem_path` | key gauge settings |
|---|---|---|---|
| DEM only | `coops` | your tif | `year`; `stations_csv: null` |
| Water only (1 sensor) | `prescribed` | fetch target | `prescribed_file`, `time_col` |
| Water only (N sensors) | `prescribed` | fetch target | `+ station_order`, `stations_csv`, `gauge_zones_path` |
| DEM + gauge subset (CO-OPS) | `coops` | your tif | `stations_csv` = the subset |
| DEM + gauge subset (prescribed) | `prescribed` | your tif | `station_order` = the subset |
| DEM + in-situ **and** CO-OPS | `hybrid` | your tif | `prescribed_file` + `sensor_meta_csv` + `year` (+ `stations_csv`) |
