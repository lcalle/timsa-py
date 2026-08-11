# TiMSA

**T**idal **I**nundation **M**odel of **S**hallow-water **A**vailability.

## What it is

Coastal habitat isn't just *where* the water is shallow — it's *how long* a
place stays at a usable depth as the tide moves through it. A mudflat a wading
bird can forage on for six hours a day is worth far more than one of the same
size exposed for only one. TiMSA measures that difference.

Give it two things — the shape of the seabed (a DEM) and a record of how water
level rises and falls over a year (from a tide gauge or your own sensor) — and
for every grid cell it computes how much *time* that cell spends within a depth
of water you care about. The output is a set of maps: hours per year in a
"shallow band," hours spent as shallow refugia under a threshold depth, and so
on. The premise comes from
[Calle et al. 2018](https://doi.org/10.1002/ecm.1305): for an intertidal
forager, habitat is an area available for a *duration*, and two sites of
identical extent can differ several-fold in the hours they're actually usable.

Under the hood it's a "bathtub" inundation model — it raises and lowers a flat
water surface over the terrain and counts time — deliberately simple, fast, and
reproducible. It reads NOAA CO-OPS tide data and CUDEM bathymetry for U.S.
coasts automatically, or runs entirely on data you supply. It's a Python port
of the C reference at [lcalle/timsa](https://github.com/lcalle/timsa),
preserving that model's algorithm and sign convention.

**Who it's for:** coastal ecologists, habitat and restoration planners, and
anyone asking not "is this underwater?" but "how long is this the right depth?"

## Install

```bash
pip install -e ".[all]"     # or ".[geo]" for raster I/O without dev tools
```

The core simulation needs only numpy, pandas, pyyaml, and requests. Raster I/O,
DEM acquisition, and vector boundaries are extras.

## Use

```bash
timsa fetch configs/example_present_day.yaml   # stations, datums, DEM, zones
timsa run   configs/example_present_day.yaml   # one simulation
```

`fetch` acquires whatever inputs a run needs and doesn't already have; `run`
executes one simulation. If you bring your own DEM and water-level data you can
skip straight to `run`. As a library:

```python
from timsa import TimsaInputs, TimsaConfig, TimsaSimulation

inputs = TimsaInputs(dem=dem, gauge_zones=zones, gauge_wdepths=wl,
                     nodata_mask=nodata, record_interval_min=6)
config = TimsaConfig(depth_windows={"shallow_band": (-1.5, 0.0)},
                     refugia_thresholds=[0.2, 0.5, 1.0], timestep_min=6)
sim = TimsaSimulation(inputs, config)
sim.run()
```

## Defining a domain

Declare a simulation extent one of three ways — a bounding box, a vector
boundary, or a center point and a radius:

```yaml
domain:
  name: my_site
  bbox: [-81.95, 24.45, -80.95, 24.83]   # [lon_min, lat_min, lon_max, lat_max]
  gauge_search_buffer_km: 40
  resolution_m: 30
  crs: null        # null -> UTM zone derived from the centroid
```

```yaml
domain:
  name: my_site
  boundary_file: aoi/my_site.geojson     # .geojson / .shp / .gpkg
```

```yaml
domain:
  name: my_site
  center: [-82.40, 27.85]                # [lon, lat]
  radius_km: 8                           # half-width of the derived bbox
  shape: box                             # box (default) | circle
```

A `center` + `radius_km` domain derives a square bounding box `2 * radius_km`
across, using the same latitude-corrected km→degree conversion as the gauge
search buffer. `shape: circle` additionally attaches a circular AOI polygon for
geometry-based clipping; the enclosing bbox still drives DEM tiling and station
discovery. Exactly one of `bbox`, `boundary_file`, or `center` may be given.

`timsa fetch` then discovers CO-OPS stations within the buffered extent,
resolves MLLW→NAVD88 offsets, downloads and mosaics the intersecting CUDEM
tiles, and builds the gauge-zone raster. Tide gauges are sparse, so the buffer
matters: a tight one around a small domain often returns no stations, and that
fails immediately with an explanation rather than surfacing later as an index
error.

## Bringing your own data

TiMSA can supply the DEM and the tide record itself, or accept either or both
from you — an existing DEM with fetched tide data, in-situ sensor records over
downloaded bathymetry, or a hand-picked subset of gauges. Those combinations,
and the config keys that select them, are laid out with worked examples in
[`docs/configuration.md`](docs/configuration.md).

## Two things to get right

**Sign convention: positive is dry.** `depth = elevation - water_surface`, so a
cell 0.3 m under water has depth `-0.3`. Depth windows carry negative bounds —
a band covering 0 to 1.5 m of water is `[-1.5, 0.0]`. Refugia thresholds stay
positive, because they name a depth of water rather than an elevation. Getting
this backwards produces a run that completes and reports zeros; the config
parser rejects an all-positive window for that reason.

**Timestep and record interval are independent.** `record_interval_min` is the
spacing of your water-level data; `timestep_min` is how often the simulation
evaluates. Step finer than the record and the gauge series is interpolated up
(`gauge_interp: linear | sinusoidal | hold`); step coarser and it is decimated.
`sinusoidal` reproduces `tide_wdchange.c` and is the right choice when the
records are tidal extrema rather than regular observations.

| record | timestep | behaviour |
|---|---|---|
| 360 min (hi/lo) | 1 min | interpolated up |
| 6 min (CO-OPS obs) | 1 min | interpolated up |
| 6 min | 6 min | used directly |
| 6 min | 30 min | decimated |

## Metrics

Per cell, per depth window and threshold:

- **Area availability** — binary; the cell entered the window at least once.
  The traditional metric, kept as a comparator.
- **Time-integrated availability** — minutes inside the window over the run.
- **Refugia time** — minutes under at most *d* metres of water, at several
  thresholds.

Annual accumulators are written once at the end. Daily rasters are opt-in
(`output.write_daily` or `--daily-rasters`) and stream to disk from a single
preallocated buffer as float32 + LZW.

## Relationship to the other repositories

- **[lcalle/timsa](https://github.com/lcalle/timsa)** — the original C
  implementation. This port preserves its algorithm and sign convention; see
  `docs/porting_from_c.md`.
- **lcalle/timsa_SLR** — the sea-level-rise analysis pipeline supporting
  *The temporal signature of sea-level rise in shallow-water ecosystems*. It
  keeps the scenario sweep, interaction-rate model, and manuscript figures, and
  is frozen at the version of record.

## Citation

If you use this software or its outputs, cite both the software and the
publications it implements — see `CITATION.cff`. The methods come from
Calle et al. (2018), *Ecological Monographs* 88(4):600–620
(https://doi.org/10.1002/ecm.1305) and Calle et al. (2016), *The Auk*
133(3):378–396 (https://doi.org/10.1642/AUK-15-234.1). Citation is required
for all uses, including commercial ones.

## License

TiMSA is **source-available under the PolyForm Noncommercial License 1.0.0**,
not an open-source license. It is free to use, modify, and share for
**noncommercial purposes** — research, teaching, personal study, and use by
nonprofit, educational, government, and environmental-protection organizations,
regardless of funding source. It is provided **as-is, with no warranty and no
liability** for results. See `LICENSE`.

**Commercial use requires a separate license.** Selling the software, selling
its outputs, or using it in a paid product or service is not granted by the
noncommercial license and is available from CalleEcology, Inc. on a royalty
basis — see `COMMERCIAL.md` or contact leo@calleecology.com.
