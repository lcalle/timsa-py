# Porting notes: C reference to Python

## Preserved

- Gauge zones via a reference raster; each cell assigned to one tide gauge.
- Day-1 height adjustment sets the water surface from the gauge. **Day 1 only** —
  the surface is not re-anchored on subsequent days.
- The surface then evolves by the gauge-prescribed increment per step.
- Depth-window indicator accumulation per step.
- Optional daylight constraint from a per-day sunrise/sunset table.
- Sinusoidal interpolation between tidal extrema (`tide_wdchange.c`).
- Prescribed water-level records at their native interval
  (`iterateday_prescribewd_NADV88.c`).
- Sign convention: `depth = dem - water_surface`, positive dry.

## Changed

- Vectorized over cells; the C per-cell loop becomes NumPy broadcasting.
- Multi-threshold metrics accumulate in a single pass over the time series.
- Daily rasters are opt-in and streamed from a preallocated buffer.
- Water surface assigned directly rather than accumulated per cell. These are
  mathematically equivalent — the C increments telescope, and the surface is
  spatially uniform within a gauge zone — but direct assignment avoids drift
  over ~87,600 steps. `surface_update='incremental'` restores the C arithmetic
  for parity checks.

## Reproducing a C run

```yaml
simulation:
  surface_update: incremental
  gauge_interp: sinusoidal      # if the record is hi/lo
  gap_policy: hold              # C adds zero change across no-data
  daylight_bounds: exclusive    # C skips a step at exactly sunrise/sunset
```

Window bounds map directly: a C window of `(lowerbound, upperbound)` is used
unchanged, since both use positive-dry.

## Issues found in the C reference

**Out-of-bounds read on the final row.** In `iterateday_prescribewd_NADV88.c`:

```c
for(i = 0; i < nrows; i++) {
    ...
    if(i < nrows) {                     /* always true */
        depths_sim->data[bb] += gaugewdepths[i+1][g] - gaugewdepths[i][g];
```

`i < nrows` holds on every iteration including the last, so `gaugewdepths[i+1]`
reads one row past the allocation. Should be `i < nrows - 1`. Harmless in effect
— the value lands on a water surface never read again — but a real OOB read.

**`dayminute % dayminute == 0`** in the `save_waterdepth` block is always true
when `dayminute != 0` and undefined at 0. Generate golden-run fixtures with
`save_waterdepth == FALSE`.

**Daily raster leak.** `dayFHA = rastercopy(defaultRaster)` runs once per day
with no matching `free()`, leaking a raster per simulated day — separate from
the disk the written files consume.
