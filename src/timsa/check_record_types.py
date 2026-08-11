"""
Classify each station in a TiMSA config as 6-min observations vs hi/lo
predictions, without running the simulation. Also warms the CO-OPS cache.

    python check_record_types.py configs/example_present_day.yaml
"""
import sys
from pathlib import Path

from timsa.config import RunConfig
from timsa.ingest import stations as st
from timsa.ingest.coops import fetch_station_record


def main(cfg_path: str) -> int:
    cfg = RunConfig.from_yaml(cfg_path)
    g = cfg.gauges

    if g.stations_csv and Path(g.stations_csv).exists():
        table = st.load_stations(g.stations_csv)
    else:
        table = st.discover_stations(cfg.domain, cache_dir=Path(g.cache_dir).parent)

    rows, n_obs, n_hilo, n_err = [], 0, 0, 0
    for _, r in table.iterrows():
        sid = str(r["StationID"])
        try:
            df, rec_type = fetch_station_record(
                sid, g.year, datum=g.datum, cache_dir=g.cache_dir,
                min_obs_coverage_pct=g.min_obs_coverage_pct,
            )
            n = len(df)
            if rec_type == "obs_6min":
                n_obs += 1
            else:
                n_hilo += 1
        except Exception as e:
            rec_type, n = f"ERROR ({e})", 0
            n_err += 1
        rows.append((sid, r.get("Name", ""), rec_type, n))

    w = max(len(x[1]) for x in rows) if rows else 4
    print(f"\n{'StationID':<10} {'Name':<{w}} {'record_type':<12} {'n':>7}")
    print("-" * (10 + w + 12 + 9))
    for sid, name, rec, n in rows:
        print(f"{sid:<10} {name:<{w}} {rec:<12} {n:>7}")
    print(f"\nobs_6min: {n_obs}   pred_hilo: {n_hilo}   errors: {n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else
                          "configs/example_present_day.yaml"))
