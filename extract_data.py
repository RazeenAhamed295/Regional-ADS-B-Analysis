#!/usr/bin/env python3
"""
extract_data.py -- extract ground trajectories from an adsb.lol globe_history
tar and segment them into usable flights.

Verified against the real archive (2.90 GB, 2026-10-20 dump):

  * tar is UNCOMPRESSED      -> tarfile.open(path, "r:")
  * trace members ARE gzip-compressed inside it, despite ending in .json
  * member paths look like  ./traces/<xx>/trace_full_<hex>.json
    (<xx> is a 2-char shard directory; there are many)
  * member payload is readsb trace format:
        {"icao": "3ecefc", "timestamp": 1792368000.0,
         "trace": [[dt, lat, lon, alt, gs, track, flags, vrate, extra,
                    type, alt_geom, geom_rate, ias, roll], ...]}
    - top-level "timestamp" is the dump base (absolute epoch)
    - each point's element 0 is SECONDS AFTER that base -> must be ADDED
    - altitude is FEET, may be the string "ground" or null (null -> skip)
    - icao may carry a "~" prefix for non-ICAO addresses
  * NOT every .json member is a trace: ./acas/acas.json (5.6 MB) is not one,
    so the .json extension alone is not a usable filter.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tarfile

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAR_PATH = "adsblol.tar"          # or pass a path as argv[1]
MAX_JSON_FILES = 5000
OUTPUT_CSV = "usable_flights.csv"

GROUND_ALT_FT = 5000              # ground-trajectory ceiling, feet
NEW_FLIGHT_GAP_S = 600            # a gap larger than this starts a new flight
MAX_INTERNAL_GAP_S = 30           # usable flights must not exceed this gap
MIN_POINTS = 20                   # usable flights must have at least this many

# Regional aircraft ICAO type designators (DOC 8643)
# Includes types actually present in the data + user-specified examples
REGIONAL_TYPES = {
    # Embraer E-Jets (E170/E175/E190/E195 family)
    "E170", "E75S", "E75L", "E75R", "E190", "E195", "E290",
    # Bombardier CRJ family
    "CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX",
    # ATR turboprops
    "AT42", "AT43", "AT44", "AT45", "AT46", "AT72", "AT73", "AT75", "AT76", "ATP",
    # De Havilland Canada Dash 8
    "DH8A", "DH8B", "DH8C", "DH8D",
    # Other regional types
    "F100", "F70", "SF34", "S210", "D228", "D328", "E110", "E120", "JSTE",
    # User examples (marketing names that might appear verbatim)
    "E175", "CRJ7", "CRJ9", "AT72", "AT76", "DH8D",
}

FALLBACK_PATHS = [
    r"C:\Users\razee\AppData\Local\Cline\adsblol.tar",
]


# ---------------------------------------------------------------------------
# Locate the archive
# ---------------------------------------------------------------------------
def find_tar(argv_path: str | None) -> str:
    for cand in [argv_path, TAR_PATH, *FALLBACK_PATHS]:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    raise SystemExit(
        f"error: {TAR_PATH!r} not found in {os.getcwd()!r}.\n"
        f"       Pass the path: python extract_data.py <path-to-tar>"
    )


# ---------------------------------------------------------------------------
# Streaming members
# ---------------------------------------------------------------------------
def is_trace_member(name: str) -> bool:
    """True for ./traces/<xx>/trace_full_<hex>.json (optionally .json.gz)."""
    low = name.replace("\\", "/").lower()
    if not (low.endswith(".json") or low.endswith(".json.gz")):
        return False
    if "/traces/" not in low:
        return False
    base = low.rsplit("/", 1)[-1]
    return base.startswith("trace_full_")


def iter_trace_members(tar_path: str, limit: int = MAX_JSON_FILES):
    """Yield (member_name, payload_bytes) for up to `limit` trace members."""
    for mode in ("r:", "r|", "r:*"):
        try:
            with tarfile.open(tar_path, mode) as tar:
                emitted = 0
                for member in tar:
                    if not member.isfile() or not is_trace_member(member.name):
                        continue
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    emitted += 1
                    if emitted > limit:
                        print(f"stopped after {limit} trace members")
                        return
                    yield member.name, handle.read()
            return
        except (tarfile.ReadError, tarfile.CompressionError, EOFError, OSError) as exc:
            print(f"mode {mode!r} failed: {exc}")
            continue
    raise SystemExit(f"error: {tar_path} is not a readable tar archive")


def maybe_gunzip(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def normalise_altitude(value):
    """-> altitude in feet, or None if the point must be dropped."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "ground":
            return 0.0
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def icao_from_name(name: str) -> str:
    """'.../trace_full_~2b0ffc.json' -> '~2b0ffc'."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".json.gz"):
        base = base[:-8]
    elif base.lower().endswith(".json"):
        base = base[:-5]
    prefix = "trace_full_"
    if base.lower().startswith(prefix):
        base = base[len(prefix):]
    return base.upper()


def parse_member(name: str, raw: bytes):
    """-> (icao24, aircraft_type, [point, ...]) ; raises ValueError on a bad payload."""
    data = json.loads(maybe_gunzip(raw).decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("top-level JSON is not an object")

    icao = data.get("icao") or icao_from_name(name)
    icao = str(icao).strip().upper()
    if not icao:
        raise ValueError("no usable icao24")

    # Extract top-level aircraft type code (ICAO designator)
    aircraft_type = data.get("t")

    base_ts = data.get("timestamp")
    if base_ts is None:
        raise ValueError("member has no top-level timestamp")
    base_ts = float(base_ts)

    trace = data.get("trace")
    if not isinstance(trace, list) or not trace:
        return icao, aircraft_type, []

    points = []
    for point in trace:
        rec = parse_point(point, base_ts)
        if rec is not None:
            rec["icao24"] = icao
            points.append(rec)
    return icao, aircraft_type, points


def parse_point(point, base_ts: float):
    """One trace point -> record, or None.

    readsb arrays: [dt, lat, lon, alt, gs, track, flags, vrate, extra,
                    type, alt_geom, geom_rate, ias, roll]
    """
    if isinstance(point, (list, tuple)):
        if len(point) < 4:
            return None
        dt, lat, lon, alt = point[0], point[1], point[2], point[3]
        try:
            timestamp = base_ts + float(dt)
        except (TypeError, ValueError):
            return None
    elif isinstance(point, dict):
        lat = point.get("lat")
        lon = point.get("lon")
        alt = point.get("altitude")
        raw_ts = point.get("timestamp")
        if raw_ts is None:
            raw_ts = point.get("dt", 0.0)
        try:
            raw_ts = float(raw_ts)
        except (TypeError, ValueError):
            return None
        timestamp = raw_ts if raw_ts > 1e6 else base_ts + raw_ts
    else:
        return None

    if lat is None or lon is None:
        return None
    altitude = normalise_altitude(alt)
    if altitude is None:
        return None

    return {"timestamp": timestamp, "lat": float(lat), "lon": float(lon),
            "altitude": altitude}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def segment_flights(df: pd.DataFrame) -> pd.DataFrame:
    """flight_id increments when the gap for that icao24 exceeds 600 s.

    Two distinct gaps are needed. The boundary gap (across the whole icao24)
    decides where a new flight starts; the internal gap (within a flight_id) is
    what the usability test measures. A single groupby("icao24").diff() puts the
    boundary gap on the first row of the NEW flight, which would make every
    flight following a long gap look like it has a huge internal gap.
    """
    df = df.sort_values(["icao24", "timestamp"]).reset_index(drop=True)

    boundary = df.groupby("icao24")["timestamp"].diff().dt.total_seconds()
    df["_new"] = boundary.isna() | (boundary > NEW_FLIGHT_GAP_S)
    df["_seg"] = df.groupby("icao24")["_new"].cumsum()
    df["flight_id"] = df["icao24"] + "_" + df["_seg"].astype(int).astype(str)

    df["gap"] = df.groupby("flight_id")["timestamp"].diff().dt.total_seconds()
    return df


def keep_usable(df: pd.DataFrame):
    stats = df.groupby("flight_id").agg(
        n_points=("timestamp", "size"), max_gap=("gap", "max"))
    ok = stats[(stats["max_gap"] <= MAX_INTERNAL_GAP_S)
               & (stats["n_points"] >= MIN_POINTS)]
    return df[df["flight_id"].isin(ok.index)].copy(), stats, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tar_path = find_tar(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"archive : {tar_path}")
    print(f"size    : {os.path.getsize(tar_path) / 1e9:.2f} GB")
    print(f"limit   : {MAX_JSON_FILES} trace members\n")

    rows: list[dict] = []
    seen = parsed = failed = regional_kept = regional_dropped = 0
    failures: list[str] = []
    type_distribution: dict[str, int] = {}

    for name, raw in iter_trace_members(tar_path, MAX_JSON_FILES):
        seen += 1
        try:
            _icao, aircraft_type, points = parse_member(name, raw)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            failed += 1
            if len(failures) < 5:
                failures.append(f"{name}: {exc}")
            continue

        # Filter: only keep regional aircraft types
        if aircraft_type not in REGIONAL_TYPES:
            regional_dropped += 1
            continue

        regional_kept += 1
        type_distribution[aircraft_type] = type_distribution.get(aircraft_type, 0) + 1
        parsed += 1
        rows.extend(p for p in points if p["altitude"] < GROUND_ALT_FT)

    print(f"trace members seen   : {seen}")
    print(f"members parsed       : {parsed}")
    print(f"members failed       : {failed}")
    print(f"non-regional dropped : {regional_dropped}")
    print(f"regional kept        : {regional_kept}")
    for line in failures:
        print(f"    ! {line}")

    if not rows:
        raise SystemExit("error: no ground points extracted")

    df = pd.DataFrame(rows, columns=["icao24", "timestamp", "lat", "lon", "altitude"])
    # Round to centiseconds before converting: readsb dt values carry 2 decimals,
    # and base_ts + dt otherwise leaves float noise like .779999971 in the CSV.
    df["timestamp"] = pd.to_datetime(df["timestamp"].round(2), unit="s", utc=True)
    ground_points = len(df)

    df = segment_flights(df)
    n_segments = df["flight_id"].nunique()

    usable, stats, ok = keep_usable(df)

    out_cols = ["flight_id", "icao24", "timestamp", "lat", "lon", "altitude"]
    usable = usable[out_cols].sort_values(["flight_id", "timestamp"])
    usable.to_csv(OUTPUT_CSV, index=False)

    ts_min, ts_max = usable["timestamp"].min(), usable["timestamp"].max()
    span_s = (ts_max - ts_min).total_seconds()
    span_h = span_s / 3600.0
    n_flights = usable["flight_id"].nunique()

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"ground points (< {GROUND_ALT_FT} ft) : {ground_points:>10,}")
    print(f"flight segments before filtering     : {n_segments:>10,}")
    print(f"usable flights                       : {n_flights:>10,}")
    print(f"usable points                        : {len(usable):>10,}")
    print(f"dropped segments                     : {n_segments - n_flights:>10,}")
    if n_flights:
        sizes = usable.groupby("flight_id").size()
        print(f"points per flight  min/median/max    : "
              f"{sizes.min()} / {int(sizes.median())} / {sizes.max()}")
        print(f"max internal gap across kept flights : {ok['max_gap'].max():>10.1f} s")
        print(f"largest gap that dropped a segment  : {stats['max_gap'].max():>10.1f} s")
    print("-" * 62)
    print(f"timestamp min : {ts_min}")
    print(f"timestamp max : {ts_max}")
    print(f"span          : {span_h:.2f} hours ({span_s:,.0f} s)")
    print("=" * 62)
    print(f"wrote {OUTPUT_CSV} ({len(usable):,} rows)")

    # Regional aircraft type distribution
    if type_distribution:
        print("\nREGIONAL AIRCRAFT TYPE DISTRIBUTION")
        print("=" * 62)
        total_regional = sum(type_distribution.values())
        for ac_type, count in sorted(type_distribution.items(), key=lambda x: -x[1]):
            pct = (count / total_regional) * 100
            print(f"  {ac_type:>6} : {count:>6,} ({pct:>5.1f}%)")
        print(f"  {'TOTAL':>6} : {total_regional:>6,}")
        print("=" * 62)

    if span_h < 1:
        print("\nWARNING: span under one hour -- this looks like a single "
              "slice, not hours of data.")


if __name__ == "__main__":
    main()
