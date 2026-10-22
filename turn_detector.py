"""
turn_detector.py
================

Ground-turn detection from ADS-B coordinates, following:

    Hoole, J., Ramachandra, S., Booker, J.D., Cooper, J.E.
    "Characterising the Variability in Taxiway Turns of Regional Airliners
    using Air Traffic Data", EUCASS 2025. DOI: 10.13009/EUCASS2025-149

TWO-SCALE ARCHITECTURE
----------------------
A ground-turn study needs two things that cannot be obtained at one sampling
scale:

  * detection  - is this a turn at all? Position noise makes consecutive-point
                 bearings unusable below ~150-200 m spacing, because the bearing
                 error is roughly atan(sigma / spacing) and swamps the 10 deg
                 retention threshold.
  * severity   - how tight was it? A tight turn is ~35 m of arc at 90 deg.
                 Any decimated track has a turn arc of at least 2 x the
                 decimation spacing, so severity is capped at about
                 180 / (2 * spacing) and can never reach the 2.6 deg/m
                 tight-turn threshold. At 200 m the cap is 0.45 deg/m.

So the two are computed at different scales:

  detect_turns(df, spatial_threshold_m=200)
      Detects on a decimated track (200 m) - robust against noise.
      For each detected turn, measures severity on the RAW track inside the
      turn's window, using a noise-robust heading series (a short baseline,
      not consecutive points) so the fine-scale measurement does not simply
      reintroduce the noise problem the decimation was solving.

  detect_turns_multipoint(df, baseline=2)
      Single-scale alternative. Kept for comparison; a short baseline is not
      enough to suppress the noise at taxi update rates.

  main()
      Real-data driver. Reads usable_flights.csv, filters to the ground phase,
      splits tracks at position glitches, runs the two-scale detector, and
      writes detected_turns.csv.

Input  : pandas.DataFrame with columns  icao24, timestamp, lat, lon
         (main() additionally uses altitude, flight_id)
Output : pandas.DataFrame with columns  icao24, turn_id, heading_change,
         severity, is_tight

Pipeline
--------
1. Smoothing      - centred rolling median (window=5) on lat/lon, per aircraft.
2. Bearing        - great-circle bearing; signed bearing change (delta) between
                    consecutive segments, wrapped to [-180, 180) so that
                    355 -> 5 is +10 deg, not -350.
3. Detection      - consecutive same-sign deltas form a candidate; retained if
                    |total heading change| exceeds 10 deg.
4. Severity       - |heading change| / path length. Path length is the distance
                    travelled over the sub-window where the heading actually
                    changed, measured on the RAW track - not the chord between
                    the endpoints (which inflates severity on paths that curve
                    back on themselves) and not the decimated span (which caps
                    severity below the tight-turn threshold).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0

# --- Thresholds (Hoole et al. 2025 / 2026) -----------------------------------
SMOOTHING_WINDOW = 5              # rolling median window, in points
MIN_HEADING_CHANGE_DEG = 10.0     # a candidate turn must exceed this
TIGHT_TURN_DEG_PER_M = 2.6        # regional-aircraft tight-turn threshold

# --- Real-data driver settings -------------------------------------------------
GROUND_ALT_FT = 1000.0            # ground-phase filter, Mode-S feet
GLITCH_STEP_M = 500.0             # split the track where a step exceeds this
SPATIAL_THRESHOLD_M = 200.0       # decimation spacing (detection scale)
MIN_SEGMENT_POINTS = 3            # drop split fragments shorter than this

# --- Two-scale settings ---------------------------------------------------------
# Baseline half-width, in raw fixes, used to estimate heading on the raw track.
# A baseline of 2 spans 4 raw fixes; at the ~26 m ground spacing in this dataset
# that is ~100 m, which is long enough to beat the 5 m position noise but short
# enough to resolve a 35 m tight turn's arc.
SEVERITY_BASELINE = 2
SEVERITY_MIN_SPAN_M = 25.0        # a turn's arc must cover at least this

# --- Detector-specific defaults ------------------------------------------------
DEFAULT_SPATIAL_THRESHOLD_M = 200.0
DEFAULT_BASELINE = 2

REQUIRED_COLUMNS = {"icao24", "timestamp", "lat", "lon"}
OUTPUT_COLUMNS = ["icao24", "turn_id", "heading_change", "severity", "is_tight"]
FLIGHT_COLUMNS = ["flight_id", "turn_id", "heading_change", "severity", "is_tight"]

FLIGHTS_CSV = r"C:\Users\razee\AppData\Local\Cline\usable_flights.csv"
TURNS_CSV = r"C:\Users\razee\AppData\Local\Cline\detected_turns.csv"

# A taxiing aircraft does not exceed ~20 m/s; 30 m/s is already implausible.
MAX_PLAUSIBLE_SPEED_MS = 30.0
# 0.5 deg is ~55 km - far more than one ground movement at one airport.
MAX_PLAUSIBLE_SPAN_DEG = 0.5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from (lat1, lon1) to (lat2, lon2), [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in metres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def delta_bearing(b1: float, b2: float) -> float:
    """Signed bearing change b2 - b1, wrapped to [-180, 180).

    Positive = right turn, negative = left turn. 355 -> 5 gives +10, not -350.
    """
    diff = (b2 - b1 + 180.0) % 360.0 - 180.0
    return 180.0 if diff == -180.0 else float(diff)


# ---------------------------------------------------------------------------
# Step 1 - smoothing
# ---------------------------------------------------------------------------
def smooth_trajectory(df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    """Centred rolling-median filter on lat/lon, applied per aircraft."""
    out = df.copy()

    def _median(s: pd.Series) -> pd.Series:
        return s.rolling(window, center=True, min_periods=1).median()

    out["lat"] = out.groupby("icao24", sort=False)["lat"].transform(_median)
    out["lon"] = out.groupby("icao24", sort=False)["lon"].transform(_median)
    return out


def add_step_m(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'step_m': the distance from each fix to the next, per aircraft."""
    g = df.copy()
    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)
    n = len(g)
    step = np.full(n, np.nan)
    for i in range(n - 1):
        step[i] = haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
    g["step_m"] = step
    return g


# ---------------------------------------------------------------------------
# Remediation 1 - spatial decimation (detection scale)
# ---------------------------------------------------------------------------
def decimate_by_distance(df: pd.DataFrame, threshold_m: float) -> pd.DataFrame:
    """Keep a fix only once the aircraft has travelled >= threshold_m from the
    previous kept fix. Greedy forward pass, per aircraft."""
    kept = []
    for _, g in df.groupby("icao24", sort=False):
        g = g.reset_index(drop=True)
        if g.empty:
            continue
        keep = [0]
        last_lat, last_lon = g.at[0, "lat"], g.at[0, "lon"]
        for i in range(1, len(g)):
            if haversine(last_lat, last_lon, g.at[i, "lat"], g.at[i, "lon"]) >= threshold_m:
                keep.append(i)
                last_lat, last_lon = g.at[i, "lat"], g.at[i, "lon"]
        kept.append(g.iloc[keep])

    if not kept:
        return df.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


# ---------------------------------------------------------------------------
# Remediation 2 - glitch splitting
# ---------------------------------------------------------------------------
def split_at_glitches(df: pd.DataFrame, max_step_m: float = GLITCH_STEP_M,
                      min_points: int = MIN_SEGMENT_POINTS) -> pd.DataFrame:
    """Break each track wherever the step to the next fix exceeds max_step_m."""
    out = []
    for _, g in df.groupby("icao24", sort=False):
        g = g.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if len(g) < min_points:
            continue
        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        step = np.array([haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
                         for i in range(len(g) - 1)] + [0.0])
        seg = (step > max_step_m).cumsum()
        for _, s in g.assign(_seg=seg).groupby("_seg", sort=False):
            if len(s) >= min_points:
                out.append(s.drop(columns="_seg"))

    if not out:
        return df.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 2 - bearings
# ---------------------------------------------------------------------------
def compute_bearings(group: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-point bearings, plus 'step_m' for path-length severity."""
    g = group.reset_index(drop=True).copy()
    n = len(g)
    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)

    bearing = np.full(n, np.nan)
    step = np.full(n, np.nan)
    for i in range(n - 1):
        bearing[i] = calculate_bearing(lat[i], lon[i], lat[i + 1], lon[i + 1])
        step[i] = haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])

    delta = np.full(n, np.nan)
    for i in range(1, n - 1):
        delta[i] = delta_bearing(bearing[i - 1], bearing[i])

    g["bearing"] = bearing
    g["delta_bearing"] = delta
    g["step_m"] = step
    return g


def compute_bearings_multipoint(group: pd.DataFrame, baseline: int) -> pd.DataFrame:
    """Baseline bearings: bearing of the leg P[i-baseline] -> P[i+baseline]."""
    if baseline < 1:
        raise ValueError("baseline must be >= 1")

    g = group.reset_index(drop=True).copy()
    n = len(g)
    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)

    bearing = np.full(n, np.nan)
    step = np.full(n, np.nan)
    for i in range(n - 1):
        step[i] = haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
    for i in range(baseline, n - baseline):
        bearing[i] = calculate_bearing(
            lat[i - baseline], lon[i - baseline], lat[i + baseline], lon[i + baseline]
        )

    delta = np.full(n, np.nan)
    for i in range(baseline + 1, n - baseline):
        delta[i] = delta_bearing(bearing[i - 1], bearing[i])

    g["bearing"] = bearing
    g["delta_bearing"] = delta
    g["step_m"] = step
    return g


# ---------------------------------------------------------------------------
# Two-scale severity
# ---------------------------------------------------------------------------
def raw_scale_turn(raw: pd.DataFrame, t0, t1,
                   baseline: int = SEVERITY_BASELINE) -> tuple[float, float]:
    """Measure a detected turn on the RAW track.

    Returns (heading_change_deg, path_length_m) or (nan, nan) if the window
    does not hold enough fixes.

    The heading series is estimated over a short baseline (P[i-b] -> P[i+b])
    rather than between consecutive points, so the 5 m position noise does not
    simply reappear at fine scale - that is the whole reason the coarse scale
    existed. The window is then trimmed to the sub-span over which the heading
    actually moves, which is the arc the severity should be normalised by.
    """
    win = raw[(raw["timestamp"].to_numpy() >= t0)
              & (raw["timestamp"].to_numpy() <= t1)].reset_index(drop=True)
    n = len(win)
    if n < 2 * baseline + 2:
        return np.nan, np.nan

    lat = win["lat"].to_numpy(dtype=float)
    lon = win["lon"].to_numpy(dtype=float)

    heading = np.full(n, np.nan)
    for i in range(baseline, n - baseline):
        heading[i] = calculate_bearing(lat[i - baseline], lon[i - baseline],
                                       lat[i + baseline], lon[i + baseline])

    # unwrap so a turn crossing north (355 -> 5) reads as one continuous change
    idx = np.where(~np.isnan(heading))[0]
    if len(idx) < 3:
        return np.nan, np.nan
    unwrapped = np.unwrap(np.radians(heading[idx]))
    m = len(unwrapped)
    diff = np.diff(unwrapped)

    # collect the maximal monotone runs and keep the one with the largest
    # heading change - that run is the turn. A run must end at the last fix
    # where the heading actually moved, not at the end of the window, or the
    # flat straight legs after the turn get folded into the arc.
    runs = []
    cur_a, cur_sign, last_k = None, 0.0, None
    for k in range(len(diff)):
        d = diff[k]
        if abs(d) < 1e-12:
            continue
        s = 1.0 if d > 0 else -1.0
        if cur_sign == 0.0:
            cur_sign, cur_a = s, k
        elif s != cur_sign:
            runs.append((cur_a, k + 1))
            cur_sign, cur_a = s, k
        last_k = k
    if cur_a is not None:
        runs.append((cur_a, last_k + 1))

    a_i, b_i = max(runs, key=lambda r: abs(unwrapped[r[1]] - unwrapped[r[0]]))
    change = float(abs(np.degrees(unwrapped[b_i] - unwrapped[a_i])))
    a, b = int(idx[a_i]), int(idx[b_i])

    # path length over that sub-span
    step = np.full(n, np.nan)
    for i in range(n - 1):
        step[i] = haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
    path = float(np.nansum(step[a:b]))

    return change, path


# ---------------------------------------------------------------------------
# Steps 3 and 4 - turn grouping and severity
# ---------------------------------------------------------------------------
def _extract_turns(g: pd.DataFrame, icao: str,
                   raw: pd.DataFrame | None = None,
                   min_heading_change_deg: float = MIN_HEADING_CHANGE_DEG,
                   tight_deg_per_m: float = TIGHT_TURN_DEG_PER_M) -> list[dict]:
    """Group same-sign delta_bearing runs into candidate turns and score them.

    raw : un-decimated track. When given, heading change and path length are
    re-measured on the raw fixes inside each candidate's window (two-scale);
    otherwise they are taken from the detection track itself.
    """
    n = len(g)
    direction = np.sign(g["delta_bearing"].to_numpy(dtype=float))
    delta = g["delta_bearing"].to_numpy(dtype=float)
    step = g["step_m"].to_numpy(dtype=float) if "step_m" in g else np.full(n, np.nan)
    ts = g["timestamp"].to_numpy() if "timestamp" in g else None

    results = []
    i = 0
    turn_id = 0

    while i < n:
        if np.isnan(direction[i]):
            i += 1
            continue

        sign = direction[i]
        start = i
        changes = []

        while i < n and not np.isnan(direction[i]) and direction[i] == sign:
            if not np.isnan(delta[i]):
                changes.append(delta[i])
            i += 1

        end = i - 1
        coarse_change = float(np.sum(changes))

        if abs(coarse_change) > min_heading_change_deg:
            turn_id += 1

            if raw is not None and ts is not None:
                fine_change, path = raw_scale_turn(raw, ts[max(start - 1, 0)],
                                                   ts[min(end + 1, n - 1)])
                if np.isnan(fine_change) or not path or path <= 0:
                    # fall back to the coarse measure rather than dropping a
                    # turn that the coarse scale did detect
                    lo, hi = max(start - 1, 0), min(end + 1, n - 1)
                    fine_change = coarse_change
                    path = float(np.nansum(step[lo:hi]))
            else:
                lo, hi = max(start - 1, 0), min(end + 1, n - 1)
                fine_change = coarse_change
                path = float(np.nansum(step[lo:hi]))

            severity = abs(fine_change) / path if path > 0 else 0.0

            results.append({
                "icao24": icao,
                "turn_id": turn_id,
                "heading_change": round(fine_change, 4),
                "severity": round(severity, 6),
                "is_tight": bool(severity > tight_deg_per_m),
            })

    return results


def _collect(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    results = []
    for icao, g in df.groupby("icao24", sort=False):
        results.extend(_extract_turns(g, icao, raw=raw))
    return pd.DataFrame(results, columns=OUTPUT_COLUMNS).reset_index(drop=True)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["icao24", "timestamp"], kind="stable").reset_index(drop=True)
    return add_step_m(smooth_trajectory(df))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def detect_turns(df: pd.DataFrame, spatial_threshold_m: float | None = None) -> pd.DataFrame:
    """Two-scale turn detection.

    Detection runs on a decimated track so position noise cannot fabricate
    turns; severity is then re-measured on the raw fixes inside each detected
    turn's window, so a ~35 m tight turn is not diluted by the decimation.

    spatial_threshold_m : float, optional
        Decimation spacing for detection. None detects on the raw track, which
        over-segments at taxi update rates.
    """
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = _prepare(df)
    track = decimate_by_distance(raw, spatial_threshold_m) if spatial_threshold_m else raw

    track = pd.concat(
        [compute_bearings(g) for _, g in track.groupby("icao24", sort=False)],
        ignore_index=True,
    )
    return _collect(track, raw=raw)


def detect_turns_multipoint(df: pd.DataFrame, baseline: int = DEFAULT_BASELINE) -> pd.DataFrame:
    """Single-scale baseline-bearing method, kept for comparison."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = _prepare(df)
    df = pd.concat(
        [compute_bearings_multipoint(g, baseline) for _, g in df.groupby("icao24", sort=False)],
        ignore_index=True,
    )
    return _collect(df)


def detect_turns_from_adsb(df: pd.DataFrame,
                           spatial_threshold_m: float | None = None) -> pd.DataFrame:
    """Entry point for detect_turns: validate columns, then detect."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    return detect_turns(df, spatial_threshold_m=spatial_threshold_m)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def synthetic_track(spacing_m: float = 10.0, noise_m: float = 5.0,
                    seed: int = 42, n_straight: int = 100,
                    turn_segments: int = 5) -> pd.DataFrame:
    """100 points straight north, a 90-degree right turn, 100 points straight east."""
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * np.cos(np.radians(51.0))
    lat0, lon0 = 51.0, 0.0

    rng = np.random.default_rng(seed)

    east_1 = np.zeros(n_straight)
    north_1 = np.arange(n_straight) * spacing_m
    level = north_1[-1]

    radius = 2.0 * spacing_m * turn_segments / np.pi
    s = np.arange(turn_segments + 1) * spacing_m
    phi = np.pi - s / radius
    east_t = radius + radius * np.cos(phi)
    north_t = level + radius * np.sin(phi)

    east_2 = radius + np.arange(1, n_straight) * spacing_m
    north_2 = np.full(n_straight - 1, level + radius)

    east = np.concatenate([east_1, east_t[1:], east_2])
    north = np.concatenate([north_1, north_t[1:], north_2])

    east += rng.normal(0.0, noise_m, east.size)
    north += rng.normal(0.0, noise_m, north.size)

    return pd.DataFrame({
        "icao24": "TEST01",
        "timestamp": pd.date_range("2024-01-01", periods=east.size, freq="1s"),
        "lat": lat0 + north / lat_scale,
        "lon": lon0 + east / lon_scale,
    })


# ---------------------------------------------------------------------------
# Real-data driver
# ---------------------------------------------------------------------------
def report_data_quality(df: pd.DataFrame) -> None:
    """Print a short data-quality block (time-series integrity + phase mix)."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    n_rows = len(df)
    n_flights = df["flight_id"].nunique()
    n_ts = df["timestamp"].nunique()
    dup_ts = int(df.duplicated(["flight_id", "timestamp"]).sum())

    print("data quality")
    print(f"  rows / flights / unique timestamps : {n_rows:,} / {n_flights:,} / {n_ts:,}")
    print(f"  duplicate (flight_id, timestamp)   : {dup_ts:,}")

    if "altitude" in df.columns:
        alt = df["altitude"]
        print(f"  altitude (Mode-S ft) : min {alt.min():.0f}  median {alt.median():.0f}  "
              f"max {alt.max():.0f}")
        n_ground = int((alt <= GROUND_ALT_FT).sum())
        print(f"  points at or below {GROUND_ALT_FT:.0f} ft     : {n_ground:,} "
              f"({n_ground / n_rows * 100:.1f}%)")

    speeds, spans = [], []
    for _, g in df.groupby("flight_id", sort=False):
        g = g.sort_values("timestamp", kind="stable").reset_index(drop=True)
        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        d = np.array([haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
                      for i in range(len(g) - 1)])
        dt = g["timestamp"].diff().dt.total_seconds().dropna().to_numpy()
        pos = dt > 0
        if pos.any():
            speeds.append(float(np.median(d[pos] / dt[pos])))
        spans.append(max(lat.max() - lat.min(), lon.max() - lon.min()))

    speeds = np.array(speeds)
    spans = np.array(spans)
    n_fast = int((speeds > MAX_PLAUSIBLE_SPEED_MS).sum())
    n_wide = int((spans > MAX_PLAUSIBLE_SPAN_DEG).sum())

    print(f"  median implied speed : {np.median(speeds):,.0f} m/s "
          f"(min {speeds.min():,.0f} / max {speeds.max():,.0f})")
    print(f"  flights faster than {MAX_PLAUSIBLE_SPEED_MS:.0f} m/s      : {n_fast}/{n_flights}")
    print(f"  flights spanning > {MAX_PLAUSIBLE_SPAN_DEG} deg      : {n_wide}/{n_flights}")

    broken = []
    if n_ts < n_rows * 0.5:
        broken.append(f"timestamps are not a real time series "
                      f"({n_ts:,} distinct values for {n_rows:,} rows)")
    if dup_ts > n_rows * 0.01:
        broken.append(f"{dup_ts:,} rows share a timestamp inside one flight")

    print()
    if broken:
        print("  -> NOT usable as a time series:")
        for p in broken:
            print(f"     - {p}")
    else:
        print("  -> usable as a time series")

    if n_fast > n_flights / 2:
        print(f"  -> NOTE: {n_fast}/{n_flights} flights exceed taxi speed - these are "
              f"full flight tracks,\n"
              f"     not ground segments. The ground-phase filter in main() is what "
              f"separates them.")
    print()


def _turn_windows(track: pd.DataFrame, spatial_threshold_m: float) -> pd.DataFrame:
    """Bounding window of each retained turn on a decimated track.

    Returns one row per detected turn: icao24, turn_id, start_time, end_time,
    plus the coarse heading change and path for fallback. A candidate running
    start..end is the turn at those fixes, formed from the segments either side
    of them, so the window is taken one fix either side.
    """
    raw = _prepare(track)
    dec = decimate_by_distance(raw, spatial_threshold_m)
    dec = pd.concat(
        [compute_bearings(g) for _, g in dec.groupby("icao24", sort=False)],
        ignore_index=True,
    )

    out = []
    for icao, g in dec.groupby("icao24", sort=False):
        g = g.reset_index(drop=True)
        d = g["delta_bearing"].to_numpy(dtype=float)
        step = g["step_m"].to_numpy(dtype=float)
        n = len(g)
        i = 0
        turn_id = 0
        while i < n:
            if np.isnan(d[i]):
                i += 1
                continue
            sign = np.sign(d[i])
            start = i
            total = 0.0
            while i < n and not np.isnan(d[i]) and np.sign(d[i]) == sign:
                total += d[i]
                i += 1
            end = i - 1
            if abs(total) > MIN_HEADING_CHANGE_DEG:
                turn_id += 1
                lo, hi = max(start - 1, 0), min(end + 1, n - 1)
                out.append({
                    "icao24": icao,
                    "turn_id": turn_id,
                    "start_time": g.at[lo, "timestamp"],
                    "end_time": g.at[hi, "timestamp"],
                    "coarse_heading_change": round(total, 4),
                    "coarse_path_m": round(float(np.nansum(step[lo:hi])), 1),
                })

    return pd.DataFrame(
        out,
        columns=["icao24", "turn_id", "start_time", "end_time",
                 "coarse_heading_change", "coarse_path_m"],
    )


def two_scale_severity(df_raw: pd.DataFrame,
                       df_decimated: pd.DataFrame,
                       tight_deg_per_m: float = TIGHT_TURN_DEG_PER_M) -> pd.DataFrame:
    """Re-score turns detected on the decimated track using the raw fixes.

    Detection is coarse (200 m decimation) because consecutive-point bearings
    amplify position noise by roughly atan(sigma / spacing), which swamps the
    10 deg retention threshold at taxi update rates. Severity is fine: both the
    heading change and the path length are measured on the RAW fixes inside each
    turn's window, so a ~35 m tight turn is not diluted by the decimation spacing.

    df_decimated must carry one row per detected turn with at least icao24,
    turn_id, start_time and end_time. If it also carries coarse_heading_change
    and coarse_path_m those are used when the raw window holds too few fixes to
    measure a heading series.

    Returns columns: icao24, turn_id, heading_change, severity, is_tight.
    """
    raw = df_raw.copy()
    if not pd.api.types.is_datetime64_any_dtype(raw["timestamp"]):
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="mixed", utc=True)

    raw_by_icao = {
        icao: g.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for icao, g in raw.groupby("icao24", sort=False)
    }

    out = []
    for row in df_decimated.itertuples(index=False):
        g = raw_by_icao.get(row.icao24)
        if g is None:
            continue

        heading_change, path = raw_scale_turn(
            g, row.start_time, row.end_time, baseline=SEVERITY_BASELINE
        )

        if (not np.isfinite(heading_change) or not path or path <= 0
                or path < SEVERITY_MIN_SPAN_M):
            # fall back to the coarse measure rather than dropping a turn the
            # coarse scale did detect
            heading_change = getattr(row, "coarse_heading_change", heading_change)
            path = getattr(row, "coarse_path_m", path)

        if not np.isfinite(heading_change) or not path or path <= 0:
            continue

        severity = abs(heading_change) / path
        out.append({
            "icao24": row.icao24,
            "turn_id": row.turn_id,
            "heading_change": round(heading_change, 4),
            "severity": round(severity, 6),
            "is_tight": bool(severity > tight_deg_per_m),
        })

    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)


def main() -> None:
    """Two-scale ground-phase turn detection on the real flight CSV.

    1. load usable_flights.csv and parse timestamps
    2. keep the ground phase (altitude <= 1000 ft)
    3. split tracks at position glitches (step > 500 m)
    4. detect turns on the 200 m decimated track
    5. re-score each turn on the raw fixes (two_scale_severity)
    6. write detected_turns.csv and print the counts
    """
    df = pd.read_csv(FLIGHTS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")

    report_data_quality(df)

    ground = df[df["altitude"] <= GROUND_ALT_FT].copy()
    print(f"ground phase: {len(ground):,} of {len(df):,} fixes "
          f"({len(ground) / len(df) * 100:.1f}%), "
          f"{ground['flight_id'].nunique():,} flights\n")

    windows = []
    n_segments = n_split_out = 0

    for flight_id, g in ground.groupby("flight_id", sort=False):
        g = (g[["icao24", "timestamp", "lat", "lon"]]
             .sort_values("timestamp", kind="stable")
             .reset_index(drop=True))
        if len(g) < MIN_SEGMENT_POINTS:
            continue

        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        step = np.array([haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
                         for i in range(len(g) - 1)] + [0.0])
        seg = (step > GLITCH_STEP_M).cumsum()

        for _, s in g.assign(_seg=seg).groupby("_seg", sort=False):
            if len(s) < MIN_SEGMENT_POINTS:
                n_split_out += 1
                continue
            n_segments += 1
            seg_track = s.drop(columns="_seg")

            if not len(detect_turns(seg_track,
                                    spatial_threshold_m=SPATIAL_THRESHOLD_M)):
                continue

            windows.append(
                _turn_windows(seg_track, SPATIAL_THRESHOLD_M)
                .assign(flight_id=flight_id)
            )

    if windows:
        decimated = pd.concat(windows, ignore_index=True)
    else:
        decimated = pd.DataFrame(
            columns=["icao24", "turn_id", "start_time", "end_time",
                     "coarse_heading_change", "coarse_path_m", "flight_id"]
        )

    # re-score every detected turn on the raw fixes inside its window
    turns = two_scale_severity(ground, decimated)

    if len(turns) and "flight_id" in decimated:
        turns.insert(0, "flight_id",
                     decimated["flight_id"].to_numpy()[:len(turns)])

    turns = (turns.reindex(columns=FLIGHT_COLUMNS) if len(turns)
             else pd.DataFrame(columns=FLIGHT_COLUMNS))

    turns.to_csv(TURNS_CSV, index=False)

    n_flights = df["flight_id"].nunique()
    n_ground_flights = ground["flight_id"].nunique()
    n_with_turns = turns["flight_id"].nunique() if len(turns) else 0
    n_tight = int(turns["is_tight"].sum()) if len(turns) else 0
    n_circuit = int((turns["heading_change"].abs() > 360).sum()) if len(turns) else 0

    print("turn detection")
    print(f"  ground filter    : altitude <= {GROUND_ALT_FT:.0f} ft")
    print(f"  glitch split     : step > {GLITCH_STEP_M:.0f} m "
          f"({n_segments:,} segments kept, {n_split_out:,} fragments dropped)")
    print(f"  detection scale  : decimated at {SPATIAL_THRESHOLD_M:.0f} m")
    print(f"  severity scale   : two-scale (raw fixes, baseline "
          f"+/-{SEVERITY_BASELINE})")
    print(f"  input            : {FLIGHTS_CSV}")
    print(f"  fixes read       : {len(df):,}")
    print(f"  flights read     : {n_flights:,}")
    print(f"  ground flights   : {n_ground_flights:,}")
    print(f"  flights w/ turns : {n_with_turns:,}")
    print(f"  TURNS DETECTED   : {len(turns):,}")
    print(f"  TIGHT TURNS      : {n_tight:,}")
    if len(turns):
        print(f"  turns per flight : {len(turns) / max(n_with_turns, 1):.2f}")
        print(f"  |heading_change| : median "
              f"{turns['heading_change'].abs().median():.1f} deg, "
              f"max {turns['heading_change'].abs().max():.1f} deg")
        print(f"  severity         : median {turns['severity'].median():.3f}, "
              f"max {turns['severity'].max():.3f} deg/m "
              f"(tight threshold {TIGHT_TURN_DEG_PER_M})")
        print(f"  circuit-shaped (> 360 deg, not taxiway turns): {n_circuit:,}")
        if n_tight:
            tight = turns[turns["is_tight"]]
            print(f"  tight turns per flight : "
                  f"{len(tight) / max(tight['flight_id'].nunique(), 1):.2f}")
    print(f"  output           : {TURNS_CSV}")


# ---------------------------------------------------------------------------
# Synthetic self-test (run with `python turn_detector.py --self-test`)
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    SPACING, NOISE, SEED = 10.0, 5.0, 42   # 1 Hz taxi feed, 5 m GPS noise

    track = synthetic_track(spacing_m=SPACING, noise_m=NOISE, seed=SEED)

    print(f"Fixture: 100 pts north, 90 deg right turn, 100 pts east")
    print(f"         {len(track)} fixes, {SPACING:.0f} m spacing, "
          f"{NOISE:.0f} m noise, seed {SEED}\n")

    clean = synthetic_track(spacing_m=SPACING, noise_m=0.0, seed=SEED)
    print("Stage 1 - noiseless (logic check, expect 1 turn @ ~90 deg):")
    for label, t in [
        ("consecutive", detect_turns(clean)),
        ("spatial 200m (two-scale)", detect_turns(clean, spatial_threshold_m=200.0)),
        ("multipoint b=2", detect_turns_multipoint(clean, baseline=2)),
    ]:
        print(f"  {label:<24} -> {len(t)} turn(s)  "
              f"hdg={t['heading_change'].round(1).tolist()}  "
              f"sev={t['severity'].round(2).tolist()}")
    print()

    print(f"Stage 2 - {NOISE:.0f} m noise, {SPACING:.0f} m spacing (1 Hz taxi):")
    rows = [
        ("consecutive (published)", detect_turns(track)),
        ("spatial >=200 m", detect_turns(track, spatial_threshold_m=200.0)),
        ("multipoint b=2", detect_turns_multipoint(track, baseline=2)),
    ]
    print()
    print(f"  {'detector':<24} {'turns':>5} {'max|hdg|':>9} {'tight':>5}  verdict")
    print("  " + "-" * 62)
    for label, t in rows:
        n = len(t)
        mx = f"{t['heading_change'].abs().max():.1f}" if n else "-"
        tk = str(int(t["is_tight"].sum())) if n else "-"
        verdict = "correct" if n == 1 else f"{n} spurious"
        print(f"  {label:<24} {n:>5} {mx:>9} {tk:>5}  {verdict}")

    print("\nStage 3 - spatial threshold sweep (5 m noise, 10 m spacing):")
    print(f"  {'threshold_m':>11} {'turns':>5} {'tight':>5} {'max sev':>9}")
    for th in [50, 100, 150, 200, 250, 300]:
        t = detect_turns(track, spatial_threshold_m=float(th))
        mx = f"{t['severity'].max():.2f}" if len(t) else "-"
        print(f"  {th:>11} {len(t):>5} {int(t['is_tight'].sum()) if len(t) else 0:>5} {mx:>9}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
    else:
        main()
