"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Model (see STRATEGY.md): a log-ratio regression fit at runtime from the
rolling-midnight console launches among the case's comparables. For each
hour h = 2..24 the log ratio log(v_h / v_1) is regressed on log(v_1),
centred at the fit set's median log hour-1 print, so a small hour-1 print
predicts a big ramp into the US evening and a huge hour-1 print predicts
a nearly flat day. Coefficients and spreads are smoothed across
neighbouring hours, and the median is floored at the 10th percentile of
the big-launch log-ratios so extrapolation to a GTA-VI-sized hour 1
cannot fall below what big launches actually did. Quantiles are
log-normal around that median with a calibrated spread.

Python 3.13, standard library only, offline, deterministic.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Spread calibration: the residual sd is scaled by this before it becomes
# the log-normal width of the forecast (0.9 was the stable middle of
# 0.85-1.0 on the practice cases).
SPREAD = 0.9
# Floor on the residual sd: no hour's spread is trusted below this, in
# log units. Keeps a lucky-tight hour from forecasting a razor-thin band.
SD_FLOOR = 0.20
# A fit launch needs a real hour-1 audience; below this the hour-1 print
# is sparse-sampling noise with extreme leverage on the slope.
MIN_FIRST_HOUR = 5000
# "Big" launches, used for the extrapolation floor.
BIG_FIRST_HOUR = 150000
# Rolling-midnight console launches: the fit set. Practice cases leave
# the four practice launches out of their comparables, so the fit set
# there is the 20 of these that are present; the final case holds all 24.
ROLLING = (
    "Resident Evil Requiem", "Ghost of Y\u014dtei",
    "The Legend of Zelda: Tears of the Kingdom",
    "Call of Duty: Black Ops 6", "Call of Duty: Modern Warfare III",
    "Call of Duty: Black Ops 7", "Call of Duty: Modern Warfare 4",
    "FINAL FANTASY VII REBIRTH", "The Last of Us Part II",
    "Ghost of Tsushima", "Hogwarts Legacy", "Monster Hunter Wilds",
    "ELDEN RING", "Stellar Blade", "Kingdom Come: Deliverance II",
    "Borderlands 4", "Clair Obscur: Expedition 33", "Saros", "Marathon",
    "Elden Ring Nightreign",
    "Marvel's Wolverine", "Red Dead Redemption 2",
    "God of War Ragnar\u00f6k", "Marvel's Spider-Man 2",
)


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _round2(value):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    cents = math.floor(value * 100 + 0.5) / 100
    return int(cents) if cents == int(cents) else cents


def _release_day(comp):
    """Map release-day hour -> viewers for one comparable, or None.

    Only hours 1..24 of the launch's own day count, and only when the
    hour-1 print exists and at least 20 of the day's hours are present.
    """
    start = _parse_time(comp["day_start"])
    hours = {}
    for point in comp["hourly"]:
        viewers = point["viewers"]
        if viewers <= 0:
            continue
        h = int((_parse_time(point["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            hours[h] = viewers
    if 1 not in hours or len(hours) < 20:
        return None
    return hours


def _smooth3(series):
    """Three-point moving average over an hour-ordered list."""
    out = []
    for i in range(len(series)):
        window = series[max(0, i - 1):i + 2]
        out.append(sum(window) / len(window))
    return out


def _low_quantile(values, frac=0.10):
    ordered = sorted(values)
    return ordered[max(0, int(frac * (len(ordered) - 1)))]


def _fit_hour(curves, names, hour, ref):
    """OLS of log(v_h/v_1) on centred log v_1 for one hour.

    Returns (intercept, slope, residual_sd). With too few points the
    regression is meaningless, so fall back to a flat median with a
    wide spread instead of failing the case.
    """
    xs, ys = [], []
    for name in names:
        curve = curves[name]
        if hour in curve:
            xs.append(math.log(curve[1]) - ref)
            ys.append(math.log(curve[hour] / curve[1]))
    if len(xs) < 3:
        return 0.0, 0.0, 0.5
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
             if denom > 0 else 0.0)
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd = max(SD_FLOOR, statistics.pstdev(resid)) if len(resid) > 1 else 0.5
    return intercept, slope, sd


def _regression_quantiles(p1, curves):
    """Quantiles per hour from the fitted regression, or None if unusable."""
    names = [n for n in ROLLING
             if n in curves and curves[n][1] >= MIN_FIRST_HOUR]
    if len(names) < 3:
        return None
    ref = statistics.median([math.log(curves[n][1]) for n in names])
    hours = list(range(2, 25))
    inter, slope, sd = {}, {}, {}
    for h in hours:
        a, b, s = _fit_hour(curves, names, h, ref)
        inter[h], slope[h], sd[h] = a, b, s
    for table in (inter, slope, sd):
        smoothed = _smooth3([table[h] for h in hours])
        for h, value in zip(hours, smoothed):
            table[h] = value
    x = math.log(max(1, p1)) - ref
    out = {}
    for h in hours:
        big = [math.log(curves[n][h] / curves[n][1]) for n in names
               if h in curves[n] and curves[n][1] >= BIG_FIRST_HOUR]
        pool = big if len(big) >= 3 else [
            math.log(curves[n][h] / curves[n][1])
            for n in names if h in curves[n]]
        floor = _low_quantile(pool) if pool else float("-inf")
        mu = max(inter[h] + slope[h] * x, floor)
        row, prev = [], 0.0
        for z in Z:
            q = max(1.0, min(1e9, p1 * math.exp(mu + z * sd[h] * SPREAD)))
            q = max(_round2(q), prev)  # keep the row non-decreasing
            prev = q
            row.append(q)
        out[h] = row
    return out


def _persistence_quantiles(p1):
    """House baseline: every hour at the hour-1 print, widening spread.

    Fallback when the regression has no fit set; keeps the case valid.
    """
    out = {}
    for h in range(2, 25):
        row, prev = [], 0.0
        for z in Z:
            q = _round2(max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1))))
            q = max(q, prev)
            prev = q
            row.append(q)
        out[h] = row
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = {}
    for comp in case.get("comparables", []):
        curve = _release_day(comp)
        if curve is not None:
            curves[comp["name"]] = curve
    quantiles = _regression_quantiles(p1, curves) or _persistence_quantiles(p1)
    return {"hours": [{"hour": row["hour"], "q": quantiles[row["hour"]]}
                     for row in case["forecast"]]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
