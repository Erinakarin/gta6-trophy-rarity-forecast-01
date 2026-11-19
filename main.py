"""GTA VI trophy rarity forecast -- make.fun gta6-trophy-rarity-forecast-01.

Forecasts each PlayStation 5 trophy's Exophase earn rate at the case's
target. Python 3.13, standard library only, no network.

Approach (v1, safe and valid first):
- Read every *.in case in $MAKEFUN_INPUT_DIR (else /input), write one
  <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output). One JSON object
  with exactly one key, "trophies": one row per trophy, {id, q: [q05, q25,
  q50, q75, q95]}, each 0-100 and non-decreasing.
- Deduplicate the history to real refresh points: Exophase refreshes on
  its own schedule, so the reads between two refreshes are byte-identical
  and carry no signal. Only the points where values actually changed are
  kept; fitting a line to every read would be fitting noise.
- Cross-trophy structure: a refresh moves almost every trophy's logit in
  the same direction (dilution as the tracked player base widens, partly
  offset by players progressing). So each trophy's total logit drift over
  the observed window is shrunk toward its tier-group mean, and the drift
  is projected forward with a saturating factor -- a 4.5-day horizon does
  not extrapolate linearly, because player-base growth and progress both
  slow down.
- Spread is wide and grows with the horizon: a random-walk term plus a
  refresh-jump term estimated from the observed refreshes, with a floor so
  an unseen refresh cannot fall outside the band. Pinball loss punishes a
  too-narrow spread harder than a slightly-wide one, and the 4.5-day
  horizon cannot be backtested (only ~2 days of data), so the calibration
  is conservative on purpose.
- Safety: everything is wrapped in try/except. On any failure a case falls
  back to its last-read rates with a wide spread, so the output is always
  valid. A missing or invalid output, or a run over 8 seconds, cannot win.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, and the standard normal's quantile
# at each (Acklam's approximation, accurate to ~1e-9, so any level works).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Rates are clamped to this band before the logit: the page prints 2
# decimals, and 0% and 100% have no logit.
LO, HI = 0.01, 99.99

# Spread model constants (logit units).
SIGMA_RW = 0.5        # random-walk spread per square-root day (house baseline)
JUMP_DEFAULT = 0.7    # assumed logit jump per refresh when none observed
SIGMA_FLOOR = 1.0     # never narrower than this: a refresh the history has
                      # not seen yet must still fit inside the band. The
                      # launch-night refresh moved almost every rate down
                      # by more than a logit, so a floor of 1.0 is the
                      # minimum honest width for a short horizon.
SIGMA_CAP = 3.0       # never wider than this
# A trophy essentially everyone has (last rate >= 99.9%) stays there: the
# house's own data shows the 100% trophy does not move. Give it a narrow
# spread instead of the normal one, or the wide band would pay pinball
# loss on every level for a rate that never moves.
NEAR_FULL = 99.9
SIGMA_NEAR_FULL = 0.15


def clamp(x):
    return min(HI, max(LO, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s):
    """Parse an ISO 8601 UTC timestamp to a timezone-aware datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def dedupe_history(history, n_trophies):
    """Keep only the reads where values actually changed (real refreshes).

    Returns a list of (time, rates) with rates as a list of floats, nulls
    carried forward from the previous kept read. Identical consecutive
    reads are dropped.
    """
    points = []
    last_rates = None
    for read in history or []:
        raw = read.get("rates") or []
        rates = []
        for i in range(n_trophies):
            v = raw[i] if i < len(raw) else None
            if v is None:
                v = last_rates[i] if last_rates is not None else 0.0
            rates.append(float(v))
        if last_rates is not None:
            changed = any(abs(a - b) > 0.005 for a, b in zip(rates, last_rates))
            players = read.get("players_tracked")
            prev_players = points[-1][2] if points else None
            if not changed and players == prev_players:
                continue
        points.append((parse_time(read.get("at")), rates, read.get("players_tracked")))
        last_rates = rates
    return points


def forecast_case(case):
    """Forecast one case. Returns {"trophies": [{"id": n, "q": [...]}, ...]}."""
    trophies = case["trophies"]
    n = len(trophies)
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(p) for p in levels]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0

    points = dedupe_history(case.get("history"), n)
    if not points:
        raise ValueError("no history")
    last_rates = points[-1][1]
    t_first = points[0][0]
    t_last = points[-1][0]
    t_target = parse_time(case.get("target"))

    # Observed window (hours) and horizon from the last read (hours).
    window_h = 0.0
    if t_first and t_last:
        window_h = max(0.0, (t_last - t_first).total_seconds() / 3600.0)
    horizon_h = horizon_days * 24.0
    if t_last and t_target:
        horizon_h = max(horizon_h, (t_target - t_last).total_seconds() / 3600.0)
    # Saturating projection: the future drift is a fraction of the observed
    # drift, approaching it as the horizon dwarfs the observed window.
    frac = horizon_h / (window_h + horizon_h) if (window_h + horizon_h) > 0 else 0.0

    # Logit of every trophy at every distinct point.
    logits = [[logit(r) for r in rates] for (_t, rates, _p) in points]
    n_pts = len(points)

    # Per-trophy total logit drift over the observed window, shrunk toward
    # its tier-group mean (refresh shifts are correlated across trophies).
    drift = [logits[-1][i] - logits[0][i] for i in range(n)]
    tiers = {}
    for i, tr in enumerate(trophies):
        tiers.setdefault(tr.get("tier", "?"), []).append(drift[i])
    group_mean = {}
    overall = sum(drift) / n if n else 0.0
    for tier, vals in tiers.items():
        group_mean[tier] = sum(vals) / len(vals) if len(vals) >= 3 else overall
    shrink = (n_pts - 1) / (n_pts + 2) if n_pts >= 2 else 0.0
    drift_s = []
    for i, tr in enumerate(trophies):
        gm = group_mean.get(tr.get("tier", "?"), overall)
        drift_s.append(gm + shrink * (drift[i] - gm))

    # Refresh-jump scale: the typical absolute logit move per refresh.
    if n_pts >= 2:
        jumps = []
        for j in range(n_pts - 1):
            for i in range(n):
                jumps.append(abs(logits[j + 1][i] - logits[j][i]))
        jump = median(jumps)
        interval_days = (window_h / 24.0) / max(1, n_pts - 1)
        jump_scale = jump / math.sqrt(interval_days) if interval_days > 0 else JUMP_DEFAULT
    else:
        jump_scale = JUMP_DEFAULT

    sigma_rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    sigma_jump = jump_scale * math.sqrt(max(horizon_days, 0.0))
    sigma = math.sqrt(sigma_rw * sigma_rw + sigma_jump * sigma_jump)
    # Extra uncertainty when the last read is stale: if the recorder went
    # quiet before the checkpoint, the rates have already had time to move
    # unseen. (The checkpoint is case["at"]; the last read should sit just
    # before it.)
    t_checkpoint = parse_time(case.get("at"))
    if t_last and t_checkpoint:
        stale_h = max(0.0, (t_checkpoint - t_last).total_seconds() / 3600.0)
        if stale_h > 1.0:
            sigma = math.sqrt(sigma * sigma + 0.01 * stale_h)
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    rows = []
    for i, tr in enumerate(trophies):
        if last_rates[i] >= NEAR_FULL:
            # A trophy essentially everyone has does not move: pin it at
            # the top with a narrow spread, and do not let the group drift
            # (which pulls every other trophy down) drag it with it.
            med = logit(HI)
            sig_i = SIGMA_NEAR_FULL
        else:
            med = logits[-1][i] + drift_s[i] * frac
            sig_i = sigma
        q = [round4(clamp(sigmoid(med + zj * sig_i))) for zj in z]
        # Enforce non-decreasing (rounding can only tie, never invert, but
        # keep the guarantee explicit).
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}


def fallback_forecast(case):
    """Last-read rates with a wide spread. Always valid, never crashes."""
    trophies = case["trophies"]
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(p) for p in levels]
    history = case.get("history") or []
    last = history[-1].get("rates") if history else []
    rows = []
    for i, tr in enumerate(trophies):
        r = last[i] if i < len(last) and last[i] is not None else 0.0
        med = logit(r)
        q = [round4(clamp(sigmoid(med + zj * 1.5))) for zj in z]
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            try:
                forecast = forecast_case(case)
            except Exception:
                forecast = fallback_forecast(case)
        except Exception:
            # Unreadable case file: write a minimal valid forecast if we can
            # still name its trophies, otherwise skip the file.
            continue
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
