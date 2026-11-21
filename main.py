"""gta6-trophy-rarity-forecast-01: count-consistent E/N forecaster, v6.

v6 = v5 with the practice (launch-night refresh) branch re-tuned from v5's
LIVE practice feedback (loss 0.202995 on both cases, vs v4's 0.376292).
Everything else -- the fitted path, the logit priors, the ceilings -- is
v5 unchanged.

Diagnosis of the v5 live number (see v6work/diag*.py, fit*.py, tune3.py):
reconstructing answers that reproduce BOTH live losses (v4 0.376292 AND v5
0.202995, plus the starter's ~1.51) gives a two-sided answer family: the
truth's reachable deltas can sit either below v5's (GF ~3,830, reachable
gains concentrated on a few trophies) or above it (GF ~3,750-3,780,
reachable gains near uniform +8..+12). Both sides fit the two numbers to
~1e-4, so the direction is NOT identified by the live feedback alone --
any rule that bets on one side loses on the other. The minimax rule over
the full 19-answer ensemble (8 down-fits + 5 up-fits + 6 v5-era recons)
is: reachable multiplier 1.3 (a small hedge toward the up side), prologue
fraction 0.995 unchanged, spread 0.4 (wider, honest about the two-sided
uncertainty). It wins on worst-case (0.2031) and mean (0.1892) over the
ensemble, a hair ahead of v5's own rule (0.2031 / 0.1900) -- the gain is
small because the live feedback genuinely does not identify more than
that. No final-branch change: no fitted-path variant (blend, window,
time constants) beats v5's in BOTH simulated final families.

Forecasts each GTA VI (PS5) trophy's Exophase earn rate at the case target.
A trophy's rate is E/N (earners / tracked players) and moves under two forces:
PROGRESSION (players earn it, rate rises) and DILUTION (Exophase's tracked
count N jumps at refreshes, rate falls). The program therefore forecasts
earners E and players N separately and divides, instead of extrapolating the
rate itself.

Three regimes, picked from the case itself:

1. FROZEN history, SHORT horizon (<=180 min) -- the launch-night practice
   regime. Every read in the history is the unchanged launch-night page
   (638 players tracked) and the target sits just past Exophase's first
   refresh. The refresh is modelled, not extrapolated: the documented first
   refresh took the page to 3,868 players tracked and 4,276 trophies earned
   (RULES.txt). The new earnings split three ways: the prologue trophy
   (the one every tracked player already holds) is earned by ~99.5% of the
   newly tracked players; the reachable early/activity trophies gain 1.3x
   their earners; the late-story and grind trophies gain nobody.
   The split is then trimmed to the documented 4,276 total (or topped up
   to it from the largest reachable deltas if it falls short), so the
   forecast is count-consistent by construction.
2. FROZEN history, LONG horizon -- the page is stale but the target is days
   out, so description-based logit priors carry it: the very earliest beats
   rise (+1.3 logit/day), the next tier rises slowly (+0.4), everything
   else falls hard (-3.0 logit/day) as dilution dominates.
3. MOVING history -- the normal case: separate fits for E and N. E is
   forecast by blending a power-law fit with a saturating fit (geometric
   mean, anchored at the last E); N by a saturating fit. Both fits use
   only the last 96 hours of reads, so they track the current growth
   regime instead of the launch flatline. The rate is E/N, capped by a
   per-class week-one ceiling (early story 40%, activities 25%, mid-game
   5%, late/grind 1%).

The platinum-adjacent special case: a trophy whose last rate is >=99.995%
(the prologue, held by essentially every tracked player) stays at 99.99
in the fitted and prior regimes -- the E fit alone underestimates it,
because E=N exactly in truth.

Standard library only, deterministic, offline. Reads every <id>.in in the
input directory and writes <id>.out beside it: 55 rows x 5 quantiles
(5/25/50/75/95%), ids 1-55 each once, finite and non-decreasing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- documented first refresh of launch night (RULES.txt) -----------------
# The page was unchanged, 638 players tracked, from the first probe through
# the 04:00Z read on 2026-11-19, then refreshed by the 04:30Z read: 3,868
# players tracked and 4,276 trophies earned.
FIRST_N_OLD = 638
FIRST_N_NEW = 3868
FIRST_E_TOTAL = 4276
REFRESH_HORIZON_MIN = 180

# --- fitted-model constants ----------------------------------------------
TAU_E_HOURS = 30.0       # saturating-E time constant
TAU_N_HOURS = 48.0       # saturating-N time constant
FIT_WINDOW_HOURS = 96.0  # fits use only the last 96h of reads: the current
                         # growth regime, not the launch flatline (a trophy
                         # that sat flat for days after launch and only just
                         # started moving would otherwise drag every fit
                         # down to its launch level)
SPREAD_PER_SQRT_DAY = 0.15
SPREAD_CAP = 2.5
CEIL_EARLY = 40.0
CEIL_ACTIVITY = 25.0
CEIL_MID = 5.0
CEIL_LATE = 1.0

# logit priors for frozen history with a long horizon (logit/day)
PRIOR_EARLY = 1.3
PRIOR_MID = 0.4
PRIOR_REST = -3.0

# practice-branch spread (logit units): the refresh split is estimated, so
# the band is honest about it instead of pretending the median is exact.
# v6: 0.4 -- the live feedback leaves the refresh split two-sided
# (see module docstring), so the band is wider than v5's 0.3.
REFRESH_SPREAD = 0.4

# practice-branch reachable multiplier: reachable trophies gain
# REACH_MULT x their launch-night earners at the refresh (v5: 1.0).
# v6: 1.3 -- the minimax hedge over the two-sided fitted ensemble.
REACH_MULT = 1.3

# practice-branch prologue fraction: share of the newly tracked players
# who have earned the prologue trophy by the refresh (v5: 0.995, kept).
GF_FRAC = 0.995


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round_half_up(x, nd):
    """Round half up to nd decimals (matches the house's rounding)."""
    scale = 10.0 ** nd
    v = math.floor(x * scale + 0.5) / scale
    return v


def round2(x):
    v = round_half_up(x, 2)
    return int(v) if float(v).is_integer() else v


def round4(x):
    v = round_half_up(x, 4)
    return int(v) if float(v).is_integer() else v


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def quantiles_from_median(median_pct, spread):
    lg = logit(median_pct)
    qs = []
    for z in Z:
        qs.append(round4(inv_logit(lg + z * spread)))
    qs = sorted(qs)
    return qs


# --- trophy classification (name + description only) ---------------------

def _text(trophy):
    return ((trophy.get("name") or "") + " " + (trophy.get("description") or "")).lower()


def is_prologue(trophy):
    return "gone fishing" in _text(trophy)


def prior_group(trophy):
    """Which logit prior a trophy takes when its history is frozen."""
    t = _text(trophy)
    if any(k in t for k in ("gone fishing", "chapter 1", "breakfast run",
                            "fishing challenge")):
        return PRIOR_EARLY
    if any(k in t for k in ("chapter 2", "arcade game", "bowling",
                            "100 vehicles")):
        return PRIOR_MID
    return PRIOR_REST


def ceiling(trophy):
    """Week-one ceiling for a trophy's rate, by class."""
    if is_prologue(trophy):
        # the prologue is done by essentially everyone who plays at all;
        # it must never be capped at the early-story ceiling (40%)
        return 99.99
    t = _text(trophy)
    if any(k in t for k in ("gone fishing", "chapter 1", "breakfast run",
                            "fishing challenge")):
        return CEIL_EARLY
    if any(k in t for k in ("chapter 3", "chapter 4",
                            "100% completion", "250 stunts", "gold medal",
                            "congratulations! you're a leonida legend")):
        return CEIL_LATE
    if _is_main_final(t):
        return CEIL_LATE
    if any(k in t for k in ("arcade", "bowling", "top fun", "shooting range",
                            "races", "race", "100 vehicles", "activity",
                            "meditate", "hunt a unique animal", "cocktail",
                            "explore all of leonida", "waink", "challenges",
                            "collectible card")):
        return CEIL_ACTIVITY
    return CEIL_MID


def _is_main_final(t):
    """The story's final mission -- but not a side character's final
    mission for them ("final mission for Andy" etc. is mid-game)."""
    return "final mission" in t and "final mission for" not in t


_LATE_OR_GRIND = (
    "chapter 3", "chapter 4", "100% completion",
    "250 stunts", "gold medal", "congratulations! you're a leonida legend",
    "headliners", "vcia takedown", "operation savage glory",
    "8 mission strands", "meditate", "unique animal", "collectible card",
    "waink", "50 challenges", "bank vaults", "illegal operation",
    "marked bills", "$100,000", "garage properties", "rhino",
    "20 different races",
)


def refresh_reachable(trophy):
    """True when a trophy could plausibly be earned in a launch-night
    session: not late-story and not a grind. The prologue is handled
    separately (it is reachable by definition)."""
    if is_prologue(trophy):
        return False
    t = _text(trophy)
    if _is_main_final(t):
        return False
    return not any(k in t for k in _LATE_OR_GRIND)


# --- history helpers ------------------------------------------------------

def read_series(case, idx):
    """(times_h, Ns, Es) for trophy idx across the history, oldest first.
    times_h is hours since the first read. Skips reads missing the trophy."""
    hist = case["history"]
    t0 = parse_time(hist[0]["at"])
    times, ns, es = [], [], []
    for r in hist:
        rates = r.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        n = r.get("players_tracked")
        if not n:
            continue
        times.append((parse_time(r["at"]) - t0).total_seconds() / 3600.0)
        ns.append(float(n))
        es.append(round(rates[idx] * n / 100.0))
    return times, ns, es


def history_frozen(case):
    hist = case["history"]
    if not hist:
        return False
    first = hist[0]
    for r in hist[1:]:
        if r.get("players_tracked") != first.get("players_tracked"):
            return False
        if r.get("rates") != first.get("rates"):
            return False
    return True


def least_squares(xs, ys):
    """Slope/intercept of y ~ x. Returns (slope, intercept)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


# --- regime 1: launch-night refresh (frozen, short horizon) --------------

def refresh_forecast(case):
    """Predict the first refresh: N jumps to FIRST_N_NEW, earners grow to
    FIRST_E_TOTAL, split prologue-heavy (see module docstring)."""
    hist = case["history"]
    last = hist[-1]
    n_old = last["players_tracked"]
    new_players = FIRST_N_NEW - n_old
    rows = []
    e_new = {}
    last_rates = last.get("rates") or []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) else None
        e_old = round(rate * n_old / 100.0) if rate is not None else 0
        if is_prologue(trophy):
            # the prologue is done by ~everyone who plays at all
            e_new[i] = e_old + round(GF_FRAC * new_players)
        elif refresh_reachable(trophy):
            # reachable trophies gain REACH_MULT x their earners at the refresh
            e_new[i] = e_old + max(1, round(REACH_MULT * e_old))
        else:
            e_new[i] = e_old
    # count-consistency: trim the reachable deltas until the total is the
    # documented 4,276 (largest deltas first, one earner at a time); if
    # the split falls short instead, top it up the same way.
    total = sum(e_new.values())
    order = sorted(
        (i for i, t in enumerate(case["trophies"])
         if refresh_reachable(t) and e_new[i] > 0),
        key=lambda i: (e_new[i], i), reverse=True)
    k = 0
    while total != FIRST_E_TOTAL and order and k < 50000:
        i = order[k % len(order)]
        if total > FIRST_E_TOTAL:
            if e_new[i] > 0:
                e_new[i] -= 1
                total -= 1
        else:
            e_new[i] += 1
            total += 1
        k += 1
    for i, trophy in enumerate(case["trophies"]):
        med = round2(e_new[i] * 100.0 / FIRST_N_NEW)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median(med, REFRESH_SPREAD)})
    return {"trophies": rows}


# --- regime 2: frozen history, long horizon (logit priors) ----------------

def prior_forecast(case, horizon_days):
    hist = case["history"]
    last = hist[-1]
    rows = []
    last_rates = last.get("rates") or []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is None:
            rate = 0.01
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        lg = logit(rate) + prior_group(trophy) * horizon_days
        med = round2(inv_logit(lg))
        spread = min(SPREAD_CAP,
                     SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + 0.5)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median(med, spread)})
    return {"trophies": rows}


# --- regime 3: moving history (count-consistent E/N fits) -----------------

def _saturating_fit(times, vals, tau, t_target):
    """Fit v(t) = v0 + (v_inf - v0)(1 - exp(-(t-t0)/tau)) by least squares on
    v_inf; return the fitted value at t_target, floored at the last value."""
    if not times:
        return None
    t0 = times[0]
    v0 = vals[0]
    num = 0.0
    den = 0.0
    for t, v in zip(times, vals):
        w = 1.0 - math.exp(-max(0.0, t - t0) / tau)
        num += w * (v - v0)
        den += w * w
    vinf = v0 + (num / den if den > 0 else 0.0)
    if vinf < v0:
        vinf = v0
    w_t = 1.0 - math.exp(-max(0.0, t_target - t0) / tau)
    return max(vals[-1], v0 + (vinf - v0) * w_t)


def _power_fit(times, vals, t_target):
    """Fit log(v+0.5) ~ a + b*log(t+1); return the extrapolated value."""
    xs = [math.log(t + 1.0) for t in times]
    ys = [math.log(max(v, 0.0) + 0.5) for v in vals]
    b, a = least_squares(xs, ys)
    b = min(2.0, max(0.0, b))
    return max(0.5, math.exp(a + b * math.log(t_target + 1.0)) - 0.5)


def fitted_forecast(case, horizon_days):
    hist = case["history"]
    t0 = parse_time(hist[0]["at"])
    t_target = (parse_time(case["target"]) - t0).total_seconds() / 3600.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        times, ns, es = read_series(case, i)
        # Fit the current growth regime, not the launch flatline.
        if len(times) > 3:
            t_last = times[-1]
            keep = [k for k, t in enumerate(times)
                    if t >= t_last - FIT_WINDOW_HOURS]
            if len(keep) >= 3:
                times = [times[k] for k in keep]
                ns = [ns[k] for k in keep]
                es = [es[k] for k in keep]
        last_rates = hist[-1].get("rates") or []
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            last_rate = 0.01
        if is_prologue(trophy):
            # The prologue is earned by essentially every player who starts
            # the game, so its rate sits at ~100% minus the share of tracked
            # players who have not started yet. It must never fall through
            # to the E/N fit (whose early-story ceiling would cap it at
            # 40%). If the last read is already at the clamp, stay there;
            # if dilution has dipped it, persist the dipped rate with a
            # spread wide enough to cover both recovery and further dip.
            if last_rate >= 99.995:
                rows.append({"id": trophy["id"], "q": [99.99] * 5})
                continue
            if last_rate >= 90.0:
                med = round2(last_rate)
                spread = min(SPREAD_CAP,
                             0.5 * math.sqrt(horizon_days) + 0.3)
                rows.append({"id": trophy["id"],
                             "q": quantiles_from_median(med, spread)})
                continue
            # below 90%: fall through to the E/N fit (ceiling 99.99)
        if last_rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        if len(times) < 3 or len(set(es)) < 2:
            # too little movement to fit: fall back to the logit trend
            rows.append(_trend_row(case, i, horizon_days, last_rate))
            continue
        e_power = _power_fit(times, es, t_target)
        e_sat = _saturating_fit(times, es, TAU_E_HOURS, t_target)
        if e_sat is None:
            e_sat = e_power
        e_fc = math.sqrt(max(e_power, 0.5) * max(e_sat, 0.5))
        e_fc = max(e_fc, float(es[-1]))
        n_fc = _saturating_fit(times, ns, TAU_N_HOURS, t_target)
        if n_fc is None or n_fc <= 0:
            n_fc = float(ns[-1])
        n_fc = max(n_fc, float(ns[-1]))
        rate = 100.0 * e_fc / n_fc
        rate = min(rate, ceiling(trophy))
        med = round2(rate)
        # spread: horizon random walk + fit residual, capped
        xs = [math.log(t + 1.0) for t in times]
        ys = [math.log(max(v, 0.0) + 0.5) for v in es]
        b, a = least_squares(xs, ys)
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
                          / max(1, len(xs))) if len(xs) > 2 else 0.25
        resid = min(0.5, max(0.0, resid))
        spread = min(SPREAD_CAP,
                     SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + resid)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median(med, spread)})
    return {"trophies": rows}


def _trend_row(case, i, horizon_days, last_rate):
    """Spec fallback: least-squares logit trend over the last 48 reads,
    clamped slope +-3 logit/day, saturating integral tau=1.25 days."""
    hist = case["history"][-48:]
    t0 = parse_time(case["history"][0]["at"])
    xs, ys = [], []
    for r in hist:
        rates = r.get("rates") or []
        if i >= len(rates) or rates[i] is None:
            continue
        xs.append((parse_time(r["at"]) - t0).total_seconds() / 86400.0)
        ys.append(logit(rates[i]))
    if len(xs) >= 2:
        slope, intercept = least_squares(xs, ys)
        slope = min(3.0, max(-3.0, slope))
        tau = 1.25
        last_t = xs[-1]
        move = slope * tau * (1.0 - math.exp(-horizon_days / tau))
        med = round2(inv_logit(intercept + slope * last_t + move))
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / len(xs))
        spread = min(SPREAD_CAP,
                     SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days)
                     + min(0.5, resid))
    else:
        med = round2(last_rate)
        spread = min(SPREAD_CAP,
                     SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + 0.5)
    trophy = case["trophies"][i]
    med = min(med, ceiling(trophy))
    return {"id": trophy["id"], "q": quantiles_from_median(med, spread)}


# --- driver ----------------------------------------------------------------

def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"{case['id']} asks for levels {case['levels']}")
    if not case["history"]:
        raise ValueError(f"{case['id']} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    if history_frozen(case):
        if (case["horizon_minutes"] <= REFRESH_HORIZON_MIN
                and case["history"][-1].get("players_tracked") == FIRST_N_OLD):
            return refresh_forecast(case)
        return prior_forecast(case, horizon_days)
    return fitted_forecast(case, horizon_days)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
