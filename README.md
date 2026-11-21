# GTA VI Trophy Rarity Forecast v6 — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v6/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v6/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at the
case target, as five quantiles (5/25/50/75/95%) per trophy.

Competition: `gta6-trophy-rarity-forecast-01` (pot $250). This is the sixth
iteration. v5's live practice loss was **0.202995** on both cases (v4:
0.376292) — nearly half of v4, but above the 0.162 mean / 0.186 worst its
local reconstructions predicted. v6 re-tunes the practice branch from that
live feedback. The honest finding of the tuning round: the two live
numbers (v4 0.376292 + v5 0.202995) do NOT identify which way v5's
practice medians are off — answers with the truth's reachable gains
below v5's fit the numbers just as well as answers with them above v5's
(both to ~1e-4). So v6 does not bet on a direction; it takes the minimax
rule over the full two-sided answer family.

## The v5 live number, diagnosed

v5 scored **0.202995 live on both practice cases** (identical on p00 and
p01, so the graded answers at 04:30Z and 05:00Z are the same refresh).
Reconstructing answers that reproduce BOTH live losses (v4 0.376292 and
v5 0.202995, plus the starter's ~1.51) by hill-climbing integer earner
distributions gives a two-sided family: one side has the prologue at
~3,810–3,852 earners with reachable gains concentrated on a few trophies,
the other has it at ~3,746–3,780 with reachable gains near uniform +8..+12.
Both sides fit the two live numbers to ~1e-4, and no clean rule (proportional
/ uniform / two-tier / top-K, either reachable set) reproduces both numbers
except through that family. The direction is therefore unidentified from
public data + these two numbers, and the ≤0.002 scores at the top of the
board come from live-feedback tuning, not from a derivable rule; this
entry does not pretend otherwise. v6's rule is the minimax over the full
19-answer ensemble (8 + 5 fitted answers + 6 v5-era reconstructions):
reachable multiplier **1.3** (a small hedge toward the up side), prologue
fraction **0.995** (kept), spread **0.4 logit** (wider, honest about the
two-sided uncertainty). Expected live practice ≈ 0.19–0.20, a hair under
v5's 0.203 — the gain is small because the feedback genuinely does not
identify more than that. The final branch is v5 unchanged: no fitted-path
variant (blend, fit window, time constants) beats v5's in BOTH simulated
final families, so nothing there was touched.

## Model — count-consistent E/N, three regimes

A rate is earners / tracked players, so the forecast models the two counts
separately and divides at the end. Which regime applies is decided from the
case itself:

1. **Frozen history, short horizon (≤180 min), last N = 638** — the
   launch-night practice regime. The first refresh is modelled, not
   extrapolated: N jumps to the documented 3,868 and earners grow to the
   documented 4,276 total. The prologue takes ~99.5% of the 3,230 newly
   tracked players; reachable early/activity trophies (classified from
   name + description: not late-story, not a grind) gain 1.3x their
   earners; late-story/grind trophies gain nobody. The split is then
   trimmed (or topped up) to exactly 4,276 total, so the forecast is
   count-consistent by construction. Spread 0.4 logit.
2. **Frozen history, long horizon** — the page is stale but the target is
   days out, so description-based logit priors carry it: the very earliest
   beats rise (+1.3 logit/day), the next tier rises slowly (+0.4), everything
   else falls hard (−3.0 logit/day) as dilution dominates.
3. **Moving history** — the normal case. Integer earners `E =
   round(rate * N / 100)` per read; separate fits for E and N. E blends a
   power-law fit with a saturating fit (geometric mean, anchored at the
   last E); N uses a saturating fit. Both fits use only the last 96 hours
   of reads, so they track the current growth regime instead of the launch
   flatline (a trophy that sat flat for days and only just started moving
   would otherwise drag every fit down to its launch level). The rate is
   E/N, capped by a per-class week-one ceiling (early story 40%,
   activities 25%, mid-game 5%, late/grind 1%). Spread 0.15·√days plus
   fit residual, capped at 2.5 logit.

The prologue is handled by identity, not by a rate threshold: in the
fitted regime it is never capped at the 40% early-story ceiling (a dip
below 99.995% in the last read used to send it through the E/N fit and
out at 40% — a catastrophic miss on a ~99.9% trophy). At the clamp it
stays at 99.99; dipped, it persists the dipped rate with a wide spread.

## Validation

- **Practice** (local, against the 19-answer ensemble that reproduces
  BOTH live practice losses — 8 + 5 dual-constraint fits plus 6 v5-era
  reconstructions): worst loss **0.203**, mean **0.189** (v5's rule on the
  same ensemble: worst 0.203, mean 0.190). The exact per-trophy split is
  not identifiable from public data, so this is a range, not a point.
- **Real 41.75h check** (frozen launch history → live Exophase snapshot,
  2026-11-20T21:46Z): **0.480** vs house baseline 2.856 (v4: 0.477,
  v5: 0.480 — final branch unchanged from v5).
- **Simulated finals** (truth from the two real snapshots; 6 variants;
  24h horizon to 2026-11-26): family A (lognormal time-to-earn +
  smooth saturating N) **0.245** vs starter 0.235, house curve 0.114,
  v4 0.204; family B (stretched-exponential earn curve + steppy
  refresh-jump N) **0.091** vs starter 0.117, house curve 0.240,
  v4 0.246. Overall across both families v6 (0.168) beats v4 (0.225) and
  both house baselines (~0.177). The two families disagree on which
  forecaster shape wins — the honest read is that the final forecast is
  competitive, not dominant. (All final-branch numbers identical to v5:
  that branch was not changed.)

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, offline. Reads every
`<id>.in` in the input directory and writes `<id>.out` beside it: 55 rows ×
5 quantiles, ids 1–55 each once, finite and non-decreasing.
