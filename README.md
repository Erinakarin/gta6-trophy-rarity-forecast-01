# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Erinakarin/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Erinakarin/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each PlayStation 5 trophy's Exophase earn rate at the case's
target, for make.fun's `gta6-trophy-rarity-forecast-01` bounty.

Referral: https://make.fun/r/gq6jtptn

## Run

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Python
3.13, standard library only, no network. Runs in ~0.1 s per case.

## Approach

The score is the mean pinball loss on the logit scale, and a prize needs a
loss strictly below the best house entry — so safety comes first, then
calibration.

- **Deduplicate the history.** Exophase refreshes on its own schedule, so
  the reads between two refreshes are identical and carry no signal. Only
  the points where values actually changed are kept.
- **Cross-trophy structure.** A refresh moves almost every trophy's logit
  in the same direction (dilution as the tracked player base widens, partly
  offset by players progressing). Each trophy's total logit drift over the
  observed window is shrunk toward its tier-group mean, then projected
  forward with a saturating factor, so a 4.5-day horizon does not
  extrapolate linearly.
- **Wide, conservative spread.** A random-walk term plus a refresh-jump
  term estimated from the observed refreshes, with a floor of 1.0 logit
  units, so a refresh the history has not seen yet still fits inside the
  band. Pinball loss punishes a too-narrow spread harder than a
  slightly-wide one.
- **The 100% trophy is pinned.** The trophy everyone has stays at 100%
  (the house's own data shows this); it gets a narrow spread and is never
  dragged down by the group drift.
- **Never invalid.** Everything is wrapped in try/except. On any failure a
  case falls back to its last-read rates with a wide spread, so the output
  is always a valid 55-row forecast with five non-decreasing quantiles in
  0–100.

## Validation (practice cases p00/p01)

- Output shape valid: 55 rows, 5 non-decreasing quantiles each, 0–100,
  unique ids, one key.
- Run time ~0.06 s per case (target under 2 s, limit 8 s).
- Two runs byte-identical.
- The true practice answers are not published (Exophase only shows current
  rates), so the loss cannot be computed exactly. Against a constructed
  dilution-only refreshed page (earners unchanged, players 638 → 3,868),
  the estimated loss is ~1.06 vs ~1.69 for the house baseline on the same
  page — the wider spread pays off when the refresh lands outside the
  house's narrow band.
