# Forecasting the heat pump's electricity (study, 2026-11-09)

Which estimate of the pump's electricity is best, how it was tested, and what it cannot do. Built in `core/theria/plan/energy_forecast.py`, shown on the Energy tab, and since 2026-11-10 the plan's energy per hour.
The scripts are in [`code/`](code/) and the independent reviewer's in [`code/review/`](code/review/). The data files they
build are not kept in the repository.

## Result

**Use the simple model below.** With known setpoint and a next-day weather forecast it misses a day by 4.6 kWh (8 %) on
average. With the measured weather it misses by 3.8 kWh (6.7 %). A more complex ensemble was not better on data it had not
seen.

The model is good enough to plan with. **It is not yet valid for measuring savings** (see [Limits](#limits)).

## What is forecast

Compressor electricity per hour, heating plus hot water, from `sensor.compr_cons_heating` and `sensor.compr_cons_hotwat`
in InfluxDB, 2022-10 to 2026-11. The electric heater is left out: its counter is unreliable from May 2026.

The counters step in 1 kWh. Hourly values are a linear interpolation of a cleaned cumulative counter, so hourly scores are
indicative only. Daily and 4-hour scores are what count.

Why InfluxDB and not Home Assistant's long-term statistics, as in BESS: these counters have no `state_class`, so Home
Assistant keeps no statistics for them.

## The model

Issued at 14:00 local on the day before, when prices arrive. Only data before that moment is used, except the day's setpoint,
which the plan decides.

**Heating, per hour:**

```text
heating = max(0, ridge(features)) * ratio^0.5
```

Features, with `h(x) = max(16 - x, 0)`:

- constant, and `max(k - T, 0)` for k = 16, 10, 4, -2
- trailing outdoor means `h(T_24h)`, `h(T_72h)`
- estimated flow target `F - 35` and `(F - 35) * h(T) / 10`
- sun, `sun/100` and `sun/100 * h(T)/10`; wind, `wind/10 * h(T)/10`

Fit: ridge with lambda 1, sample weights with a half-life of 365 days, refit every 7 days. `ratio` is actual over predicted
electricity in the last 3 days up to the issue time, with 5 kWh added to both, clipped to 0.5-2.

**Estimated flow target:** a least-squares fit of the logged target on 1, `T`, `T²` and `setpoint - 18`, over the last 60 days.
Its error a day ahead is 0.52 C (mean absolute, measured weather).

**Hot water, per hour:** the mean per clock hour over the last 28 days, separately for weekdays and weekends.

**Cost:** electricity per hour times the hour's price.

## How it was tested

Walk-forward: each day is forecast only from what was known at 14:00 the day before. Scored on October to April.

| Period | Use | Days |
|---|---|---|
| Oct 2022 - Sep 2023 | training only | - |
| Oct-Apr 2023/24, 2024/25 | choosing the model | 344 |
| Oct 2025 - Apr 2026, Oct - 8 Nov 2026 | one final test, after the model was frozen | 227 |

Day error is the mean absolute error of the day's total. "Cold days" are days at least 4 C below the mean of the previous 28.

### Choosing the model (development seasons, 344 days)

| Heating model | Day error, kWh |
|---|---|
| Flow target only | 14.0 |
| Emitter from flow, divided by COP from the datasheet | 12.5 |
| Degree-hours divided by COP | 9.2 |
| Outdoor temperature hinges | 9.05 |
| Hinges + setpoint | 5.86 |
| Hinges + flow | 5.70 |
| + trailing outdoor, sun and wind | 5.14 |
| + 3-day level correction (**SIMPLE**, with the hot-water profile above) | 4.75 |
| Mean of SIMPLE and gradient boosting | 4.40 |

The flow target can be computed almost exactly, but on its own it does not tell how much heat the house draws: how many loops
and valves are open also matters. It belongs in the model together with the outdoor temperature.

### Final test (227 days)

| Method | Day error, kWh | Bias, kWh/day | Share of the day | With forecast weather | Cost error, kr/day, spot price |
|---|---|---|---|---|---|
| Mean of SIMPLE and gradient boosting | 3.73 | +0.90 | 6.6 % | 4.56 | 3.93 |
| **SIMPLE** | **3.79** | **+0.63** | **6.7 %** | **4.61** | **3.86** |
| Straight line against outdoor, 28 days, + hot water 7 days | 4.39 | -0.05 | 7.8 % | 5.10 | 4.37 |
| The plan's energy model today + hot water | 8.25 | +7.67 | 14.7 % | 8.63 | 9.43 |
| BESS, 48 h mean | 8.35 | -0.23 | 14.9 % | 8.35 | 8.10 |
| BESS, 7-day hourly profile | 9.94 | -1.57 | 17.7 % | 9.94 | 9.54 |

- The ensemble's gain over SIMPLE is -0.06 kWh/day, 95 % interval -0.31 to +0.17 (14-day blocks). **SIMPLE was chosen**: it
  needs only numpy, and gradient boosting weakens the response to the setpoint.
- The ensemble's gain over the 28-day line is -0.66 kWh/day (block interval -1.24 to -0.04), and is not significant with
  forecast weather (-1.19 to +0.11).
- Cold days, 39: SIMPLE 6.12 kWh (bias +2.62), ensemble 6.07 (+3.76), 28-day line 5.38 (-0.97).

The plan's energy model is a weak comparison: fixed coefficients, heating only.

## Independent review

An independent agent tried to break the test. **Verdict: sound enough to document and build for planning, with changes; not
yet validated for the savings baseline.**

**Checked and clean:**

- **No leakage.** Replacing all data from the issue time onward changed no prediction.
- **Counters.** Heating plus hot water against the separately cleaned total counter (`sensor.compr_consump_tot`), 227 days: 12746
  against 12740 kWh, mean day difference 0.03 kWh, mean absolute 0.52, 17 days over 1 kWh. Rejected values move energy in time but
  do not lose it. Both counters come from the same H60.
- **Day selection** does not flatter the result: dropped days are warmer than kept ones.
- **The bootstrap** was redone with 14-day blocks. The conclusions stand.

**Findings, and what was done:**

| # | Finding | Severity | Done |
|---|---|---|---|
| 1 | The final test barely tests the setpoint: it moved by 0.75 C or more on 16 % of days, against 56 % in development. The fitted response is observational and unstable between refits | High | Limit below; needs deliberate steps |
| 2 | Once control runs, the level correction, hot-water profile and refits would learn from controlled days. A real 5 % cut pulls the baseline down by about 2.5 % | High | Baseline must be frozen on pre-control data, without level correction |
| 3 | The forecast's error is as large as the likely saving: rolling 30-day cost error mean 3.1 %, p95 6.2 %, max 7.9 % | High | Limit below |
| 4 | "The ensemble is best" is not supported | Medium | SIMPLE chosen |
| 5 | The lead over the 28-day line depends on the period and on knowing the setpoint. Without the true setpoint (prior 7-day mean instead): final test 3.77 to 4.14 kWh/day | Medium | Limit below |
| 6 | Cold-day bias is real but its cause is not established; it varies by month (Jan 2026 +7.9, Feb 2026 -17.7 kWh/day, few days) | Medium | Open |
| 7 | "The final test was untouched" is overstated: an earlier exploratory run covered the same period with the baselines. The final model's features were not in it | Low | Stated here |
| 8 | Hourly scores rest on interpolated values | Low | Stated above |

A setpoint-only version without the flow estimate scores 4.88 against 4.75 on the development seasons (reproduced). It is the
natural simplification, and would need its own test: the final test has been used.

Compressor saturation looks unlikely as the cause of the cold-day bias.

## Limits

Forecast of compressor electricity (heating and hot water) for the next day. Day error about 4.6 kWh (8 %) with forecast
weather and known setpoint, 3.8 kWh with measured weather, on one held-out winter when the setpoint hardly varied.

- Accuracy degrades on cold-snap days (about +5 %, unstable) and without the true setpoint.
- Hourly values are indicative only.
- The response to a setpoint change is an observational estimate of about 0.15-0.25 kWh/h per degree. It is unvalidated, and
  below the 0.23-0.37 measured earlier.
- Monthly savings below about 6 % cannot be told from model error.
- The baseline is valid only if fitted on data without control.
- The electric heater, the stove and other unmetered heat are outside the model.

## What is left

1. **To plan:** done 2026-11-10, the plan reads its heating part per hour and setpoint. Forecast against actual is built: each estimate is stored when it is made, and the error
   is shown per day. The built code, run on the final test's days with its own data loading, gives 3.87 kWh/day (224 days).
2. **To measure savings:** freeze a baseline on pre-control data without level correction, run deliberate setpoint steps to
   validate the setpoint response, and keep control days.
3. Fix the setpoint-to-flow relation from the pump's curve instead of a 60-day fit, so that the response stays stable.
