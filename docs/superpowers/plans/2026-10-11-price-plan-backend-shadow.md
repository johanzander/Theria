# Price-Driven Heating Plan - Backend (Shadow Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute, every 15 minutes, an hourly room-setpoint plan for the ground floor that minimises electricity cost under comfort limits, expose it through a REST API, and never write anything to Home Assistant.

**Architecture:** Pure, unit-tested modules under `core/theria/plan/` (settings, prices, pump model, room model, dynamic-programming planner, weather client) are orchestrated by a `PlanService` that reads live data from Home Assistant and Open-Meteo, keeps recent snapshots in memory, and is exposed by a small FastAPI router (`backend/plan_api.py`). Models start from documented priors; fitting, scoring and persistence come in plan 2.

**Tech Stack:** Python 3.13, numpy, requests, FastAPI, pytest (+ httpx for `TestClient`), ruff, black, mypy.

**Spec:** `docs/superpowers/specs/2026-10-11-price-plan-design.md` (read it first). Measurements behind every default: `documentation/ivt-heat-pump.md`.

## Plan series

The spec covers four independent pieces, each producing working software:

1. **This plan - backend, shadow mode.** Models on priors, planner, API. Nothing is written to HA.
2. *Plan 2 - calibration and scoring.* Load history from InfluxDB, fit `PumpParams` / `RoomParams` nightly, compare predictions with actuals, persist snapshots and mode.
3. *Plan 3 - React frontend* (`frontend/`, Vite + React + TypeScript + Tailwind + Recharts, copied from bess-manager's setup) with the "Price and plan" page.
4. *Plan 4 - active mode.* Executor with write guards, read-back, fallback, kill switch, controlled test steps.

## Deviations from the approved spec (decided while planning; spec updated to match)

- **Hourly decisions.** The planner decides once per hour and uses the hourly mean of the 15-minute prices. The spec said 15-minute periods; hourly steps keep the DP state grid fine enough to avoid discretisation drift and match the 1.5 C/h rate limit. Prices are still parsed at 15-minute resolution.
- **LK setpoint.** The room model includes the LK valves: loops close as the room approaches the LK setpoint. With LK at 19.5 C the floor cannot store heat above 19.5 C, so pre-heating would be impossible. This plan models LK as one fixed input, `lk_setpoint_c = 21.0`. Johan decided that in active mode Theria controls the LK setpoint as a second variable (between target and ceiling) to avoid spill heat; that joint decision and the LK writes belong to plan 4. Nothing here writes to LK.
- **Active mode is refused.** `PlanSettings` accepts only `off` and `shadow` until plan 4 exists.

## Global Constraints

- Python 3.13 (`.python-version`), `black`, `ruff` and `mypy` clean (`disallow_untyped_defs`); union syntax `X | None`; no `hasattr` or silent fallbacks; specific exception classes from `core/theria/exceptions.py`; never match on exception message text.
- This plan **must not write to Home Assistant**: no `set_temperature`, `set_hvac_mode` or service calls anywhere in the new code.
- Defaults from the spec, exactly: floor 18.0 C, extreme floor 17.0 C, target 19.5 C, ceiling 21.0 C, comfort cost 1.0 SEK per C-hour, room setpoint range 15.0-22.0 C in 0.5 C steps, base setpoint 18.0 C, max change 1.5 C per hour, extreme price = at least the 95th percentile of the last 14 days and at least 3.0 SEK/kWh, mode default `off`.
- Heating only, ground floor only. Hot water, boutique and guest house are out of scope.
- Prices: use the `today` / `tomorrow` lists (SEK/kWh). The `price` inside `raw_today` / `raw_tomorrow` is SEK/MWh and must not be used for values. (The existing `PriceOptimizer` parses that raw field; do not copy it.)
- Tests: behaviour only, no implementation details. Run with the project venv: `.venv/bin/python -m pytest`.
- Commits: never without Johan's approval; no AI attribution in messages.

## Review Focus

Failure modes the spec implies but a first implementation tends to miss. Each has a test in the task that owns the code.

1. **Clock change days** (23 or 25 hours, 92 or 100 quarters): parsing and hourly aggregation must work on UTC timestamps, not on counts of 96. (Task 2)
2. **Negative prices** (as low as -0.86 SEK/kWh in the history): the planner may use them but must never push the room above the ceiling. (Task 5)
3. **Unreachable floor** (cold, low start temperature): the plan must report floor-violation hours instead of failing. (Task 5)
4. **Tomorrow's prices not published yet**: the horizon is shorter, not an error. (Task 7)
5. **A sensor that is `unavailable` / `unknown`, or a weather gap**: a specific exception, the previous snapshot stays available, and the API reports its age. (Tasks 7, 8)

## File Structure

| File | Responsibility |
|---|---|
| `core/theria/exceptions.py` (modify) | Add `PriceDataError`, `ModelFitError`, `WeatherDataError` |
| `core/theria/plan/__init__.py` (create) | Package marker |
| `core/theria/plan/settings.py` (create) | `PlanSettings`: validated configuration |
| `core/theria/plan/prices.py` (create) | Nordpool parsing, hourly mean, extreme-price rule, price history |
| `core/theria/plan/pump_model.py` (create) | Flow target and heating electricity; fit function |
| `core/theria/plan/room_model.py` (create) | Slab and room dynamics with sun and LK valves; fit function |
| `core/theria/plan/planner.py` (create) | Dynamic-programming planner, baseline, reasons |
| `core/theria/plan/weather.py` (create) | Open-Meteo parser and client |
| `core/theria/plan/service.py` (create) | `PlanService`: live inputs, planning loop, snapshots |
| `backend/plan_api.py` (create) | FastAPI router |
| `backend/app.py` (modify) | Start the service, include the router |
| `config.yaml` (modify) | `plan` options and schema |
| `Dockerfile` (modify) | Copy `plan_api.py` into the image |
| `requirements-dev.txt` (modify) | Add `httpx` for `TestClient` |
| `tests/core/theria/plan/` (create) | One test module per source module plus `conftest.py` |

---

### Task 1: Exceptions, package skeleton and `PlanSettings`

**Files:**
- Modify: `core/theria/exceptions.py`, `requirements-dev.txt`
- Create: `core/theria/plan/__init__.py`, `core/theria/plan/settings.py`, `tests/core/theria/plan/__init__.py`, `tests/core/theria/plan/conftest.py`, `tests/core/theria/plan/test_settings.py`

**Interfaces:**
- Produces: `PriceDataError`, `ModelFitError`, `WeatherDataError` (all `TheriaError`); `PlanSettings` (frozen dataclass) with `from_dict(data: dict) -> PlanSettings`, `from_options(options: dict) -> PlanSettings | None`, constant `SUPPORTED_MODES = ("off", "shadow")`. Fields: `price_entity, outdoor_entity, room_setpoint_entity, flow_setpoint_entity: str`, `room_temp_entities: tuple[str, ...]`, `latitude, longitude: float`, then defaults `mode="off", floor_c=18.0, extreme_floor_c=17.0, target_c=19.5, ceiling_c=21.0, lk_setpoint_c=21.0, comfort_cost_sek_per_c_hour=1.0, rts_min_c=15.0, rts_max_c=22.0, rts_base_c=18.0, rts_step_c=0.5, max_rts_step_per_hour_c=1.5, extreme_price_percentile=95.0, extreme_price_min_sek_per_kwh=3.0, extreme_history_days=14`.
- Produces test fixture `settings` (a valid `PlanSettings`) in `tests/core/theria/plan/conftest.py`.

- [ ] **Step 1: Add `httpx` to the dev requirements and create the test package**

Append `httpx` to `requirements-dev.txt` (needed by `fastapi.testclient` in Task 8) and install it:

```bash
echo "httpx" >> requirements-dev.txt
.venv/bin/pip install -r requirements-dev.txt
mkdir -p core/theria/plan tests/core/theria/plan
touch tests/core/theria/plan/__init__.py
```

Create `core/theria/plan/__init__.py`:

```python
"""Price-driven heating plan (shadow mode)."""
```

Create `tests/core/theria/plan/conftest.py` (more fixtures are appended in Task 5):

```python
import pytest

from core.theria.plan.settings import PlanSettings


@pytest.fixture
def settings() -> PlanSettings:
    return PlanSettings(
        price_entity="sensor.price",
        outdoor_entity="sensor.outdoor",
        room_setpoint_entity="climate.room_temp_setpoint",
        flow_setpoint_entity="sensor.heating_setpoint",
        room_temp_entities=("sensor.lk_bibliotek_temperature",),
        latitude=55.578,
        longitude=12.927,
    )
```

- [ ] **Step 2: Write the failing test**

Create `tests/core/theria/plan/test_settings.py`:

```python
import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import PlanSettings

BASE = {
    "price_entity": "sensor.price",
    "outdoor_entity": "sensor.outdoor",
    "room_setpoint_entity": "climate.room_temp_setpoint",
    "flow_setpoint_entity": "sensor.heating_setpoint",
    "room_temp_entities": ["sensor.a", "sensor.b"],
    "latitude": 55.578,
    "longitude": 12.927,
}


def test_defaults_match_the_agreed_design():
    s = PlanSettings.from_dict(BASE)
    assert (s.floor_c, s.extreme_floor_c, s.target_c, s.ceiling_c) == (
        18.0,
        17.0,
        19.5,
        21.0,
    )
    assert s.comfort_cost_sek_per_c_hour == 1.0
    assert s.mode == "off"
    assert s.room_temp_entities == ("sensor.a", "sensor.b")


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "active"},
        {"floor_c": 19.9},  # above target
        {"extreme_floor_c": 18.5},  # above floor
        {"rts_base_c": 30.0},  # outside range
        {"rts_step_c": 0.0},
        {"max_rts_step_per_hour_c": 0.1},
        {"comfort_cost_sek_per_c_hour": -1.0},
        {"extreme_price_percentile": 100.0},
        {"room_temp_entities": []},
        {"unknown_option": 1},
    ],
)
def test_invalid_settings_raise_configuration_error(override):
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict({**BASE, **override})


def test_missing_required_setting_raises_configuration_error():
    data = dict(BASE)
    del data["price_entity"]
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict(data)


def test_from_options_returns_none_without_a_plan_section():
    assert PlanSettings.from_options({"zones": []}) is None
    assert PlanSettings.from_options({"plan": BASE}).mode == "off"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_settings.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.settings'`.

- [ ] **Step 4: Add the exceptions and implement `PlanSettings`**

Append to `core/theria/exceptions.py`:

```python
class PriceDataError(TheriaError):
    """Electricity price data is missing or malformed."""

    pass


class ModelFitError(TheriaError):
    """A model cannot be fitted from the supplied data."""

    pass


class WeatherDataError(TheriaError):
    """Weather forecast data is missing or malformed."""

    pass
```

Create `core/theria/plan/settings.py`:

```python
"""Configuration for the price-driven heating plan."""

from dataclasses import dataclass, fields

from core.theria.exceptions import ConfigurationError

SUPPORTED_MODES = ("off", "shadow")


@dataclass(frozen=True)
class PlanSettings:
    """Plan configuration. Defaults are the values agreed in the design spec."""

    price_entity: str
    outdoor_entity: str
    room_setpoint_entity: str
    flow_setpoint_entity: str
    room_temp_entities: tuple[str, ...]
    latitude: float
    longitude: float
    mode: str = "off"
    floor_c: float = 18.0
    extreme_floor_c: float = 17.0
    target_c: float = 19.5
    ceiling_c: float = 21.0
    lk_setpoint_c: float = 21.0
    comfort_cost_sek_per_c_hour: float = 1.0
    rts_min_c: float = 15.0
    rts_max_c: float = 22.0
    rts_base_c: float = 18.0
    rts_step_c: float = 0.5
    max_rts_step_per_hour_c: float = 1.5
    extreme_price_percentile: float = 95.0
    extreme_price_min_sek_per_kwh: float = 3.0
    extreme_history_days: int = 14

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"plan mode must be one of {SUPPORTED_MODES}, got {self.mode!r}"
            )
        if not self.extreme_floor_c <= self.floor_c <= self.target_c <= self.ceiling_c:
            raise ConfigurationError(
                "plan temperatures must satisfy extreme_floor <= floor <= target <= ceiling"
            )
        if not self.rts_min_c <= self.rts_base_c <= self.rts_max_c:
            raise ConfigurationError(
                "plan room setpoint range must satisfy rts_min <= rts_base <= rts_max"
            )
        if self.rts_step_c <= 0 or self.max_rts_step_per_hour_c < self.rts_step_c:
            raise ConfigurationError(
                "rts_step_c must be positive and not larger than max_rts_step_per_hour_c"
            )
        if self.comfort_cost_sek_per_c_hour < 0:
            raise ConfigurationError("comfort_cost_sek_per_c_hour must not be negative")
        if not 0 < self.extreme_price_percentile < 100:
            raise ConfigurationError("extreme_price_percentile must be in (0, 100)")
        if not self.room_temp_entities:
            raise ConfigurationError("room_temp_entities must not be empty")

    @classmethod
    def from_dict(cls, data: dict) -> "PlanSettings":
        """Create from the `plan` section of config.yaml."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown plan settings: {sorted(unknown)}")
        converted = dict(data)
        if "room_temp_entities" in converted:
            converted["room_temp_entities"] = tuple(converted["room_temp_entities"])
        try:
            return cls(**converted)
        except TypeError as e:
            raise ConfigurationError(f"invalid plan settings: {e}") from e

    @classmethod
    def from_options(cls, options: dict) -> "PlanSettings | None":
        """Settings from the add-on options, or None when no `plan` section exists."""
        section = options.get("plan")
        return None if section is None else cls.from_dict(section)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_settings.py -v`
Expected: 13 passed.

- [ ] **Step 6: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/exceptions.py core/theria/plan tests/core/theria/plan requirements-dev.txt
git commit -m "feat: add plan settings and plan exceptions"
```

---

### Task 2: Prices

**Files:**
- Create: `core/theria/plan/prices.py`, `tests/core/theria/plan/test_prices.py`

**Interfaces:**
- Consumes: `PriceDataError`.
- Produces: `PricePeriod(start: datetime, price_sek_per_kwh: float)` (frozen, tz-aware); `parse_nordpool_attributes(attrs: dict) -> list[PricePeriod]`; `hourly_prices(periods: list[PricePeriod], first_hour: datetime) -> list[tuple[datetime, float]]`; `extreme_threshold(history_sek_per_kwh: list[float], percentile: float, min_sek_per_kwh: float) -> float`; `PriceHistory(days: int)` with `add(periods)`, `prune(now)`, `values() -> list[float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_prices.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import PriceDataError
from core.theria.plan.prices import (
    PriceHistory,
    PricePeriod,
    extreme_threshold,
    hourly_prices,
    parse_nordpool_attributes,
)

DAY = datetime(2026, 10, 12, 0, tzinfo=UTC)


def attrs(values, day=DAY, key="today"):
    raw = [
        {"start": (day + timedelta(minutes=15 * i)).isoformat(), "price": v * 1000.0}
        for i, v in enumerate(values)
    ]
    return {key: list(values), f"raw_{key}": raw}


def test_values_come_from_sek_per_kwh_list_not_the_sek_per_mwh_raw_price():
    periods = parse_nordpool_attributes(attrs([1.5, 2.5]))
    assert [p.price_sek_per_kwh for p in periods] == [1.5, 2.5]
    assert periods[1].start == DAY + timedelta(minutes=15)


def test_tomorrow_is_optional_and_days_are_merged_in_order():
    both = {
        **attrs([1.0] * 96),
        **attrs([2.0] * 96, DAY + timedelta(days=1), "tomorrow"),
    }
    periods = parse_nordpool_attributes(both)
    assert len(periods) == 192
    assert periods[0].start < periods[-1].start
    assert parse_nordpool_attributes(
        {**attrs([1.0] * 96), "tomorrow": [], "raw_tomorrow": []}
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"today": [1.0, 2.0], "raw_today": [{"start": DAY.isoformat()}]},
        {"today": [1.0], "raw_today": [{"start": "2026-10-12T00:00:00"}]},  # naive time
        {"today": ["x"], "raw_today": [{"start": DAY.isoformat()}]},
        {
            "today": [1.0, 1.0],
            "raw_today": [{"start": DAY.isoformat()}] * 2,
        },  # duplicate
        {"today": [1.0]},  # raw missing
    ],
)
def test_malformed_attributes_raise(bad):
    with pytest.raises(PriceDataError):
        parse_nordpool_attributes(bad)


def test_hourly_prices_average_quarters_and_drop_incomplete_trailing_hour():
    periods = [
        PricePeriod(DAY + timedelta(minutes=15 * i), float(i)) for i in range(10)
    ]
    hourly = hourly_prices(periods, DAY)
    assert hourly == [(DAY, 1.5), (DAY + timedelta(hours=1), 5.5)]


def test_hourly_prices_without_a_complete_hour_raises():
    with pytest.raises(PriceDataError):
        hourly_prices([PricePeriod(DAY, 1.0)], DAY)


def test_extreme_threshold_is_the_larger_of_percentile_and_minimum():
    quiet = [1.0] * 100
    assert extreme_threshold(quiet, 95.0, 3.0) == 3.0
    volatile = [1.0] * 90 + [8.0] * 10
    assert extreme_threshold(volatile, 95.0, 3.0) == 8.0


def test_extreme_threshold_needs_a_day_of_history():
    with pytest.raises(PriceDataError):
        extreme_threshold([1.0] * 95, 95.0, 3.0)


def test_history_prunes_old_periods_and_deduplicates():
    history = PriceHistory(days=2)
    old = PricePeriod(DAY - timedelta(days=3), 9.0)
    recent = PricePeriod(DAY, 1.0)
    history.add([old, recent, PricePeriod(DAY, 2.0)])
    history.prune(DAY + timedelta(hours=1))
    assert history.values() == [2.0]


def test_dst_change_day_with_100_quarters_is_parsed_and_aggregated():
    day = datetime(2026, 10, 24, 22, tzinfo=UTC)  # local midnight, CEST
    periods = parse_nordpool_attributes(attrs([1.0] * 100, day))
    hourly = hourly_prices(periods, day)
    assert len(hourly) == 25
    assert hourly[-1][0] == day + timedelta(hours=24)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_prices.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.prices'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/prices.py`:

```python
"""Electricity prices for planning: parsing, hourly aggregation, extreme-price rule."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.theria.exceptions import PriceDataError

QUARTERS_PER_HOUR = 4


@dataclass(frozen=True)
class PricePeriod:
    """One 15-minute price period."""

    start: datetime  # timezone-aware
    price_sek_per_kwh: float


def parse_nordpool_attributes(attrs: dict) -> list[PricePeriod]:
    """Parse the Nordpool sensor attributes into 15-minute periods.

    Uses `today`/`tomorrow` (SEK/kWh) for the values and `raw_today`/`raw_tomorrow`
    only for the period start times. The `price` field inside the raw entries is in
    SEK/MWh and must not be used. `tomorrow` is empty until prices are published.
    """
    periods: list[PricePeriod] = []
    for day in ("today", "tomorrow"):
        values = attrs.get(day)
        if not values:
            continue
        raw = attrs.get(f"raw_{day}")
        if raw is None or len(raw) != len(values):
            raise PriceDataError(
                f"{day}: expected raw_{day} with {len(values)} entries to match the values"
            )
        for entry, value in zip(raw, values, strict=True):
            if not isinstance(value, int | float):
                raise PriceDataError(f"{day}: non-numeric price {value!r}")
            start = datetime.fromisoformat(entry["start"])
            if start.tzinfo is None:
                raise PriceDataError(f"{day}: period start without timezone")
            periods.append(PricePeriod(start, float(value)))
    periods.sort(key=lambda p: p.start)
    starts = [p.start for p in periods]
    if len(set(starts)) != len(starts):
        raise PriceDataError("duplicate price period start times")
    return periods


def hourly_prices(
    periods: list[PricePeriod], first_hour: datetime
) -> list[tuple[datetime, float]]:
    """Mean price per complete hour, starting at `first_hour`, until data runs out.

    A trailing hour with fewer than four quarters is dropped.
    """
    by_start = {p.start: p.price_sek_per_kwh for p in periods}
    out: list[tuple[datetime, float]] = []
    hour = first_hour
    while True:
        quarters = [hour + timedelta(minutes=15 * i) for i in range(QUARTERS_PER_HOUR)]
        if not all(q in by_start for q in quarters):
            break
        out.append((hour, float(np.mean([by_start[q] for q in quarters]))))
        hour += timedelta(hours=1)
    if not out:
        raise PriceDataError(f"no complete price hour available from {first_hour}")
    return out


def extreme_threshold(
    history_sek_per_kwh: list[float], percentile: float, min_sek_per_kwh: float
) -> float:
    """Price at or above which a period counts as extreme.

    The larger of the given percentile of recent prices and the absolute minimum.
    """
    if len(history_sek_per_kwh) < 96:
        raise PriceDataError(
            "need at least one day (96 periods) of price history for the extreme rule"
        )
    return max(float(np.percentile(history_sek_per_kwh, percentile)), min_sek_per_kwh)


class PriceHistory:
    """Rolling store of 15-minute prices used for the extreme-price rule."""

    def __init__(self, days: int):
        self._days = days
        self._prices: dict[datetime, float] = {}

    def add(self, periods: list[PricePeriod]) -> None:
        for p in periods:
            self._prices[p.start] = p.price_sek_per_kwh

    def prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self._days)
        self._prices = {s: v for s, v in self._prices.items() if s >= cutoff}

    def values(self) -> list[float]:
        return [self._prices[s] for s in sorted(self._prices)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_prices.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/prices.py tests/core/theria/plan/test_prices.py
git commit -m "feat: parse Nordpool prices and detect extreme periods"
```

---

### Task 3: Pump model

**Files:**
- Create: `core/theria/plan/pump_model.py`, `tests/core/theria/plan/test_pump_model.py`

**Interfaces:**
- Consumes: `ModelFitError`.
- Produces: `PumpParams` (frozen dataclass: `flow_intercept, flow_slope, rts_ref, flow_per_rts, energy_intercept, energy_per_outdoor, energy_per_rts`); `PRIOR_PUMP_PARAMS`; `DEFAULT_FLOW_PER_RTS = 1.9`; `flow_target(params, t_out_c, rts_c)` and `energy_kwh_per_hour(params, t_out_c, rts_c)` (scalars or numpy arrays; energy never negative); `fit_pump_params(t_out_c, flow_target_c, rts_c, energy_kwh_per_h, flow_per_rts=1.9) -> PumpParams` (numpy arrays of hourly heating-only samples).

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_pump_model.py`:

```python
import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.pump_model import (
    PRIOR_PUMP_PARAMS,
    energy_kwh_per_hour,
    fit_pump_params,
    flow_target,
)


def test_prior_reproduces_measured_curve_and_room_setpoint_gain():
    p = PRIOR_PUMP_PARAMS
    assert flow_target(p, 12.0, p.rts_ref) == pytest.approx(32.64)
    assert flow_target(p, 5.0, 19.0) - flow_target(p, 5.0, 18.0) == pytest.approx(1.9)


def test_energy_is_higher_with_higher_setpoint_and_never_negative():
    p = PRIOR_PUMP_PARAMS
    assert energy_kwh_per_hour(p, 5.0, 20.0) > energy_kwh_per_hour(p, 5.0, 16.0)
    assert float(energy_kwh_per_hour(p, 40.0, 13.0)) == 0.0


def make_data(n=200):
    rng = np.random.default_rng(0)
    t_out = rng.uniform(-2, 12, n)
    rts = rng.uniform(15, 22, n)
    flow = 41.0 - 0.7 * t_out + 1.9 * (rts - 18.5)
    energy = 0.4 - 0.12 * t_out + 0.11 * rts
    return t_out, flow, rts, energy


def test_fit_recovers_the_generating_parameters():
    t_out, flow, rts, energy = make_data()
    fitted = fit_pump_params(t_out, flow, rts, energy)
    assert fitted.flow_slope == pytest.approx(-0.7, abs=1e-6)
    assert flow_target(fitted, 5.0, 18.5) == pytest.approx(41.0 - 0.7 * 5.0, abs=1e-6)
    assert fitted.energy_per_rts == pytest.approx(0.11, abs=1e-6)
    assert fitted.energy_per_outdoor == pytest.approx(-0.12, abs=1e-6)


def test_fit_refuses_data_without_variation_or_enough_samples():
    t_out, flow, rts, energy = make_data()
    with pytest.raises(ModelFitError):
        fit_pump_params(t_out[:10], flow[:10], rts[:10], energy[:10])
    with pytest.raises(ModelFitError):
        fit_pump_params(t_out, flow, np.full_like(rts, 18.0), energy)
    with pytest.raises(ModelFitError):
        fit_pump_params(np.full_like(t_out, 5.0), flow, rts, energy)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_pump_model.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.pump_model'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/pump_model.py`:

```python
"""Heat pump model: flow target and heating electricity versus outdoor temperature and room setpoint."""

from dataclasses import dataclass

import numpy as np

from core.theria.exceptions import ModelFitError

Array = float | np.ndarray  # scalars and numpy arrays are both accepted

DEFAULT_FLOW_PER_RTS = 1.9  # measured median, 227 steps 2022-2025


@dataclass(frozen=True)
class PumpParams:
    """Parameters of the pump model.

    flow_target = flow_intercept + flow_slope * t_out + flow_per_rts * (rts - rts_ref)
    energy      = energy_intercept + energy_per_outdoor * t_out + energy_per_rts * rts
    """

    flow_intercept: float
    flow_slope: float
    rts_ref: float
    flow_per_rts: float
    energy_intercept: float
    energy_per_outdoor: float
    energy_per_rts: float


# Priors from the measured winter 2025/26 curve and the rough counter regression
# (documentation/ivt-heat-pump.md). Replaced by fit_pump_params once data is loaded.
PRIOR_PUMP_PARAMS = PumpParams(
    flow_intercept=40.8,
    flow_slope=-0.68,
    rts_ref=18.0,
    flow_per_rts=DEFAULT_FLOW_PER_RTS,
    energy_intercept=0.31,
    energy_per_outdoor=-0.15,
    energy_per_rts=0.13,
)


def flow_target(params: PumpParams, t_out_c: Array, rts_c: Array) -> Array:
    """Pump flow temperature target (C). Works on scalars and numpy arrays."""
    return (
        params.flow_intercept
        + params.flow_slope * t_out_c
        + params.flow_per_rts * (rts_c - params.rts_ref)
    )


def energy_kwh_per_hour(params: PumpParams, t_out_c: Array, rts_c: Array) -> Array:
    """Heating electricity (kWh per hour), never negative."""
    raw = (
        params.energy_intercept
        + params.energy_per_outdoor * t_out_c
        + params.energy_per_rts * rts_c
    )
    return np.maximum(raw, 0.0)


def fit_pump_params(
    t_out_c: np.ndarray,
    flow_target_c: np.ndarray,
    rts_c: np.ndarray,
    energy_kwh_per_h: np.ndarray,
    flow_per_rts: float = DEFAULT_FLOW_PER_RTS,
) -> PumpParams:
    """Fit the pump model on hourly heating-only data."""
    n = len(t_out_c)
    if n < 24 or not (len(flow_target_c) == len(rts_c) == len(energy_kwh_per_h) == n):
        raise ModelFitError(
            "pump fit needs equally long arrays with at least 24 hourly samples"
        )
    if np.ptp(t_out_c) < 3.0:
        raise ModelFitError("pump fit needs at least 3 C spread in outdoor temperature")
    if np.ptp(rts_c) < 1.0:
        raise ModelFitError("pump fit needs at least 1 C spread in room setpoint")
    rts_ref = float(np.mean(rts_c))
    adjusted = flow_target_c - flow_per_rts * (rts_c - rts_ref)
    flow_slope, flow_intercept = np.polyfit(t_out_c, adjusted, 1)
    design = np.column_stack([np.ones(n), t_out_c, rts_c])
    coef, *_ = np.linalg.lstsq(design, energy_kwh_per_h, rcond=None)
    return PumpParams(
        flow_intercept=float(flow_intercept),
        flow_slope=float(flow_slope),
        rts_ref=rts_ref,
        flow_per_rts=flow_per_rts,
        energy_intercept=float(coef[0]),
        energy_per_outdoor=float(coef[1]),
        energy_per_rts=float(coef[2]),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_pump_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/pump_model.py tests/core/theria/plan/test_pump_model.py
git commit -m "feat: add heat pump flow and energy model"
```

---

### Task 4: Room model

**Files:**
- Create: `core/theria/plan/room_model.py`, `tests/core/theria/plan/test_room_model.py`

**Interfaces:**
- Consumes: `ModelFitError`.
- Produces: `RoomParams` (frozen: `tau_slab_h, k_slab, k_out, k_sun, sun_lag_h, drift_c_per_h`); `PRIOR_ROOM_PARAMS`; `RoomState(room_c, slab_c, sun_filtered_wm2)` (frozen); `valve_open(room_c, lk_setpoint_c)`; `step(params, room_c, slab_c, sun_filtered_wm2, flow_c, t_out_c, sun_wm2, lk_setpoint_c, dt_h=1.0) -> (room_c, slab_c, sun_filtered_wm2)` (scalars or numpy arrays); `simulate(params, initial: RoomState, flow_c, t_out_c, sun_wm2, lk_setpoint_c) -> np.ndarray` (room temperature at the end of each hour); `fit_room_params(room_c, flow_c, t_out_c, sun_wm2, lk_setpoint_c) -> RoomParams` (contiguous hourly arrays, `room_c[i]` = temperature at the start of hour i, at least one week). Constant `VALVE_RAMP_C = 0.5`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_room_model.py`:

```python
import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.room_model import (
    PRIOR_ROOM_PARAMS,
    RoomParams,
    RoomState,
    fit_room_params,
    simulate,
    step,
    valve_open,
)


def test_valves_close_as_room_reaches_the_lk_setpoint():
    assert valve_open(19.0, 21.0) == 1.0
    assert valve_open(20.75, 21.0) == pytest.approx(0.5)
    assert valve_open(21.5, 21.0) == 0.0


def test_hot_slab_warms_room_and_closed_valves_stop_it():
    open_room, _, _ = step(PRIOR_ROOM_PARAMS, 19.0, 30.0, 0.0, 35.0, 5.0, 0.0, 21.0)
    closed_room, closed_slab, _ = step(
        PRIOR_ROOM_PARAMS, 19.0, 30.0, 0.0, 35.0, 5.0, 0.0, 18.0
    )
    assert open_room > 19.0 - PRIOR_ROOM_PARAMS.k_out * (19.0 - 5.0)
    assert closed_room < 19.0  # only loss to outdoors
    assert closed_slab < 30.0  # slab relaxes towards the room


def test_sun_warms_the_room_with_delay():
    params = PRIOR_ROOM_PARAMS
    state = (19.0, 19.0, 0.0)
    first = step(params, *state, 19.0, 19.0, 600.0, 19.0)
    assert first[0] - 19.0 < params.k_sun * 600.0  # not instant
    assert first[2] > 0.0


def forcing(n=24 * 21):
    hrs = np.arange(n)
    t_out = (
        5
        + 3 * np.sin(2 * np.pi * hrs / 24 - 1)
        + 2 * np.sin(2 * np.pi * hrs / (24 * 5))
    )
    sun = np.clip(500 * np.sin(2 * np.pi * (hrs % 24 - 6) / 24), 0, None) * (
        0.4 + 0.6 * (np.sin(hrs / 37) > 0)
    )
    flow = 36 + 4 * np.sin(2 * np.pi * hrs / (24 * 3)) + 2 * np.sin(hrs / 7)
    return flow, t_out, sun


@pytest.mark.parametrize(
    "true,initial",
    [
        (RoomParams(8.0, 0.05, 0.06, 0.0006, 4.0, 0.01), RoomState(19.0, 25.0, 0.0)),
        (RoomParams(4.0, 0.03, 0.05, 0.0004, 2.0, 0.0), RoomState(19.0, 30.0, 0.0)),
    ],
)
def test_fit_recovers_parameters_from_simulated_history(true, initial):
    flow, t_out, sun = forcing()
    room = np.concatenate(
        [[initial.room_c], simulate(true, initial, flow, t_out, sun, 21.0)]
    )
    pad = lambda a: np.append(a, a[-1])  # noqa: E731
    fitted = fit_room_params(room, pad(flow), pad(t_out), pad(sun), 21.0)
    assert fitted.tau_slab_h == true.tau_slab_h
    assert fitted.sun_lag_h == true.sun_lag_h
    assert fitted.k_slab == pytest.approx(true.k_slab, rel=0.02)
    assert fitted.k_out == pytest.approx(true.k_out, rel=0.02)
    assert fitted.k_sun == pytest.approx(true.k_sun, rel=0.02)


def test_fit_needs_a_week_of_equally_long_series():
    flow, t_out, sun = forcing(24 * 3)
    with pytest.raises(ModelFitError):
        fit_room_params(np.full(len(flow), 19.0), flow, t_out, sun, 21.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_room_model.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.room_model'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/room_model.py`:

```python
"""Ground-floor room model: slab (floor) and room temperature with sun and LK valves."""

from dataclasses import dataclass

import numpy as np

from core.theria.exceptions import ModelFitError

Array = float | np.ndarray  # scalars and numpy arrays are both accepted

VALVE_RAMP_C = 0.5  # valves close over the last 0.5 C below the LK setpoint


@dataclass(frozen=True)
class RoomParams:
    """Parameters of the room model (per hour).

    d_room = valve * k_slab * (slab - room) - k_out * (room - t_out)
             + k_sun * sun_filtered + drift
    slab relaxes towards the flow temperature while valves are open and towards
    the room temperature while they are closed, with time constant tau_slab_h.
    sun_filtered is the irradiance smoothed with time constant sun_lag_h.
    """

    tau_slab_h: float
    k_slab: float
    k_out: float
    k_sun: float
    sun_lag_h: float
    drift_c_per_h: float


# Priors chosen so that at 5 C outdoors and room setpoint 18 C the room settles near
# 19.5 C. Replaced by fit_room_params once LK history is available.
PRIOR_ROOM_PARAMS = RoomParams(
    tau_slab_h=6.0,
    k_slab=0.04,
    k_out=0.05,
    k_sun=0.0004,
    sun_lag_h=3.0,
    drift_c_per_h=0.0,
)


@dataclass(frozen=True)
class RoomState:
    room_c: float
    slab_c: float
    sun_filtered_wm2: float


def valve_open(room_c: Array, lk_setpoint_c: float) -> Array:
    """Fraction (0-1) of the loops open; scalars and numpy arrays."""
    return np.clip((lk_setpoint_c - room_c) / VALVE_RAMP_C, 0.0, 1.0)


def step(
    params: RoomParams,
    room_c: Array,
    slab_c: Array,
    sun_filtered_wm2: Array,
    flow_c: Array,
    t_out_c: Array,
    sun_wm2: Array,
    lk_setpoint_c: float,
    dt_h: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Advance one step. Works on scalars and numpy arrays.

    Returns (room_c, slab_c, sun_filtered_wm2) at the end of the step.
    """
    valve = valve_open(room_c, lk_setpoint_c)
    d_room = (
        valve * params.k_slab * (slab_c - room_c)
        - params.k_out * (room_c - t_out_c)
        + params.k_sun * sun_filtered_wm2
        + params.drift_c_per_h
    ) * dt_h
    slab_target = valve * flow_c + (1.0 - valve) * room_c
    slab_next = slab_c + (slab_target - slab_c) * (
        1.0 - np.exp(-dt_h / params.tau_slab_h)
    )
    sun_next = sun_filtered_wm2 + (sun_wm2 - sun_filtered_wm2) * (
        1.0 - np.exp(-dt_h / params.sun_lag_h)
    )
    return room_c + d_room, slab_next, sun_next


def simulate(
    params: RoomParams,
    initial: RoomState,
    flow_c: np.ndarray,
    t_out_c: np.ndarray,
    sun_wm2: np.ndarray,
    lk_setpoint_c: float,
) -> np.ndarray:
    """Room temperature at the end of each hour for the given hourly forcing."""
    room: Array = initial.room_c
    slab: Array = initial.slab_c
    sun_f: Array = initial.sun_filtered_wm2
    out = np.empty(len(flow_c))
    for i in range(len(flow_c)):
        room, slab, sun_f = step(
            params, room, slab, sun_f, flow_c[i], t_out_c[i], sun_wm2[i], lk_setpoint_c
        )
        out[i] = room
    return out


BURN_IN_H = 48  # slab temperature is unobserved; let the initial guess decay
TAU_SLAB_GRID_H = (2.0, 4.0, 6.0, 8.0, 12.0)
SUN_LAG_GRID_H = (1.0, 2.0, 3.0, 4.0, 6.0)


def fit_room_params(
    room_c: np.ndarray,
    flow_c: np.ndarray,
    t_out_c: np.ndarray,
    sun_wm2: np.ndarray,
    lk_setpoint_c: float,
) -> RoomParams:
    """Fit on contiguous hourly data (room_c[i] is the temperature at the start of hour i).

    Teacher-forced one-step regression over a grid of slab and sun time constants.
    """
    n = len(room_c) - 1
    if n < 7 * 24 or not (len(flow_c) == len(t_out_c) == len(sun_wm2) == len(room_c)):
        raise ModelFitError(
            "room fit needs equally long arrays with at least one week of hourly samples"
        )
    delta = np.diff(room_c)
    best: tuple[float, RoomParams] | None = None
    for tau in TAU_SLAB_GRID_H:
        slab = np.empty(len(room_c))
        slab[0] = room_c[0]
        for lag in SUN_LAG_GRID_H:
            sun_f = np.empty(len(room_c))
            sun_f[0] = sun_wm2[0]
            for i in range(n):
                valve = valve_open(room_c[i], lk_setpoint_c)
                target = valve * flow_c[i] + (1.0 - valve) * room_c[i]
                slab[i + 1] = slab[i] + (target - slab[i]) * (1.0 - np.exp(-1.0 / tau))
                sun_f[i + 1] = sun_f[i] + (sun_wm2[i] - sun_f[i]) * (
                    1.0 - np.exp(-1.0 / lag)
                )
            window = slice(BURN_IN_H, n)
            valve = valve_open(room_c[window], lk_setpoint_c)
            design = np.column_stack(
                [
                    valve * (slab[window] - room_c[window]),
                    -(room_c[window] - t_out_c[window]),
                    sun_f[window],
                    np.ones(n - BURN_IN_H),
                ]
            )
            coef, *_ = np.linalg.lstsq(design, delta[window], rcond=None)
            if coef[0] < 0 or coef[1] < 0 or coef[2] < 0:
                continue
            sse = float(np.sum((design @ coef - delta[window]) ** 2))
            if best is None or sse < best[0]:
                best = (
                    sse,
                    RoomParams(
                        tau_slab_h=tau,
                        k_slab=float(coef[0]),
                        k_out=float(coef[1]),
                        k_sun=float(coef[2]),
                        sun_lag_h=lag,
                        drift_c_per_h=float(coef[3]),
                    ),
                )
    if best is None:
        raise ModelFitError(
            "no physically valid room model found (negative coefficients)"
        )
    return best[1]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_room_model.py -v`
Expected: 6 passed. The fit tests recover the generating parameters (time constants exactly, gains within 2 %).

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/room_model.py tests/core/theria/plan/test_room_model.py
git commit -m "feat: add ground-floor room model with sun and LK valves"
```

---

### Task 5: Planner

**Files:**
- Create: `core/theria/plan/planner.py`, `tests/core/theria/plan/test_planner.py`
- Modify: `tests/core/theria/plan/conftest.py` (append the `make_inputs` fixture)

**Interfaces:**
- Consumes: `PlanSettings`, `PumpParams`, `flow_target`, `energy_kwh_per_hour`, `RoomParams`, `RoomState`, `step`, `OptimizationError`.
- Produces: `PlanInputs(hour_starts, price_sek_per_kwh, extreme, t_out_c, sun_wm2, initial: RoomState, previous_rts_c)` (frozen; arrays have one entry per hour); `HourPlan(start, rts_c, price_sek_per_kwh, extreme, floor_c, flow_target_c, energy_kwh, room_c_end, slab_c_end, energy_cost_sek, comfort_cost_sek, tail, reason)`; `Plan(hours, baseline_hours, objective_sek, baseline_objective_sek, floor_violation_hours)`; `plan(inputs, settings, pump, room) -> Plan`; `sun_filtered_sequence(room, initial, sun) -> np.ndarray`; constant `TAIL_HOURS = 6`. `HourPlan.reason` keys: `binding` (`floor` | `ceiling` | `rts_min` | `rts_max` | `price_trade_off` | `baseline`), `price_rank` (0 cheapest, 1 most expensive hour of the horizon), `extreme_price`, `marginal_objective_sek` (`minus_step_sek` / `plus_step_sek`: extra objective if the setpoint were one step lower / higher).

- [ ] **Step 1: Append the `make_inputs` fixture and write the failing tests**

Replace the imports at the top of `tests/core/theria/plan/conftest.py` with the block below and append the `make_inputs` fixture after the existing `settings` fixture. The complete file is:

```python
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.planner import PlanInputs
from core.theria.plan.room_model import RoomState
from core.theria.plan.settings import PlanSettings

HORIZON = 36
START = datetime(2026, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def settings() -> PlanSettings:
    return PlanSettings(
        price_entity="sensor.price",
        outdoor_entity="sensor.outdoor",
        room_setpoint_entity="climate.room_temp_setpoint",
        flow_setpoint_entity="sensor.heating_setpoint",
        room_temp_entities=("sensor.lk_bibliotek_temperature",),
        latitude=55.578,
        longitude=12.927,
    )


@pytest.fixture
def make_inputs():
    def _make(
        price,
        extreme=None,
        t_out=5.0,
        room_c=19.5,
        slab_c=28.0,
        previous_rts_c=18.0,
    ) -> PlanInputs:
        return PlanInputs(
            hour_starts=[START + timedelta(hours=i) for i in range(HORIZON)],
            price_sek_per_kwh=np.asarray(price, dtype=float),
            extreme=np.zeros(HORIZON, bool) if extreme is None else np.asarray(extreme),
            t_out_c=np.full(HORIZON, float(t_out)),
            sun_wm2=np.zeros(HORIZON),
            initial=RoomState(room_c, slab_c, 0.0),
            previous_rts_c=previous_rts_c,
        )

    return _make
```

Create `tests/core/theria/plan/test_planner.py`:

```python
import numpy as np
import pytest

from core.theria.exceptions import OptimizationError
from core.theria.plan.planner import TAIL_HOURS, plan
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS

from .conftest import HORIZON

HOUR_OF_DAY = np.arange(HORIZON) % 24
EVENING = (HOUR_OF_DAY >= 16) & (HOUR_OF_DAY < 20)


def run(inputs, settings):
    return plan(inputs, settings, PRIOR_PUMP_PARAMS, PRIOR_ROOM_PARAMS)


def test_flat_price_plan_respects_rate_limit_and_setpoint_range(settings, make_inputs):
    result = run(make_inputs(np.full(HORIZON, 1.0)), settings)
    rts = [h.rts_c for h in result.hours]
    assert all(settings.rts_min_c <= r <= settings.rts_max_c for r in rts)
    steps = np.abs(np.diff([settings.rts_base_c, *rts]))
    assert steps.max() <= settings.max_rts_step_per_hour_c + 1e-9


def test_price_spike_lowers_setpoint_during_spike_and_cuts_cost(settings, make_inputs):
    price = np.where(EVENING, 4.0, 0.8)
    result = run(make_inputs(price), settings)
    during = np.mean([h.rts_c for h, e in zip(result.hours, EVENING, strict=True) if e])
    outside = np.mean(
        [h.rts_c for h, e in zip(result.hours, EVENING, strict=True) if not e]
    )
    assert during < outside - 1.0
    assert result.objective_sek < result.baseline_objective_sek


def test_every_hour_has_a_reason_and_last_hours_are_flagged_as_tail(
    settings, make_inputs
):
    result = run(make_inputs(np.where(EVENING, 4.0, 0.8)), settings)
    assert all("binding" in h.reason and "price_rank" in h.reason for h in result.hours)
    assert [h.tail for h in result.hours].count(True) == TAIL_HOURS
    assert result.hours[-1].tail and not result.hours[0].tail


def test_floor_is_held_when_it_is_reachable(settings, make_inputs):
    price = np.where(EVENING, 6.0, 0.7)
    result = run(make_inputs(price, t_out=4.0, room_c=19.5, slab_c=30.0), settings)
    assert result.floor_violation_hours == 0
    assert min(h.room_c_end for h in result.hours) >= settings.floor_c - 0.05


def test_unreachable_floor_is_reported_not_raised(settings, make_inputs):
    result = run(
        make_inputs(np.full(HORIZON, 1.0), t_out=-5.0, room_c=17.0, slab_c=20.0),
        settings,
    )
    assert result.floor_violation_hours > 0


def test_extreme_hours_use_the_lower_floor(settings, make_inputs):
    price = np.where(EVENING, 8.0, 0.8)
    result = run(make_inputs(price, extreme=EVENING, t_out=2.0, slab_c=33.0), settings)
    for hour, is_extreme in zip(result.hours, EVENING, strict=True):
        assert hour.floor_c == (
            settings.extreme_floor_c if is_extreme else settings.floor_c
        )


def test_too_short_horizon_raises(settings, make_inputs):
    inputs = make_inputs(np.full(HORIZON, 1.0))
    short = type(inputs)(
        hour_starts=inputs.hour_starts[:1],
        price_sek_per_kwh=inputs.price_sek_per_kwh[:1],
        extreme=inputs.extreme[:1],
        t_out_c=inputs.t_out_c[:1],
        sun_wm2=inputs.sun_wm2[:1],
        initial=inputs.initial,
        previous_rts_c=inputs.previous_rts_c,
    )
    with pytest.raises(OptimizationError):
        run(short, settings)


def test_negative_prices_never_push_the_room_above_the_ceiling(settings, make_inputs):
    price = np.where(EVENING, -0.8, 1.0)
    result = run(make_inputs(price), settings)
    assert max(h.room_c_end for h in result.hours) <= settings.ceiling_c + 0.05
    during = np.mean([h.rts_c for h, e in zip(result.hours, EVENING, strict=True) if e])
    assert during > settings.rts_base_c  # being paid to heat is used
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_planner.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.planner'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/planner.py`. How it works: backward induction over a grid of (room temperature 0.1 C, slab temperature 1 C, previous setpoint). Each hour it tries every setpoint within the rate limit, simulates the room model, adds price x energy plus comfort cost (soft below target, 100 SEK per C-hour below the floor or above the ceiling), and interpolates the cost-to-go bilinearly. A small switching cost breaks ties so plans are smooth. The last 6 hours of a horizon are flagged `tail` (low confidence, because stored heat has no value after the horizon).

```python
"""Dynamic-programming planner: choose the room setpoint for each coming hour."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from core.theria.exceptions import OptimizationError
from core.theria.plan.pump_model import PumpParams, energy_kwh_per_hour, flow_target
from core.theria.plan.room_model import RoomParams, RoomState, step
from core.theria.plan.settings import PlanSettings

FloatArray = npt.NDArray[np.float64]
FloatGrid = npt.NDArray[np.floating[Any]]  # cost-to-go tables are float32

ROOM_GRID_C = np.arange(14.0, 24.0 + 1e-9, 0.1)
SLAB_GRID_C = np.arange(14.0, 50.0 + 1e-9, 1.0)
ROOM_BIN_C = 0.1
SLAB_BIN_C = 1.0
VIOLATION_COST_SEK_PER_C_HOUR = 100.0  # floor / ceiling are effectively hard
SWITCH_COST_SEK_PER_C = 0.02  # tie-breaker: prefer smooth plans, avoids pump jitter
TERMINAL_HOURS = 4.0  # terminal penalty: ending below target counts as this many hours
TAIL_HOURS = 6  # last hours of a horizon are shown but flagged low-confidence


@dataclass(frozen=True)
class PlanInputs:
    """Hourly forcing for the horizon. Arrays have one entry per hour."""

    hour_starts: list[datetime]
    price_sek_per_kwh: FloatArray
    extreme: npt.NDArray[np.bool_]
    t_out_c: FloatArray
    sun_wm2: FloatArray
    initial: RoomState
    previous_rts_c: float


@dataclass(frozen=True)
class HourPlan:
    start: datetime
    rts_c: float
    price_sek_per_kwh: float
    extreme: bool
    floor_c: float
    flow_target_c: float
    energy_kwh: float
    room_c_end: float
    slab_c_end: float
    energy_cost_sek: float
    comfort_cost_sek: float
    tail: bool
    reason: dict[str, Any]


@dataclass(frozen=True)
class Plan:
    hours: list[HourPlan]
    baseline_hours: list[HourPlan]  # same model, flat rts_base
    objective_sek: float
    baseline_objective_sek: float
    floor_violation_hours: int


def _interp2(grid: FloatGrid, r_idx: FloatArray, s_idx: FloatArray) -> FloatArray:
    """Bilinear lookup of `grid` (NR, NS) at fractional indices, clipped to the grid."""
    r_idx = np.clip(r_idx, 0, grid.shape[0] - 1.000001)
    s_idx = np.clip(s_idx, 0, grid.shape[1] - 1.000001)
    r0 = np.floor(r_idx).astype(int)
    s0 = np.floor(s_idx).astype(int)
    fr = r_idx - r0
    fs = s_idx - s0
    return np.asarray(
        grid[r0, s0] * (1 - fr) * (1 - fs)
        + grid[r0 + 1, s0] * fr * (1 - fs)
        + grid[r0, s0 + 1] * (1 - fr) * fs
        + grid[r0 + 1, s0 + 1] * fr * fs,
        dtype=float,
    )


def sun_filtered_sequence(
    room: RoomParams, initial: float, sun: FloatArray
) -> FloatArray:
    """Filtered sun at the start of each hour (exogenous, so not a DP state)."""
    out = np.empty(len(sun))
    value = initial
    k = 1.0 - np.exp(-1.0 / room.sun_lag_h)
    for i, s in enumerate(sun):
        out[i] = value
        value = value + (s - value) * k
    return out


class _Problem:
    """Everything the DP needs for one horizon, with the model transition and costs."""

    def __init__(
        self,
        inputs: PlanInputs,
        settings: PlanSettings,
        pump: PumpParams,
        room: RoomParams,
    ):
        self.inputs = inputs
        self.settings = settings
        self.pump = pump
        self.room = room
        self.horizon = len(inputs.hour_starts)
        self.actions = np.arange(
            settings.rts_min_c, settings.rts_max_c + 1e-9, settings.rts_step_c
        )
        self.sun_f = sun_filtered_sequence(
            room, inputs.initial.sun_filtered_wm2, inputs.sun_wm2
        )
        self.floors = np.where(
            inputs.extreme, settings.extreme_floor_c, settings.floor_c
        )

    def transition(
        self, t: int, a: int, room_c: FloatArray, slab_c: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Room and slab temperature at the end of hour t for action index a."""
        flow = float(flow_target(self.pump, self.inputs.t_out_c[t], self.actions[a]))
        room_next, slab_next, _ = step(
            self.room,
            room_c,
            slab_c,
            self.sun_f[t],
            flow,
            self.inputs.t_out_c[t],
            self.inputs.sun_wm2[t],
            self.settings.lk_setpoint_c,
        )
        return np.asarray(room_next, dtype=float), np.asarray(slab_next, dtype=float)

    def comfort_cost(self, room_c: FloatArray, floor_c: float) -> FloatArray:
        s = self.settings
        return np.asarray(
            s.comfort_cost_sek_per_c_hour * np.maximum(0.0, s.target_c - room_c)
            + VIOLATION_COST_SEK_PER_C_HOUR * np.maximum(0.0, floor_c - room_c)
            + VIOLATION_COST_SEK_PER_C_HOUR * np.maximum(0.0, room_c - s.ceiling_c),
            dtype=float,
        )

    def energy_kwh(self, t: int, rts_c: float) -> float:
        return float(energy_kwh_per_hour(self.pump, self.inputs.t_out_c[t], rts_c))

    def stage_cost(self, t: int, a: int, room_next: FloatArray) -> FloatArray:
        energy_cost = float(self.inputs.price_sek_per_kwh[t]) * self.energy_kwh(
            t, float(self.actions[a])
        )
        return energy_cost + self.comfort_cost(room_next, float(self.floors[t]))

    def q_value(
        self,
        t: int,
        a: int,
        room_c: FloatArray,
        slab_c: FloatArray,
        next_value: FloatGrid,
    ) -> FloatArray:
        """Immediate cost plus interpolated cost-to-go for action a from the given states."""
        room_next, slab_next = self.transition(t, a, room_c, slab_c)
        future = _interp2(
            next_value[:, :, a],
            (room_next - ROOM_GRID_C[0]) / ROOM_BIN_C,
            (slab_next - SLAB_GRID_C[0]) / SLAB_BIN_C,
        )
        return self.stage_cost(t, a, room_next) + future

    def solve(self) -> tuple[FloatGrid, npt.NDArray[np.int8]]:
        """Backward induction. Returns cost-to-go and policy, indexed by previous action."""
        s = self.settings
        n_actions = len(self.actions)
        max_move = round(s.max_rts_step_per_hour_c / s.rts_step_c)
        rr, ss = np.meshgrid(ROOM_GRID_C, SLAB_GRID_C, indexing="ij")
        shape = (len(ROOM_GRID_C), len(SLAB_GRID_C), n_actions)
        value = np.empty((self.horizon + 1, *shape), dtype=np.float32)
        terminal = (
            TERMINAL_HOURS
            * s.comfort_cost_sek_per_c_hour
            * np.maximum(0.0, s.target_c - rr)
        )
        value[self.horizon] = terminal[:, :, None]
        policy = np.empty((self.horizon, *shape), dtype=np.int8)
        for t in range(self.horizon - 1, -1, -1):
            q = np.empty(shape)
            for a in range(n_actions):
                q[:, :, a] = self.q_value(t, a, rr, ss, value[t + 1])
            best = np.full(shape, np.inf)
            best_a = np.zeros(shape, dtype=np.int8)
            for offset in range(-max_move, max_move + 1):
                shifted = np.full(shape, np.inf)
                lo, hi = max(0, -offset), min(n_actions, n_actions - offset)
                shifted[:, :, lo:hi] = q[:, :, lo + offset : hi + offset] + (
                    SWITCH_COST_SEK_PER_C * abs(offset) * s.rts_step_c
                )
                better = shifted < best
                best = np.where(better, shifted, best)
                best_a = np.where(
                    better, (np.arange(n_actions) + offset)[None, None, :], best_a
                ).astype(np.int8)
            value[t] = best
            policy[t] = best_a
        return value, policy

    def hour_plan(
        self,
        t: int,
        rts_c: float,
        room_end: float,
        slab_end: float,
        reason: dict[str, Any],
        tail: bool,
    ) -> HourPlan:
        energy = self.energy_kwh(t, rts_c)
        price = float(self.inputs.price_sek_per_kwh[t])
        floor_c = float(self.floors[t])
        return HourPlan(
            start=self.inputs.hour_starts[t],
            rts_c=rts_c,
            price_sek_per_kwh=price,
            extreme=bool(self.inputs.extreme[t]),
            floor_c=floor_c,
            flow_target_c=float(flow_target(self.pump, self.inputs.t_out_c[t], rts_c)),
            energy_kwh=energy,
            room_c_end=room_end,
            slab_c_end=slab_end,
            energy_cost_sek=price * energy,
            comfort_cost_sek=float(self.comfort_cost(np.array([room_end]), floor_c)[0]),
            tail=tail,
            reason=reason,
        )

    def follow_policy(
        self, value: FloatGrid, policy: npt.NDArray[np.int8]
    ) -> list[HourPlan]:
        """Roll the optimal policy forward from the initial state, with explanations."""
        room_c, slab_c = self.inputs.initial.room_c, self.inputs.initial.slab_c
        prev = int(np.argmin(np.abs(self.actions - self.inputs.previous_rts_c)))
        rank = np.argsort(np.argsort(self.inputs.price_sek_per_kwh)) / max(
            1, self.horizon - 1
        )
        hours: list[HourPlan] = []
        for t in range(self.horizon):
            r_i = int(
                np.clip(
                    round((room_c - ROOM_GRID_C[0]) / ROOM_BIN_C),
                    0,
                    len(ROOM_GRID_C) - 1,
                )
            )
            s_i = int(
                np.clip(
                    round((slab_c - SLAB_GRID_C[0]) / SLAB_BIN_C),
                    0,
                    len(SLAB_GRID_C) - 1,
                )
            )
            a = int(policy[t, r_i, s_i, prev])
            here = (np.array([[room_c]]), np.array([[slab_c]]))
            chosen = float(self.q_value(t, a, *here, value[t + 1])[0, 0])
            marginal = {
                label: float(self.q_value(t, b, *here, value[t + 1])[0, 0]) - chosen
                for label, b in (("minus_step_sek", a - 1), ("plus_step_sek", a + 1))
                if 0 <= b < len(self.actions)
            }
            room_next, slab_next = self.transition(t, a, *here)
            room_end, slab_end = float(room_next[0, 0]), float(slab_next[0, 0])
            hours.append(
                self.hour_plan(
                    t,
                    float(self.actions[a]),
                    room_end,
                    slab_end,
                    self.reason(a, room_end, t, float(rank[t]), marginal),
                    tail=t >= self.horizon - TAIL_HOURS,
                )
            )
            room_c, slab_c, prev = room_end, slab_end, a
        return hours

    def flat_plan(self, rts_c: float) -> list[HourPlan]:
        """The same models with a constant room setpoint: the comparison baseline."""
        room_c, slab_c = self.inputs.initial.room_c, self.inputs.initial.slab_c
        hours: list[HourPlan] = []
        for t in range(self.horizon):
            flow = float(flow_target(self.pump, self.inputs.t_out_c[t], rts_c))
            room_next, slab_next, _ = step(
                self.room,
                room_c,
                slab_c,
                self.sun_f[t],
                flow,
                self.inputs.t_out_c[t],
                self.inputs.sun_wm2[t],
                self.settings.lk_setpoint_c,
            )
            room_c, slab_c = float(room_next), float(slab_next)
            hours.append(
                self.hour_plan(
                    t, rts_c, room_c, slab_c, {"binding": "baseline"}, tail=False
                )
            )
        return hours

    def reason(
        self,
        a: int,
        room_end: float,
        t: int,
        price_rank: float,
        marginal: dict[str, float],
    ) -> dict[str, Any]:
        s = self.settings
        floor_c = float(self.floors[t])
        if room_end <= floor_c + 0.05:
            binding = "floor"
        elif room_end >= s.ceiling_c - 0.05:
            binding = "ceiling"
        elif a == 0:
            binding = "rts_min"
        elif a == len(self.actions) - 1:
            binding = "rts_max"
        else:
            binding = "price_trade_off"
        return {
            "binding": binding,
            # 0 = cheapest hour of the horizon, 1 = most expensive
            "price_rank": price_rank,
            "extreme_price": bool(self.inputs.extreme[t]),
            # extra objective if the setpoint were one step lower / higher
            "marginal_objective_sek": marginal,
        }


def plan(
    inputs: PlanInputs, settings: PlanSettings, pump: PumpParams, room: RoomParams
) -> Plan:
    """Backward dynamic programming over (room, slab, previous setpoint)."""
    horizon = len(inputs.hour_starts)
    arrays = (inputs.price_sek_per_kwh, inputs.extreme, inputs.t_out_c, inputs.sun_wm2)
    if horizon < 2 or any(len(a) != horizon for a in arrays):
        raise OptimizationError(
            "planner needs at least 2 hours and equally long inputs"
        )
    problem = _Problem(inputs, settings, pump, room)
    value, policy = problem.solve()
    hours = problem.follow_policy(value, policy)
    baseline = problem.flat_plan(settings.rts_base_c)
    objective = sum(h.energy_cost_sek + h.comfort_cost_sek for h in hours)
    base_objective = sum(h.energy_cost_sek + h.comfort_cost_sek for h in baseline)
    violations = sum(1 for h in hours if h.room_c_end < h.floor_c - 0.05)
    return Plan(hours, baseline, float(objective), float(base_objective), violations)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_planner.py -v`
Expected: 8 passed in about a second.

- [ ] **Step 5: Type check**

Run: `.venv/bin/mypy core/theria/plan`
Expected: no errors in `core/theria/plan` (a pre-existing error in `core/theria/ha_client.py:53` may be listed; it is not part of this work).

- [ ] **Step 6: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/planner.py tests/core/theria/plan/test_planner.py tests/core/theria/plan/conftest.py
git commit -m "feat: add dynamic-programming price planner"
```

---

### Task 6: Weather client

**Files:**
- Create: `core/theria/plan/weather.py`, `tests/core/theria/plan/test_weather.py`

**Interfaces:**
- Consumes: `WeatherDataError`.
- Produces: `WeatherHour(start: datetime, shortwave_wm2: float, cloud_cover_pct: float, temperature_c: float)` where `start` is the UTC start of the hour; `parse_open_meteo(payload: dict) -> list[WeatherHour]`; `OpenMeteoClient(latitude, longitude, session=None)` with `fetch(past_days=1, forecast_days=3) -> list[WeatherHour]`.

Note: Open-Meteo labels each value with the end of the hour it averages, so `start = label - 1 h`. The earlier sun/lag analysis in `documentation/ivt-heat-pump.md` used labels as they came, so its 4 h lag may really be about 3.5 h.

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_weather.py`:

```python
from datetime import UTC, datetime

import pytest

from core.theria.exceptions import WeatherDataError
from core.theria.plan.weather import parse_open_meteo


def payload(**overrides):
    base = {
        "hourly": {
            "time": ["2026-10-12T13:00", "2026-10-12T14:00"],
            "shortwave_radiation": [372.0, 329.0],
            "cloud_cover": [52, 87],
            "temperature_2m": [12.7, 11.8],
        }
    }
    base["hourly"].update(overrides)
    return base


def test_label_is_end_of_hour_so_start_is_one_hour_earlier():
    hours = parse_open_meteo(payload())
    assert hours[0].start == datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert hours[0].shortwave_wm2 == 372.0
    assert hours[1].temperature_c == 11.8


def test_missing_value_raises():
    with pytest.raises(WeatherDataError):
        parse_open_meteo(payload(shortwave_radiation=[372.0, None]))


def test_missing_field_raises():
    with pytest.raises(WeatherDataError):
        parse_open_meteo({"hourly": {"time": []}})


def test_mismatched_lengths_raise():
    with pytest.raises(WeatherDataError):
        parse_open_meteo(payload(cloud_cover=[52]))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_weather.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.weather'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/weather.py`:

```python
"""Open-Meteo forecast: outdoor temperature and solar irradiance per hour."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests

from core.theria.exceptions import WeatherDataError

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = "shortwave_radiation,cloud_cover,temperature_2m"


@dataclass(frozen=True)
class WeatherHour:
    """Weather for the hour starting at `start` (UTC)."""

    start: datetime
    shortwave_wm2: float
    cloud_cover_pct: float
    temperature_c: float


def parse_open_meteo(payload: dict) -> list[WeatherHour]:
    """Parse a forecast response requested with timezone=UTC.

    Open-Meteo labels each value with the END of the hour it averages (the value
    labelled 13:00 is the mean over 12:00-13:00), so the start is label - 1 h.
    """
    try:
        hourly = payload["hourly"]
        times = hourly["time"]
        columns = (
            hourly["shortwave_radiation"],
            hourly["cloud_cover"],
            hourly["temperature_2m"],
        )
    except KeyError as e:
        raise WeatherDataError(f"missing field in Open-Meteo response: {e}") from e
    if any(len(c) != len(times) for c in columns):
        raise WeatherDataError("Open-Meteo columns have different lengths")
    hours: list[WeatherHour] = []
    for label, sw, cc, temp in zip(times, *columns, strict=True):
        if sw is None or cc is None or temp is None:
            raise WeatherDataError(f"missing value at {label}")
        end = datetime.fromisoformat(label).replace(tzinfo=UTC)
        hours.append(
            WeatherHour(end - timedelta(hours=1), float(sw), float(cc), float(temp))
        )
    return hours


class OpenMeteoClient:
    """Fetches the hourly forecast for one location."""

    def __init__(
        self, latitude: float, longitude: float, session: requests.Session | None = None
    ):
        self._latitude = latitude
        self._longitude = longitude
        self._session = session or requests.Session()

    def fetch(self, past_days: int = 1, forecast_days: int = 3) -> list[WeatherHour]:
        params: dict[str, str | float | int] = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": HOURLY_VARIABLES,
            "timezone": "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
        try:
            response = self._session.get(OPEN_METEO_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as e:
            raise WeatherDataError(f"Open-Meteo request failed: {e}") from e
        return parse_open_meteo(payload)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_weather.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/weather.py tests/core/theria/plan/test_weather.py
git commit -m "feat: add Open-Meteo weather client"
```

---

### Task 7: PlanService

**Files:**
- Create: `core/theria/plan/service.py`, `tests/core/theria/plan/test_service.py`

**Interfaces:**
- Consumes: `HAClient.get_state(entity_id) -> dict` and `get_climate_state(entity_id) -> dict` (key `target_temperature`); everything from Tasks 1-6.
- Produces: `PlanSnapshot(computed_at, mode, plan, extreme_threshold_sek_per_kwh, outdoor_bias_c, initial: RoomState, model_status)` (frozen); `PlanService(ha_client, settings, weather, pump_params=PRIOR_PUMP_PARAMS, room_params=PRIOR_ROOM_PARAMS, now=<utc clock>, interval_seconds=900)` with `now()`, `mode` (property), `set_mode(mode)` (only `off` / `shadow`, in memory only), `latest()`, `history(hours)`, `model_status()`, `run_once() -> PlanSnapshot | None` (None when mode is `off`), `async start()`, `async stop()`.

Behaviour to keep in mind: the horizon runs from the start of the current hour to the last complete hour with prices (and weather). The outdoor forecast is shifted by the difference between the measured outdoor sensor and the forecast for the current hour. The initial slab temperature is a rough prior (half way between the room and the pump's flow target); plan 2 replaces it with an estimate. The service only reads from Home Assistant.

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_service.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import (
    ConfigurationError,
    PriceDataError,
    SensorError,
    WeatherDataError,
)
from core.theria.plan.service import PlanService
from core.theria.plan.weather import WeatherHour

NOW = datetime(2026, 10, 12, 12, 20, tzinfo=UTC)
DAY_START = datetime(2026, 10, 12, 0, tzinfo=UTC)


def price_attributes(days=2, base=1.0, evening=3.0):
    values, raw = [], []
    for q in range(96 * days):
        start = DAY_START + timedelta(minutes=15 * q)
        hour = start.hour
        values.append(evening if 16 <= hour < 20 else base)
        raw.append(
            {
                "start": start.isoformat(),
                "end": (start + timedelta(minutes=15)).isoformat(),
                "price": 1000.0,
            }
        )
    return {
        "today": values[:96],
        "raw_today": raw[:96],
        "tomorrow": values[96:],
        "raw_tomorrow": raw[96:],
    }


class FakeHA:
    def __init__(self, attrs):
        self.attrs = attrs
        self.reads = []

    def get_state(self, entity_id):
        self.reads.append(entity_id)
        states = {
            "sensor.price": {"state": "1.0", "attributes": self.attrs},
            "sensor.outdoor": {"state": "6.0", "attributes": {}},
            "sensor.lk_bibliotek_temperature": {"state": "19.8", "attributes": {}},
            "sensor.heating_setpoint": {"state": "36.0", "attributes": {}},
        }
        return states[entity_id]

    def get_climate_state(self, entity_id):
        return {"target_temperature": 18.0}


class FakeWeather:
    def __init__(self, hours=96, first=DAY_START - timedelta(hours=24)):
        self._hours = [
            WeatherHour(
                first + timedelta(hours=i),
                200.0 if 9 <= (first + timedelta(hours=i)).hour < 15 else 0.0,
                50.0,
                5.0,
            )
            for i in range(hours)
        ]

    def fetch(self):
        return self._hours


def make_service(settings, attrs=None, weather=None, mode="shadow", clock=None):
    from dataclasses import replace

    return PlanService(
        FakeHA(attrs or price_attributes()),
        replace(settings, mode=mode),
        weather or FakeWeather(),
        now=clock or (lambda: NOW),
    )


def test_run_once_plans_from_current_hour_to_end_of_known_prices(settings):
    service = make_service(settings)
    snapshot = service.run_once()
    hours = snapshot.plan.hours
    assert hours[0].start == datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert len(hours) == 36  # 12:00 today to 24:00 tomorrow
    assert service.latest() is snapshot


def test_evening_price_hours_get_lower_setpoint_than_cheap_hours(settings):
    snapshot = make_service(settings).run_once()
    evening = [h.rts_c for h in snapshot.plan.hours if 16 <= h.start.hour < 20]
    other = [h.rts_c for h in snapshot.plan.hours if not 16 <= h.start.hour < 20]
    assert sum(evening) / len(evening) < sum(other) / len(other)


def test_off_mode_does_nothing_and_reads_nothing(settings):
    service = make_service(settings, mode="off")
    assert service.run_once() is None
    assert service._ha.reads == []


def test_set_mode_rejects_active_until_the_executor_exists(settings):
    service = make_service(settings)
    with pytest.raises(ConfigurationError):
        service.set_mode("active")
    service.set_mode("off")
    assert service.mode == "off"


def test_missing_tomorrow_prices_shortens_the_horizon(settings):
    attrs = price_attributes(days=1)
    attrs["tomorrow"], attrs["raw_tomorrow"] = [], []
    snapshot = make_service(settings, attrs=attrs).run_once()
    assert len(snapshot.plan.hours) == 12


def test_weather_gap_raises_instead_of_guessing(settings):
    weather = FakeWeather(hours=30)  # ends before the horizon does
    with pytest.raises(WeatherDataError):
        make_service(settings, weather=weather).run_once()


def test_malformed_price_data_raises(settings):
    attrs = price_attributes()
    attrs["raw_today"] = attrs["raw_today"][:-1]
    with pytest.raises(PriceDataError):
        make_service(settings, attrs=attrs).run_once()


def test_history_returns_only_recent_snapshots(settings):
    current = [NOW]
    service = make_service(settings, clock=lambda: current[0])
    service.run_once()
    assert len(service.history(hours=1)) == 1
    current[0] = NOW + timedelta(hours=2)
    assert service.history(hours=1) == []
    assert len(service.history(hours=3)) == 1


def test_unavailable_room_sensor_raises_sensor_error(settings):
    service = make_service(settings)
    original = service._ha.get_state

    def unavailable(entity_id):
        if entity_id == "sensor.lk_bibliotek_temperature":
            return {"state": "unavailable", "attributes": {}}
        return original(entity_id)

    service._ha.get_state = unavailable
    with pytest.raises(SensorError):
        service.run_once()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_service.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.service'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/service.py`:

```python
"""Plan service: reads live data, computes the plan and keeps the latest snapshots."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from core.theria.exceptions import ConfigurationError, SensorError, WeatherDataError
from core.theria.ha_client import HAClient
from core.theria.plan.planner import Plan, PlanInputs, plan, sun_filtered_sequence
from core.theria.plan.prices import (
    PriceHistory,
    extreme_threshold,
    hourly_prices,
    parse_nordpool_attributes,
)
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, PumpParams
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS, RoomParams, RoomState
from core.theria.plan.settings import SUPPORTED_MODES, PlanSettings
from core.theria.plan.weather import OpenMeteoClient, WeatherHour

logger = logging.getLogger(__name__)

SLAB_INIT_FRACTION = 0.5  # rough prior: slab sits half way between room and flow target
SUN_WARMUP_HOURS = 24
SNAPSHOTS_KEPT = 96 * 2  # two days at one snapshot per 15 minutes


@dataclass(frozen=True)
class PlanSnapshot:
    computed_at: datetime
    mode: str
    plan: Plan
    extreme_threshold_sek_per_kwh: float
    outdoor_bias_c: float
    initial: RoomState
    model_status: dict  # which model parameters are priors vs fitted


def _floor_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


class PlanService:
    """Computes the price-driven plan. Shadow mode: never writes to Home Assistant."""

    def __init__(
        self,
        ha_client: HAClient,
        settings: PlanSettings,
        weather: OpenMeteoClient,
        pump_params: PumpParams = PRIOR_PUMP_PARAMS,
        room_params: RoomParams = PRIOR_ROOM_PARAMS,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        interval_seconds: int = 900,
    ):
        self._ha = ha_client
        self._settings = settings
        self._weather = weather
        self._pump = pump_params
        self._room = room_params
        self._now = now
        self._interval_seconds = interval_seconds
        self._mode = settings.mode
        self._history = PriceHistory(settings.extreme_history_days)
        self._snapshots: deque[PlanSnapshot] = deque(maxlen=SNAPSHOTS_KEPT)
        self._task: asyncio.Task | None = None

    def now(self) -> datetime:
        return self._now()

    def _read_float(self, entity_id: str) -> float:
        """Numeric state of an entity; raises SensorError for unavailable/unknown."""
        raw = self._ha.get_state(entity_id)["state"]
        try:
            return float(raw)
        except (TypeError, ValueError) as e:
            raise SensorError(f"{entity_id} has non-numeric state {raw!r}") from e

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Change mode in memory only; a restart returns to the configured mode."""
        if mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"mode must be one of {SUPPORTED_MODES}, got {mode!r}"
            )
        self._mode = mode

    def latest(self) -> PlanSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def history(self, hours: int) -> list[PlanSnapshot]:
        cutoff = self._now() - timedelta(hours=hours)
        return [s for s in self._snapshots if s.computed_at >= cutoff]

    def model_status(self) -> dict:
        return {
            "pump": {"source": "prior", "params": self._pump},
            "room": {"source": "prior", "params": self._room},
        }

    def run_once(self) -> PlanSnapshot | None:
        """Compute and store one plan; returns None when mode is off."""
        if self._mode == "off":
            return None
        now = self._now()
        s = self._settings

        attrs = self._ha.get_state(s.price_entity)["attributes"]
        periods = parse_nordpool_attributes(attrs)
        self._history.add(periods)
        self._history.prune(now)
        threshold = extreme_threshold(
            self._history.values(),
            s.extreme_price_percentile,
            s.extreme_price_min_sek_per_kwh,
        )
        hourly = hourly_prices(periods, _floor_hour(now))
        starts = [h for h, _ in hourly]
        prices = np.array([p for _, p in hourly])

        weather = {w.start: w for w in self._weather.fetch()}
        measured_outdoor = self._read_float(s.outdoor_entity)
        bias = measured_outdoor - self._forecast_at(weather, starts[0]).temperature_c
        t_out = np.array(
            [self._forecast_at(weather, h).temperature_c + bias for h in starts]
        )
        sun = np.array([self._forecast_at(weather, h).shortwave_wm2 for h in starts])
        warmup = [
            self._forecast_at(weather, starts[0] - timedelta(hours=k)).shortwave_wm2
            for k in range(SUN_WARMUP_HOURS, 0, -1)
        ]
        sun_filtered_now = float(
            sun_filtered_sequence(self._room, 0.0, np.array([*warmup, 0.0]))[-1]
        )

        room_c = float(np.mean([self._read_float(e) for e in s.room_temp_entities]))
        flow_now = self._read_float(s.flow_setpoint_entity)
        previous_rts = float(
            self._ha.get_climate_state(s.room_setpoint_entity)["target_temperature"]
        )
        initial = RoomState(
            room_c=room_c,
            slab_c=room_c + SLAB_INIT_FRACTION * (flow_now - room_c),
            sun_filtered_wm2=sun_filtered_now,
        )

        inputs = PlanInputs(
            hour_starts=starts,
            price_sek_per_kwh=prices,
            extreme=prices >= threshold,
            t_out_c=t_out,
            sun_wm2=sun,
            initial=initial,
            previous_rts_c=previous_rts,
        )
        result = plan(inputs, s, self._pump, self._room)
        snapshot = PlanSnapshot(
            computed_at=now,
            mode=self._mode,
            plan=result,
            extreme_threshold_sek_per_kwh=threshold,
            outdoor_bias_c=bias,
            initial=initial,
            model_status={"pump": "prior", "room": "prior"},
        )
        self._snapshots.append(snapshot)
        return snapshot

    @staticmethod
    def _forecast_at(
        weather: dict[datetime, WeatherHour], hour: datetime
    ) -> WeatherHour:
        try:
            return weather[hour]
        except KeyError as e:
            raise WeatherDataError(f"no weather forecast for {hour.isoformat()}") from e

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.run_once)
            except Exception:
                logger.exception("plan computation failed")
            await asyncio.sleep(self._interval_seconds)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_service.py -v`
Expected: 9 passed.

- [ ] **Step 5: Confirm the service never writes to Home Assistant**

Run: `grep -rn "set_temperature\|set_hvac_mode\|\.post(" core/theria/plan/`
Expected: no output.

- [ ] **Step 6: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/service.py tests/core/theria/plan/test_service.py
git commit -m "feat: add shadow-mode plan service"
```

---

### Task 8: Plan API

**Files:**
- Create: `backend/plan_api.py`, `tests/core/theria/plan/test_plan_api.py`

**Interfaces:**
- Consumes: `PlanService`, `PlanSnapshot`, `ConfigurationError`.
- Produces: `router` (prefix `/api`) with `GET /api/plan` (404 until a plan exists; body is the snapshot plus `age_seconds`), `GET /api/plan/history?hours=1..48` (compact summaries), `GET /api/model/health`, `POST /api/plan/mode` with body `{"mode": "off" | "shadow"}` (422 for anything else, including `active`); module variable `service: PlanService | None` set by `app.py`; dependency `get_service()` (503 when unset).

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_plan_api.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api
from tests.core.theria.plan.test_service import make_service


@pytest.fixture
def client(settings):
    service = make_service(settings)
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    return TestClient(app), service


def test_plan_is_404_until_one_is_computed(client):
    http, service = client
    assert http.get("/api/plan").status_code == 404
    service.run_once()
    body = http.get("/api/plan").json()
    assert len(body["plan"]["hours"]) == 36
    assert body["plan"]["hours"][0]["reason"]["binding"]
    assert body["mode"] == "shadow"
    assert body["age_seconds"] == 0


def test_history_lists_summaries(client):
    http, service = client
    service.run_once()
    body = http.get("/api/plan/history?hours=24").json()
    assert len(body) == 1
    assert set(body[0]) >= {
        "computed_at",
        "rts_now_c",
        "objective_sek",
        "baseline_objective_sek",
    }


def test_history_rejects_out_of_range_hours(client):
    http, _ = client
    assert http.get("/api/plan/history?hours=0").status_code == 422


def test_model_health_reports_priors(client):
    http, _ = client
    body = http.get("/api/model/health").json()
    assert body["models"]["pump"]["source"] == "prior"
    assert body["models"]["room"]["source"] == "prior"


def test_mode_can_be_switched_but_active_is_refused(client):
    http, service = client
    assert http.post("/api/plan/mode", json={"mode": "off"}).json() == {"mode": "off"}
    assert service.mode == "off"
    assert http.post("/api/plan/mode", json={"mode": "active"}).status_code == 422
    assert service.mode == "off"


def test_unconfigured_service_is_503():
    app = FastAPI()
    app.include_router(plan_api.router)
    plan_api.service = None
    assert TestClient(app).get("/api/plan").status_code == 503
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_plan_api.py -v`
Expected: collection error `ImportError` / `ModuleNotFoundError` for `backend.plan_api`.

- [ ] **Step 3: Implement**

Create `backend/plan_api.py`:

```python
"""REST endpoints for the price-driven heating plan (shadow mode)."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from core.theria.exceptions import ConfigurationError
from core.theria.plan.service import PlanService, PlanSnapshot

router = APIRouter(prefix="/api")

# Set by app.py during startup, like the other services in api.py.
service: PlanService | None = None


def get_service() -> PlanService:
    if service is None:
        raise HTTPException(status_code=503, detail="Plan service is not configured")
    return service


class ModeRequest(BaseModel):
    mode: str


def _summary(snapshot: PlanSnapshot) -> dict:
    first = snapshot.plan.hours[0]
    return {
        "computed_at": snapshot.computed_at,
        "mode": snapshot.mode,
        "rts_now_c": first.rts_c,
        "objective_sek": snapshot.plan.objective_sek,
        "baseline_objective_sek": snapshot.plan.baseline_objective_sek,
        "floor_violation_hours": snapshot.plan.floor_violation_hours,
    }


@router.get("/plan")
async def get_plan(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Latest plan with predictions, baseline and per-hour reasons."""
    snapshot = svc.latest()
    if snapshot is None:
        raise HTTPException(
            status_code=404, detail=f"No plan computed yet (mode: {svc.mode})"
        )
    body = jsonable_encoder(snapshot)
    body["age_seconds"] = (svc.now() - snapshot.computed_at).total_seconds()
    return body


@router.get("/plan/history")
async def get_plan_history(
    svc: Annotated[PlanService, Depends(get_service)],
    hours: Annotated[int, Query(ge=1, le=48)] = 24,
) -> Any:
    """Compact summaries of recent plans."""
    return jsonable_encoder([_summary(s) for s in svc.history(hours)])


@router.get("/model/health")
async def get_model_health(
    svc: Annotated[PlanService, Depends(get_service)],
) -> Any:
    """Model parameters and whether they are priors or fitted."""
    return jsonable_encoder({"mode": svc.mode, "models": svc.model_status()})


@router.post("/plan/mode")
async def set_plan_mode(
    svc: Annotated[PlanService, Depends(get_service)], request: ModeRequest
) -> dict[str, str]:
    """Switch between off and shadow. Active mode is not available yet."""
    try:
        svc.set_mode(request.mode)
    except ConfigurationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"mode": svc.mode}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_plan_api.py -v`
Expected: 6 passed (one `StarletteDeprecationWarning` about `httpx` is harmless).

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add backend/plan_api.py tests/core/theria/plan/test_plan_api.py
git commit -m "feat: add plan REST endpoints"
```

---

### Task 9: Wiring, configuration and live shadow check

**Files:**
- Modify: `backend/app.py`, `config.yaml`, `Dockerfile`, `documentation/TODO.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a running service in the add-on (shadow mode) and the endpoints under `/api/plan`.

- [ ] **Step 1: Add the `plan` options to `config.yaml`**

Under `options:` (after `heat_pump:` and before `zones:`) add:

```yaml
  plan:
    mode: "shadow"  # off | shadow. Shadow computes and logs plans, never writes to HA.
    price_entity: "sensor.nordpool_kwh_se4_sek_2_10_025"
    outdoor_entity: "sensor.outdoor"
    room_setpoint_entity: "climate.room_temp_setpoint"
    flow_setpoint_entity: "sensor.heating_setpoint"
    room_temp_entities:
      - "sensor.lk_bibliotek_temperature"
      - "sensor.lk_vardagsrum_temperature"
    latitude: 55.578
    longitude: 12.927
```

Under `schema:` (after `heat_pump:` and before `zones:`) add:

```yaml
  plan:
    mode: list(off|shadow)
    price_entity: str
    outdoor_entity: str
    room_setpoint_entity: str
    flow_setpoint_entity: str
    room_temp_entities:
      - str
    latitude: float
    longitude: float
```

- [ ] **Step 2: Verify the settings load from the real file**

Run:

```bash
.venv/bin/python -c "import yaml; from core.theria.plan.settings import PlanSettings; o = yaml.safe_load(open('config.yaml'))['options']; print(PlanSettings.from_options(o))"
```

Expected: a `PlanSettings(...)` line with `mode='shadow'`, `floor_c=18.0`, `target_c=19.5`, `ceiling_c=21.0` and no exception.

- [ ] **Step 3: Include `plan_api.py` in the Docker image**

In `Dockerfile` change the copy line to:

```dockerfile
COPY backend/app.py backend/api.py backend/plan_api.py backend/log_config.py backend/requirements.txt ./
```

- [ ] **Step 4: Start the service from `app.py`**

Add imports next to the existing core imports:

```python
import plan_api
from core.theria.exceptions import TheriaError
from core.theria.plan.service import PlanService
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import OpenMeteoClient
```

In `lifespan`, directly before `yield`, add:

```python
    # Start the price-driven plan service (shadow mode: never writes to HA)
    plan_service = None
    if ha_client:
        try:
            import yaml

            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            with open(config_path) as f:
                options = yaml.safe_load(f).get("options", {})
            plan_settings = PlanSettings.from_options(options)
            if plan_settings:
                plan_service = PlanService(
                    ha_client,
                    plan_settings,
                    OpenMeteoClient(plan_settings.latitude, plan_settings.longitude),
                )
                plan_api.service = plan_service
                await plan_service.start()
                logger.info(f"📈 Price plan service started (mode: {plan_settings.mode})")
        except TheriaError as e:
            logger.error(f"Price plan service not started: {e}")
```

In the shutdown part of `lifespan` (after the other `stop()` calls) add:

```python
    if plan_service:
        await plan_service.stop()
```

Next to `app.include_router(api_router)` add:

```python
app.include_router(plan_api.router)
```

- [ ] **Step 5: Lint, format, type check and run the whole suite**

Run:

```bash
.venv/bin/ruff check --fix backend core
.venv/bin/black backend core tests
.venv/bin/mypy core/theria/plan backend/plan_api.py
.venv/bin/python -m pytest -q
```

Expected: ruff and black clean; mypy reports no errors in the new files (the pre-existing `ha_client.py:53` error may appear); pytest shows all tests passing, including the 63 new ones and the existing thermal-learner tests.

- [ ] **Step 6: Live shadow check against Home Assistant (read-only)**

`.env` in the project root holds `HA_URL` and `HA_TOKEN`. In one terminal run `./dev-start.sh`. In another:

```bash
sleep 20
curl -s localhost:8081/api/plan | .venv/bin/python -m json.tool | head -80
curl -s localhost:8081/api/model/health | .venv/bin/python -m json.tool
```

Expected: `mode: "shadow"`; `plan.hours` runs from the current hour to the last hour with published prices (up to about 36 entries once tomorrow's prices are out); `rts_c` values between 15.0 and 22.0 with steps of at most 1.5 C; `reason.binding` set on every hour; `model_status` shows `prior` for pump and room; `age_seconds` small. If the response is a 404 or an error, read the server log: a `SensorError` names the unavailable entity, a `WeatherDataError` points to Open-Meteo.

Then confirm nothing was written to the pump: in Home Assistant open the history of `climate.room_temp_setpoint` and check it is unchanged since before the test.

- [ ] **Step 7: Record the follow-up plans in the TODO list**

Append to `documentation/TODO.md`:

```markdown
## Price plan (see docs/superpowers/specs/2026-10-11-price-plan-design.md)

- [ ] Plan 2: calibration (load InfluxDB history, fit pump and room models nightly), prediction scoring, persisted snapshots and mode.
- [ ] Plan 3: React frontend "Price and plan" page (copy bess-manager's Vite/React/Tailwind/Recharts setup, Docker Node stage, CI steps).
- [ ] Plan 4: active mode (executor, write guards, read-back, fallback, kill switch, controlled test steps).
- [ ] Plan 4 design: joint pump + LK decision (LK setpoint between target and ceiling per hour) and LK write guards.
- [ ] Re-check the 4 h sun lag with Open-Meteo's end-of-hour labels (may be about 3.5 h) and with stove evenings excluded.
- [ ] Existing `PriceOptimizer` parses `raw_today[].price` (SEK/MWh); retire it or fix it to use the SEK/kWh lists.
```

- [ ] **Step 8: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add backend/app.py config.yaml Dockerfile documentation/TODO.md
git commit -m "feat: run the price plan service in shadow mode"
```

---

## Self-review (against the spec)

- **Spec coverage:** prices incl. extreme rule (Task 2), pump model (3), room model with sun and valves (4), planner with floor/target/ceiling, comfort cost, rate limit and reasons (5), weather (6), shadow service with fallback to no write (7), API (8), config and deployment (9). Not in this plan, by design: calibration and scoring, persistence, frontend, active mode and guards (plans 2-4).
- **Placeholders:** none; every step has code or an exact command with expected output.
- **Type consistency:** names used across tasks match their definitions in the Interfaces blocks (`PlanInputs`, `HourPlan`, `Plan`, `PlanSnapshot`, `PumpParams`, `RoomParams`, `RoomState`, `sun_filtered_sequence`, `PlanService.now`).
