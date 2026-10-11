# Price Plan - Calibration, Scoring and Persistence (Plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the documented priors in the shadow-mode planner with models fitted to Johan's own InfluxDB history, refit them nightly with plausibility gates, score the room predictions against what happens, and keep the mode and fitted models across restarts.

**Architecture:** `calibration_data.py` assembles aligned hourly arrays from InfluxDB (through the existing `get_sensor_timeseries`) and Open-Meteo; `calibration.py` fits the pump (flow curve, electricity) and the ground-floor room model and rejects fits that are not trustworthy; `PlanService` applies accepted fits, refits on a schedule, records measured room temperatures for `ScoreBoard`, and persists mode and models through `PlanStateStore`. Still nothing is written to Home Assistant.

**Tech Stack:** Python 3.13, numpy, requests, FastAPI, pytest, ruff, black, mypy.

**Spec:** `docs/superpowers/specs/2026-10-11-price-plan-design.md` (sections 4-6 and 10). Previous plan: `docs/superpowers/plans/2026-10-11-price-plan-backend-shadow.md` (done). Measurements and caveats: `documentation/ivt-heat-pump.md`.

## Global Constraints

- Python 3.13, `black`, `ruff`, `mypy` (`disallow_untyped_defs`) clean on new code; `X | None` unions; no `hasattr` or silent fallbacks; specific exception classes from `core/theria/exceptions.py`; never match on exception message text.
- **No writes to Home Assistant or InfluxDB.** Only reads (`get_sensor_timeseries`, HA `get_state`/`get_climate_state`, Open-Meteo). The only file written is the plan state file (`/data/plan_state.json` in the add-on, `.plan_state.json` in development).
- `PlanSettings` keeps refusing `active` mode; persisted state with any mode other than `off` / `shadow` is an error.
- Fits are accepted only when plausible; a rejected fit keeps the previous models and is retried after 6 hours; accepted fits are refreshed every 24 hours.
- The 45 C flow cap (`max_flow_c`) is unchanged and applies to the plan, not to fitting.
- Hot-water hours (`binary_sensor.switch_valve_1` on) are excluded from heating fits.
- Tests: behaviour only; run with `.venv/bin/python -m pytest`.
- Commits: never without Johan's approval; no AI attribution.

## What the real data showed (prototype run, 2026-10-11, 1095 days)

These are facts about the fits this plan produces on Johan's data; the plan's tests use a synthetic house, not these numbers.

- **Pump, good:** heat curve `40.68 - 0.707 * T_out` at room setpoint 18.0 C (flow RMSE 0.54 C, 1747 hours); heating electricity `-0.49 - 0.177 * T_out + 0.189 * RTS` kWh/h (11 767 hours, RMSE 0.87; the counter has 1 kWh resolution).
- **Room, weak:** time constant of the slab 2 h and sun filter 0.5 h (both at the lowest grid values), `k_slab 0.0076`, `k_out 0.0071`, `k_sun 0.00068`; the 6 h temperature change is predicted with RMSE 0.68 C against 0.89 C for "no change" (23 % better) on 1485 windows. The model's winter equilibrium (40 C flow, 0 C outdoors) is 18.3 C, matching what the rooms did last winter.
- **Consequence:** with the fitted room model the planner sets the lowest room setpoint for the whole 30-hour horizon, because the model says the house is very slow (about 68 h time constant). That may be true or may be over-optimism; `ScoreBoard` is how we find out in shadow mode. **Do not use this for active control until the 6 h and 24 h prediction errors are small enough** (decide the threshold with Johan when real scores exist).
- The earlier "rooms lag the sun by 4 h" is the integrating effect of heating the room, not a filter lag: the fitted sun filter is short.

## Review Focus

Failure modes the spec implies that a first implementation tends to miss; each has a test in the task that owns the code.

1. **Counter resets and spikes** in the cumulative energy counter must become unknown hours, not huge or negative consumption. (Task 2)
2. **Hot-water hours** must not pollute the flow, energy or room fits. (Task 5)
3. **A fit that looks good but is physically wrong** (no room variation, house that would settle at 32 C in winter, constant room setpoint) is rejected. (Task 5)
4. **A damaged or unwritable state file** raises `StateStoreError`; a missing file is an empty state; persisted `active` is refused. (Task 6)
5. **A rejected refit keeps the previous models**, records the error, retries later, and the API says "previous models kept". (Task 7)

## File Structure

| File | Responsibility |
|---|---|
| `core/theria/exceptions.py` (modify) | Add `StateStoreError` |
| `core/theria/plan/settings.py` (modify) | Four more required entity settings |
| `core/theria/plan/series.py` (create) | Hourly alignment, gap filling, spike removal, counter differences |
| `core/theria/plan/pump_model.py` (modify) | Split fitting into flow curve and energy model |
| `core/theria/plan/room_model.py` (modify) | Gap-aware fit on 6 h changes, hourly LK setpoints, error scoring, equilibrium |
| `core/theria/plan/weather.py` (modify) | `OpenMeteoClient.history` (archive + recent) |
| `core/theria/plan/calibration_data.py` (create) | `HistorySource`, `InfluxHistorySource`, `CalibrationInputs`, loader |
| `core/theria/plan/calibration.py` (create) | `fit_pump`, `fit_room`, `Calibrator`, plausibility gates |
| `core/theria/plan/snapshot.py` (create) | `PlanSnapshot` (moved out of `service.py`) |
| `core/theria/plan/scoring.py` (create) | `ScoreBoard`, `LeadScore` |
| `core/theria/plan/state_store.py` (create) | `PlanStateStore`, `PlanState` |
| `core/theria/plan/service.py` (modify) | Refit, scheduling, persistence, scoring |
| `backend/plan_api.py` (modify) | Fit metadata and scores in `/api/model/health`; `POST /api/model/refit` |
| `backend/app.py` (modify) | Build calibrator and state store |
| `config.yaml`, `.gitignore` (modify) | New plan options; ignore `.plan_state.json` |
| `scripts/calibration_report.py` (create) | Read-only: fit on real history and print the result |
| `tests/core/theria/plan/` | New and updated tests, plus `world.py` (synthetic house) |

---

### Task 1: Entity settings for history

**Files:**
- Modify: `core/theria/plan/settings.py`, `config.yaml`, `tests/core/theria/plan/conftest.py`, `tests/core/theria/plan/test_settings.py`

**Interfaces:**
- Produces: `PlanSettings` gains required fields `room_thermostat_entities: tuple[str, ...]` (LK climate entities whose `temperature` attribute is the LK setpoint), `heating_mode_entity: str` (on = hot water, off = heating), `flow_measured_entity: str`, `energy_counter_entity: str` (cumulative heating electricity, kWh); `room_thermostat_entities` must not be empty.

- [ ] **Step 1: Update the tests first**

Replace `tests/core/theria/plan/conftest.py`:

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
        room_thermostat_entities=("climate.lk_bibliotek_thermostat",),
        heating_mode_entity="binary_sensor.switch_valve_1",
        flow_measured_entity="sensor.radiator_forward",
        energy_counter_entity="sensor.compr_cons_heating",
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

Replace `tests/core/theria/plan/test_settings.py`:

```python
import json

import pytest
import yaml

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import PlanSettings, load_plan_settings

BASE = {
    "price_entity": "sensor.price",
    "outdoor_entity": "sensor.outdoor",
    "room_setpoint_entity": "climate.room_temp_setpoint",
    "flow_setpoint_entity": "sensor.heating_setpoint",
    "room_temp_entities": ["sensor.a", "sensor.b"],
    "room_thermostat_entities": ["climate.a"],
    "heating_mode_entity": "binary_sensor.valve",
    "flow_measured_entity": "sensor.flow",
    "energy_counter_entity": "sensor.energy",
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
        {"room_thermostat_entities": []},
        {"max_flow_c": 29.0},  # below any useful flow temperature
        {"max_flow_c": 63.0},  # above the pump limit of 62 C
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


def test_load_prefers_addon_options_json_over_config_yaml(tmp_path):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"plan": {**BASE, "mode": "shadow"}}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"plan": {**BASE, "mode": "off"}}}))
    assert load_plan_settings(str(options), str(config)).mode == "shadow"


def test_load_falls_back_to_config_yaml_in_development(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"plan": BASE}}))
    settings = load_plan_settings(str(tmp_path / "missing.json"), str(config))
    assert settings.price_entity == "sensor.price"


def test_load_returns_none_when_no_file_exists_as_in_the_container_image(tmp_path):
    assert (
        load_plan_settings(str(tmp_path / "a.json"), str(tmp_path / "b.yaml")) is None
    )


@pytest.mark.parametrize("text", ["{not json", "[1, 2]"])
def test_load_raises_configuration_error_for_unreadable_options(tmp_path, text):
    options = tmp_path / "options.json"
    options.write_text(text)
    with pytest.raises(ConfigurationError):
        load_plan_settings(str(options), str(tmp_path / "b.yaml"))


def test_load_raises_configuration_error_for_invalid_yaml(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("options: [unclosed")
    with pytest.raises(ConfigurationError):
        load_plan_settings(str(tmp_path / "a.json"), str(config))


def test_max_flow_default_is_the_flow_temperature_lk_designed_for():
    assert PlanSettings.from_dict(BASE).max_flow_c == 45.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/core/theria/plan -q`
Expected: errors/failures with `TypeError: PlanSettings.__init__() got an unexpected keyword argument 'room_thermostat_entities'`.

- [ ] **Step 3: Implement**

Replace `core/theria/plan/settings.py`:

```python
"""Configuration for the price-driven heating plan."""

import json
import os
from dataclasses import dataclass, fields

import yaml

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
    room_thermostat_entities: tuple[str, ...]  # LK climate entities (setpoint history)
    heating_mode_entity: str  # on = hot water, off = heating
    flow_measured_entity: str
    energy_counter_entity: str  # cumulative heating electricity (kWh)
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
    # Highest pump flow temperature the plan may ask for: the 45 C flow temperature LK
    # designed the floor for (LK023393). Above it the floor is outside its design and the
    # parquet surface leaves the 25 C range LK calculated.
    max_flow_c: float = 45.0

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
        if not 30.0 <= self.max_flow_c <= 62.0:
            raise ConfigurationError(
                "max_flow_c must be between 30 and 62 C (the pump limit)"
            )
        if self.comfort_cost_sek_per_c_hour < 0:
            raise ConfigurationError("comfort_cost_sek_per_c_hour must not be negative")
        if not 0 < self.extreme_price_percentile < 100:
            raise ConfigurationError("extreme_price_percentile must be in (0, 100)")
        if not self.room_temp_entities or not self.room_thermostat_entities:
            raise ConfigurationError(
                "room_temp_entities and room_thermostat_entities must not be empty"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "PlanSettings":
        """Create from the `plan` section of config.yaml."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown plan settings: {sorted(unknown)}")
        converted = dict(data)
        for key in ("room_temp_entities", "room_thermostat_entities"):
            if key in converted:
                converted[key] = tuple(converted[key])
        try:
            return cls(**converted)
        except TypeError as e:
            raise ConfigurationError(f"invalid plan settings: {e}") from e

    @classmethod
    def from_options(cls, options: dict) -> "PlanSettings | None":
        """Settings from the add-on options, or None when no `plan` section exists."""
        section = options.get("plan")
        return None if section is None else cls.from_dict(section)


def load_plan_settings(options_path: str, config_path: str) -> PlanSettings | None:
    """Load the `plan` section: add-on options.json in production, config.yaml in development.

    Returns None when neither file exists (for example the container image has no
    config.yaml) or when the file has no `plan` section. Unreadable or invalid files
    raise ConfigurationError.
    """
    try:
        if os.path.exists(options_path):
            with open(options_path) as f:
                options = json.load(f)
        elif os.path.exists(config_path):
            with open(config_path) as f:
                options = yaml.safe_load(f).get("options", {})
        else:
            return None
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as e:
        raise ConfigurationError(f"cannot read plan configuration: {e}") from e
    if not isinstance(options, dict):
        raise ConfigurationError("plan configuration must be a mapping")
    return PlanSettings.from_options(options)
```

In `config.yaml`, in the `plan:` options add after `room_temp_entities` the keys below, and in the `plan:` schema the matching types:

```yaml
    room_thermostat_entities:
      - "climate.lk_bibliotek_thermostat"
      - "climate.lk_vardagsrum_thermostat"
    heating_mode_entity: "binary_sensor.switch_valve_1"
    flow_measured_entity: "sensor.radiator_forward"
    energy_counter_entity: "sensor.compr_cons_heating"
```

```yaml
    room_thermostat_entities:
      - str
    heating_mode_entity: str
    flow_measured_entity: str
    energy_counter_entity: str
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan -q`
Expected: all plan tests pass (the settings tests are updated, the rest unchanged).

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/settings.py config.yaml tests/core/theria/plan
git commit -m "feat: add history entity settings for plan calibration"
```

---

### Task 2: Series helpers

**Files:**
- Create: `core/theria/plan/series.py`, `tests/core/theria/plan/test_series.py`

**Interfaces:**
- Produces: `FloatArray`, `Point = tuple[datetime, float]`; `hour_floor(moment) -> datetime` (UTC); `hourly_grid(start, n_hours) -> list[datetime]`; `to_hourly(points, start, n_hours, how="mean"|"last", ffill_hours=0) -> FloatArray` (NaN where empty); `interpolate_gaps(values, max_gap_hours) -> FloatArray`; `despike(values, window, max_dev) -> FloatArray`; `hourly_energy_from_counter(counter_kwh, max_kwh_per_hour) -> FloatArray` (hour i = counter[i] - counter[i-1]; resets and spikes become NaN).

- [ ] **Step 1: Write the failing test**

Create `tests/core/theria/plan/test_series.py`:

```python
from datetime import UTC, datetime, timedelta, timezone

import numpy as np

from core.theria.plan.series import (
    despike,
    hourly_energy_from_counter,
    interpolate_gaps,
    to_hourly,
)

START = datetime(2026, 1, 10, 0, tzinfo=UTC)


def at(minutes: int, value: float):
    return (START + timedelta(minutes=minutes), value)


def test_mean_and_last_within_the_hour_and_nan_for_empty_hours():
    points = [at(5, 1.0), at(35, 3.0), at(125, 10.0)]
    mean = to_hourly(points, START, 4, how="mean")
    last = to_hourly(points, START, 4, how="last")
    assert mean[0] == 2.0 and last[0] == 3.0
    assert np.isnan(mean[1]) and mean[2] == 10.0 and np.isnan(mean[3])


def test_local_timestamps_land_in_the_right_utc_hour():
    local = timezone(timedelta(hours=2))
    point = (datetime(2026, 1, 10, 3, 30, tzinfo=local), 7.0)  # 01:30 UTC
    assert to_hourly([point], START, 3)[1] == 7.0


def test_forward_fill_carries_values_only_for_the_allowed_hours():
    out = to_hourly([at(10, 5.0)], START, 6, how="last", ffill_hours=2)
    assert list(out[:3]) == [5.0, 5.0, 5.0]
    assert np.isnan(out[3])


def test_non_finite_points_are_ignored():
    assert np.isnan(to_hourly([at(1, float("nan"))], START, 1)[0])


def test_interpolation_fills_short_gaps_only():
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])
    out = interpolate_gaps(values, max_gap_hours=2)
    assert out[1] == 2.0
    assert np.isnan(out[3:6]).all()


def test_despike_removes_isolated_garbage_and_keeps_real_steps():
    values = np.array([20.0, 20.1, 240.0, 20.2, 20.3, 20.4, 20.5])
    out = despike(values, window=5, max_dev=3.0)
    assert np.isnan(out[2])
    assert np.isfinite(np.delete(out, 2)).all()


def test_counter_differences_drop_resets_and_spikes():
    counter = np.array([100.0, 101.0, 103.0, 5.0, 6.0, 900.0, 7.0, 8.5])
    out = hourly_energy_from_counter(counter, max_kwh_per_hour=4.0)
    assert out[1] == 1.0 and out[2] == 2.0
    assert np.isnan(out[3])  # reset (negative difference)
    assert out[4] == 1.0
    assert np.isnan(out[5]) and np.isnan(out[6])  # spike up and back down
    assert out[7] == 1.5
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_series.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.series'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/series.py`:

```python
"""Aligning Home Assistant history to a regular hourly UTC grid and cleaning it."""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Point = tuple[datetime, float]


def hour_floor(moment: datetime) -> datetime:
    """The start of the UTC hour containing `moment` (must be timezone-aware)."""
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def hourly_grid(start: datetime, n_hours: int) -> list[datetime]:
    first = hour_floor(start)
    return [first + timedelta(hours=i) for i in range(n_hours)]


def to_hourly(
    points: list[Point],
    start: datetime,
    n_hours: int,
    how: str = "mean",
    ffill_hours: int = 0,
) -> FloatArray:
    """Resample irregular points to hourly values; NaN where there is no data.

    how: "mean" or "last" within the hour. `ffill_hours` carries the last value forward
    over that many empty hours (for sensors that only report on change).
    """
    if how not in ("mean", "last"):
        raise ValueError(f"how must be 'mean' or 'last', got {how!r}")
    first = hour_floor(start)
    sums = np.zeros(n_hours)
    counts = np.zeros(n_hours)
    last = np.full(n_hours, np.nan)
    for moment, value in sorted(points, key=lambda p: p[0]):
        index = int((moment.astimezone(UTC) - first) / timedelta(hours=1))
        if 0 <= index < n_hours and np.isfinite(value):
            sums[index] += value
            counts[index] += 1
            last[index] = value
    out = np.full(n_hours, np.nan)
    filled = counts > 0
    out[filled] = sums[filled] / counts[filled] if how == "mean" else last[filled]
    if ffill_hours > 0:
        gap = 0
        for i in range(1, n_hours):
            if np.isnan(out[i]) and np.isfinite(out[i - 1]) and gap < ffill_hours:
                out[i] = out[i - 1]
                gap += 1
            elif np.isfinite(out[i]):
                gap = 0
    return out


def interpolate_gaps(values: FloatArray, max_gap_hours: int) -> FloatArray:
    """Linear interpolation over NaN runs of at most `max_gap_hours` between finite values."""
    out = np.array(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(out))
    for left, right in pairwise(finite):
        gap = right - left - 1
        if 0 < gap <= max_gap_hours:
            out[left + 1 : right] = np.linspace(out[left], out[right], gap + 2)[1:-1]
    return out


def despike(values: FloatArray, window: int, max_dev: float) -> FloatArray:
    """NaN out values further than `max_dev` from the rolling median of `window` hours."""
    out = np.array(values, dtype=float)
    half = window // 2
    for i in range(len(values)):
        if not np.isfinite(values[i]):
            continue
        neighbourhood = values[max(0, i - half) : i + half + 1]
        neighbourhood = neighbourhood[np.isfinite(neighbourhood)]
        if (
            len(neighbourhood) >= 3
            and abs(values[i] - np.median(neighbourhood)) > max_dev
        ):
            out[i] = np.nan
    return out


def hourly_energy_from_counter(
    counter_kwh: FloatArray, max_kwh_per_hour: float
) -> FloatArray:
    """Hourly consumption from a cumulative counter sampled at the end of each hour.

    Hour i gets counter[i] - counter[i-1]. Counter spikes and resets show up as a
    negative or implausibly large difference and are NaN (the heat pump cannot draw
    more than `max_kwh_per_hour`).
    """
    out = np.full(len(counter_kwh), np.nan)
    diff = np.diff(counter_kwh)
    ok = np.isfinite(diff) & (diff >= 0.0) & (diff <= max_kwh_per_hour)
    out[1:][ok] = diff[ok]
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_series.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/series.py tests/core/theria/plan/test_series.py
git commit -m "feat: add hourly series helpers for calibration"
```

---

### Task 3: Model fitting building blocks

**Files:**
- Modify: `core/theria/plan/pump_model.py`, `core/theria/plan/room_model.py`, `tests/core/theria/plan/test_room_model.py`

**Interfaces:**
- Produces in `pump_model`: `fit_flow_curve(t_out_c, flow_target_c, rts_c, flow_per_rts=1.9) -> (intercept, slope, rts_ref)` (the room setpoint need not vary), `fit_energy_model(t_out_c, rts_c, energy_kwh_per_h) -> (intercept, per_outdoor, per_rts)` (needs at least 1 C spread in the room setpoint); `fit_pump_params` is unchanged for callers.
- Produces in `room_model`: `fit_room_params(room_c, flow_c, t_out_c, sun_wm2, lk_setpoint_c: float | array, valid=None) -> RoomParams` (NaN gaps allowed, 6 h changes, larger time-constant grids); `window_errors(params, room_c, flow_c, t_out_c, sun_wm2, lk_setpoint_c, valid=None) -> (model_rmse, persistence_rmse, windows)`; `equilibrium_room_c(params, flow_c, t_out_c) -> float`; `FIT_HORIZON_H = 6`.

- [ ] **Step 1: Add the failing test**

Append to `tests/core/theria/plan/test_room_model.py`:

```python
def test_equilibrium_is_where_floor_heat_balances_loss_to_outdoors():
    params = RoomParams(6.0, 0.05, 0.06, 0.0006, 3.0, 0.0)
    assert equilibrium_room_c(params, flow_c=40.0, t_out_c=0.0) == pytest.approx(
        0.05 * 40.0 / 0.11
    )
    colder = equilibrium_room_c(params, flow_c=40.0, t_out_c=-10.0)
    assert colder < equilibrium_room_c(params, flow_c=40.0, t_out_c=0.0)
```

and add `equilibrium_room_c` to the import list from `core.theria.plan.room_model` at the top of that file.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_room_model.py -q`
Expected: collection error `ImportError: cannot import name 'equilibrium_room_c'`.

- [ ] **Step 3: Implement**

Replace `core/theria/plan/pump_model.py`:

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


def fit_flow_curve(
    t_out_c: np.ndarray,
    flow_target_c: np.ndarray,
    rts_c: np.ndarray,
    flow_per_rts: float = DEFAULT_FLOW_PER_RTS,
) -> tuple[float, float, float]:
    """Fit the heat curve; returns (flow_intercept, flow_slope, rts_ref).

    The room setpoint does not need to vary: its effect is the fixed `flow_per_rts`.
    """
    if len(t_out_c) < 24 or not (len(t_out_c) == len(flow_target_c) == len(rts_c)):
        raise ModelFitError(
            "flow curve fit needs equally long arrays, at least 24 samples"
        )
    if np.ptp(t_out_c) < 3.0:
        raise ModelFitError(
            "flow curve fit needs at least 3 C spread in outdoor temperature"
        )
    rts_ref = float(np.mean(rts_c))
    adjusted = flow_target_c - flow_per_rts * (rts_c - rts_ref)
    slope, intercept = np.polyfit(t_out_c, adjusted, 1)
    return float(intercept), float(slope), rts_ref


def fit_energy_model(
    t_out_c: np.ndarray, rts_c: np.ndarray, energy_kwh_per_h: np.ndarray
) -> tuple[float, float, float]:
    """Fit heating electricity; returns (intercept, per outdoor C, per room setpoint C)."""
    n = len(t_out_c)
    if n < 24 or not (n == len(rts_c) == len(energy_kwh_per_h)):
        raise ModelFitError("energy fit needs equally long arrays, at least 24 samples")
    if np.ptp(t_out_c) < 3.0:
        raise ModelFitError(
            "energy fit needs at least 3 C spread in outdoor temperature"
        )
    if np.ptp(rts_c) < 1.0:
        raise ModelFitError("energy fit needs at least 1 C spread in room setpoint")
    design = np.column_stack([np.ones(n), t_out_c, rts_c])
    coef, *_ = np.linalg.lstsq(design, energy_kwh_per_h, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def fit_pump_params(
    t_out_c: np.ndarray,
    flow_target_c: np.ndarray,
    rts_c: np.ndarray,
    energy_kwh_per_h: np.ndarray,
    flow_per_rts: float = DEFAULT_FLOW_PER_RTS,
) -> PumpParams:
    """Fit both parts on the same hourly heating-only samples."""
    intercept, slope, rts_ref = fit_flow_curve(
        t_out_c, flow_target_c, rts_c, flow_per_rts
    )
    e0, e_out, e_rts = fit_energy_model(t_out_c, rts_c, energy_kwh_per_h)
    return PumpParams(
        flow_intercept=intercept,
        flow_slope=slope,
        rts_ref=rts_ref,
        flow_per_rts=flow_per_rts,
        energy_intercept=e0,
        energy_per_outdoor=e_out,
        energy_per_rts=e_rts,
    )
```

Replace `core/theria/plan/room_model.py`:

```python
"""Ground-floor room model: slab (floor) and room temperature with sun and LK valves."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from core.theria.exceptions import ModelFitError

Array = float | np.ndarray  # scalars and numpy arrays are both accepted
FloatArray = npt.NDArray[np.float64]

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


def valve_open(room_c: Array, lk_setpoint_c: Array) -> Array:
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
TAU_SLAB_GRID_H = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
SUN_LAG_GRID_H = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
FIT_HORIZON_H = (
    6  # hourly changes are dominated by sensor quantisation; fit 6 h changes
)


def _slab_and_sun(
    room_c: FloatArray,
    flow_c: FloatArray,
    sun_wm2: FloatArray,
    lk_setpoint_c: FloatArray,
    tau_slab_h: float,
    sun_lag_h: float,
) -> tuple[FloatArray, FloatArray]:
    """Teacher-forced slab temperature and filtered sun for the whole series."""
    n = len(room_c)
    slab = np.empty(n)
    sun_f = np.empty(n)
    slab[0] = room_c[0]
    sun_f[0] = sun_wm2[0]
    slab_gain = 1.0 - np.exp(-1.0 / tau_slab_h)
    sun_gain = 1.0 - np.exp(-1.0 / sun_lag_h)
    for i in range(n - 1):
        valve = valve_open(room_c[i], lk_setpoint_c[i])
        target = valve * flow_c[i] + (1.0 - valve) * room_c[i]
        slab[i + 1] = slab[i] + (target - slab[i]) * slab_gain
        sun_f[i + 1] = sun_f[i] + (sun_wm2[i] - sun_f[i]) * sun_gain
    return slab, sun_f


def _design(
    room_c: FloatArray,
    t_out_c: FloatArray,
    slab: FloatArray,
    sun_f: FloatArray,
    lk_setpoint_c: FloatArray,
    rows: slice,
) -> FloatArray:
    valve = valve_open(room_c[rows], lk_setpoint_c[rows])
    return np.column_stack(
        [
            valve * (slab[rows] - room_c[rows]),
            -(room_c[rows] - t_out_c[rows]),
            sun_f[rows],
            np.ones(len(room_c[rows])),
        ]
    )


class _Series:
    """Hourly series prepared for fitting: gaps filled for the recursion, rows marked valid."""

    def __init__(
        self,
        room_c: FloatArray,
        flow_c: FloatArray,
        t_out_c: FloatArray,
        sun_wm2: FloatArray,
        lk_setpoint_c: float | FloatArray,
        valid: npt.NDArray[np.bool_] | None,
    ):
        n = len(room_c)
        lk = np.broadcast_to(np.asarray(lk_setpoint_c, dtype=float), (n,))
        if not (len(flow_c) == len(t_out_c) == len(sun_wm2) == n):
            raise ModelFitError("room fit needs equally long arrays")
        finite = (
            np.isfinite(room_c)
            & np.isfinite(flow_c)
            & np.isfinite(t_out_c)
            & np.isfinite(sun_wm2)
            & np.isfinite(lk)
        )
        # a one-step row i needs hours i and i+1
        rows = finite[:-1] & finite[1:]
        if valid is not None:
            rows &= valid[:-1]
        self.rows = rows
        self.room = _hold(room_c)
        self.flow = _hold(flow_c)
        self.t_out = _hold(t_out_c)
        self.sun = _hold(sun_wm2)
        self.lk = _hold(np.array(lk))
        self.delta = np.diff(self.room)

    def usable(self) -> npt.NDArray[np.bool_]:
        mask = self.rows.copy()
        mask[:BURN_IN_H] = False
        return np.asarray(mask, dtype=bool)


def _hold(values: FloatArray) -> FloatArray:
    """Forward-fill NaN (leading NaN take the first finite value) for the recursion only."""
    out = np.array(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(out))
    if len(finite) == 0:
        raise ModelFitError("series has no finite values")
    out[: finite[0]] = out[finite[0]]
    for i in range(finite[0] + 1, len(out)):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out


def _windows(
    design: FloatArray, delta: FloatArray, usable: npt.NDArray[np.bool_], horizon_h: int
) -> tuple[FloatArray, FloatArray]:
    """Sums of `horizon_h` consecutive rows (design and delta) where every row is usable."""
    weight = usable.astype(float)
    cum_design = np.vstack(
        [np.zeros(design.shape[1]), np.cumsum(design * weight[:, None], axis=0)]
    )
    cum_delta = np.concatenate([[0.0], np.cumsum(delta * weight)])
    cum_ok = np.concatenate([[0.0], np.cumsum(weight)])
    ok = (cum_ok[horizon_h:] - cum_ok[:-horizon_h]) == horizon_h
    x = (cum_design[horizon_h:] - cum_design[:-horizon_h])[ok]
    y = (cum_delta[horizon_h:] - cum_delta[:-horizon_h])[ok]
    return x, y


def fit_room_params(
    room_c: FloatArray,
    flow_c: FloatArray,
    t_out_c: FloatArray,
    sun_wm2: FloatArray,
    lk_setpoint_c: float | FloatArray,
    valid: npt.NDArray[np.bool_] | None = None,
) -> RoomParams:
    """Fit on hourly data (room_c[i] is the temperature at the start of hour i).

    Gaps are NaN; rows touching a gap are left out of the regression. Teacher-forced
    one-step regression over a grid of slab and sun time constants. `lk_setpoint_c` may
    be one number or an hourly array (the LK setpoint has changed over time).
    """
    series = _Series(room_c, flow_c, t_out_c, sun_wm2, lk_setpoint_c, valid)
    usable = series.usable()
    if int(usable.sum()) < 7 * 24:
        raise ModelFitError("room fit needs at least one week of valid hourly samples")
    best: tuple[float, RoomParams] | None = None
    rows = slice(0, len(series.delta))
    for tau in TAU_SLAB_GRID_H:
        for lag in SUN_LAG_GRID_H:
            slab, sun_f = _slab_and_sun(
                series.room, series.flow, series.sun, series.lk, tau, lag
            )
            design = _design(series.room, series.t_out, slab, sun_f, series.lk, rows)
            x, y = _windows(design, series.delta, usable, FIT_HORIZON_H)
            if len(y) < 7 * 24:
                continue
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            if coef[0] < 0 or coef[1] < 0 or coef[2] < 0:
                continue
            sse = float(np.sum((x @ coef - y) ** 2))
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
            "no physically valid room model found (negative coefficients or too few windows)"
        )
    return best[1]


def window_errors(
    params: RoomParams,
    room_c: FloatArray,
    flow_c: FloatArray,
    t_out_c: FloatArray,
    sun_wm2: FloatArray,
    lk_setpoint_c: float | FloatArray,
    valid: npt.NDArray[np.bool_] | None = None,
) -> tuple[float, float, int]:
    """(model RMSE, persistence RMSE, windows) of the 6 h room temperature change."""
    series = _Series(room_c, flow_c, t_out_c, sun_wm2, lk_setpoint_c, valid)
    slab, sun_f = _slab_and_sun(
        series.room,
        series.flow,
        series.sun,
        series.lk,
        params.tau_slab_h,
        params.sun_lag_h,
    )
    design = _design(
        series.room, series.t_out, slab, sun_f, series.lk, slice(0, len(series.delta))
    )
    x, y = _windows(design, series.delta, series.usable(), FIT_HORIZON_H)
    if len(y) == 0:
        raise ModelFitError("no valid windows to score the room model")
    coef = np.array([params.k_slab, params.k_out, params.k_sun, params.drift_c_per_h])
    return (
        float(np.sqrt(np.mean((x @ coef - y) ** 2))),
        float(np.sqrt(np.mean(y**2))),
        len(y),
    )


def equilibrium_room_c(params: RoomParams, flow_c: float, t_out_c: float) -> float:
    """Room temperature the model settles at with open valves and no sun."""
    return (params.k_slab * flow_c + params.k_out * t_out_c + params.drift_c_per_h) / (
        params.k_slab + params.k_out
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_room_model.py tests/core/theria/plan/test_pump_model.py -q`
Expected: 11 passed (the existing fit-recovery tests still recover the generating parameters exactly).

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/pump_model.py core/theria/plan/room_model.py tests/core/theria/plan/test_room_model.py
git commit -m "feat: gap-aware model fitting on multi-hour changes"
```

---

### Task 4: Weather history

**Files:**
- Modify: `core/theria/plan/weather.py`

**Interfaces:**
- Produces: `OpenMeteoClient.history(start, stop) -> list[WeatherHour]` (archive API for days older than 6 days, forecast API `past_days` for newer ones); constants `OPEN_METEO_ARCHIVE_URL`, `ARCHIVE_DELAY_DAYS = 6`. Network code only; the parser is already covered by `test_weather.py`.

- [ ] **Step 1: Implement**

Replace `core/theria/plan/weather.py`:

```python
"""Open-Meteo forecast: outdoor temperature and solar irradiance per hour."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests

from core.theria.exceptions import WeatherDataError

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_DELAY_DAYS = (
    6  # the archive lags; newer days come from the forecast API (past_days)
)
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

    def history(self, start: datetime, stop: datetime) -> list[WeatherHour]:
        """Hourly weather for [start, stop): archive for older days, forecast API for recent ones."""
        today = datetime.now(UTC).date()
        archive_end = min(stop.date(), today - timedelta(days=ARCHIVE_DELAY_DAYS))
        hours: dict[datetime, WeatherHour] = {}
        if start.date() <= archive_end:
            for hour in self._get(
                OPEN_METEO_ARCHIVE_URL,
                {
                    "start_date": start.date().isoformat(),
                    "end_date": archive_end.isoformat(),
                },
            ):
                hours[hour.start] = hour
        recent_days = (
            today - max(start.date(), archive_end + timedelta(days=1))
        ).days + 1
        if stop.date() > archive_end and recent_days > 0:
            for hour in self._get(
                OPEN_METEO_URL, {"past_days": min(recent_days, 92), "forecast_days": 1}
            ):
                hours[hour.start] = hour
        return [hours[k] for k in sorted(hours)]

    def _get(self, url: str, extra: dict[str, str | int]) -> list[WeatherHour]:
        params: dict[str, str | float | int] = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": HOURLY_VARIABLES,
            "timezone": "UTC",
            **extra,
        }
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as e:
            raise WeatherDataError(f"Open-Meteo request failed: {e}") from e
        return parse_open_meteo(payload)
```

- [ ] **Step 2: Verify**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_weather.py -q && .venv/bin/mypy core/theria/plan/weather.py 2>&1 | grep -E "plan/weather" || echo mypy-clean`
Expected: 4 passed, `mypy-clean`.

Run the live check (needs network): `.venv/bin/python -c "from datetime import datetime, timedelta, UTC; from core.theria.plan.weather import OpenMeteoClient; now = datetime.now(UTC); h = OpenMeteoClient(55.578, 12.927).history(now - timedelta(days=30), now); print(len(h), h[0].start, h[-1].start)"`
Expected: about 720 hours covering the last 30 days, without gaps at the archive/recent boundary.

- [ ] **Step 3: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/weather.py
git commit -m "feat: fetch hourly weather history from Open-Meteo"
```

---

### Task 5: Calibration

**Files:**
- Create: `core/theria/plan/calibration_data.py`, `core/theria/plan/calibration.py`, `tests/core/theria/plan/world.py`, `tests/core/theria/plan/test_calibration.py`, `scripts/calibration_report.py`

**Interfaces:**
- Consumes: Tasks 1-4; `core.theria.influxdb_helper.get_sensor_timeseries(sensor_name, start, stop, timezone, domain, field_name) -> {"status", "message", "data": [(datetime, float)]}`.
- Produces: `HistorySource.series(entity_id, start, stop, field=None) -> list[Point]` (Protocol), `WeatherHistory.history(start, stop)` (Protocol), `InfluxHistorySource`, `CalibrationInputs` (aligned hourly arrays, `.tail(n_hours)`), `load_calibration_inputs(source, weather, settings, stop, days)`; `PumpFit`, `RoomFit`, `FittedModels(pump, room, fitted_at, window_start, window_end)`; `fit_pump(inputs)`, `fit_room(inputs)`, `Calibrator(source, weather, settings, days=1095).run(now) -> FittedModels` which raises `ModelFitError` when a fit is implausible (too few hours, implausible curve slope, energy that falls with room setpoint, mean consumption outside 0.2-4 kWh/h, room history without variation, room model that does not beat "no change" by 5 % on 6 h changes, or a house that would settle outside 16-20.5 C at 40 C flow and 0 C outdoors).

Timestamp convention: hour i of the history is labelled by its start; the temperature of hour i is taken as the temperature at the start of hour i.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/theria/plan/world.py` (a synthetic house with known parameters, served as history):

```python
"""A synthetic house with known pump and room parameters, served as history."""

from datetime import UTC, datetime, timedelta

import numpy as np

from core.theria.plan.pump_model import PumpParams, energy_kwh_per_hour, flow_target
from core.theria.plan.room_model import RoomParams, RoomState, simulate
from core.theria.plan.weather import WeatherHour

END = datetime(2026, 3, 1, 0, tzinfo=UTC)

TRUE_PUMP = PumpParams(
    flow_intercept=41.0,
    flow_slope=-0.7,
    rts_ref=18.0,
    flow_per_rts=1.9,
    energy_intercept=0.4,
    energy_per_outdoor=-0.15,
    energy_per_rts=0.12,
)
TRUE_ROOM = RoomParams(
    tau_slab_h=6.0,
    k_slab=0.05,
    k_out=0.06,
    k_sun=0.0006,
    sun_lag_h=3.0,
    drift_c_per_h=0.0,
)
LK_SETPOINT = 19.5
ENTITIES = {
    "outdoor": "sensor.outdoor",
    "room": "sensor.lk_bibliotek_temperature",
    "lk": "climate.lk_bibliotek_thermostat",
    "rts": "climate.room_temp_setpoint",
    "setpoint": "sensor.heating_setpoint",
    "valve": "binary_sensor.switch_valve_1",
    "flow": "sensor.radiator_forward",
    "counter": "sensor.compr_cons_heating",
}


class World:
    """Hourly truth for `days` days ending at END, plus the sources that serve it."""

    def __init__(
        self,
        days: int = 150,
        seed: int = 3,
        hot_water_hours: bool = False,
        room_params: RoomParams = TRUE_ROOM,
    ):
        rng = np.random.default_rng(seed)
        n = days * 24
        self.start = END - timedelta(hours=n)
        hrs = np.arange(n)
        self.t_out = (
            3
            + 5 * np.sin(2 * np.pi * hrs / 24 - 1)
            + 3 * np.sin(2 * np.pi * hrs / (24 * 9))
        )
        self.sun = np.clip(300 * np.sin(2 * np.pi * (hrs % 24 - 6) / 24), 0, None) * (
            0.4 + 0.6 * (np.sin(hrs / 41) > 0)
        )
        steps = rng.uniform(15.5, 21.5, size=n // 60 + 1)
        self.rts = np.repeat(steps, 60)[:n].round(1)
        self.setpoint = flow_target(TRUE_PUMP, self.t_out, self.rts)
        self.hot_water = np.zeros(n)
        if hot_water_hours:
            self.hot_water[(hrs % 24) == 4] = 1.0
        self.energy = energy_kwh_per_hour(TRUE_PUMP, self.t_out, self.rts)
        self.counter = 1000.0 + np.cumsum(np.round(self.energy * 4) / 4)
        self.flow = self.setpoint - 1.0
        end_of_hour = simulate(
            room_params,
            RoomState(19.0, 24.0, 0.0),
            self.flow,
            self.t_out,
            self.sun,
            LK_SETPOINT,
        )
        # the history is labelled by hour: hour i holds the temperature at the START of hour i
        self.room = np.concatenate([[19.0], end_of_hour[:-1]])

    def points(self, values: np.ndarray) -> list[tuple[datetime, float]]:
        return [
            (self.start + timedelta(hours=i, minutes=30), float(v))
            for i, v in enumerate(values)
        ]


class FakeSource:
    def __init__(
        self,
        world: World,
        overrides: dict[tuple[str, str | None], np.ndarray] | None = None,
    ):
        self._world = world
        w = world
        self._series = {
            (ENTITIES["outdoor"], None): w.t_out,
            (ENTITIES["room"], None): w.room,
            (ENTITIES["lk"], "temperature"): np.full(len(w.t_out), LK_SETPOINT),
            (ENTITIES["rts"], "temperature"): w.rts,
            (ENTITIES["setpoint"], None): w.setpoint,
            (ENTITIES["valve"], None): w.hot_water,
            (ENTITIES["flow"], None): w.flow,
            (ENTITIES["counter"], None): w.counter,
        }
        self._series.update(overrides or {})

    def series(self, entity_id, start, stop, field=None):
        return self._world.points(self._series[(entity_id, field)])


class FakeWeatherHistory:
    def __init__(self, world: World):
        self._world = world

    def history(self, start, stop):
        w = self._world
        return [
            WeatherHour(
                w.start + timedelta(hours=i), float(w.sun[i]), 50.0, float(w.t_out[i])
            )
            for i in range(len(w.sun))
        ]
```

Create `tests/core/theria/plan/test_calibration.py`:

```python
import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import load_calibration_inputs
from core.theria.plan.room_model import RoomParams

from .world import (
    END,
    ENTITIES,
    TRUE_PUMP,
    TRUE_ROOM,
    FakeSource,
    FakeWeatherHistory,
    World,
)


def calibrator(settings, world, overrides=None):
    return Calibrator(
        FakeSource(world, overrides), FakeWeatherHistory(world), settings, days=150
    )


def test_inputs_are_aligned_hour_by_hour_and_hot_water_hours_are_marked(settings):
    world = World(days=20, hot_water_hours=True)
    inputs = load_calibration_inputs(
        FakeSource(world), FakeWeatherHistory(world), settings, END, days=20
    )
    assert len(inputs.hours) == 20 * 24
    assert inputs.hours[0] == world.start and inputs.hours[-1].hour == 23
    assert np.allclose(inputs.t_out_c, world.t_out, atol=1e-9)
    assert np.allclose(inputs.sun_wm2, world.sun)
    assert inputs.hot_water[4] == 1.0 and inputs.hot_water[5] == 0.0
    # energy comes from counter differences: hour 0 has no previous counter value
    assert np.isnan(inputs.energy_kwh[0])
    assert np.allclose(inputs.energy_kwh[1:], np.round(world.energy[1:] * 4) / 4)


def test_calibration_recovers_the_pump_and_room_that_generated_the_history(settings):
    world = World()
    fitted = calibrator(settings, world).run(END)
    pump = fitted.pump.params
    assert pump.flow_slope == pytest.approx(TRUE_PUMP.flow_slope, abs=0.01)
    assert pump.flow_intercept + pump.flow_per_rts * (
        18.0 - pump.rts_ref
    ) == pytest.approx(TRUE_PUMP.flow_intercept, abs=0.05)
    assert pump.energy_per_rts == pytest.approx(TRUE_PUMP.energy_per_rts, abs=0.01)
    room = fitted.room.params
    assert (
        room.tau_slab_h == TRUE_ROOM.tau_slab_h
        and room.sun_lag_h == TRUE_ROOM.sun_lag_h
    )
    assert room.k_slab == pytest.approx(TRUE_ROOM.k_slab, rel=0.1)
    assert room.k_out == pytest.approx(TRUE_ROOM.k_out, rel=0.1)
    assert fitted.room.rmse_c < fitted.room.persistence_rmse_c
    assert fitted.window_end == inputs_end(world)


def inputs_end(world):
    return END - __import__("datetime").timedelta(hours=1)


def test_hot_water_hours_do_not_break_the_fit(settings):
    world = World(hot_water_hours=True)
    fitted = calibrator(settings, world).run(END)
    assert fitted.pump.params.flow_slope == pytest.approx(
        TRUE_PUMP.flow_slope, abs=0.02
    )


def test_a_constant_room_setpoint_cannot_identify_the_energy_model(settings):
    world = World()
    constant = np.full(len(world.rts), 18.0)
    cal = calibrator(settings, world, {(ENTITIES["rts"], "temperature"): constant})
    with pytest.raises(ModelFitError):
        cal.run(END)


def test_too_little_heating_history_is_rejected(settings):
    world = World(days=20)
    with pytest.raises(ModelFitError):
        Calibrator(FakeSource(world), FakeWeatherHistory(world), settings, days=20).run(
            END
        )


def test_a_flat_room_temperature_is_rejected_because_it_beats_nothing(settings):
    world = World()
    flat = np.full(len(world.room), 19.5)
    with pytest.raises(ModelFitError):
        calibrator(settings, world, {(ENTITIES["room"], None): flat}).run(END)


def test_a_room_model_that_cannot_reproduce_winter_temperatures_is_rejected(settings):
    # Floor heat dominates loss: the house would settle far above anything observed
    # (equilibrium about 32 C at 40 C flow), although it fits the history well.
    strong_floor = RoomParams(6.0, 0.12, 0.03, 0.0006, 3.0, 0.0)
    world = World(room_params=strong_floor)
    with pytest.raises(ModelFitError, match="winter"):
        calibrator(settings, world).run(END)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_calibration.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'core.theria.plan.calibration'`.

- [ ] **Step 3: Implement**

Create `core/theria/plan/calibration_data.py`:

```python
"""Hourly calibration data assembled from InfluxDB history and Open-Meteo."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.influxdb_helper import get_sensor_timeseries
from core.theria.plan.series import (
    FloatArray,
    Point,
    despike,
    hour_floor,
    hourly_energy_from_counter,
    hourly_grid,
    interpolate_gaps,
    to_hourly,
)
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import WeatherHour

MAX_ENERGY_KWH_PER_HOUR = (
    4.0  # the pump draws at most 3.6 kW (AirX installation manual)
)
SETPOINT_FFILL_HOURS = 24 * 60  # setpoints are only logged when they change
SMALL_GAP_HOURS = 3


class HistorySource(Protocol):
    def series(
        self, entity_id: str, start: datetime, stop: datetime, field: str | None = None
    ) -> list[Point]:
        """Raw (time, value) points for an entity; `field` picks an attribute."""
        ...


class WeatherHistory(Protocol):
    def history(self, start: datetime, stop: datetime) -> list[WeatherHour]: ...


class InfluxHistorySource:
    """HistorySource backed by the existing InfluxDB helper."""

    def series(
        self, entity_id: str, start: datetime, stop: datetime, field: str | None = None
    ) -> list[Point]:
        domain, name = entity_id.split(".", 1)
        result = get_sensor_timeseries(
            name, start, stop, domain=domain, field_name=field
        )
        if result["status"] == "error":
            # An entity with no points in the window is "no data", not a failure.
            if result["message"] == NO_DATA_MESSAGE:
                return []
            raise ModelFitError(f"InfluxDB query for {entity_id} failed")
        return [(moment, float(value)) for moment, value in result["data"]]


NO_DATA_MESSAGE = "No valid data points found"


@dataclass(frozen=True)
class CalibrationInputs:
    """Aligned hourly arrays; hour i covers [hours[i], hours[i] + 1 h). NaN = unknown."""

    hours: list[datetime]
    t_out_c: FloatArray
    heating_setpoint_c: FloatArray
    rts_c: FloatArray
    hot_water: FloatArray  # 1.0 while the pump makes hot water, 0.0 while heating
    energy_kwh: FloatArray
    room_c: FloatArray  # mean of the ground-floor room sensors
    flow_c: FloatArray
    lk_setpoint_c: FloatArray
    sun_wm2: FloatArray

    def tail(self, n_hours: int) -> "CalibrationInputs":
        """The last `n_hours` hours."""
        return CalibrationInputs(
            hours=self.hours[-n_hours:],
            t_out_c=self.t_out_c[-n_hours:],
            heating_setpoint_c=self.heating_setpoint_c[-n_hours:],
            rts_c=self.rts_c[-n_hours:],
            hot_water=self.hot_water[-n_hours:],
            energy_kwh=self.energy_kwh[-n_hours:],
            room_c=self.room_c[-n_hours:],
            flow_c=self.flow_c[-n_hours:],
            lk_setpoint_c=self.lk_setpoint_c[-n_hours:],
            sun_wm2=self.sun_wm2[-n_hours:],
        )


def _cleaned(
    points: list[Point], start: datetime, n: int, window: int, max_dev: float
) -> FloatArray:
    hourly = to_hourly(points, start, n, how="mean")
    return interpolate_gaps(despike(hourly, window, max_dev), SMALL_GAP_HOURS)


def _mean_of(columns: list[FloatArray]) -> FloatArray:
    stacked = np.vstack(columns)
    out = np.full(stacked.shape[1], np.nan)
    present = np.isfinite(stacked).any(axis=0)
    out[present] = np.nanmean(stacked[:, present], axis=0)
    return out


def load_calibration_inputs(
    source: HistorySource,
    weather: WeatherHistory,
    settings: PlanSettings,
    stop: datetime,
    days: int,
) -> CalibrationInputs:
    """Fetch and align `days` of history ending at `stop` (timezone-aware)."""
    n = days * 24
    end = hour_floor(stop)
    start = end - timedelta(hours=n)
    hours = hourly_grid(start, n)

    def last(entity: str, field: str | None, ffill: int) -> FloatArray:
        return to_hourly(
            source.series(entity, start, end, field),
            start,
            n,
            how="last",
            ffill_hours=ffill,
        )

    t_out = _cleaned(
        source.series(settings.outdoor_entity, start, end), start, n, 7, 6.0
    )
    rooms = [
        _cleaned(source.series(e, start, end), start, n, 7, 2.0)
        for e in settings.room_temp_entities
    ]
    lk = [
        last(e, "temperature", SETPOINT_FFILL_HOURS)
        for e in settings.room_thermostat_entities
    ]
    counter = last(settings.energy_counter_entity, None, SMALL_GAP_HOURS)
    sun = np.full(n, np.nan)
    by_start = {
        hour_floor(w.start): w.shortwave_wm2 for w in weather.history(start, end)
    }
    for i, hour in enumerate(hours):
        sun[i] = by_start.get(hour, np.nan)
    return CalibrationInputs(
        hours=hours,
        t_out_c=t_out,
        heating_setpoint_c=last(settings.flow_setpoint_entity, None, 48),
        rts_c=last(settings.room_setpoint_entity, "temperature", SETPOINT_FFILL_HOURS),
        hot_water=last(settings.heating_mode_entity, None, 12),
        energy_kwh=hourly_energy_from_counter(counter, MAX_ENERGY_KWH_PER_HOUR),
        room_c=_mean_of(rooms),
        flow_c=_cleaned(
            source.series(settings.flow_measured_entity, start, end), start, n, 7, 8.0
        ),
        lk_setpoint_c=_mean_of(lk),
        sun_wm2=sun,
    )
```

Create `core/theria/plan/calibration.py`:

```python
"""Fitting the pump and room models to history, with plausibility checks."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.calibration_data import (
    CalibrationInputs,
    HistorySource,
    WeatherHistory,
    load_calibration_inputs,
)
from core.theria.plan.pump_model import (
    DEFAULT_FLOW_PER_RTS,
    PumpParams,
    energy_kwh_per_hour,
    fit_energy_model,
    fit_flow_curve,
)
from core.theria.plan.room_model import (
    RoomParams,
    equilibrium_room_c,
    fit_room_params,
    window_errors,
)
from core.theria.plan.settings import PlanSettings

# The heat curve is changed by hand; only recent data describes it.
FLOW_WINDOW_DAYS = 240
FLOW_MAX_OUTDOOR_C = (
    12.0  # the range the plan works in; the curve bends in mild weather
)
ROOM_WINDOW_DAYS = 365
MIN_FLOW_ROWS = 200
MIN_ENERGY_ROWS = 500
MIN_FLOW_SLOPE, MAX_FLOW_SLOPE = -1.5, -0.2  # C of flow per C outdoors
MIN_MEAN_KWH, MAX_MEAN_KWH = 0.2, 4.0  # plausible mean heating electricity per hour
MIN_ROOM_SIGNAL_C = 0.1  # 6 h change the room must show for a fit to mean anything
# Winter sanity check: at 0 C outdoors and about 40 C flow (the measured heat curve) the
# ground-floor rooms settled between 16 and 19.5 C last winter.
WINTER_CHECK_T_OUT_C, WINTER_CHECK_FLOW_C = 0.0, 40.0
WINTER_EQUILIBRIUM_RANGE_C = (16.0, 20.5)
MIN_ROOM_SKILL = 0.05  # model must beat "same temperature next hour" by at least this


@dataclass(frozen=True)
class PumpFit:
    params: PumpParams
    flow_rows: int
    flow_rmse_c: float
    energy_rows: int
    energy_rmse_kwh: float


@dataclass(frozen=True)
class RoomFit:
    params: RoomParams
    rows: int
    rmse_c: float
    persistence_rmse_c: float


@dataclass(frozen=True)
class FittedModels:
    pump: PumpFit
    room: RoomFit
    fitted_at: datetime
    window_start: datetime
    window_end: datetime


def fit_pump(inputs: CalibrationInputs) -> PumpFit:
    """Fit flow curve (recent heating hours) and energy model (all heating hours)."""
    heating = (
        (inputs.hot_water == 0.0)
        & np.isfinite(inputs.t_out_c)
        & np.isfinite(inputs.heating_setpoint_c)
        & np.isfinite(inputs.rts_c)
        & (inputs.heating_setpoint_c > 15.0)
    )
    cold_enough = np.nan_to_num(inputs.t_out_c, nan=99.0) <= FLOW_MAX_OUTDOOR_C
    recent = np.zeros(len(heating), dtype=bool)
    recent[-FLOW_WINDOW_DAYS * 24 :] = True
    flow_rows = heating & recent & cold_enough
    if int(flow_rows.sum()) < MIN_FLOW_ROWS:
        raise ModelFitError(
            f"flow curve needs {MIN_FLOW_ROWS} heating hours below {FLOW_MAX_OUTDOOR_C} C "
            f"in the last {FLOW_WINDOW_DAYS} days, found {int(flow_rows.sum())}"
        )
    intercept, slope, rts_ref = fit_flow_curve(
        inputs.t_out_c[flow_rows],
        inputs.heating_setpoint_c[flow_rows],
        inputs.rts_c[flow_rows],
    )
    if not MIN_FLOW_SLOPE <= slope <= MAX_FLOW_SLOPE:
        raise ModelFitError(f"implausible heat curve slope {slope:.2f} C/C")
    energy_rows = heating & np.isfinite(inputs.energy_kwh)
    if int(energy_rows.sum()) < MIN_ENERGY_ROWS:
        raise ModelFitError(
            f"energy model needs {MIN_ENERGY_ROWS} heating hours with energy, "
            f"found {int(energy_rows.sum())}"
        )
    e0, e_out, e_rts = fit_energy_model(
        inputs.t_out_c[energy_rows],
        inputs.rts_c[energy_rows],
        inputs.energy_kwh[energy_rows],
    )
    if e_rts <= 0.0 or e_out >= 0.0:
        raise ModelFitError(
            f"implausible energy model (per room setpoint {e_rts:.3f}, per outdoor {e_out:.3f})"
        )
    params = PumpParams(
        flow_intercept=intercept,
        flow_slope=slope,
        rts_ref=rts_ref,
        flow_per_rts=DEFAULT_FLOW_PER_RTS,
        energy_intercept=e0,
        energy_per_outdoor=e_out,
        energy_per_rts=e_rts,
    )
    predicted = energy_kwh_per_hour(
        params, inputs.t_out_c[energy_rows], inputs.rts_c[energy_rows]
    )
    mean_kwh = float(np.mean(predicted))
    if not MIN_MEAN_KWH <= mean_kwh <= MAX_MEAN_KWH:
        raise ModelFitError(
            f"implausible mean heating electricity {mean_kwh:.2f} kWh/h"
        )
    flow_pred = (
        intercept
        + slope * inputs.t_out_c[flow_rows]
        + DEFAULT_FLOW_PER_RTS * (inputs.rts_c[flow_rows] - rts_ref)
    )
    return PumpFit(
        params=params,
        flow_rows=int(flow_rows.sum()),
        flow_rmse_c=float(
            np.sqrt(np.mean((flow_pred - inputs.heating_setpoint_c[flow_rows]) ** 2))
        ),
        energy_rows=int(energy_rows.sum()),
        energy_rmse_kwh=float(
            np.sqrt(np.mean((predicted - inputs.energy_kwh[energy_rows]) ** 2))
        ),
    )


def fit_room(inputs: CalibrationInputs) -> RoomFit:
    """Fit the ground-floor room model on heating hours (hot-water hours are excluded)."""
    inputs = inputs.tail(ROOM_WINDOW_DAYS * 24)
    valid = inputs.hot_water == 0.0
    args = (
        inputs.room_c,
        inputs.flow_c,
        inputs.t_out_c,
        inputs.sun_wm2,
        inputs.lk_setpoint_c,
    )
    params = fit_room_params(*args, valid=valid)
    model_rmse, persistence_rmse, rows = window_errors(params, *args, valid=valid)
    if persistence_rmse < MIN_ROOM_SIGNAL_C:
        raise ModelFitError(
            f"room temperature shows no variation ({persistence_rmse:.3f} C over 6 h)"
        )
    if model_rmse > persistence_rmse * (1.0 - MIN_ROOM_SKILL):
        raise ModelFitError(
            f"room model ({model_rmse:.3f} C) does not beat persistence "
            f"({persistence_rmse:.3f} C) on 6 h changes"
        )
    equilibrium = equilibrium_room_c(params, WINTER_CHECK_FLOW_C, WINTER_CHECK_T_OUT_C)
    low, high = WINTER_EQUILIBRIUM_RANGE_C
    if not low <= equilibrium <= high:
        raise ModelFitError(
            f"room model settles at {equilibrium:.1f} C in winter conditions, "
            f"outside the observed {low}-{high} C"
        )
    return RoomFit(params, rows, model_rmse, persistence_rmse)


class Calibrator:
    """Loads history and fits both models. Raises ModelFitError when a fit is not trustworthy."""

    def __init__(
        self,
        source: HistorySource,
        weather: WeatherHistory,
        settings: PlanSettings,
        days: int = 1095,
    ):
        self._source = source
        self._weather = weather
        self._settings = settings
        self._days = days

    def run(self, now: datetime) -> FittedModels:
        inputs = load_calibration_inputs(
            self._source, self._weather, self._settings, now, self._days
        )
        return FittedModels(
            pump=fit_pump(inputs),
            room=fit_room(inputs),
            fitted_at=now,
            window_start=inputs.hours[0],
            window_end=inputs.hours[-1],
        )
```

Create `scripts/calibration_report.py` (read-only helper for Johan and for the live check):

```python
"""Fit the plan models on real history and print the result (read-only).

Run from the repository root:  .venv/bin/python scripts/calibration_report.py [days]
Needs .env with HA_DB_URL / HA_DB_USER_NAME / HA_DB_PASSWORD (InfluxDB) and the
`plan` section of config.yaml.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import InfluxHistorySource
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import OpenMeteoClient


def main() -> None:
    load_dotenv()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    options = yaml.safe_load(Path("config.yaml").read_text())["options"]
    settings = PlanSettings.from_options(options)
    if settings is None:
        raise SystemExit("config.yaml has no plan section")
    calibrator = Calibrator(
        InfluxHistorySource(),
        OpenMeteoClient(settings.latitude, settings.longitude),
        settings,
        days=days,
    )
    fitted = calibrator.run(datetime.now(UTC))
    print(fitted)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_calibration.py -q`
Expected: 7 passed in a few seconds.

- [ ] **Step 5: Prove the winter check is what rejects the strong-floor house**

Run:

```bash
.venv/bin/python - <<'EOF'
import core.theria.plan.calibration as c
c.WINTER_EQUILIBRIUM_RANGE_C = (-100, 100)
import pytest, sys
sys.exit(pytest.main(["-q", "tests/core/theria/plan/test_calibration.py::test_a_room_model_that_cannot_reproduce_winter_temperatures_is_rejected", "-p", "no:cacheprovider"]))
EOF
```

Expected: the test FAILS (`DID NOT RAISE ModelFitError`) with the check disabled, which shows the check is what the test exercises.

- [ ] **Step 6: Run on the real history (read-only)**

Needs `HA_DB_URL`, `HA_DB_USER_NAME`, `HA_DB_PASSWORD` in `.env` (the local InfluxDB: `http://192.168.2.9:8086/api/v2/query`).

Run: `.venv/bin/python scripts/calibration_report.py`
Expected: a `FittedModels(...)` line in about 20-30 seconds, with a flow curve near `40.7 - 0.71 * T_out`, `energy_per_rts` between 0.1 and 0.25, and the room fit's `rmse_c` below `persistence_rmse_c`. A `ModelFitError` here is a finding, not a failure of the task: record the reason in the ledger.

- [ ] **Step 7: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/calibration_data.py core/theria/plan/calibration.py scripts/calibration_report.py tests/core/theria/plan/world.py tests/core/theria/plan/test_calibration.py
git commit -m "feat: fit pump and room models to InfluxDB history"
```

---

### Task 6: Snapshot, scoring and state store

**Files:**
- Create: `core/theria/plan/snapshot.py`, `core/theria/plan/scoring.py`, `core/theria/plan/state_store.py`, `tests/core/theria/plan/test_scoring.py`, `tests/core/theria/plan/test_state_store.py`
- Modify: `core/theria/exceptions.py`

**Interfaces:**
- Produces: `StateStoreError(TheriaError)`; `PlanSnapshot` (same fields as before, `model_status: dict[str, Any]`) now in `snapshot.py`; `ScoreBoard(max_readings=672)` with `record(moment, room_c)` and `scores(snapshots) -> list[LeadScore]`, `LeadScore(lead_h, n, mae_c, bias_c)` for leads 1, 6 and 24 h (bias = predicted minus actual; a prediction is scored against a reading within 10 minutes of the end of the predicted hour); `PlanState(mode: str | None, models: FittedModels | None)`; `PlanStateStore(path)` with `load()` (missing file -> empty state; damaged file or unsupported mode -> `StateStoreError`) and `save(state)` (atomic replace; unwritable -> `StateStoreError`).

- [ ] **Step 1: Write the failing tests**

Create `tests/core/theria/plan/test_scoring.py`:

```python
from datetime import UTC, datetime, timedelta

import numpy as np

from core.theria.plan.planner import HourPlan, Plan
from core.theria.plan.room_model import RoomState
from core.theria.plan.scoring import ScoreBoard
from core.theria.plan.snapshot import PlanSnapshot

T0 = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)


def snapshot(computed_at: datetime, predicted: list[float]) -> PlanSnapshot:
    hours = [
        HourPlan(
            start=T0 + timedelta(hours=i),
            rts_c=18.0,
            price_sek_per_kwh=1.0,
            extreme=False,
            floor_c=18.0,
            flow_target_c=36.0,
            energy_kwh=1.0,
            room_c_end=value,
            slab_c_end=28.0,
            energy_cost_sek=1.0,
            comfort_cost_sek=0.0,
            tail=False,
            reason={},
        )
        for i, value in enumerate(predicted)
    ]
    return PlanSnapshot(
        computed_at=computed_at,
        mode="shadow",
        plan=Plan(hours, hours, 0.0, 0.0, 0),
        extreme_threshold_sek_per_kwh=3.0,
        outdoor_bias_c=0.0,
        initial=RoomState(19.5, 28.0, 0.0),
        model_status={},
    )


def test_errors_are_grouped_by_lead_time_and_signed_as_predicted_minus_actual():
    # Plan made at 06:00 predicts the room at 07:00 (1 h) and 12:00 (6 h).
    plan = snapshot(T0, [19.8, 19.7, 19.6, 19.5, 19.4, 19.3])
    board = ScoreBoard()
    board.record(T0 + timedelta(hours=1, minutes=3), 19.5)  # 1 h: predicted 19.8
    board.record(T0 + timedelta(hours=6), 19.0)  # 6 h: predicted 19.3
    scores = {s.lead_h: s for s in board.scores([plan])}
    assert scores[1].n == 1 and np.isclose(scores[1].bias_c, 0.3)
    assert scores[6].n == 1 and np.isclose(scores[6].mae_c, 0.3)
    assert 24 not in scores


def test_predictions_without_a_nearby_reading_are_not_scored():
    plan = snapshot(T0, [19.8, 19.7])
    board = ScoreBoard()
    board.record(T0 + timedelta(hours=1, minutes=40), 19.0)
    assert board.scores([plan]) == []


def test_mean_absolute_error_ignores_sign():
    board = ScoreBoard()
    board.record(T0 + timedelta(hours=1), 19.0)
    plans = [snapshot(T0, [19.5]), snapshot(T0, [18.5])]
    (score,) = board.scores(plans)
    assert score.n == 2
    assert np.isclose(score.mae_c, 0.5) and np.isclose(score.bias_c, 0.0)
```

Create `tests/core/theria/plan/test_state_store.py`:

```python
from datetime import UTC, datetime

import pytest

from core.theria.exceptions import StateStoreError
from core.theria.plan.calibration import FittedModels, PumpFit, RoomFit
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS
from core.theria.plan.state_store import PlanState, PlanStateStore

NOW = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)


def fitted() -> FittedModels:
    return FittedModels(
        pump=PumpFit(PRIOR_PUMP_PARAMS, 400, 0.5, 5000, 0.8),
        room=RoomFit(PRIOR_ROOM_PARAMS, 1500, 0.6, 0.9),
        fitted_at=NOW,
        window_start=datetime(2023, 10, 12, tzinfo=UTC),
        window_end=NOW,
    )


def test_missing_file_is_an_empty_state(tmp_path):
    assert PlanStateStore(str(tmp_path / "state.json")).load() == PlanState(None, None)


def test_mode_and_models_survive_a_restart(tmp_path):
    path = str(tmp_path / "state.json")
    PlanStateStore(path).save(PlanState(mode="off", models=fitted()))
    loaded = PlanStateStore(path).load()
    assert loaded.mode == "off"
    assert loaded.models == fitted()


@pytest.mark.parametrize(
    "text",
    [
        "{broken",
        '{"mode": "active", "models": null}',
        '{"mode": "off", "models": {"pump": {}}}',
    ],
)
def test_damaged_files_raise_instead_of_being_ignored(tmp_path, text):
    path = tmp_path / "state.json"
    path.write_text(text)
    with pytest.raises(StateStoreError):
        PlanStateStore(str(path)).load()


def test_unwritable_location_raises(tmp_path):
    store = PlanStateStore(str(tmp_path / "missing_dir" / "state.json"))
    with pytest.raises(StateStoreError):
        store.save(PlanState(mode="shadow", models=None))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_scoring.py tests/core/theria/plan/test_state_store.py -q`
Expected: collection errors `ModuleNotFoundError` for `core.theria.plan.snapshot` / `state_store`.

- [ ] **Step 3: Implement**

Append to `core/theria/exceptions.py`:

```python
class StateStoreError(TheriaError):
    """The persisted plan state cannot be read or written."""

    pass
```

Create `core/theria/plan/snapshot.py`:

```python
"""One computed plan with the context it was computed in."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.theria.plan.planner import Plan
from core.theria.plan.room_model import RoomState


@dataclass(frozen=True)
class PlanSnapshot:
    computed_at: datetime
    mode: str
    plan: Plan
    extreme_threshold_sek_per_kwh: float
    outdoor_bias_c: float
    initial: RoomState
    model_status: dict[str, Any]  # which model parameters are priors vs fitted
```

Create `core/theria/plan/scoring.py`:

```python
"""Scoring of the room temperature predictions against what actually happened."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.theria.plan.snapshot import PlanSnapshot

LEADS_H = (1, 6, 24)
LEAD_TOLERANCE_H = {1: 0.5, 6: 0.5, 24: 1.0}
READING_TOLERANCE = timedelta(minutes=10)


@dataclass(frozen=True)
class LeadScore:
    """Prediction error of the room temperature `lead_h` hours ahead (C)."""

    lead_h: int
    n: int
    mae_c: float
    bias_c: float  # predicted minus actual


class ScoreBoard:
    """Collects measured room temperatures and scores snapshots against them."""

    def __init__(self, max_readings: int = 96 * 7):
        self._readings: deque[tuple[datetime, float]] = deque(maxlen=max_readings)

    def record(self, moment: datetime, room_c: float) -> None:
        self._readings.append((moment, room_c))

    def scores(self, snapshots: Sequence[PlanSnapshot]) -> list[LeadScore]:
        errors: dict[int, list[float]] = {lead: [] for lead in LEADS_H}
        readings = list(self._readings)
        for snapshot in snapshots:
            for hour in snapshot.plan.hours:
                end = hour.start + timedelta(hours=1)
                lead_h = (end - snapshot.computed_at) / timedelta(hours=1)
                lead = _bucket(lead_h)
                if lead is None:
                    continue
                actual = _reading_near(readings, end)
                if actual is not None:
                    errors[lead].append(hour.room_c_end - actual)
        return [
            LeadScore(
                lead_h=lead,
                n=len(values),
                mae_c=float(np.mean(np.abs(values))),
                bias_c=float(np.mean(values)),
            )
            for lead, values in errors.items()
            if values
        ]


def _bucket(lead_h: float) -> int | None:
    for lead in LEADS_H:
        if abs(lead_h - lead) <= LEAD_TOLERANCE_H[lead]:
            return lead
    return None


def _reading_near(
    readings: list[tuple[datetime, float]], moment: datetime
) -> float | None:
    close = [
        (abs(t - moment), value)
        for t, value in readings
        if abs(t - moment) <= READING_TOLERANCE
    ]
    return min(close)[1] if close else None
```

Create `core/theria/plan/state_store.py`:

```python
"""Persisting the plan mode and fitted models across restarts."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.theria.exceptions import StateStoreError
from core.theria.plan.calibration import FittedModels, PumpFit, RoomFit
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.room_model import RoomParams
from core.theria.plan.settings import SUPPORTED_MODES


@dataclass(frozen=True)
class PlanState:
    mode: str | None
    models: FittedModels | None


def _models_to_dict(m: FittedModels) -> dict[str, Any]:
    return {
        "pump": {
            "params": m.pump.params.__dict__,
            "flow_rows": m.pump.flow_rows,
            "flow_rmse_c": m.pump.flow_rmse_c,
            "energy_rows": m.pump.energy_rows,
            "energy_rmse_kwh": m.pump.energy_rmse_kwh,
        },
        "room": {
            "params": m.room.params.__dict__,
            "rows": m.room.rows,
            "rmse_c": m.room.rmse_c,
            "persistence_rmse_c": m.room.persistence_rmse_c,
        },
        "fitted_at": m.fitted_at.isoformat(),
        "window_start": m.window_start.isoformat(),
        "window_end": m.window_end.isoformat(),
    }


def _models_from_dict(d: dict[str, Any]) -> FittedModels:
    pump, room = d["pump"], d["room"]
    return FittedModels(
        pump=PumpFit(
            params=PumpParams(**pump["params"]),
            flow_rows=int(pump["flow_rows"]),
            flow_rmse_c=float(pump["flow_rmse_c"]),
            energy_rows=int(pump["energy_rows"]),
            energy_rmse_kwh=float(pump["energy_rmse_kwh"]),
        ),
        room=RoomFit(
            params=RoomParams(**room["params"]),
            rows=int(room["rows"]),
            rmse_c=float(room["rmse_c"]),
            persistence_rmse_c=float(room["persistence_rmse_c"]),
        ),
        fitted_at=datetime.fromisoformat(d["fitted_at"]),
        window_start=datetime.fromisoformat(d["window_start"]),
        window_end=datetime.fromisoformat(d["window_end"]),
    )


class PlanStateStore:
    """A small JSON file. A missing file is an empty state; a damaged file is an error."""

    def __init__(self, path: str):
        self._path = path

    def load(self) -> PlanState:
        if not os.path.exists(self._path):
            return PlanState(mode=None, models=None)
        try:
            with open(self._path) as f:
                raw = json.load(f)
            mode = raw.get("mode")
            if mode is not None and mode not in SUPPORTED_MODES:
                raise StateStoreError(f"unsupported mode {mode!r} in {self._path}")
            models = raw.get("models")
            return PlanState(
                mode=mode,
                models=None if models is None else _models_from_dict(models),
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ) as e:
            raise StateStoreError(f"cannot read plan state {self._path}: {e}") from e

    def save(self, state: PlanState) -> None:
        payload = {
            "mode": state.mode,
            "models": None if state.models is None else _models_to_dict(state.models),
        }
        temporary = f"{self._path}.tmp"
        try:
            with open(temporary, "w") as f:
                json.dump(payload, f, indent=1)
            os.replace(temporary, self._path)
        except OSError as e:
            raise StateStoreError(f"cannot write plan state {self._path}: {e}") from e
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_scoring.py tests/core/theria/plan/test_state_store.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/exceptions.py core/theria/plan/snapshot.py core/theria/plan/scoring.py core/theria/plan/state_store.py tests/core/theria/plan
git commit -m "feat: add prediction scoring and persisted plan state"
```

---

### Task 7: Service and API integration

**Files:**
- Modify: `core/theria/plan/service.py`, `backend/plan_api.py`, `tests/core/theria/plan/test_service.py`, `tests/core/theria/plan/test_plan_api.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: `PlanService(..., calibrator: Calibrator | None = None, state_store: PlanStateStore | None = None, refit_interval_hours=24.0, retry_interval_hours=6.0)`; `refit() -> FittedModels` (re-raises `TheriaError` after recording `last_fit_error` and scheduling a retry; keeps the previous models), `refit_due() -> bool`, `scores() -> list[LeadScore]`; `set_mode` persists; `model_status()` returns `{"pump": {"source": "prior"|"fitted", "params", "fit"}, "room": {...}, "fitted_at", "window", "last_fit_error"}`; `GET /api/model/health` adds `scores`; `POST /api/model/refit` returns the fitted models (422 `...; previous models kept` when rejected, 409 when calibration is not configured).

- [ ] **Step 1: Update the failing tests**

Replace `tests/core/theria/plan/test_service.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import (
    ConfigurationError,
    ModelFitError,
    PriceDataError,
    SensorError,
    WeatherDataError,
)
from core.theria.plan.calibration import FittedModels, PumpFit, RoomFit
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS, RoomParams
from core.theria.plan.service import PlanService
from core.theria.plan.state_store import PlanStateStore
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


def make_service(
    settings,
    attrs=None,
    weather=None,
    mode="shadow",
    clock=None,
    calibrator=None,
    state_store=None,
):
    from dataclasses import replace

    return PlanService(
        FakeHA(attrs or price_attributes()),
        replace(settings, mode=mode),
        weather or FakeWeather(),
        now=clock or (lambda: NOW),
        calibrator=calibrator,
        state_store=state_store,
    )


FITTED_PUMP = PumpParams(39.0, -0.8, 18.0, 1.9, 0.5, -0.1, 0.15)
FITTED_ROOM = RoomParams(4.0, 0.03, 0.04, 0.0004, 2.0, 0.0)


def fitted_models(fitted_at=NOW):
    return FittedModels(
        pump=PumpFit(FITTED_PUMP, 400, 0.5, 5000, 0.8),
        room=RoomFit(FITTED_ROOM, 1500, 0.6, 0.9),
        fitted_at=fitted_at,
        window_start=fitted_at - timedelta(days=365),
        window_end=fitted_at,
    )


class FakeCalibrator:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def run(self, now):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


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


def test_models_are_priors_until_a_fit_is_accepted(settings):
    status = make_service(settings).model_status()
    assert status["pump"]["source"] == "prior" and status["room"]["source"] == "prior"
    assert status["fitted_at"] is None and status["last_fit_error"] is None


def test_refit_switches_to_fitted_models_and_plans_with_them(settings):
    service = make_service(settings, calibrator=FakeCalibrator(fitted_models()))
    before = service.run_once().plan.hours[0].flow_target_c
    service.refit()
    status = service.model_status()
    assert (
        status["pump"]["source"] == "fitted" and status["pump"]["params"] == FITTED_PUMP
    )
    assert status["room"]["params"] == FITTED_ROOM and status["fitted_at"] == NOW
    snapshot = service.run_once()
    assert snapshot.model_status == {"pump": "fitted", "room": "fitted"}
    assert snapshot.plan.hours[0].flow_target_c != before


def test_a_rejected_fit_keeps_the_previous_models_and_schedules_a_retry(settings):
    current = [NOW]
    calibrator = FakeCalibrator(error=ModelFitError("not enough heating hours"))
    service = make_service(settings, calibrator=calibrator, clock=lambda: current[0])
    assert service.refit_due()
    with pytest.raises(ModelFitError):
        service.refit()
    assert service.model_status()["pump"]["source"] == "prior"
    assert "ModelFitError" in service.model_status()["last_fit_error"]
    assert not service.refit_due()  # retry only after the retry interval
    current[0] = NOW + timedelta(hours=7)
    assert service.refit_due()


def test_a_successful_fit_is_not_repeated_before_the_refit_interval(settings):
    current = [NOW]
    service = make_service(
        settings, calibrator=FakeCalibrator(fitted_models()), clock=lambda: current[0]
    )
    service.refit()
    current[0] = NOW + timedelta(hours=23)
    assert not service.refit_due()
    current[0] = NOW + timedelta(hours=25)
    assert service.refit_due()


def test_refit_without_a_calibrator_is_a_configuration_error(settings):
    with pytest.raises(ConfigurationError):
        make_service(settings).refit()
    assert not make_service(settings).refit_due()


def test_mode_and_fitted_models_survive_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first = make_service(
        settings,
        calibrator=FakeCalibrator(fitted_models()),
        state_store=PlanStateStore(path),
    )
    first.refit()
    first.set_mode("off")
    second = make_service(settings, state_store=PlanStateStore(path), mode="shadow")
    assert second.mode == "off"
    assert second.model_status()["pump"]["params"] == FITTED_PUMP
    assert second.model_status()["room"]["params"] != PRIOR_ROOM_PARAMS


def test_scores_compare_earlier_predictions_with_later_readings(settings):
    current = [NOW]
    service = make_service(settings, clock=lambda: current[0])
    service.run_once()
    current[0] = NOW + timedelta(
        minutes=45
    )  # 13:05, five minutes after the plan hour ends
    service.run_once()
    scores = {s.lead_h: s for s in service.scores()}
    assert scores[1].n >= 1
```

Replace `tests/core/theria/plan/test_plan_api.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api
from core.theria.exceptions import ModelFitError
from tests.core.theria.plan.test_service import (
    FakeCalibrator,
    fitted_models,
    make_service,
)


@pytest.fixture
def client(settings):
    service = make_service(settings, calibrator=FakeCalibrator(fitted_models()))
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


def test_model_health_includes_fit_metadata_and_scores(client):
    http, service = client
    service.run_once()
    service.refit()
    body = http.get("/api/model/health").json()
    assert body["models"]["pump"]["source"] == "fitted"
    assert body["models"]["pump"]["fit"]["flow_rows"] == 400
    assert body["models"]["last_fit_error"] is None
    assert isinstance(body["scores"], list)


def test_refit_endpoint_returns_the_fitted_models(client):
    http, _ = client
    response = http.post("/api/model/refit")
    assert response.status_code == 200
    assert response.json()["room"]["rmse_c"] == 0.6


def test_rejected_refit_is_422_and_says_the_old_models_are_kept(settings):
    service = make_service(
        settings, calibrator=FakeCalibrator(error=ModelFitError("too little data"))
    )
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    response = TestClient(app).post("/api/model/refit")
    assert response.status_code == 422
    assert "previous models kept" in response.json()["detail"]


def test_refit_without_calibration_is_409(settings):
    service = make_service(settings)
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    assert TestClient(app).post("/api/model/refit").status_code == 409
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/core/theria/plan/test_service.py tests/core/theria/plan/test_plan_api.py -q`
Expected: failures/errors such as `TypeError: PlanService.__init__() got an unexpected keyword argument 'calibrator'`.

- [ ] **Step 3: Implement**

Replace `core/theria/plan/service.py`:

```python
"""Plan service: reads live data, computes the plan and keeps the latest snapshots."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from core.theria.exceptions import (
    ConfigurationError,
    SensorError,
    TheriaError,
    WeatherDataError,
)
from core.theria.ha_client import HAClient
from core.theria.plan.calibration import Calibrator, FittedModels
from core.theria.plan.planner import PlanInputs, plan, sun_filtered_sequence
from core.theria.plan.prices import (
    PriceHistory,
    extreme_threshold,
    hourly_prices,
    parse_nordpool_attributes,
)
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, PumpParams
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS, RoomParams, RoomState
from core.theria.plan.scoring import LeadScore, ScoreBoard
from core.theria.plan.settings import SUPPORTED_MODES, PlanSettings
from core.theria.plan.snapshot import PlanSnapshot
from core.theria.plan.state_store import PlanState, PlanStateStore
from core.theria.plan.weather import OpenMeteoClient, WeatherHour

logger = logging.getLogger(__name__)

SLAB_INIT_FRACTION = 0.5  # rough prior: slab sits half way between room and flow target
SUN_WARMUP_HOURS = 24
SNAPSHOTS_KEPT = 96 * 2  # two days at one snapshot per 15 minutes


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
        calibrator: Calibrator | None = None,
        state_store: PlanStateStore | None = None,
        refit_interval_hours: float = 24.0,
        retry_interval_hours: float = 6.0,
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
        self._calibrator = calibrator
        self._state_store = state_store
        self._refit_interval = timedelta(hours=refit_interval_hours)
        self._retry_interval = timedelta(hours=retry_interval_hours)
        self._models: FittedModels | None = None
        self._last_fit_error: str | None = None
        self._next_refit_at: datetime | None = None
        self._scoreboard = ScoreBoard()
        if state_store is not None:
            state = state_store.load()
            if state.mode is not None:
                self._mode = state.mode
            if state.models is not None:
                self._apply_models(state.models)
                self._next_refit_at = state.models.fitted_at + self._refit_interval

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
        """Change the mode; it is persisted so that a restart keeps it."""
        if mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"mode must be one of {SUPPORTED_MODES}, got {mode!r}"
            )
        self._mode = mode
        self._save_state()

    def _save_state(self) -> None:
        if self._state_store is not None:
            self._state_store.save(PlanState(mode=self._mode, models=self._models))

    def _apply_models(self, models: FittedModels) -> None:
        self._models = models
        self._pump = models.pump.params
        self._room = models.room.params

    def refit(self) -> FittedModels:
        """Fit both models to history now; keeps the old models if the fit is rejected."""
        if self._calibrator is None:
            raise ConfigurationError("calibration is not configured")
        now = self._now()
        try:
            models = self._calibrator.run(now)
        except TheriaError as e:
            self._last_fit_error = f"{type(e).__name__}: {e}"
            self._next_refit_at = now + self._retry_interval
            raise
        self._apply_models(models)
        self._last_fit_error = None
        self._next_refit_at = now + self._refit_interval
        self._save_state()
        return models

    def refit_due(self) -> bool:
        return self._calibrator is not None and (
            self._next_refit_at is None or self._now() >= self._next_refit_at
        )

    def latest(self) -> PlanSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def history(self, hours: int) -> list[PlanSnapshot]:
        cutoff = self._now() - timedelta(hours=hours)
        return [s for s in self._snapshots if s.computed_at >= cutoff]

    def model_status(self) -> dict[str, Any]:
        fitted = self._models
        return {
            "pump": {
                "source": "prior" if fitted is None else "fitted",
                "params": self._pump,
                "fit": None if fitted is None else fitted.pump,
            },
            "room": {
                "source": "prior" if fitted is None else "fitted",
                "params": self._room,
                "fit": None if fitted is None else fitted.room,
            },
            "fitted_at": None if fitted is None else fitted.fitted_at,
            "window": (
                None
                if fitted is None
                else {"start": fitted.window_start, "end": fitted.window_end}
            ),
            "last_fit_error": self._last_fit_error,
        }

    def scores(self) -> list[LeadScore]:
        """Room temperature prediction error by lead time, from the stored snapshots."""
        return self._scoreboard.scores(list(self._snapshots))

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
        self._scoreboard.record(now, room_c)
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
            model_status={
                "pump": "prior" if self._models is None else "fitted",
                "room": "prior" if self._models is None else "fitted",
            },
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
            if self.refit_due():
                try:
                    await asyncio.to_thread(self.refit)
                    logger.info("plan models refitted")
                except TheriaError:
                    logger.exception(
                        "plan model refit rejected; keeping previous models"
                    )
            try:
                await asyncio.to_thread(self.run_once)
            except Exception:
                logger.exception("plan computation failed")
            await asyncio.sleep(self._interval_seconds)
```

Replace `backend/plan_api.py`:

```python
"""REST endpoints for the price-driven heating plan (shadow mode)."""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from core.theria.exceptions import ConfigurationError, TheriaError
from core.theria.plan.service import PlanService
from core.theria.plan.snapshot import PlanSnapshot

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
    return jsonable_encoder(
        {"mode": svc.mode, "models": svc.model_status(), "scores": svc.scores()}
    )


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


@router.post("/model/refit")
async def refit_models(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Fit the pump and room models to history now (takes several seconds)."""
    try:
        models = await asyncio.to_thread(svc.refit)
    except ConfigurationError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except TheriaError as e:
        raise HTTPException(
            status_code=422, detail=f"{type(e).__name__}: {e}; previous models kept"
        ) from e
    return jsonable_encoder(models)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/core/theria/plan -q`
Expected: all plan tests pass.

- [ ] **Step 5: Confirm nothing writes to Home Assistant**

Run: `grep -rn "set_temperature\|set_hvac_mode\|\.post(" core/theria/plan/ || echo none`
Expected: `none`.

- [ ] **Step 6: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add core/theria/plan/service.py backend/plan_api.py tests/core/theria/plan
git commit -m "feat: refit, score and persist plan models in the service"
```

---

### Task 8: Wiring, ignore file and live check

**Files:**
- Modify: `backend/app.py`, `.gitignore`, `documentation/TODO.md`, `documentation/ivt-heat-pump.md`

- [ ] **Step 1: Build the calibrator and state store in `app.py`**

Add imports next to the other plan imports:

```python
from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import InfluxHistorySource
from core.theria.plan.state_store import PlanStateStore
```

Replace the `if plan_settings:` block in `lifespan` with:

```python
            if plan_settings:
                weather_client = OpenMeteoClient(
                    plan_settings.latitude, plan_settings.longitude
                )
                state_path = (
                    "/data/plan_state.json"
                    if os.path.isdir("/data")
                    else os.path.join(
                        os.path.dirname(__file__), "..", ".plan_state.json"
                    )
                )
                plan_service = PlanService(
                    ha_client,
                    plan_settings,
                    weather_client,
                    calibrator=Calibrator(
                        InfluxHistorySource(), weather_client, plan_settings
                    ),
                    state_store=PlanStateStore(state_path),
                )
                plan_api.service = plan_service
                await plan_service.start()
                logger.info(
                    f"📈 Price plan service started (mode: {plan_settings.mode})"
                )
```

Append `.plan_state.json` to `.gitignore`.

- [ ] **Step 2: Lint, format, type check, whole suite**

Run:

```bash
.venv/bin/ruff check --fix backend core scripts
.venv/bin/black backend core scripts tests
.venv/bin/mypy core/theria/plan backend/plan_api.py scripts/calibration_report.py
.venv/bin/python -m pytest -q
```

Expected: ruff and black clean; mypy shows nothing for the new files (a pre-existing `ha_client.py:53` error may appear); the whole suite passes (135 tests at the time of writing).

- [ ] **Step 3: Live check against the real system (read-only)**

Run `./dev-start.sh` in one terminal, then:

```bash
sleep 60
curl -s -X POST localhost:8081/api/model/refit | .venv/bin/python -m json.tool | head -40
curl -s localhost:8081/api/model/health | .venv/bin/python -m json.tool | head -60
curl -s localhost:8081/api/plan | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['model_status'], len(d['plan']['hours']))"
```

Expected: the refit returns `pump` and `room` fits; health shows `source: "fitted"` for both and `scores: []` at first (scores appear after plans at 1, 6 and 24 hours have a reading to compare with); the plan reports `model_status` `{'pump': 'fitted', 'room': 'fitted'}`. Stop the server (`pkill -f "uvicorn.*app:app"; pkill -f watchmedo`) and check that `.plan_state.json` exists and `climate.room_temp_setpoint` in Home Assistant is unchanged.

- [ ] **Step 4: Record the findings**

Add to `documentation/ivt-heat-pump.md` a section "Fitted models (calibration, 2026-10-11)" with the numbers from "What the real data showed" above (copy the values the live run actually produced if they differ) and the caveat that the room fit is weakly identified. Replace the TODO item "Plan 2: calibration ..." in `documentation/TODO.md` with:

```markdown
- [x] Plan 2: calibration, scoring and persistence (nightly refit, ScoreBoard, persisted mode and models).
- [ ] Watch `/api/model/health` scores in shadow mode for a few days; agree with Johan what 6 h and 24 h room prediction error is good enough before plan 4.
- [ ] Improve the room model identification: sensors are quantised to 0.1 C, hot-water hours remove many windows, stove evenings add unexplained heat; consider excluding stove evenings and a free-run (multi-step) fit.
- [ ] Room model: fitted slab and sun time constants sit at the lowest grid values; revisit the model structure once scores exist.
```

- [ ] **Step 5: Commit (only after Johan approves)**

Project rule: never commit without approval, no AI attribution in the message.

```bash
git add backend/app.py .gitignore documentation
git commit -m "feat: run plan calibration and persistence in the add-on"
```

---

## Amendments after the final review (applied; the code in the repo is authoritative)

The embedded code above is the version reviewed first. The final review found five Important issues, fixed with
tests written first, so the repository differs from the listings in these ways:

1. **Hot water as a share of the hour.** `series.step_fraction` turns the valve's change events into the share of each
   hour the pump made hot water (`CalibrationInputs.hot_water` is now 0-1, heating hours need exactly 0); in the room
   fit hot-water time counts as closed valves (`hot_water` argument of `fit_room_params`, `window_errors`,
   `free_run_errors`; `step(..., valve_scale)`).
2. **Honest room gate.** `free_run_errors` runs the model freely 6 h and 24 h ahead; `fit_room` fits on all but the newest
   45 days, requires it to beat "no change" at both horizons on the held-out days, then refits on all data.
   `RoomFit` carries the held-out errors. `window_errors` is kept for diagnostics.
3. **Partial acceptance.** `FittedModels.room` may be `None` with `room_error` set: the pump fit is applied, the room model
   stays on the prior (or on the previously accepted fit); `model_status()["room"]` reports `fit_error`.
4. **No-data by reason code.** `influxdb_helper` adds `reason: "no_data"` to its no-data results and
   `InfluxHistorySource` checks it instead of message text.
5. **Supervisor cycle.** `PlanService.cycle()` survives any refit error; `.gitignore` entries fixed.

## Self-review (against the spec)

- **Spec coverage:** nightly fitting of pump and room models from InfluxDB (Tasks 3-5, 7), plausibility gates and fallback to priors (5, 7), prediction scoring by lead time (6, 7), persistence of mode and models (6, 7), API exposure (7). Out of scope by design: using the fitted models for control (plan 4), the frontend (plan 3), a free-run multi-step fit (TODO).
- **Placeholders:** none; every step has code or an exact command with expected output.
- **Type consistency:** `FittedModels`, `PumpFit`, `RoomFit`, `PlanSnapshot`, `LeadScore`, `PlanState` are defined once in the Interfaces blocks and used with the same field names in later tasks.
