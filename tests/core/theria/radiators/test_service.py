import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import PriceDataError
from core.theria.plan.event_log import EventLog
from core.theria.plan.prices import PricePeriod
from core.theria.radiators.service import RadiatorService
from core.theria.radiators.settings import GuestRoom, RadiatorPrefs, RadiatorSettings
from core.theria.radiators.store import PrefsStore
from core.theria.sensor_guard import (
    STATUS_FAULT,
    STATUS_OK,
    GuardSettings,
    SensorReport,
    SensorSpec,
)

NOW = datetime(2026, 11, 2, 12, 20, tzinfo=UTC)  # Monday 13:20 local, the salon is open
VANTSIDAN, KLIPPSIDAN = "climate.vantsidan", "climate.klippsidan"
HALLWAY, KITCHEN = "climate.hallway", "climate.kitchen"
SENSORS = {
    "sensor.v": 20.5,
    "sensor.k": 20.0,
    "sensor.floor": 19.7,
    "sensor.ceiling": 19.1,
}


def settings() -> RadiatorSettings:
    guard = GuardSettings(
        sensors=tuple(
            SensorSpec(
                name=e, room="salon", temperature_entity=e, reference=e == "sensor.v"
            )
            for e in SENSORS
        ),
        notify=False,
    )
    return RadiatorSettings(
        salon_thermostats=(VANTSIDAN, KLIPPSIDAN),
        salon_sensors=("sensor.floor", "sensor.ceiling"),
        salon_sensor_guard=guard,
        outdoor_entity="sensor.outdoor",
        defaults=RadiatorPrefs(
            guest_rooms=(GuestRoom(HALLWAY, 19.5), GuestRoom(KITCHEN, 20.0)),
        ),
    )


def climate(setpoint, temperature, action="idle"):
    return {
        "state": "heat",
        "attributes": {
            "temperature": setpoint,
            "current_temperature": temperature,
            "hvac_action": action,
        },
    }


class FakeHA:
    def __init__(self):
        self.states = {
            VANTSIDAN: climate(21.0, 20.5, "heating"),
            KLIPPSIDAN: climate(21.0, 20.0, "heating"),
            HALLWAY: climate(19.5, 20.9),
            KITCHEN: climate(20.0, 20.9),
            "sensor.outdoor": {"state": "8.0", "attributes": {}},
        }
        self.writes = []

    def get_state(self, entity_id):
        if entity_id not in self.states:
            raise RuntimeError(f"no such entity {entity_id}")
        return self.states[entity_id]

    def set_temperature(self, entity_id, temperature):
        self.writes.append((entity_id, temperature))

    def call_service(self, domain, service, data):
        self.writes.append((domain, service, data))


class FakeGuard:
    def __init__(self, values=None):
        self.values = dict(SENSORS if values is None else values)

    def check(self):
        return [
            SensorReport(
                name=e,
                room="salon",
                temperature_entity=e,
                status=STATUS_OK if v is not None else STATUS_FAULT,
                issues=(),
                value_c=v,
                age_minutes=3.0,
                battery_percent=None,
                signal_dbm=None,
                offset_c=None,
                reference=e == "sensor.v",
                checked_at=NOW,
                usable=v is not None,
            )
            for e, v in self.values.items()
        ]


def prices(hours=30, cheap=0.5, dear_at=None):
    start = NOW.replace(minute=0)
    out = []
    for h in range(hours):
        p = 3.0 if dear_at is not None and h in dear_at else cheap
        out += [
            PricePeriod(start + timedelta(hours=h, minutes=15 * q), p) for q in range(4)
        ]
    return out


def make(tmp_path=None, ha=None, guard=None, periods=None, mode="shadow", log=None):
    ha = ha or FakeHA()
    cfg = settings()
    store = PrefsStore(cfg.defaults, None)
    if mode != "shadow":
        store.update({"mode": mode})
    service = RadiatorService(
        ha,
        cfg,
        store,
        guard or FakeGuard(),
        (lambda: periods) if not callable(periods) else periods,
        event_log=log,
        now=lambda: NOW,
    )
    return service, ha


def test_a_cycle_makes_a_snapshot_with_a_would_be_setpoint_per_thermostat():
    service, _ = make(periods=prices(dear_at={3, 4}))
    snap = service.run_once()
    assert snap is not None and snap is service.latest()
    assert set(snap.would_write) == {VANTSIDAN, KLIPPSIDAN, HALLWAY, KITCHEN}
    assert snap.would_write[VANTSIDAN] == snap.would_write[KLIPPSIDAN]
    assert snap.read[VANTSIDAN] == 21.0
    assert snap.salon.reading.mean_c == pytest.approx(19.825)
    assert snap.salon.hours and not snap.salon.unknown
    assert snap.difference[HALLWAY] == pytest.approx(snap.would_write[HALLWAY] - 19.5)


def test_a_cycle_never_writes_to_home_assistant():
    service, ha = make(periods=prices(dear_at={3}))
    service.run_once()
    assert ha.writes == []


def test_off_mode_returns_none_and_reads_nothing():
    service, ha = make(periods=prices(), mode="off")
    ha.get_state = lambda e: pytest.fail("read in off mode")
    assert service.run_once() is None
    assert service.latest() is None


def test_unknown_salon_holds_the_current_setpoints():
    guard = FakeGuard(
        {**SENSORS, "sensor.k": None, "sensor.floor": None, "sensor.ceiling": None}
    )
    service, _ = make(guard=guard, periods=prices(dear_at={3}))
    snap = service.run_once()
    assert snap.salon.unknown
    assert snap.would_write[VANTSIDAN] == 21.0
    assert snap.would_write[KLIPPSIDAN] == 21.0
    assert snap.salon.hours == []


def test_an_unavailable_thermostat_is_skipped_and_recorded():
    ha = FakeHA()
    ha.states[KITCHEN] = {"state": "unavailable", "attributes": {}}
    service, _ = make(ha=ha, periods=prices(dear_at={3}))
    snap = service.run_once()
    assert snap.availability[KITCHEN] is False
    assert KITCHEN not in snap.would_write
    assert HALLWAY in snap.would_write
    assert service.availability_share()[KITCHEN] == 0.0
    assert service.availability_share()[HALLWAY] == 1.0


def test_the_log_records_the_difference_between_current_and_would_be_setpoint(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    service, _ = make(periods=prices(dear_at={3}), log=log)
    snap = service.run_once()
    [event] = log.read(10)
    assert event["kind"] == "radiators" and event["action"] == "shadow"
    assert VANTSIDAN in event["detail"]
    assert f"{snap.would_write[HALLWAY]}" in event["detail"]


def test_missing_tomorrow_prices_still_give_a_plan():
    service, _ = make(periods=prices(hours=9))
    snap = service.run_once()
    assert len(snap.salon.hours) == 9


def test_cycle_survives_a_failing_price_feed():
    def failing():
        raise PriceDataError("no prices")

    service, _ = make(periods=failing)
    asyncio.run(service.cycle())
    assert service.latest() is None
    assert "no prices" in service.last_error


def test_an_unavailable_outdoor_sensor_reuses_the_last_value_and_says_so():
    service, ha = make(periods=prices(dear_at={3}))
    assert service.run_once().outdoor_stale is False
    ha.states["sensor.outdoor"] = {"state": "unavailable", "attributes": {}}
    snap = service.run_once()
    assert snap.outdoor_stale is True and snap.outdoor_c == 8.0


def test_an_unavailable_outdoor_sensor_without_history_fails_the_cycle_visibly():
    ha = FakeHA()
    ha.states["sensor.outdoor"] = {"state": "unavailable", "attributes": {}}
    service, _ = make(ha=ha, periods=prices())
    asyncio.run(service.cycle())
    assert service.latest() is None
    assert "outdoor" in service.last_error


def test_non_numeric_thermostat_attributes_do_not_fail_the_cycle():
    ha = FakeHA()
    ha.states[HALLWAY] = climate("x", "abc", "idle")
    ha.states[KITCHEN] = {"state": "heat"}  # no attributes at all
    service, _ = make(ha=ha, periods=prices(dear_at={3}))
    snap = service.run_once()
    assert snap.read[HALLWAY] is None
    assert snap.availability[KITCHEN] is False
    assert KITCHEN not in snap.would_write


def test_a_room_set_to_off_is_left_alone():
    ha = FakeHA()
    ha.states[KITCHEN] = {
        "state": "off",
        "attributes": {"temperature": None, "current_temperature": 18.0},
    }
    service, _ = make(ha=ha, periods=prices(dear_at={3}))
    snap = service.run_once()
    assert KITCHEN not in snap.would_write
    assert snap.left_alone == [KITCHEN]
    assert snap.availability[KITCHEN] is True


def with_targets(vantsidan, klippsidan):
    ha = FakeHA()
    ha.states[VANTSIDAN] = climate(vantsidan, 20.5)
    ha.states[KLIPPSIDAN] = climate(klippsidan, 20.0)
    service, _ = make(ha=ha, periods=prices())
    return service, ha


def test_the_salon_status_has_the_comfort_floor_when_netatmo_asks_for_comfort():
    service, _ = with_targets(21.0, 21.0)
    status = service.salon_status()
    assert status.floor_c == 19.0
    assert status.mean_c == pytest.approx(sum(SENSORS.values()) / len(SENSORS))


def test_the_salon_status_has_the_closed_floor_when_netatmo_stands_low():
    service, _ = with_targets(19.0, 19.0)
    assert service.salon_status().floor_c == 17.0


def test_netatmo_decides_and_not_the_hours_set_in_theria():
    service, _ = with_targets(16.0, 16.0)  # Sunday's eco hours; Theria's hours say open
    assert service.salon_status().floor_c == 17.0
    evening, _ = with_targets(21.0, 21.0)
    evening._now = lambda: (
        NOW + timedelta(hours=8)
    )  # 21:20 local, closed by Theria's hours
    assert evening.salon_status().floor_c == 19.0


def test_one_thermostat_asking_for_comfort_is_enough():
    service, _ = with_targets(19.0, 21.0)
    assert service.salon_status().floor_c == 19.0


def test_one_thermostat_that_cannot_be_read_does_not_matter():
    service, ha = with_targets(21.0, 21.0)
    del ha.states[VANTSIDAN]
    assert service.salon_status().floor_c == 19.0


def test_with_no_thermostat_readable_the_salon_is_unknown():
    service, ha = with_targets(21.0, 21.0)
    del ha.states[VANTSIDAN], ha.states[KLIPPSIDAN]
    assert service.salon_status().mean_c is None
