from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.lk_executor import LKExecutor

NOW = datetime(2026, 11, 3, 1, 20, tzinfo=UTC)
THERMOSTATS = ("climate.lk_a", "climate.lk_b")


class FakeLK:
    def __init__(self, value=19.5):
        self.values = dict.fromkeys(THERMOSTATS, value)
        self.writes = []
        self.fail = set()
        self.ignore = set()  # accepts the call but keeps the old value

    def get_climate_state(self, entity_id):
        return {"target_temperature": self.values[entity_id]}

    def set_temperature(self, entity_id, temperature):
        if entity_id in self.fail:
            raise RuntimeError("unreachable")
        self.writes.append((entity_id, temperature))
        if entity_id not in self.ignore:
            self.values[entity_id] = temperature


@pytest.fixture
def lk_settings(settings):
    from dataclasses import replace

    return replace(settings, room_thermostat_entities=THERMOSTATS)


def make(lk_settings, ha=None, clock=None):
    ha = ha or FakeLK()
    current = [NOW]
    return LKExecutor(ha, lk_settings, now=clock or (lambda: current[0])), ha, current


def test_the_thermostats_go_up_while_the_pump_setpoint_is_above_the_base(lk_settings):
    executor, ha, _ = make(lk_settings)
    result = executor.execute(pump_setpoint_c=19.0)
    assert result.action == "wrote" and result.target_c == 21.0
    assert ha.values == {"climate.lk_a": 21.0, "climate.lk_b": 21.0}


def test_they_go_back_to_the_base_when_the_pump_is_at_or_below_its_base(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(21.0))
    assert executor.execute(pump_setpoint_c=18.0).action == "wrote"
    assert set(ha.values.values()) == {19.5}


def test_nothing_is_written_when_they_are_already_right(lk_settings):
    executor, ha, _ = make(lk_settings)
    assert executor.execute(pump_setpoint_c=17.0).action == "kept"
    assert ha.writes == []


def test_a_hand_change_pauses_the_lever_for_a_while(lk_settings):
    executor, ha, now = make(lk_settings)
    executor.execute(pump_setpoint_c=19.0)
    ha.values["climate.lk_a"] = 20.0  # Johan changes one thermostat
    now[0] += timedelta(minutes=15)
    assert executor.execute(pump_setpoint_c=17.0).action == "paused"
    now[0] += timedelta(hours=7)
    assert executor.execute(pump_setpoint_c=17.0).action == "wrote"


def test_one_unreachable_thermostat_does_not_stop_the_other(lk_settings):
    executor, ha, _ = make(lk_settings)
    ha.fail.add("climate.lk_a")
    result = executor.execute(pump_setpoint_c=19.0)
    assert result.action == "failed" and ha.values["climate.lk_b"] == 21.0


def test_a_write_that_is_not_accepted_is_reported_not_assumed(lk_settings):
    executor, ha, _ = make(lk_settings)
    ha.ignore.add("climate.lk_b")
    assert executor.execute(pump_setpoint_c=19.0).action == "failed"


def test_restore_writes_the_base_everywhere_and_reports_failure(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(21.0))
    assert executor.restore_base().action == "wrote"
    assert set(ha.values.values()) == {19.5}
    ha.values["climate.lk_a"] = 21.0
    ha.fail.add("climate.lk_a")
    assert executor.restore_base().action == "failed"


def test_restore_overrides_a_manual_pause(lk_settings):
    executor, ha, _ = make(lk_settings)
    executor.execute(pump_setpoint_c=19.0)
    ha.values["climate.lk_a"] = 20.0
    executor.execute(pump_setpoint_c=17.0)  # paused
    assert executor.restore_base().action == "wrote"


def test_the_base_must_not_exceed_the_boost(settings):
    from dataclasses import replace

    with pytest.raises(ConfigurationError):
        replace(settings, lk_base_c=22.0)


def test_a_thermostat_set_below_the_base_by_hand_is_left_alone(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(19.0))
    assert executor.execute(pump_setpoint_c=17.0).action == "kept"
    assert executor.restore_base().action == "kept"
    assert ha.writes == []
