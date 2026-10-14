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
        self.lag = 0  # reads that still show the old value after a write
        self._pending: dict[str, tuple[float, int]] = {}

    def get_climate_state(self, entity_id):
        if entity_id in self._pending:
            value, reads_left = self._pending[entity_id]
            if reads_left > 0:
                self._pending[entity_id] = (value, reads_left - 1)
            else:
                self.values[entity_id] = value
                del self._pending[entity_id]
        return {"target_temperature": self.values[entity_id]}

    def set_temperature(self, entity_id, temperature):
        if entity_id in self.fail:
            raise RuntimeError("unreachable")
        self.writes.append((entity_id, temperature))
        if entity_id in self.ignore:
            return
        if self.lag:
            self._pending[entity_id] = (temperature, self.lag)
        else:
            self.values[entity_id] = temperature


@pytest.fixture
def lk_settings(settings):
    from dataclasses import replace

    return replace(settings, room_thermostat_entities=THERMOSTATS)


def make(lk_settings, ha=None, clock=None):
    ha = ha or FakeLK()
    current = [NOW]
    executor = LKExecutor(
        ha, lk_settings, now=clock or (lambda: current[0]), sleep=lambda _: None
    )
    return executor, ha, current


LIMIT = 22.0


def test_every_thermostat_is_held_at_the_limit(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(19.5))
    result = executor.execute()
    assert result.action == "wrote" and result.target_c == LIMIT
    assert ha.values == {"climate.lk_a": LIMIT, "climate.lk_b": LIMIT}


def test_nothing_is_written_when_they_are_already_at_the_limit(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    assert executor.execute().action == "kept"
    assert ha.writes == []


def test_a_held_thermostat_gets_its_hold_and_goes_back_to_the_limit_when_released(
    lk_settings,
):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    executor.execute({"climate.lk_a": 20.5})
    assert ha.values == {"climate.lk_a": 20.5, "climate.lk_b": LIMIT}
    executor.execute({})
    assert ha.values == {"climate.lk_a": LIMIT, "climate.lk_b": LIMIT}


def test_a_hand_change_pauses_the_lever_for_an_hour(lk_settings):
    executor, ha, now = make(lk_settings, FakeLK(LIMIT))
    executor.execute()
    ha.values["climate.lk_a"] = 20.0  # Johan changes one thermostat
    now[0] += timedelta(minutes=15)
    assert executor.execute().action == "paused"
    now[0] += timedelta(minutes=30)
    assert executor.execute().action == "paused"
    now[0] += timedelta(minutes=31)
    assert executor.execute().action == "wrote"
    assert ha.values["climate.lk_a"] == LIMIT


def test_a_hold_that_was_written_is_not_mistaken_for_a_hand_change(lk_settings):
    executor, ha, now = make(lk_settings, FakeLK(LIMIT))
    executor.execute({"climate.lk_a": 20.5})
    now[0] += timedelta(minutes=15)
    assert executor.execute({"climate.lk_a": 20.5}).action == "kept"


def test_one_unreachable_thermostat_does_not_stop_the_other(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(19.5))
    ha.fail.add("climate.lk_a")
    result = executor.execute()
    assert result.action == "failed" and ha.values["climate.lk_b"] == LIMIT


def test_a_write_that_is_not_accepted_is_reported_not_assumed(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(19.5))
    ha.ignore.add("climate.lk_b")
    assert executor.execute().action == "failed"


def test_restore_writes_the_base_everywhere_and_reports_failure(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    assert executor.restore_base().action == "wrote"
    assert set(ha.values.values()) == {19.5}
    ha.values["climate.lk_a"] = LIMIT
    ha.fail.add("climate.lk_a")
    assert executor.restore_base().action == "failed"


def test_restore_overrides_a_manual_pause(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    executor.execute()
    ha.values["climate.lk_a"] = 20.0
    executor.execute()  # paused
    assert executor.restore_base().action == "wrote"


def test_thermostats_that_take_a_few_reads_to_confirm_are_not_a_failure(lk_settings):
    ha = FakeLK(19.5)
    ha.lag = 3
    executor, _, _ = make(lk_settings, ha)
    result = executor.execute()
    assert result.action == "wrote" and ha.values == {
        "climate.lk_a": LIMIT,
        "climate.lk_b": LIMIT,
    }


def test_the_state_survives_a_restart(lk_settings):
    executor, ha, now = make(lk_settings, FakeLK(LIMIT))
    executor.execute({"climate.lk_a": 20.5})
    saved = executor.export_state()
    again = LKExecutor(ha, lk_settings, now=lambda: now[0], sleep=lambda _: None)
    again.load_state(saved)
    assert again.execute({"climate.lk_a": 20.5}).action == "kept"
