from datetime import UTC, datetime, timedelta

import pytest

from core.theria.plan.lk_executor import LKExecutor

NOW = datetime(2026, 11, 3, 1, 20, tzinfo=UTC)
THERMOSTATS = ("climate.lk_a", "climate.lk_b")


class FakeLK:
    def __init__(self, value=21.0):  # the thermostats as set by hand, away from the normal target
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


LIMIT = 19.5  # the normal target: where the thermostats stand outside a window


def test_every_thermostat_is_held_at_the_limit(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(21.0))
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


def test_a_hand_change_pauses_the_room_and_is_kept_after_the_hour(lk_settings):
    executor, ha, now = make(lk_settings, FakeLK(LIMIT))
    executor.execute()
    ha.values["climate.lk_a"] = 20.0  # Johan changes one thermostat
    now[0] += timedelta(minutes=15)
    assert executor.execute().action == "paused"
    now[0] += timedelta(minutes=30)
    assert executor.execute().action == "paused"
    now[0] += timedelta(minutes=31)
    executor.execute()
    assert ha.values["climate.lk_a"] == 20.0  # left alone until the next window


def test_a_hold_that_was_written_is_not_mistaken_for_a_hand_change(lk_settings):
    executor, ha, now = make(lk_settings, FakeLK(LIMIT))
    executor.execute({"climate.lk_a": 20.5})
    now[0] += timedelta(minutes=15)
    assert executor.execute({"climate.lk_a": 20.5}).action == "kept"


def test_one_unreachable_thermostat_does_not_stop_the_other(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(21.0))
    ha.fail.add("climate.lk_a")
    result = executor.execute()
    assert result.action == "failed" and ha.values["climate.lk_b"] == LIMIT


def test_a_write_that_is_not_accepted_is_reported_not_assumed(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(21.0))
    ha.ignore.add("climate.lk_b")
    assert executor.execute().action == "failed"


def test_restore_writes_the_base_everywhere_and_reports_failure(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(21.0))
    assert executor.restore_base().action == "wrote"
    assert set(ha.values.values()) == {19.5}
    ha.values["climate.lk_a"] = 21.0
    ha.fail.add("climate.lk_a")
    assert executor.restore_base().action == "failed"


def test_restore_overrides_a_manual_pause(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    executor.execute()
    ha.values["climate.lk_a"] = 20.0
    executor.execute()  # paused
    assert executor.restore_base().action == "wrote"


def test_thermostats_that_take_a_few_reads_to_confirm_are_not_a_failure(lk_settings):
    ha = FakeLK(21.0)
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


def slow_cloud_failing_once(lk_settings):
    """The thermostats held the limit; the cloud is too slow to confirm a hold in time."""
    executor, ha, now = make(lk_settings, FakeLK(LIMIT))
    assert executor.execute().action == "kept"
    ha.lag = 1000
    assert executor.execute({"climate.lk_a": 20.5}).action == "failed"
    ha.lag, ha._pending = 0, {}
    now[0] += timedelta(minutes=15)
    return executor, ha, now


def test_a_thermostat_that_accepts_after_the_wait_is_not_taken_for_a_hand_change(
    lk_settings,
):
    executor, ha, _ = slow_cloud_failing_once(lk_settings)
    ha.values["climate.lk_a"] = 20.5  # the cloud caught up
    assert executor.execute({"climate.lk_a": 20.5}).action == "kept"


def test_a_value_that_is_neither_the_old_nor_the_pending_one_is_a_hand_change(
    lk_settings,
):
    executor, ha, _ = slow_cloud_failing_once(lk_settings)
    ha.values["climate.lk_a"] = 19.0
    assert executor.execute({"climate.lk_a": 20.5}).action == "paused"


def test_what_was_still_pending_survives_a_restart_of_the_lk_executor(lk_settings):
    executor, ha, now = slow_cloud_failing_once(lk_settings)
    saved = executor.export_state()
    ha.values["climate.lk_a"] = 20.5
    again = LKExecutor(ha, lk_settings, now=lambda: now[0], sleep=lambda _: None)
    again.load_state(saved)
    assert again.execute({"climate.lk_a": 20.5}).action == "kept"


def test_the_thermostat_that_took_the_value_is_not_a_hand_change_when_the_other_one_lagged(
    lk_settings,
):
    ha = FakeLK(21.0)
    executor2, ha, now2 = make(lk_settings, ha)
    executor2.restore_base()  # both thermostats are known at the base
    ha.ignore.add("climate.lk_b")  # the cloud does not deliver this one in time
    holds = {"climate.lk_a": 20.5, "climate.lk_b": 20.5}
    result = executor2.execute(holds, "w1")
    assert result.action == "failed" and ha.values["climate.lk_a"] == 20.5
    ha.ignore.discard("climate.lk_b")
    ha.values["climate.lk_b"] = 20.5  # delivered late
    now2[0] += timedelta(minutes=15)
    assert executor2.execute(holds, "w1").action == "kept"


# ---- the thermostats show their value to a tenth, and a value that is there is not written again


def test_a_thermostat_within_a_tenth_of_its_target_is_at_its_target(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    ha.values["climate.lk_a"] = LIMIT - 0.05  # the device shows a rounded value
    result = executor.execute()
    assert result.action == "kept" and ha.writes == []


def test_a_thermostat_a_quarter_degree_off_is_written(lk_settings):
    executor, ha, _ = make(lk_settings, FakeLK(LIMIT))
    ha.values["climate.lk_a"] = LIMIT - 0.25
    executor.execute()
    assert ("climate.lk_a", LIMIT) in ha.writes


# Per room, in the heat shift


@pytest.fixture
def shift_settings(lk_settings):
    from dataclasses import replace

    return replace(lk_settings)


BASE = 19.5
HOLD_BOTH = {"climate.lk_a": 19.0, "climate.lk_b": 19.0}


def test_outside_a_hold_the_heat_shift_stands_at_the_base(shift_settings):
    executor, ha, _ = make(shift_settings, FakeLK(22.0))
    executor.execute()
    assert ha.values["climate.lk_a"] == BASE


def test_a_hand_change_pauses_only_that_room(shift_settings):
    executor, ha, now = make(shift_settings, FakeLK(BASE))
    executor.execute(HOLD_BOTH, "w1")
    ha.values["climate.lk_a"] = 21.0
    now[0] += timedelta(minutes=15)
    ha.writes.clear()
    executor.execute({"climate.lk_a": 19.0, "climate.lk_b": 18.5}, "w1")
    assert ha.values["climate.lk_a"] == 21.0
    assert ha.values["climate.lk_b"] == 18.5


def test_a_room_changed_by_hand_is_left_until_the_next_window(shift_settings):
    executor, ha, now = make(shift_settings, FakeLK(BASE))
    executor.execute(HOLD_BOTH, "w1")
    ha.values["climate.lk_a"] = 21.0
    now[0] += timedelta(minutes=15)
    executor.execute(HOLD_BOTH, "w1")
    now[0] += timedelta(hours=2)  # the pause is over, the window is the same
    executor.execute(HOLD_BOTH, "w1")
    assert ha.values["climate.lk_a"] == 21.0
    executor.execute({"climate.lk_a": BASE, "climate.lk_b": BASE}, None)  # normal
    assert ha.values["climate.lk_a"] == 21.0
    executor.execute(HOLD_BOTH, "w2")  # the next window
    assert ha.values["climate.lk_a"] == 19.0


def test_one_room_that_cannot_be_read_does_not_stop_the_other(shift_settings):
    ha = FakeLK(BASE)
    executor, ha, _ = make(shift_settings, ha)
    original = ha.get_climate_state

    def broken(entity_id):
        if entity_id == "climate.lk_a":
            raise RuntimeError("gone")
        return original(entity_id)

    ha.get_climate_state = broken
    result = executor.execute(HOLD_BOTH, "w1")
    assert result.action == "failed"
    assert ha.values["climate.lk_b"] == 19.0


def test_a_late_confirmation_of_an_earlier_value_is_not_a_hand_change(shift_settings):
    executor, ha, now = make(shift_settings, FakeLK(BASE))
    executor.execute()
    ha.lag = 1000
    assert executor.execute({"climate.lk_a": 19.0}, "w1").action == "failed"
    now[0] += timedelta(minutes=15)
    assert executor.execute({"climate.lk_a": 20.5}, "w2").action == "failed"
    ha.lag, ha._pending = 0, {}
    ha.values["climate.lk_a"] = 19.0  # the first write arrives after the second was sent
    now[0] += timedelta(minutes=15)
    result = executor.execute({"climate.lk_a": 20.5}, "w2")
    assert result.action == "wrote"
    assert ha.values["climate.lk_a"] == 20.5


def test_a_pending_value_expires_after_the_grace_period(shift_settings):
    executor, ha, now = make(shift_settings, FakeLK(BASE))
    executor.execute()
    ha.lag = 1000
    executor.execute({"climate.lk_a": 19.0}, "w1")
    ha.lag, ha._pending = 0, {}
    now[0] += timedelta(hours=1, minutes=1)
    ha.values["climate.lk_a"] = 19.0
    result = executor.execute({"climate.lk_a": 19.0}, "w1")
    assert result.action == "paused"
