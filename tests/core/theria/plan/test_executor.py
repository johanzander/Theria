from datetime import UTC, datetime, timedelta

import pytest

from core.theria.plan.executor import PlanExecutor, SafetyMonitor
from core.theria.plan.planner import HourPlan, Plan
from core.theria.plan.room_model import RoomState
from core.theria.plan.snapshot import PlanSnapshot

NOW = datetime(2026, 11, 3, 12, 20, tzinfo=UTC)


def snapshot(rts_by_hour, computed_at=NOW):
    hours = [
        HourPlan(
            start=datetime(2026, 11, 3, 12, tzinfo=UTC) + timedelta(hours=i),
            rts_c=rts,
            price_sek_per_kwh=1.0,
            extreme=False,
            floor_c=18.0,
            flow_target_c=35.0,
            energy_kwh=2.0,
            room_c_end=19.9,
            slab_c_end=25.0,
            energy_cost_sek=2.0,
            comfort_cost_sek=0.0,
            tail=False,
            reason={},
        )
        for i, rts in enumerate(rts_by_hour)
    ]
    return PlanSnapshot(
        computed_at=computed_at,
        mode="active",
        plan=Plan(hours, hours, 0.0, 0.0, 0),
        extreme_threshold_sek_per_kwh=3.0,
        outdoor_bias_c=0.0,
        initial=RoomState(19.9, 25.0, 0.0),
        model_status={},
    )


class FakeHA:
    def __init__(self, setpoint=18.0):
        self.setpoint = setpoint
        self.writes = []
        self.fail_write = False
        self.ignore_write = False
        self.lag = 0  # reads that still show the old value after a write
        self._pending: tuple[float, int] | None = None

    def get_climate_state(self, entity_id):
        if self._pending is not None:
            value, reads_left = self._pending
            if reads_left > 0:
                self._pending = (value, reads_left - 1)
                return {"target_temperature": self.setpoint}
            self.setpoint, self._pending = value, None
        return {"target_temperature": self.setpoint}

    def set_temperature(self, entity_id, temperature):
        if self.fail_write:
            raise RuntimeError("HA is down")
        self.writes.append(temperature)
        if self.ignore_write:
            return
        if self.lag:
            self._pending = (temperature, self.lag)
        else:
            self.setpoint = temperature


@pytest.fixture
def make(settings):
    def _make(setpoint=18.0, clock=None):
        ha = FakeHA(setpoint)
        current = [NOW]
        executor = PlanExecutor(
            ha, settings, now=clock or (lambda: current[0]), sleep=lambda _: None
        )
        return executor, ha, current

    return _make


def test_it_writes_the_setpoint_of_the_current_plan_hour(make):
    executor, ha, _ = make(18.0)
    result = executor.execute(snapshot([17.5, 18.0]))
    assert ha.writes == [17.5] and result.action == "wrote" and result.written_c == 17.5


def test_lowering_is_not_rate_limited_but_stays_in_the_setpoint_range(make):
    executor, ha, _ = make(18.0)
    executor.execute(snapshot([12.0]))
    assert ha.writes == [executor._s.rts_min_c]


def test_the_plan_may_move_the_setpoint_as_far_as_the_range_allows(make):
    executor, ha, _ = make(18.0)
    executor.execute(snapshot([16.0]))
    assert ha.writes == [16.0]


def test_raising_is_limited_per_hour(make):
    executor, ha, current = make(18.0)
    executor.execute(snapshot([20.0]))
    assert ha.writes == [19.0]  # at most max_rts_step_per_hour_c (1 C) per hour
    current[0] += timedelta(minutes=15)
    assert executor.execute(snapshot([20.0], current[0])).action == "kept"
    current[0] += timedelta(minutes=50)
    executor.execute(snapshot([20.0], current[0]))
    assert ha.writes == [19.0, 20.0]


def test_nothing_is_written_when_the_setpoint_already_is_right(make):
    executor, ha, _ = make(17.5)
    assert executor.execute(snapshot([17.5])).action == "kept"
    assert ha.writes == []


def test_without_a_fresh_plan_it_goes_back_to_the_base(make):
    executor, ha, _ = make(17.0)
    result = executor.execute(snapshot([16.0], NOW - timedelta(hours=2)))
    assert (
        ha.writes == [18.0] and result.action == "fallback" and "plan" in result.reason
    )
    executor2, ha2, _ = make(17.0)
    assert executor2.execute(None).action == "fallback"
    assert ha2.writes == [18.0]


def test_a_manual_change_is_respected_for_an_hour(make):
    executor, ha, current = make(18.0)
    executor.execute(snapshot([17.0]))
    ha.setpoint = 20.0  # Johan turned it up by hand
    current[0] += timedelta(minutes=15)
    result = executor.execute(snapshot([17.0], current[0]))
    assert result.action == "paused" and ha.writes == [17.0]
    current[0] += timedelta(minutes=30)
    assert executor.execute(snapshot([17.0], current[0])).action == "paused"
    current[0] += timedelta(minutes=30)
    ha.setpoint = 20.0
    assert executor.execute(snapshot([17.0], current[0])).action == "wrote"


def test_a_write_that_is_not_confirmed_by_the_read_back_is_reported(make):
    executor, ha, _ = make(18.0)
    ha.ignore_write = True
    result = executor.execute(snapshot([17.0]))
    assert result.action == "failed" and "read back" in result.reason


def test_a_failing_home_assistant_does_not_raise(make):
    executor, ha, _ = make(18.0)
    ha.fail_write = True
    result = executor.execute(snapshot([17.0]))
    assert result.action == "failed" and "HA is down" in result.reason


def test_restore_base_writes_the_base_setpoint(make):
    executor, ha, _ = make(16.5)
    assert executor.restore_base().action == "wrote" and ha.writes == [18.0]
    assert executor.restore_base().action == "kept"


class FakeStates:
    def __init__(self, aux="0"):
        self.values = {"sensor.add_heat_status": aux}

    def get_state(self, entity_id):
        return {"state": self.values[entity_id]}


def test_the_electric_heater_running_while_the_setpoint_is_raised_is_trouble(settings):
    monitor = SafetyMonitor(FakeStates(aux="1"), settings)
    assert monitor.check(raised=True) is None
    assert "electric heater" in monitor.check(raised=True)
    quiet = SafetyMonitor(FakeStates(aux="1"), settings)
    assert [quiet.check(raised=False) for _ in range(4)] == [
        None
    ] * 4  # cold-weather use


def test_trouble_clears_when_the_condition_ends(settings):
    states = FakeStates(aux="1")
    monitor = SafetyMonitor(states, settings)
    monitor.check(raised=True)
    states.values["sensor.add_heat_status"] = "0"
    assert monitor.check(raised=True) is None
    states.values["sensor.add_heat_status"] = "1"
    assert monitor.check(raised=True) is None  # the count starts over


def test_trouble_is_reported_once_and_the_count_starts_over(settings):
    monitor = SafetyMonitor(FakeStates(aux="1"), settings)
    assert [monitor.check(raised=True) for _ in range(2)][-1] is not None
    assert monitor.check(raised=True) is None  # not repeated every cycle


def test_a_pump_that_takes_a_few_reads_to_confirm_is_not_a_failure(make):
    executor, ha, _ = make(18.0)
    ha.lag = 4  # the first four reads still show the old setpoint
    result = executor.execute(snapshot([17.0]))
    assert result.action == "wrote" and ha.setpoint == 17.0
    assert (
        executor.execute(snapshot([17.0])).action == "kept"
    )  # not seen as a hand change


def test_a_pump_that_never_confirms_is_a_failure_after_the_wait(make):
    executor, ha, _ = make(18.0)
    ha.lag = 1000
    result = executor.execute(snapshot([17.0]))
    assert result.action == "failed" and "read back 18.0" in result.reason
