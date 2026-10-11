from datetime import UTC, date, datetime, timedelta

import pytest

from core.theria.plan.executor import PlanExecutor, SafetyMonitor
from core.theria.plan.planner import HourPlan, Plan
from core.theria.plan.room_model import RoomState
from core.theria.plan.snapshot import PlanSnapshot
from core.theria.plan.stages import StageState

NOW = datetime(2026, 11, 3, 12, 20, tzinfo=UTC)
STAGE1 = StageState(index=0, stage_started_on=date(2026, 11, 1), revert_until=None)
STAGE2 = StageState(index=1, stage_started_on=date(2026, 11, 1), revert_until=None)


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

    def get_climate_state(self, entity_id):
        return {"target_temperature": self.setpoint}

    def set_temperature(self, entity_id, temperature):
        if self.fail_write:
            raise RuntimeError("HA is down")
        self.writes.append(temperature)
        if not self.ignore_write:
            self.setpoint = temperature


@pytest.fixture
def make(settings):
    def _make(setpoint=18.0, clock=None):
        ha = FakeHA(setpoint)
        current = [NOW]
        executor = PlanExecutor(ha, settings, now=clock or (lambda: current[0]))
        return executor, ha, current

    return _make


def test_it_writes_the_setpoint_of_the_current_plan_hour(make):
    executor, ha, _ = make(18.0)
    result = executor.execute(snapshot([17.5, 18.0]), STAGE1)
    assert ha.writes == [17.5] and result.action == "wrote" and result.written_c == 17.5


def test_lowering_is_not_rate_limited_but_is_clamped_to_the_stage(make):
    executor, ha, _ = make(18.0)
    executor.execute(snapshot([15.0]), STAGE1)
    assert ha.writes == [17.0]  # stage 1: at most 1 C from the base 18


def test_a_wider_stage_allows_the_old_manual_range(make):
    executor, ha, _ = make(18.0)
    executor.execute(snapshot([16.0]), STAGE2)
    assert ha.writes == [16.0]


def test_raising_is_limited_per_hour_by_the_stage(make):
    executor, ha, current = make(18.0)
    executor.execute(snapshot([20.0]), STAGE1)
    assert ha.writes == [18.5]  # stage 1 may raise 0.5 C per hour
    current[0] += timedelta(minutes=15)
    assert executor.execute(snapshot([20.0], current[0]), STAGE1).action == "kept"
    current[0] += timedelta(minutes=50)
    executor.execute(snapshot([20.0], current[0]), STAGE1)
    assert ha.writes == [18.5, 19.0]


def test_nothing_is_written_when_the_setpoint_already_is_right(make):
    executor, ha, _ = make(17.5)
    assert executor.execute(snapshot([17.5]), STAGE1).action == "kept"
    assert ha.writes == []


def test_without_a_fresh_plan_it_goes_back_to_the_base(make):
    executor, ha, _ = make(17.0)
    result = executor.execute(snapshot([16.0], NOW - timedelta(hours=2)), STAGE1)
    assert (
        ha.writes == [18.0] and result.action == "fallback" and "plan" in result.reason
    )
    executor2, ha2, _ = make(17.0)
    assert executor2.execute(None, STAGE1).action == "fallback"
    assert ha2.writes == [18.0]


def test_during_a_revert_the_base_setpoint_is_used(make):
    executor, ha, _ = make(17.0)
    state = StageState(0, date(2026, 11, 3), revert_until=date(2026, 11, 5))
    result = executor.execute(snapshot([16.0]), state)
    assert ha.writes == [18.0] and result.action == "fallback"


def test_a_manual_change_is_respected_for_six_hours(make):
    executor, ha, current = make(18.0)
    executor.execute(snapshot([17.0]), STAGE1)
    ha.setpoint = 20.0  # Johan turned it up by hand
    current[0] += timedelta(minutes=15)
    result = executor.execute(snapshot([17.0], current[0]), STAGE1)
    assert result.action == "paused" and ha.writes == [17.0]
    current[0] += timedelta(hours=3)
    assert executor.execute(snapshot([17.0], current[0]), STAGE1).action == "paused"
    current[0] += timedelta(hours=4)
    ha.setpoint = 20.0
    assert executor.execute(snapshot([17.0], current[0]), STAGE1).action == "wrote"


def test_a_write_that_is_not_confirmed_by_the_read_back_is_reported(make):
    executor, ha, _ = make(18.0)
    ha.ignore_write = True
    result = executor.execute(snapshot([17.0]), STAGE1)
    assert result.action == "failed" and "read back" in result.reason


def test_a_failing_home_assistant_does_not_raise(make):
    executor, ha, _ = make(18.0)
    ha.fail_write = True
    result = executor.execute(snapshot([17.0]), STAGE1)
    assert result.action == "failed" and "HA is down" in result.reason


def test_restore_base_writes_the_base_setpoint(make):
    executor, ha, _ = make(16.5)
    assert executor.restore_base().action == "wrote" and ha.writes == [18.0]
    assert executor.restore_base().action == "kept"


class FakeStates:
    def __init__(self, speed="9", valve="off", aux="0"):
        self.values = {
            "sensor.compressor_speed": speed,
            "binary_sensor.switch_valve_1": valve,
            "sensor.add_heat_status": aux,
        }

    def get_state(self, entity_id):
        return {"state": self.values[entity_id]}


def test_a_compressor_that_stays_stopped_while_heating_is_trouble(settings):
    states = FakeStates(speed="0")
    monitor = SafetyMonitor(states, settings)
    assert monitor.check() is None and monitor.check() is None
    assert "compressor" in monitor.check()


def test_a_stopped_compressor_during_hot_water_is_not_trouble(settings):
    monitor = SafetyMonitor(FakeStates(speed="0", valve="on"), settings)
    assert [monitor.check() for _ in range(5)] == [None] * 5


def test_the_electric_heater_running_while_the_setpoint_is_raised_is_trouble(settings):
    monitor = SafetyMonitor(FakeStates(aux="1"), settings)
    assert monitor.check(raised=True) is None
    assert "electric heater" in monitor.check(raised=True)
    quiet = SafetyMonitor(FakeStates(aux="1"), settings)
    assert [quiet.check(raised=False) for _ in range(4)] == [
        None
    ] * 4  # cold-weather use


def test_trouble_clears_when_the_condition_ends(settings):
    states = FakeStates(speed="0")
    monitor = SafetyMonitor(states, settings)
    monitor.check()
    monitor.check()
    states.values["sensor.compressor_speed"] = "9"
    assert monitor.check() is None
    states.values["sensor.compressor_speed"] = "0"
    assert monitor.check() is None  # the count starts over


def test_the_compressor_stopping_at_the_base_setpoint_is_not_the_plans_doing(settings):
    monitor = SafetyMonitor(FakeStates(speed="0"), settings)
    assert [monitor.check(deviated=False) for _ in range(5)] == [None] * 5


def test_trouble_is_reported_once_and_the_count_starts_over(settings):
    monitor = SafetyMonitor(FakeStates(speed="0"), settings)
    assert [monitor.check() for _ in range(3)][-1] is not None
    assert monitor.check() is None  # not repeated every cycle


def test_a_raise_never_goes_further_than_the_rate_limit_allows(make):
    executor, ha, _ = make(17.8)  # off the 0.5 grid, e.g. set by hand earlier
    executor.execute(snapshot([19.0]), STAGE1)
    assert ha.writes == [18.0]  # 17.8 + 0.5 = 18.3, and 18.5 would be too much
