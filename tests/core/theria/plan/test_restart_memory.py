from dataclasses import replace

from core.theria.plan.executor import PlanExecutor
from core.theria.plan.lk_executor import LKExecutor
from core.theria.plan.service import PlanService
from core.theria.plan.state_store import PlanStateStore

from .test_service import FakeWeather, price_attributes
from .test_service_active import FakeNotifier, FakeSafety, WritableHA, run_cycle
from .test_service_lk import LIMIT, NOW as CHEAP_BEFORE_PEAK


def build(settings, ha, path):
    now = lambda: CHEAP_BEFORE_PEAK  # noqa: E731
    return PlanService(
        ha,
        replace(settings, mode="shadow"),
        FakeWeather(),
        now=now,
        executor=PlanExecutor(ha, settings, now=now),
        lk_executor=LKExecutor(ha, settings, now=now),
        safety=FakeSafety(),
        notifier=FakeNotifier(),
        outcome_loader=lambda start, stop: [],
        state_store=PlanStateStore(path),
    )


def test_a_hand_change_made_while_the_service_was_down_is_noticed_after_restart(
    settings, tmp_path
):
    path = str(tmp_path / "state.json")
    ha = WritableHA(price_attributes(), 18.0)
    first = build(settings, ha, path)
    first.set_mode("active", confirm=True)
    run_cycle(first)
    assert ha.lk["climate.lk_bibliotek_thermostat"] == LIMIT
    ha.setpoint = 19.5  # Johan turns the pump up while the add-on is restarting
    ha.lk["climate.lk_bibliotek_thermostat"] = 20.0
    second = build(settings, ha, path)
    ha.writes.clear()
    ha.lk_writes.clear()
    run_cycle(second)
    assert ha.writes == [] and ha.lk_writes == []
    assert second.execution_status()["last_execution"].action == "paused"
    assert second.execution_status()["last_lk"].action == "paused"
