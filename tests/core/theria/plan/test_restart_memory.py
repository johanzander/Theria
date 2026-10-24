from dataclasses import replace

from core.theria.plan.executor import PlanExecutor
from core.theria.plan.lk_executor import LKExecutor
from core.theria.plan.service import PlanService
from core.theria.plan.state_store import PlanStateStore

from .test_service import FakeWeather, feed_periods
from .test_service_active import FakeNotifier, FakeSafety, WritableHA, run_cycle
from .test_service_active import EVENING
from .test_service_lk import LIMIT, NOW as CHEAP_BEFORE_PEAK


def build(settings, ha, path, at=CHEAP_BEFORE_PEAK):
    now = lambda: at  # noqa: E731
    return PlanService(
        ha,
        replace(settings, mode="shadow"),
        FakeWeather(),
        price_periods=lambda: ha.prices,
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
    ha = WritableHA(feed_periods(), 18.0)
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


def test_the_base_that_stop_wrote_is_not_taken_for_a_hand_change_after_the_restart(
    settings, tmp_path
):
    import asyncio

    path = str(tmp_path / "state.json")
    ha = WritableHA(feed_periods(), 18.0)
    first = build(settings, ha, path, at=EVENING)  # the expensive hours: the plan cuts
    first.set_mode("active", confirm=True)
    run_cycle(first)
    first_write = ha.setpoint
    assert first_write != 18.0, "precondition: the plan holds a setpoint other than the base"
    asyncio.run(first.stop())  # restores the base (the add-on is being updated)
    assert ha.setpoint == 18.0
    second = build(settings, ha, path, at=EVENING)
    run_cycle(second)
    assert second.execution_status()["last_execution"].action != "paused"


def test_the_state_is_saved_before_the_base_is_written_in_case_the_stop_is_cut_short(
    settings, tmp_path
):
    import asyncio
    import json

    path = str(tmp_path / "state.json")
    ha = WritableHA(feed_periods(), 18.0)
    service = build(settings, ha, path, at=EVENING)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.setpoint != 18.0
    saved_at_first_write = []
    real = ha.set_temperature

    def watching(entity_id, temperature):
        if not saved_at_first_write:
            with open(path) as f:
                saved_at_first_write.append(json.load(f)["executor"]["pending"])
        real(entity_id, temperature)

    ha.set_temperature = watching
    asyncio.run(service.stop())
    # the supervisor may kill the add-on while the thermostats are still being confirmed: what
    # is on disk by then must already say that the base may arrive
    assert saved_at_first_write == [18.0]


def test_a_stop_that_was_cut_short_before_the_thermostats_changed_is_not_a_hand_change(
    settings, tmp_path
):
    import asyncio

    path = str(tmp_path / "state.json")
    ha = WritableHA(feed_periods(), 18.0)
    first = build(settings, ha, path, at=EVENING)
    first.set_mode("active", confirm=True)
    run_cycle(first)
    assert ha.lk["climate.lk_bibliotek_thermostat"] == LIMIT
    ha.fail_lk_writes = True  # the cloud never got to change them before the add-on was killed
    asyncio.run(first.stop())
    ha.fail_lk_writes = False
    second = build(settings, ha, path, at=EVENING)
    run_cycle(second)
    assert second.execution_status()["last_lk"].action != "paused"
