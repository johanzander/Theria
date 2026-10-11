from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from core.theria.plan.state_store import PlanStateStore

from .test_service_active import good_days, make, run_cycle

IN_THE_NIGHT = datetime(
    2026, 10, 12, 22, 20, tzinfo=UTC
)  # 00:20 local; judged day is 12th


@pytest.fixture
def night_settings(settings):
    return replace(settings, night_start_hour=0, night_end_hour=6)


def cold_mornings(room=18.8, first=date(2026, 10, 8), n=5):
    return [
        replace(d, room_at_morning_c=room, day=first + timedelta(days=i))
        for i, d in enumerate(good_days(first, n))
    ]


def margin(service):
    return service.execution_status()["morning_margin_c"]


def test_a_cold_morning_raises_the_night_margin_and_says_so(night_settings):
    service, _, notifier, _, _ = make(
        night_settings, clock=lambda: IN_THE_NIGHT, outcomes=cold_mornings()
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert margin(service) == pytest.approx(0.25)
    assert any("night margin" in m for _, m in notifier.created)


def test_the_margin_moves_the_nights_floor_and_target_up_in_the_plan(night_settings):
    service, *_ = make(
        night_settings, clock=lambda: IN_THE_NIGHT, outcomes=cold_mornings()
    )
    service.run_once()
    before = service.latest().plan.hours[0].floor_c
    service.set_mode("active", confirm=True)
    run_cycle(service)  # the margin is learned at the end of this cycle ...
    run_cycle(service)  # ... and used by the next plan
    assert service.latest().plan.hours[0].floor_c == pytest.approx(before + 0.25)


def test_the_margin_survives_a_restart(night_settings, tmp_path):
    path = str(tmp_path / "state.json")
    first, *_ = make(
        night_settings,
        clock=lambda: IN_THE_NIGHT,
        outcomes=cold_mornings(),
        state_store=PlanStateStore(path),
    )
    first.set_mode("active", confirm=True)
    run_cycle(first)
    second, *_ = make(
        night_settings, clock=lambda: IN_THE_NIGHT, state_store=PlanStateStore(path)
    )
    assert margin(second) == pytest.approx(0.25)


def test_the_margin_never_cancels_the_night_window_completely(night_settings):
    service, *_ = make(night_settings, clock=lambda: IN_THE_NIGHT)
    service._morning_margin_c = 5.0  # as if it had been raised for a very long time
    service.run_once()
    plan = service.latest().plan
    assert plan.hours[0].floor_c <= night_settings.floor_c


def test_without_a_night_window_nothing_is_learned(settings):
    service, *_ = make(settings, outcomes=cold_mornings())
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert margin(service) == 0.0
