from datetime import UTC, datetime, timedelta

from core.theria.plan.room_model import day_heat_c_per_h

from .test_service import FakeHA, feed_periods, make_service

MIDNIGHT = datetime(2026, 10, 12, 22, 0, tzinfo=UTC)  # 00:00 local


def with_room(service, room_c):
    reading = FakeHA.get_state

    def get_state(entity_id):
        state = reading(service._ha, entity_id)
        if entity_id == "sensor.lk_bibliotek_temperature":
            return {"state": str(room_c), "attributes": {}}
        return state

    service._ha.get_state = get_state
    return service


def test_a_stove_warmed_room_is_forecast_to_cool_with_the_pump_turned_down(settings):
    service = with_room(
        make_service(settings, prices=feed_periods(days=3), clock=lambda: MIDNIGHT),
        22.0,
    )
    hours = service.run_once().plan.hours
    assert all(h.rts_c < settings.rts_base_c for h in hours[:6])
    assert hours[5].room_c_end < 21.5


def test_a_warmer_room_falls_faster(settings):
    def fall(room_c):
        service = with_room(
            make_service(settings, prices=feed_periods(days=3), clock=lambda: MIDNIGHT),
            room_c,
        )
        hours = service.run_once().plan.baseline_hours
        return room_c - hours[5].room_c_end

    assert fall(22.0) > fall(19.8) > 0


def test_the_measured_extra_heat_comes_by_day_and_not_at_night():
    assert day_heat_c_per_h(3) == 0.0
    assert [day_heat_c_per_h(h) for h in (9, 15, 21)] == [0.026, 0.105, 0.047]


def test_the_disturbance_is_not_fed_the_extra_heat_the_forecast_already_has(settings):
    now = [
        datetime(2026, 10, 12, 8, 0, tzinfo=UTC)
    ]  # 10:00 local, no sun in FakeWeather's dark
    service = make_service(settings, clock=lambda: now[0])
    service.run_once()
    now[0] += timedelta(hours=1)
    service.run_once()
    assert service._disturbance.drift_c_per_h < 0.02


def test_the_room_forecast_does_not_carry_the_measured_disturbance_forward(settings):
    """Backtest 27 Oct-4 Nov: the disturbance made the 6-12 h forecast worse (RMSE 0.78 -> 0.87)."""
    service = with_room(
        make_service(settings, prices=feed_periods(days=3), clock=lambda: MIDNIGHT),
        20.0,
    )
    quiet = service.run_once().plan.baseline_hours
    service._disturbance.drift_c_per_h = 0.15
    service._disturbance._at = MIDNIGHT
    warmed = service.run_once().plan.baseline_hours
    assert [h.room_c_end for h in warmed] == [h.room_c_end for h in quiet]
