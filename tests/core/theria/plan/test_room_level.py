import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.room_level import room_level_c

from .test_day_view import measured
from .test_service_active import make, run_cycle


def test_a_room_lifted_by_the_sun_is_judged_at_its_usual_level():
    day = [19.5] * 18 + [21.0, 22.5, 22.7, 22.0, 21.5, 21.3]
    assert room_level_c(21.3, day) == 19.5


def test_a_room_that_is_colder_than_usual_is_judged_as_it_is():
    assert room_level_c(18.6, [19.5] * 24) == 18.6


def test_too_little_history_means_the_room_is_taken_as_measured():
    assert room_level_c(21.3, [19.5] * 5) == 21.3


def test_missing_hours_are_ignored():
    day = [np.nan] * 6 + [19.5] * 18
    assert room_level_c(21.3, day) == 19.5


def room_at(level):
    def loader(start, stop):
        inputs = measured(start, stop)
        return type(inputs)(
            **{**inputs.__dict__, "room_c": np.full(len(inputs.hours), level)}
        )

    return loader


def test_the_plan_starts_from_the_room_level_not_from_a_sunny_peak(settings):
    service, *_ = make(settings, actuals_loader=room_at(19.0))  # sensors read 19.8
    run_cycle(service)
    assert service.latest().initial.room_c == 19.0


def test_the_plan_starts_from_the_measured_room_when_history_cannot_be_read(settings):
    def broken(start, stop):
        raise ModelFitError("InfluxDB is down")

    service, *_ = make(settings, actuals_loader=broken)
    run_cycle(service)
    assert service.latest().initial.room_c == 19.8
