from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.data_health import (
    counted_hours_check,
    history_checks,
    live_h60_checks,
    price_feed_check,
)
from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.prices import PricePeriod

NOW = datetime(2026, 10, 24, 10, 20, tzinfo=UTC)  # 12:20 local


def inputs(n: int = 72, **overrides: np.ndarray) -> OutcomeInputs:
    full = np.full(n, 1.0)
    base = OutcomeInputs(
        hours=[NOW.replace(minute=0) - timedelta(hours=n - i) for i in range(n)],
        price_sek_per_kwh=full.copy(), rts_c=full.copy(), t_out_c=full.copy(), compressor_kwh=full.copy(),
        aux_kwh=np.zeros(n), hot_water=np.zeros(n), room_c=full.copy(),
    )
    return replace(base, **overrides)


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_every_input_with_values_is_ok(settings):
    assert set(statuses(history_checks(inputs(), settings)).values()) == {"ok"}


def test_an_input_without_a_single_value_says_so_and_names_the_entity(settings):
    checks = history_checks(inputs(price_sek_per_kwh=np.full(72, np.nan)), settings)
    price = next(c for c in checks if c.name == "price history")
    assert price.status == "warn" and "NO DATA" in price.detail and "Nord Pool" in price.detail


def test_an_input_with_most_values_missing_is_a_warning_with_its_share(settings):
    aux = np.full(72, np.nan)
    aux[-12:] = 0.0
    checks = history_checks(inputs(aux_kwh=aux), settings)
    heater = next(c for c in checks if c.name == "electric heater counter")
    assert heater.status == "warn" and "50%" in heater.detail


def test_hours_count_for_the_savings_only_with_every_input(settings):
    assert counted_hours_check(inputs(), settings).status == "ok"


def test_the_input_that_costs_the_most_hours_is_named(settings):
    check = counted_hours_check(inputs(price_sek_per_kwh=np.full(72, np.nan)), settings)
    assert check.status == "warn" and "price history (0%)" in check.detail and "0 of 72" in check.detail


def test_hours_with_hot_water_are_not_counted_as_missing(settings):
    hot = np.where(np.arange(72) % 2 == 0, 1.0, 0.0)
    assert counted_hours_check(inputs(hot_water=hot), settings).status == "ok"


def periods(hours: int, start: datetime = NOW.replace(minute=0)) -> list[PricePeriod]:
    return [PricePeriod(start + timedelta(minutes=15 * q), 0.5) for q in range(4 * hours)]


def test_prices_ahead_are_ok(settings):
    assert price_feed_check(periods(36), NOW, settings.timezone).status == "ok"


def test_too_few_hours_of_prices_fail_because_the_plan_cannot_move_anything(settings):
    assert price_feed_check(periods(3), NOW, settings.timezone).status == "fail"


def test_no_prices_at_all_fail(settings):
    assert price_feed_check([], NOW, settings.timezone).status == "fail"


def test_tomorrows_prices_missing_in_the_afternoon_is_a_warning(settings):
    afternoon = datetime(2026, 10, 24, 14, 30, tzinfo=UTC)  # 16:30 local
    check = price_feed_check(periods(8, afternoon.replace(minute=0)), afternoon, settings.timezone)
    assert check.status == "warn" and "tomorrow" in check.detail


class Ha:
    def __init__(self, states):
        self.states = states

    def get_state(self, entity):
        value = self.states[entity]
        if isinstance(value, Exception):
            raise value
        return {"state": value}


def test_flow_readings_a_working_pump_could_give_are_ok(settings):
    ha = Ha({settings.flow_measured_entity: "34.5", settings.flow_setpoint_entity: "33.0"})
    assert set(statuses(live_h60_checks(ha, settings)).values()) == {"ok"}


@pytest.mark.parametrize("bad", ["0.0", "3.0", "unavailable"])
def test_a_zero_or_missing_flow_target_is_a_glitch(settings, bad):
    ha = Ha({settings.flow_measured_entity: "34.5", settings.flow_setpoint_entity: bad})
    assert statuses(live_h60_checks(ha, settings))["flow target"] == "warn"
