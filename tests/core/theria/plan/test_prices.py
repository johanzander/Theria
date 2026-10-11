from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import PriceDataError
from core.theria.plan.prices import (
    PriceHistory,
    PricePeriod,
    extreme_threshold,
    hourly_prices,
    parse_nordpool_attributes,
)

DAY = datetime(2026, 10, 12, 0, tzinfo=UTC)


def attrs(values, day=DAY, key="today"):
    raw = [
        {"start": (day + timedelta(minutes=15 * i)).isoformat(), "price": v * 1000.0}
        for i, v in enumerate(values)
    ]
    return {key: list(values), f"raw_{key}": raw}


def test_values_come_from_sek_per_kwh_list_not_the_sek_per_mwh_raw_price():
    periods = parse_nordpool_attributes(attrs([1.5, 2.5]))
    assert [p.price_sek_per_kwh for p in periods] == [1.5, 2.5]
    assert periods[1].start == DAY + timedelta(minutes=15)


def test_tomorrow_is_optional_and_days_are_merged_in_order():
    both = {
        **attrs([1.0] * 96),
        **attrs([2.0] * 96, DAY + timedelta(days=1), "tomorrow"),
    }
    periods = parse_nordpool_attributes(both)
    assert len(periods) == 192
    assert periods[0].start < periods[-1].start
    assert parse_nordpool_attributes(
        {**attrs([1.0] * 96), "tomorrow": [], "raw_tomorrow": []}
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"today": [1.0, 2.0], "raw_today": [{"start": DAY.isoformat()}]},
        {"today": [1.0], "raw_today": [{"start": "2026-10-12T00:00:00"}]},  # naive time
        {"today": ["x"], "raw_today": [{"start": DAY.isoformat()}]},
        {
            "today": [1.0, 1.0],
            "raw_today": [{"start": DAY.isoformat()}] * 2,
        },  # duplicate
        {"today": [1.0]},  # raw missing
    ],
)
def test_malformed_attributes_raise(bad):
    with pytest.raises(PriceDataError):
        parse_nordpool_attributes(bad)


def test_hourly_prices_average_quarters_and_drop_incomplete_trailing_hour():
    periods = [
        PricePeriod(DAY + timedelta(minutes=15 * i), float(i)) for i in range(10)
    ]
    hourly = hourly_prices(periods, DAY)
    assert hourly == [(DAY, 1.5), (DAY + timedelta(hours=1), 5.5)]


def test_hourly_prices_without_a_complete_hour_raises():
    with pytest.raises(PriceDataError):
        hourly_prices([PricePeriod(DAY, 1.0)], DAY)


def test_extreme_threshold_is_the_larger_of_percentile_and_minimum():
    quiet = [1.0] * 100
    assert extreme_threshold(quiet, 95.0, 3.0) == 3.0
    volatile = [1.0] * 90 + [8.0] * 10
    assert extreme_threshold(volatile, 95.0, 3.0) == 8.0


def test_extreme_threshold_needs_a_day_of_history():
    with pytest.raises(PriceDataError):
        extreme_threshold([1.0] * 95, 95.0, 3.0)


def test_history_prunes_old_periods_and_deduplicates():
    history = PriceHistory(days=2)
    old = PricePeriod(DAY - timedelta(days=3), 9.0)
    recent = PricePeriod(DAY, 1.0)
    history.add([old, recent, PricePeriod(DAY, 2.0)])
    history.prune(DAY + timedelta(hours=1))
    assert history.values() == [2.0]


def test_dst_change_day_with_100_quarters_is_parsed_and_aggregated():
    day = datetime(2026, 10, 24, 22, tzinfo=UTC)  # local midnight, CEST
    periods = parse_nordpool_attributes(attrs([1.0] * 100, day))
    hourly = hourly_prices(periods, day)
    assert len(hourly) == 25
    assert hourly[-1][0] == day + timedelta(hours=24)
