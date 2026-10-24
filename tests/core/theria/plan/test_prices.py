from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import PriceDataError
from core.theria.plan.prices import (
    PriceHistory,
    PricePeriod,
    extreme_threshold,
    hourly_price_array,
    hourly_prices,
)

DAY = datetime(2026, 10, 12, 0, tzinfo=UTC)


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
    periods = [PricePeriod(day + timedelta(minutes=15 * i), 1.0) for i in range(100)]
    hourly = hourly_prices(periods, day)
    assert len(hourly) == 25
    assert hourly[-1][0] == day + timedelta(hours=24)


def test_a_price_per_hour_for_the_hours_asked_nan_where_a_quarter_is_missing():
    periods = [PricePeriod(DAY + timedelta(minutes=15 * i), float(i)) for i in range(7)]
    hours = [DAY, DAY + timedelta(hours=1), DAY + timedelta(hours=2)]
    out = hourly_price_array(periods, hours)
    assert out[0] == 1.5 and np.isnan(out[1]) and np.isnan(out[2])  # hour 1 has three quarters only
