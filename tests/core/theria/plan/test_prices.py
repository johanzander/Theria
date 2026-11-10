from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import PriceDataError
from core.theria.plan.prices import (
    PricePeriod,
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
