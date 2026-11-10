"""Electricity prices for planning: parsing and hourly aggregation."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.theria.exceptions import PriceDataError
from core.theria.plan.series import FloatArray

QUARTERS_PER_HOUR = 4


@dataclass(frozen=True)
class PricePeriod:
    """One 15-minute price period."""

    start: datetime  # timezone-aware
    price_sek_per_kwh: float


def hourly_price_array(periods: list[PricePeriod], hours: list[datetime]) -> FloatArray:
    """Mean price of each of the given hours; NaN for an hour without all four quarters."""
    by_start = {p.start: p.price_sek_per_kwh for p in periods}
    out = np.full(len(hours), np.nan)
    for i, hour in enumerate(hours):
        quarters = [hour + timedelta(minutes=15 * q) for q in range(QUARTERS_PER_HOUR)]
        if all(q in by_start for q in quarters):
            out[i] = float(np.mean([by_start[q] for q in quarters]))
    return out


def hourly_prices(
    periods: list[PricePeriod], first_hour: datetime
) -> list[tuple[datetime, float]]:
    """Mean price per complete hour, starting at `first_hour`, until data runs out.

    A trailing hour with fewer than four quarters is dropped.
    """
    by_start = {p.start: p.price_sek_per_kwh for p in periods}
    out: list[tuple[datetime, float]] = []
    hour = first_hour
    while True:
        quarters = [hour + timedelta(minutes=15 * i) for i in range(QUARTERS_PER_HOUR)]
        if not all(q in by_start for q in quarters):
            break
        out.append((hour, float(np.mean([by_start[q] for q in quarters]))))
        hour += timedelta(hours=1)
    if not out:
        raise PriceDataError(f"no complete price hour available from {first_hour}")
    return out
