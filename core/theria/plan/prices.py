"""Electricity prices for planning: parsing, hourly aggregation, extreme-price rule."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.theria.exceptions import PriceDataError

QUARTERS_PER_HOUR = 4


@dataclass(frozen=True)
class PricePeriod:
    """One 15-minute price period."""

    start: datetime  # timezone-aware
    price_sek_per_kwh: float


def parse_nordpool_attributes(attrs: dict) -> list[PricePeriod]:
    """Parse the Nordpool sensor attributes into 15-minute periods.

    Uses `today`/`tomorrow` (SEK/kWh) for the values and `raw_today`/`raw_tomorrow`
    only for the period start times. The `price` field inside the raw entries is in
    SEK/MWh and must not be used. `tomorrow` is empty until prices are published.
    """
    periods: list[PricePeriod] = []
    for day in ("today", "tomorrow"):
        values = attrs.get(day)
        if not values:
            continue
        raw = attrs.get(f"raw_{day}")
        if raw is None or len(raw) != len(values):
            raise PriceDataError(
                f"{day}: expected raw_{day} with {len(values)} entries to match the values"
            )
        for entry, value in zip(raw, values, strict=True):
            if not isinstance(value, int | float):
                raise PriceDataError(f"{day}: non-numeric price {value!r}")
            start = datetime.fromisoformat(entry["start"])
            if start.tzinfo is None:
                raise PriceDataError(f"{day}: period start without timezone")
            periods.append(PricePeriod(start, float(value)))
    periods.sort(key=lambda p: p.start)
    starts = [p.start for p in periods]
    if len(set(starts)) != len(starts):
        raise PriceDataError("duplicate price period start times")
    return periods


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


def extreme_threshold(
    history_sek_per_kwh: list[float], percentile: float, min_sek_per_kwh: float
) -> float:
    """Price at or above which a period counts as extreme.

    The larger of the given percentile of recent prices and the absolute minimum.
    """
    if len(history_sek_per_kwh) < 96:
        raise PriceDataError(
            "need at least one day (96 periods) of price history for the extreme rule"
        )
    return max(float(np.percentile(history_sek_per_kwh, percentile)), min_sek_per_kwh)


class PriceHistory:
    """Rolling store of 15-minute prices used for the extreme-price rule."""

    def __init__(self, days: int):
        self._days = days
        self._prices: dict[datetime, float] = {}

    def add(self, periods: list[PricePeriod]) -> None:
        for p in periods:
            self._prices[p.start] = p.price_sek_per_kwh

    def prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self._days)
        self._prices = {s: v for s, v in self._prices.items() if s >= cutoff}

    def values(self) -> list[float]:
        return [self._prices[s] for s in sorted(self._prices)]
