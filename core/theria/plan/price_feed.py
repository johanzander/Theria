"""Day-ahead prices from Home Assistant's built-in Nord Pool integration.

The integration answers the service `nordpool.get_prices_for_date` with 15-minute prices in
SEK/MWh for one local day. Today is asked for once per day, tomorrow until it is published
(around 13:00), then kept.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from core.theria.exceptions import PriceDataError
from core.theria.plan.prices import PricePeriod

RETRY_TOMORROW = timedelta(minutes=10)
MWH_PER_KWH = 1000.0


class ServiceClient(Protocol):
    def get_service_response(
        self, domain: str, service: str, data: dict[str, Any]
    ) -> dict[str, Any]: ...


def parse_prices_for_date(response: dict[str, Any], area: str) -> list[PricePeriod]:
    """The service answer for one area as periods in SEK/kWh."""
    entries = response.get(area)
    if not entries:
        raise PriceDataError(f"no prices for area {area} in the Nord Pool answer")
    periods = []
    for entry in entries:
        start = datetime.fromisoformat(entry["start"])
        if start.tzinfo is None:
            raise PriceDataError("price period start without timezone")
        periods.append(PricePeriod(start, float(entry["price"]) / MWH_PER_KWH))
    periods.sort(key=lambda p: p.start)
    return periods


class NordpoolFeed:
    """Today's and tomorrow's 15-minute prices; today's failure is an error, tomorrow's is not."""

    def __init__(
        self,
        client: ServiceClient,
        config_entry: str,
        area: str,
        timezone: str,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._client = client
        self._entry = config_entry
        self._area = area
        self._zone = ZoneInfo(timezone)
        self._now = now
        self._days: dict[date, list[PricePeriod]] = {}
        self._tomorrow_asked_at: datetime | None = None

    def periods(self) -> list[PricePeriod]:
        now = self._now()
        today = now.astimezone(self._zone).date()
        self._days = {d: p for d, p in self._days.items() if d >= today}
        if today not in self._days:
            self._days[today] = self._fetch(today)
        tomorrow = today + timedelta(days=1)
        if tomorrow not in self._days and (
            self._tomorrow_asked_at is None
            or now - self._tomorrow_asked_at >= RETRY_TOMORROW
        ):
            self._tomorrow_asked_at = now
            try:
                self._days[tomorrow] = self._fetch(tomorrow)
            except (RuntimeError, PriceDataError):
                pass  # not published yet
        return [p for d in sorted(self._days) for p in self._days[d]]

    def _fetch(self, day: date) -> list[PricePeriod]:
        try:
            response = self._client.get_service_response(
                "nordpool",
                "get_prices_for_date",
                {
                    "config_entry": self._entry,
                    "date": day.isoformat(),
                    "areas": self._area,
                    "currency": "SEK",
                },
            )
        except RuntimeError as e:
            raise PriceDataError(
                f"cannot read the Nord Pool prices for {day}: {e}"
            ) from e
        return parse_prices_for_date(response, self._area)
