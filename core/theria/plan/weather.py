"""Open-Meteo forecast: outdoor temperature and solar irradiance per hour."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests

from core.theria.exceptions import WeatherDataError

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_DELAY_DAYS = (
    6  # the archive lags; newer days come from the forecast API (past_days)
)
HOURLY_VARIABLES = "shortwave_radiation,cloud_cover,temperature_2m"


@dataclass(frozen=True)
class WeatherHour:
    """Weather for the hour starting at `start` (UTC)."""

    start: datetime
    shortwave_wm2: float
    cloud_cover_pct: float
    temperature_c: float


def parse_open_meteo(payload: dict) -> list[WeatherHour]:
    """Parse a forecast response requested with timezone=UTC.

    Open-Meteo labels each value with the END of the hour it averages (the value
    labelled 13:00 is the mean over 12:00-13:00), so the start is label - 1 h.
    """
    try:
        hourly = payload["hourly"]
        times = hourly["time"]
        columns = (
            hourly["shortwave_radiation"],
            hourly["cloud_cover"],
            hourly["temperature_2m"],
        )
    except KeyError as e:
        raise WeatherDataError(f"missing field in Open-Meteo response: {e}") from e
    if any(len(c) != len(times) for c in columns):
        raise WeatherDataError("Open-Meteo columns have different lengths")
    hours: list[WeatherHour] = []
    for label, sw, cc, temp in zip(times, *columns, strict=True):
        if sw is None or cc is None or temp is None:
            raise WeatherDataError(f"missing value at {label}")
        end = datetime.fromisoformat(label).replace(tzinfo=UTC)
        hours.append(
            WeatherHour(end - timedelta(hours=1), float(sw), float(cc), float(temp))
        )
    return hours


class OpenMeteoClient:
    """Fetches the hourly forecast for one location."""

    def __init__(
        self, latitude: float, longitude: float, session: requests.Session | None = None
    ):
        self._latitude = latitude
        self._longitude = longitude
        self._session = session or requests.Session()

    def fetch(self, past_days: int = 1, forecast_days: int = 3) -> list[WeatherHour]:
        params: dict[str, str | float | int] = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": HOURLY_VARIABLES,
            "timezone": "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
        try:
            response = self._session.get(OPEN_METEO_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as e:
            raise WeatherDataError(f"Open-Meteo request failed: {e}") from e
        return parse_open_meteo(payload)

    def history(self, start: datetime, stop: datetime) -> list[WeatherHour]:
        """Hourly weather for [start, stop): archive for older days, forecast API for recent ones."""
        today = datetime.now(UTC).date()
        archive_end = min(stop.date(), today - timedelta(days=ARCHIVE_DELAY_DAYS))
        hours: dict[datetime, WeatherHour] = {}
        if start.date() <= archive_end:
            for hour in self._get(
                OPEN_METEO_ARCHIVE_URL,
                {
                    "start_date": start.date().isoformat(),
                    "end_date": archive_end.isoformat(),
                },
            ):
                hours[hour.start] = hour
        recent_days = (
            today - max(start.date(), archive_end + timedelta(days=1))
        ).days + 1
        if stop.date() > archive_end and recent_days > 0:
            for hour in self._get(
                OPEN_METEO_URL, {"past_days": min(recent_days, 92), "forecast_days": 1}
            ):
                hours[hour.start] = hour
        return [hours[k] for k in sorted(hours)]

    def _get(self, url: str, extra: dict[str, str | int]) -> list[WeatherHour]:
        params: dict[str, str | float | int] = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": HOURLY_VARIABLES,
            "timezone": "UTC",
            **extra,
        }
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as e:
            raise WeatherDataError(f"Open-Meteo request failed: {e}") from e
        return parse_open_meteo(payload)
