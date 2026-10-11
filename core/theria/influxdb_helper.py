"""Fetching sensor history from InfluxDB 1.x with InfluxQL (GET <url>/query).

Same endpoint and credentials as the home-analytics add-on. The measurement name is the entity
id (`sensor.outdoor`), the value of a sensor is the field `value` and attributes are fields of
their own name. Credentials come from the add-on options, falling back to `.env` locally.
"""

import json
import logging
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

_LOGGER = logging.getLogger(__name__)

DATABASE = (
    "home_assistant"  # the bucket home_assistant/autogen is database/retention policy
)
SAFE_NAME = re.compile(
    r"^[A-Za-z0-9_.]+$"
)  # entity ids and field names go into the query

# `reason` value on an error result when the query worked but returned no points.
NO_DATA = "no_data"


def get_influxdb_config():
    """Load InfluxDB config from options.json or fallback to .env."""
    config = {"url": "", "username": "", "password": ""}

    # 1. Try load from Home Assistant options.json
    try:
        if os.path.exists("/data/options.json"):
            with open("/data/options.json") as f:
                options = json.load(f)
            influxdb_config = options.get("influxdb", {})
            config.update(
                {
                    "url": influxdb_config.get("url", ""),
                    "username": influxdb_config.get("username", ""),
                    "password": influxdb_config.get("password", ""),
                }
            )
            _LOGGER.debug("Loaded InfluxDB config from options.json")
    except Exception as e:
        _LOGGER.warning("Failed to load options.json: %s", str(e))

    # 2. Fallback to .env if necessary
    if not config["url"] or not config["username"] or not config["password"]:
        try:
            load_dotenv()  # this loads from .env automatically
            config.update(
                {
                    "url": os.getenv("HA_DB_URL", ""),
                    "username": os.getenv("HA_DB_USER_NAME", ""),
                    "password": os.getenv("HA_DB_PASSWORD", ""),
                }
            )
            _LOGGER.debug("Loaded InfluxDB config from .env file")
        except Exception as e:
            _LOGGER.warning("Failed to load .env file: %s", str(e))

    # 3. Final check
    if not config["url"] or not config["username"] or not config["password"]:
        _LOGGER.error("InfluxDB configuration is incomplete.")

    return config


def get_sensor_timeseries(
    sensor_name: str,
    start_time: datetime,
    stop_time: datetime,
    timezone: str = "Europe/Stockholm",
    domain: str = "sensor",
    field_name: str | None = None,
    parse_as_string: bool = False,
) -> dict:
    """Get time series data for a single sensor or climate entity.

    Args:
        sensor_name: Entity ID without the domain prefix (e.g. 'outdoor')
        start_time: Start of the range (inclusive)
        stop_time: End of the range (exclusive)
        timezone: Timezone of the returned timestamps and of naive inputs
        domain: Entity domain ('sensor', 'climate', ...)
        field_name: Field to fetch; 'current_temperature' for climate, else 'value'
        parse_as_string: Keep values as strings (for state-like fields)

    Returns:
        {"status": "success", "data": [(timestamp, value), ...]} sorted by time, or
        {"status": "error", "message": ..., "reason": NO_DATA when the query worked but
        found nothing}.
    """
    local_tz = ZoneInfo(timezone)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=local_tz)
    if stop_time.tzinfo is None:
        stop_time = stop_time.replace(tzinfo=local_tz)

    config = get_influxdb_config()
    url, username, password = config["url"], config["username"], config["password"]
    if not url or not username or not password:
        _LOGGER.error("InfluxDB configuration is incomplete")
        return {"status": "error", "message": "Incomplete InfluxDB configuration"}

    if field_name is None:
        field_name = "current_temperature" if domain == "climate" else "value"
    measurement = f"{domain}.{sensor_name}"
    if not SAFE_NAME.match(measurement) or not SAFE_NAME.match(field_name):
        return {"status": "error", "message": f"Invalid entity or field: {measurement}"}

    start_str = start_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_str = stop_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = (
        f'SELECT "{field_name}" FROM "{measurement}" '
        f"WHERE time >= '{start_str}' AND time < '{stop_str}' ORDER BY time ASC"
    )

    try:
        response = requests.get(
            url,
            auth=(username, password),
            params={"db": DATABASE, "q": query, "epoch": "ms"},
            timeout=30,
        )
    except requests.RequestException as e:
        _LOGGER.error("Error connecting to InfluxDB: %s", e)
        return {"status": "error", "message": f"Connection error: {e!s}"}

    if response.status_code != 200:
        _LOGGER.error("Error from InfluxDB: %s", response.status_code)
        return {"status": "error", "message": f"InfluxDB error: {response.status_code}"}

    try:
        result = response.json()["results"][0]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        return {"status": "error", "message": f"Unexpected InfluxDB response: {e!s}"}
    if "error" in result:
        _LOGGER.error("InfluxDB query error: %s", result["error"])
        return {
            "status": "error",
            "message": f"InfluxDB query error: {result['error']}",
        }

    points: list[tuple[datetime, float | str]] = []
    for series in result.get("series", []):
        for moment_ms, value in series["values"]:
            if value is None:
                continue
            moment = datetime.fromtimestamp(moment_ms / 1000, ZoneInfo("UTC"))
            points.append(
                (
                    moment.astimezone(local_tz),
                    str(value) if parse_as_string else float(value),
                )
            )
    if not points:
        _LOGGER.warning("No data found for %s", measurement)
        return {"status": "error", "reason": NO_DATA, "message": "No data found"}
    points.sort(key=lambda p: p[0])
    _LOGGER.info("Fetched %d data points for %s", len(points), measurement)
    return {"status": "success", "data": points}
