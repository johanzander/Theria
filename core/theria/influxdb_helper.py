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

OPTIONS_PATH = "/data/options.json"
DEFAULT_BUCKET = (
    "home_assistant/autogen"  # "database/retention policy", as in home-analytics
)
SAFE_NAME = re.compile(
    r"^[A-Za-z0-9_.]+$"
)  # entity ids and field names go into the query

# `reason` value on an error result when the query worked but returned no points.
NO_DATA = "no_data"


def get_influxdb_config():
    """InfluxDB url, credentials and bucket: add-on options first, then .env (local runs).

    The bucket is `database/retention policy` (as in home-analytics); it falls back to
    HA_DB_BUCKET and finally to the Home Assistant default.
    """
    config = {"url": "", "username": "", "password": "", "bucket": ""}

    try:
        if os.path.exists(OPTIONS_PATH):
            with open(OPTIONS_PATH) as f:
                options = json.load(f)
            influxdb_config = options.get("influxdb", {})
            for key in config:
                config[key] = influxdb_config.get(key, "")
            _LOGGER.debug("Loaded InfluxDB config from options.json")
    except Exception as e:
        _LOGGER.warning("Failed to load options.json: %s", str(e))

    load_dotenv()  # local runs: HA_DB_URL, HA_DB_USER_NAME, HA_DB_PASSWORD, HA_DB_BUCKET
    for key, variable in (
        ("url", "HA_DB_URL"),
        ("username", "HA_DB_USER_NAME"),
        ("password", "HA_DB_PASSWORD"),
        ("bucket", "HA_DB_BUCKET"),
    ):
        config[key] = config[key] or os.getenv(variable, "")
    config["bucket"] = config["bucket"] or DEFAULT_BUCKET

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
    database = config["bucket"].split("/", 1)[0]  # InfluxQL takes the database only
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
            params={"db": database, "q": query, "epoch": "ms"},
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
