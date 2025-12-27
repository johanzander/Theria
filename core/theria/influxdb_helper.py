"""Provides helper functions to interact with InfluxDB for fetching sensor data.

The module includes functionality to parse responses, handle timezones, and process sensor readings.
This module is designed to run within either the Pyscript environment or a standard Python environment.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

_LOGGER = logging.getLogger(__name__)


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


def get_sensor_data(sensors_list, start_time=None, stop_time=None) -> dict:
    """Get sensor data with configurable time range.

    Args:
        sensors_list: List of sensor names to query
        start_time: Start time for the query (defaults to 24h before stop_time)
        stop_time: End time for the query (defaults to now)

    Returns:
        dict: Query results with status and data
    """
    # Set up timezone
    local_tz = ZoneInfo("Europe/Stockholm")

    # Determine stop time
    if stop_time is None:
        stop_time = datetime.now(local_tz)
    elif stop_time.tzinfo is None:
        stop_time = stop_time.replace(tzinfo=local_tz)

    # Determine start time - default to 24h before stop time
    if start_time is None:
        start_time = stop_time - timedelta(hours=24)
        _LOGGER.debug("Using default 24-hour window")
    elif start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=local_tz)

    # Get configuration
    influxdb_config = get_influxdb_config()

    url = influxdb_config.get("url", "")
    username = influxdb_config.get("username", "")
    password = influxdb_config.get("password", "")

    # Validate required configuration
    if not url or not username or not password:
        _LOGGER.error(
            "InfluxDB configuration is incomplete. URL: %s, Username: %s",
            url,
            username,
        )
        return {"status": "error", "message": "Incomplete InfluxDB configuration"}

    headers = {
        "Content-type": "application/vnd.flux",
        "Accept": "application/csv",
    }

    # Format times for InfluxDB query
    start_str = start_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = stop_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    sensor_filter = " or ".join(
        [f'r["_measurement"] == "sensor.{sensor}"' for sensor in sensors_list]
    )

    # Time-bounded query (always uses range since we always have start_time)
    flux_query = f"""from(bucket: "home_assistant/autogen")
                    |> range(start: {start_str}, stop: {end_str})
                    |> filter(fn: (r) => {sensor_filter})
                    |> filter(fn: (r) => r["_field"] == "value")
                    |> filter(fn: (r) => r["domain"] == "sensor")
                    |> last()
                    """

    try:
        # Use the environment-aware executor to make the request
        response = requests.post(
            url=url,
            auth=(username, password),
            headers=headers,
            data=flux_query,
            timeout=10,
        )

        if response.status_code == 204:
            _LOGGER.warning("No data found for the requested sensors")
            return {"status": "error", "message": "No data found"}

        if response.status_code != 200:
            _LOGGER.error("Error from InfluxDB: %s", response.status_code)
            return {
                "status": "error",
                "message": f"InfluxDB error: {response.status_code}",
            }

        sensor_readings = parse_influxdb_response(response.text)
        return {"status": "success", "data": sensor_readings}

    except requests.RequestException as e:
        _LOGGER.error("Error connecting to InfluxDB: %s", str(e))
        return {"status": "error", "message": f"Connection error: {e!s}"}
    except Exception as e:
        _LOGGER.error("Unexpected error: %s", str(e))
        return {"status": "error", "message": f"Unexpected error: {e!s}"}


def parse_influxdb_response(response_text) -> dict:
    """Parse InfluxDB response to extract the latest measurement for each sensor."""
    readings = {}
    lines = response_text.strip().split("\n")

    # Skip metadata rows (lines starting with '#')
    data_lines = [line for line in lines if not line.startswith("#")]

    # Process each data line
    for line in data_lines:
        parts = line.split(",")
        try:
            # Ensure the line has enough parts and the value can be converted to float
            if len(parts) < 9 or parts[6] == "_value":
                continue

            # Extract sensor name (_measurement) and value (_value)
            sensor_name = parts[
                10
            ].strip()  # _measurement is the 11th column (index 10)
            value = float(parts[6].strip())  # _value is the 7th column (index 6)

            # Store the value in the readings dictionary with the sensor name
            readings[sensor_name] = value
        except (IndexError, ValueError) as e:
            _LOGGER.error("Failed to parse line: %s, error: %s", line, e)
            continue

    _LOGGER.debug("Parsed response: %s", readings)
    return readings


def get_sensor_timeseries(
    sensor_name: str,
    start_time: datetime,
    stop_time: datetime,
    timezone: str = "Europe/Stockholm",
    domain: str = "sensor",
    field_name: str = None,
    parse_as_string: bool = False
) -> dict:
    """Get time series data for a single sensor or climate entity.

    Args:
        sensor_name: Entity ID (without domain prefix, e.g., 'outdoor' or 'vantsidan')
        start_time: Start time for the query
        stop_time: End time for the query
        timezone: Local timezone (default: Europe/Stockholm)
        domain: Entity domain ('sensor' or 'climate')
        field_name: Specific field to fetch (optional, auto-detected if not provided)
        parse_as_string: If True, parse values as strings instead of floats (for state fields)

    Returns:
        dict: {
            "status": "success" or "error",
            "message": error message if status is "error",
            "data": [(timestamp, value), ...] sorted by timestamp
        }
    """
    local_tz = ZoneInfo(timezone)

    # Ensure times have timezone
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=local_tz)
    if stop_time.tzinfo is None:
        stop_time = stop_time.replace(tzinfo=local_tz)

    # Get configuration
    influxdb_config = get_influxdb_config()

    url = influxdb_config.get("url", "")
    username = influxdb_config.get("username", "")
    password = influxdb_config.get("password", "")

    # Validate required configuration
    if not url or not username or not password:
        _LOGGER.error("InfluxDB configuration is incomplete")
        return {"status": "error", "message": "Incomplete InfluxDB configuration"}

    headers = {
        "Content-type": "application/vnd.flux",
        "Accept": "application/csv",
    }

    # Format times for InfluxDB query (UTC)
    start_str = start_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = stop_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Auto-detect field name if not provided
    if field_name is None:
        # Sensors use "value", climate entities use "current_temperature"
        field_name = "current_temperature" if domain == "climate" else "value"

    # Flux query to get all data points in time range
    flux_query = f"""from(bucket: "home_assistant/autogen")
                    |> range(start: {start_str}, stop: {end_str})
                    |> filter(fn: (r) => r["_measurement"] == "{domain}.{sensor_name}")
                    |> filter(fn: (r) => r["_field"] == "{field_name}")
                    |> filter(fn: (r) => r["domain"] == "{domain}")
                    |> sort(columns: ["_time"])
                    """

    try:
        response = requests.post(
            url=url,
            auth=(username, password),
            headers=headers,
            data=flux_query,
            timeout=30,
        )

        if response.status_code == 204:
            _LOGGER.warning(f"No data found for sensor {sensor_name}")
            return {"status": "error", "message": "No data found"}

        if response.status_code != 200:
            _LOGGER.error(f"Error from InfluxDB: {response.status_code}")
            return {
                "status": "error",
                "message": f"InfluxDB error: {response.status_code}",
            }

        # Parse CSV response
        timeseries_data = _parse_timeseries_response(response.text, local_tz, parse_as_string)

        if not timeseries_data:
            return {"status": "error", "message": "No valid data points found"}

        _LOGGER.info(
            f"Fetched {len(timeseries_data)} data points for {sensor_name} "
            f"({start_time.strftime('%Y-%m-%d %H:%M')} to {stop_time.strftime('%Y-%m-%d %H:%M')})"
        )

        return {"status": "success", "data": timeseries_data}

    except requests.RequestException as e:
        _LOGGER.error(f"Error connecting to InfluxDB: {e}")
        return {"status": "error", "message": f"Connection error: {e!s}"}
    except Exception as e:
        _LOGGER.error(f"Unexpected error: {e}")
        return {"status": "error", "message": f"Unexpected error: {e!s}"}


def _parse_timeseries_response(response_text: str, local_tz: ZoneInfo, parse_as_string: bool = False) -> list:
    """Parse InfluxDB CSV response to extract time series data.

    Args:
        response_text: CSV response from InfluxDB
        local_tz: Timezone to convert timestamps to
        parse_as_string: If True, keep values as strings instead of converting to float

    Returns:
        List of (timestamp, value) tuples sorted by timestamp
    """
    data_points = []
    lines = response_text.strip().split("\n")

    # Skip metadata rows (lines starting with '#')
    data_lines = [line for line in lines if not line.startswith("#")]

    # Process each data line
    for line in data_lines:
        parts = line.split(",")
        try:
            # Ensure the line has enough parts
            if len(parts) < 7 or parts[6] == "_value":
                continue

            # Extract timestamp and value
            timestamp_str = parts[5].strip()  # _time is the 6th column (index 5)

            # Parse value as string or float based on parameter
            if parse_as_string:
                value = parts[6].strip()  # Keep as string
            else:
                value = float(parts[6].strip())  # Convert to float

            # Parse timestamp and convert to local timezone
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            timestamp_local = timestamp.astimezone(local_tz)

            data_points.append((timestamp_local, value))

        except (IndexError, ValueError, TypeError) as e:
            _LOGGER.debug(f"Failed to parse line: {line}, error: {e}")
            continue

    # Sort by timestamp
    data_points.sort(key=lambda x: x[0])

    _LOGGER.debug(f"Parsed {len(data_points)} data points from InfluxDB response")
    return data_points
