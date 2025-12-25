"""
Temperature History Collection Service

Background service that continuously collects temperature readings for historical tracking.
Runs independently of UI - collects data every minute regardless of whether anyone is viewing.
"""

import asyncio
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

import requests
import yaml

from .ha_client import HAClient
from .history import history_tracker
from .settings import ZoneSettings

logger = logging.getLogger(__name__)


class TemperatureHistoryService:
    """
    Background service for continuous temperature history collection.

    Collects temperature readings from all zones at regular intervals
    and stores them in the history tracker for chart visualization.
    """

    def __init__(
        self,
        ha_client: HAClient,
        zones: list[ZoneSettings],
        collection_interval_seconds: int = 60
    ):
        self.ha_client = ha_client
        self.zones = zones
        self.collection_interval_seconds = collection_interval_seconds

        self._task: asyncio.Task | None = None
        self._running = False

        # Load InfluxDB configuration (optional)
        self.influxdb_config = self._load_influxdb_config()
        self.influxdb_enabled = self.influxdb_config.get("enabled", False)

    def _load_influxdb_config(self) -> dict:
        """Load InfluxDB configuration from options.json or config.yaml."""
        config = {}

        try:
            # Try Home Assistant options.json first (production)
            if os.path.exists("/data/options.json"):
                import json
                with open("/data/options.json") as f:
                    options = json.load(f)
                    config = options.get("influxdb", {})
                    logger.debug("Loaded InfluxDB config from options.json")
        except Exception as e:
            logger.debug(f"Could not load options.json: {e}")

        # Fallback to config.yaml (development)
        if not config:
            try:
                config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        yaml_config = yaml.safe_load(f)
                        config = yaml_config.get("options", {}).get("influxdb", {})
                        logger.debug("Loaded InfluxDB config from config.yaml")
            except Exception as e:
                logger.debug(f"Could not load config.yaml: {e}")

        return config

    async def start(self):
        """Start the temperature history collection service."""
        if self._running:
            logger.warning("Temperature history service already running")
            return

        self._running = True

        # Backfill historical data from Home Assistant before starting real-time collection
        logger.info("📊 Backfilling historical data from Home Assistant...")
        await self._backfill_history()

        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"📊 Temperature history service started for {len(self.zones)} zone(s)")
        logger.info(f"   Collection interval: {self.collection_interval_seconds} seconds")

    async def stop(self):
        """Stop the temperature history collection service."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("📊 Temperature history service stopped")

    async def _run_loop(self):
        """Main collection loop - collects temperature readings every interval."""
        logger.info("📊 Temperature history collection loop starting...")

        while self._running:
            try:
                await self._collect_readings()
                await self._collect_heating_power()
            except Exception as e:
                logger.error(f"Error in temperature history collection loop: {e}", exc_info=True)

            # Sleep until next interval
            await asyncio.sleep(self.collection_interval_seconds)

    async def _collect_readings(self):
        """Collect temperature readings for all zones and store in history."""
        for zone in self.zones:
            try:
                # Read all temperature sensors
                temperatures = []
                for sensor in zone.temp_sensors:
                    try:
                        temp = self.ha_client.get_temperature(sensor)
                        temperatures.append(temp)
                    except Exception as e:
                        logger.warning(f"Failed to read {sensor}: {e}")

                # Calculate average of valid temperatures
                valid_temps = [t for t in temperatures if t is not None]
                if not valid_temps:
                    continue

                avg_temp = sum(valid_temps) / len(valid_temps)

                # Read climate entity states for target temps and heating requests
                target_temps = {}
                current_temps = {}
                heating_requests = {}
                for climate_entity in zone.climate_entities:
                    try:
                        full_state = self.ha_client.get_state(climate_entity)
                        attributes = full_state.get("attributes", {})
                        target_temps[climate_entity] = attributes.get("temperature")
                        current_temps[climate_entity] = attributes.get("current_temperature")
                        heating_power = attributes.get("heating_power_request")
                        if heating_power is not None:
                            heating_requests[climate_entity] = heating_power
                    except Exception as e:
                        logger.warning(f"Failed to read {climate_entity}: {e}")

                # Store temperature reading for historical tracking
                history_tracker.add_temperature_reading(
                    zone_id=zone.id,
                    current_temp=avg_temp,
                    scheduled_temp=None,  # No scheduler in current implementation
                    target_temps=target_temps,
                    current_temps=current_temps,
                    heating_requests=heating_requests
                )

                logger.debug(f"Zone {zone.id}: Stored temp reading {avg_temp:.1f}°C")

            except Exception as e:
                logger.error(f"Error collecting temperature for zone {zone.id}: {e}", exc_info=True)

    async def _collect_heating_power(self):
        """Collect heating power requests for all zones."""
        for zone in self.zones:
            try:
                total_heating_request = 0
                heating_request_count = 0
                max_heating_request = 0.0

                for climate_entity in zone.climate_entities:
                    try:
                        full_state = self.ha_client.get_state(climate_entity)
                        heating_power_request = full_state.get("attributes", {}).get("heating_power_request")

                        if heating_power_request is not None:
                            total_heating_request += float(heating_power_request)
                            heating_request_count += 1
                            max_heating_request = max(max_heating_request, float(heating_power_request))

                    except Exception as e:
                        logger.warning(f"Failed to read heating power from {climate_entity}: {e}")

                # Store heating power snapshot
                if heating_request_count > 0:
                    avg_heating_request = total_heating_request / heating_request_count
                    is_heating = avg_heating_request > 0

                    history_tracker.add_heating_power_snapshot(
                        zone_id=zone.id,
                        avg_heating_request=avg_heating_request,
                        max_heating_request=max_heating_request,
                        heating_active=is_heating
                    )

                    logger.debug(f"Zone {zone.id}: Stored heating power {avg_heating_request:.0f}%")

            except Exception as e:
                logger.error(f"Error collecting heating power for zone {zone.id}: {e}", exc_info=True)

    async def _backfill_history(self):
        """Backfill historical data on startup.

        Default: Uses HA Recorder (7 days) - works out of the box
        Optional: Uses InfluxDB (30 days) - if configured for longer history
        """
        end_time = datetime.now()

        # Determine backfill source and time range
        if self.influxdb_enabled:
            start_time = end_time - timedelta(days=30)
            source = "InfluxDB"
            use_influxdb = True
            logger.info(f"📊 Backfilling from InfluxDB (30 days): {start_time} to {end_time}")
        else:
            start_time = end_time - timedelta(days=7)
            source = "HA Recorder"
            use_influxdb = False
            logger.info(f"📊 Backfilling from HA Recorder (7 days): {start_time} to {end_time}")

        for zone in self.zones:
            try:
                # Fetch historical data for all sensors in the zone
                sensor_histories = {}

                for sensor_entity in zone.temp_sensors:
                    try:
                        if use_influxdb:
                            history_data = self._fetch_influxdb_history(
                                sensor_entity,
                                start_time,
                                end_time
                            )
                        else:
                            history_data = self._fetch_ha_history(
                                sensor_entity,
                                start_time,
                                end_time
                            )

                        if history_data:
                            sensor_histories[sensor_entity] = history_data
                            logger.info(f"  {sensor_entity}: Retrieved {len(history_data)} historical readings")

                    except Exception as e:
                        logger.warning(f"Failed to backfill {sensor_entity}: {e}")

                # Also get climate entity history for target temps
                climate_histories = {}
                for climate_entity in zone.climate_entities:
                    try:
                        if use_influxdb:
                            history_data = self._fetch_influxdb_history(
                                climate_entity,
                                start_time,
                                end_time,
                                domain="climate"
                            )
                        else:
                            history_data = self._fetch_ha_history(
                                climate_entity,
                                start_time,
                                end_time
                            )

                        if history_data:
                            climate_histories[climate_entity] = history_data
                            logger.info(f"  {climate_entity}: Retrieved {len(history_data)} historical readings")

                    except Exception as e:
                        logger.warning(f"Failed to backfill {climate_entity}: {e}")

                # Aggregate and store the historical data
                stored_count = self._aggregate_and_store_history(
                    zone,
                    sensor_histories,
                    climate_histories,
                    start_time,
                    end_time
                )

                logger.info(f"Zone {zone.id}: Stored {stored_count} historical readings")

            except Exception as e:
                logger.error(f"Error backfilling history for zone {zone.id}: {e}", exc_info=True)

        logger.info("✅ Historical data backfill complete")

    def _fetch_ha_history(self, entity_id: str, start_time: datetime, end_time: datetime) -> list:
        """Fetch historical data for an entity from Home Assistant Recorder.

        Uses the built-in HA History API (/api/history/period) which queries
        the recorder database (SQLite/PostgreSQL/MySQL).

        Args:
            entity_id: Entity ID (e.g., 'sensor.outdoor', 'climate.living_room')
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of state change dictionaries
        """
        try:
            # Format timestamps for HA API (ISO format)
            start_iso = start_time.isoformat()

            # Query HA History API
            url = f"{self.ha_client.base_url}/api/history/period/{start_iso}"
            params = {
                "filter_entity_id": entity_id,
                "end_time": end_time.isoformat()
            }

            response = requests.get(
                url,
                headers=self.ha_client.headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                # HA returns array of arrays - one array per entity
                data = response.json()
                if data and len(data) > 0:
                    return data[0]  # First array contains our entity's history

            return []

        except Exception as e:
            logger.warning(f"Failed to fetch HA history for {entity_id}: {e}")
            return []

    def _fetch_influxdb_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
        domain: str = "sensor"
    ) -> list:
        """Fetch historical data for an entity from InfluxDB.

        Args:
            entity_id: Full entity ID (e.g., 'sensor.outdoor', 'climate.living_room')
            start_time: Start of time range
            end_time: End of time range
            domain: Entity domain ('sensor' or 'climate')

        Returns:
            List of state change dictionaries compatible with HA History API format
        """
        try:
            from .influxdb_helper import get_sensor_timeseries

            # Extract sensor name from entity_id (remove domain prefix)
            sensor_name = entity_id.replace(f"{domain}.", "")

            # Determine field name based on domain
            if domain == "climate":
                # For climate, we'll fetch both current and target temperature
                # First fetch current_temperature
                result_current = get_sensor_timeseries(
                    sensor_name=sensor_name,
                    start_time=start_time,
                    stop_time=end_time,
                    domain=domain,
                    field_name="current_temperature"
                )

                # Then fetch target temperature
                result_target = get_sensor_timeseries(
                    sensor_name=sensor_name,
                    start_time=start_time,
                    stop_time=end_time,
                    domain=domain,
                    field_name="temperature"
                )

                # Convert to HA History API format
                history_data = []

                # Merge both datasets by timestamp
                if result_current.get("status") == "success":
                    for timestamp, value in result_current.get("data", []):
                        history_data.append({
                            "last_changed": timestamp.isoformat(),
                            "last_updated": timestamp.isoformat(),
                            "state": str(value),
                            "attributes": {"current_temperature": value}
                        })

                # Add target temperature data
                if result_target.get("status") == "success":
                    target_map = {ts: val for ts, val in result_target.get("data", [])}
                    for entry in history_data:
                        ts = datetime.fromisoformat(entry["last_changed"])
                        if ts in target_map:
                            entry["attributes"]["temperature"] = target_map[ts]

                return history_data

            else:
                # For sensors, fetch the "value" field
                result = get_sensor_timeseries(
                    sensor_name=sensor_name,
                    start_time=start_time,
                    stop_time=end_time,
                    domain=domain,
                    field_name="value"
                )

                if result.get("status") != "success":
                    logger.warning(f"InfluxDB query failed for {entity_id}: {result.get('message')}")
                    return []

                # Convert to HA History API format
                history_data = []
                for timestamp, value in result.get("data", []):
                    history_data.append({
                        "last_changed": timestamp.isoformat(),
                        "last_updated": timestamp.isoformat(),
                        "state": str(value),
                        "attributes": {}
                    })

                return history_data

        except Exception as e:
            logger.error(f"Failed to fetch InfluxDB history for {entity_id}: {e}", exc_info=True)
            return []

    def _aggregate_and_store_history(
        self,
        zone,
        sensor_histories: dict,
        climate_histories: dict,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Aggregate historical sensor data and store in history tracker."""
        # Group all state changes by time bucket (1-minute intervals)
        time_buckets = defaultdict(lambda: {"temps": [], "targets": {}, "currents": {}, "heating_requests": {}})

        # Process sensor temperature history
        for entity_id, history in sensor_histories.items():
            for state_change in history:
                try:
                    timestamp_str = state_change.get("last_changed") or state_change.get("last_updated")
                    state_value = state_change.get("state")

                    if timestamp_str and state_value not in ("unknown", "unavailable", None):
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        # Round to nearest minute
                        bucket_time = timestamp.replace(second=0, microsecond=0)

                        try:
                            temp = float(state_value)
                            time_buckets[bucket_time]["temps"].append(temp)
                            # Also store individual sensor temps for visualization
                            time_buckets[bucket_time]["currents"][entity_id] = temp
                        except (ValueError, TypeError):
                            pass

                except Exception as e:
                    logger.debug(f"Error processing sensor history: {e}")

        # Process climate entity history for target temps
        for entity_id, history in climate_histories.items():
            for state_change in history:
                try:
                    timestamp_str = state_change.get("last_changed") or state_change.get("last_updated")
                    attributes = state_change.get("attributes", {})

                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        bucket_time = timestamp.replace(second=0, microsecond=0)

                        # Get target and current temperature from attributes
                        target_temp = attributes.get("temperature")
                        current_temp = attributes.get("current_temperature")
                        heating_request = attributes.get("heating_power_request")

                        if target_temp is not None:
                            time_buckets[bucket_time]["targets"][entity_id] = float(target_temp)
                        if current_temp is not None:
                            time_buckets[bucket_time]["currents"][entity_id] = float(current_temp)
                        if heating_request is not None:
                            time_buckets[bucket_time]["heating_requests"][entity_id] = float(heating_request)

                except Exception as e:
                    logger.debug(f"Error processing climate history: {e}")

        # Forward-fill target, current temps, and heating requests across all buckets
        # This ensures we have the last known values even when they don't change
        last_targets = {}
        last_currents = {}
        last_heating_requests = {}

        for bucket_time in sorted(time_buckets.keys()):
            bucket_data = time_buckets[bucket_time]

            # Update last known values
            if bucket_data["targets"]:
                last_targets.update(bucket_data["targets"])
            if bucket_data["currents"]:
                last_currents.update(bucket_data["currents"])
            if bucket_data["heating_requests"]:
                last_heating_requests.update(bucket_data["heating_requests"])

            # Apply forward-filled values to this bucket
            bucket_data["targets"] = dict(last_targets)
            bucket_data["currents"] = dict(last_currents)
            bucket_data["heating_requests"] = dict(last_heating_requests)

        # Store aggregated data in history tracker
        stored_count = 0
        for bucket_time in sorted(time_buckets.keys()):
            bucket_data = time_buckets[bucket_time]

            # Calculate average temperature from forward-filled current temps
            # This ensures the average matches what's displayed on charts
            current_temps_dict = bucket_data["currents"]
            if current_temps_dict:
                temps_list = list(current_temps_dict.values())
                avg_temp = sum(temps_list) / len(temps_list)

                # Store in history tracker with historical timestamp
                history_tracker.add_temperature_reading(
                    zone_id=zone.id,
                    current_temp=avg_temp,
                    scheduled_temp=None,
                    target_temps=bucket_data["targets"],
                    current_temps=bucket_data["currents"],
                    heating_requests=bucket_data["heating_requests"],
                    timestamp=bucket_time
                )

                # Store heating power snapshot if we have heating request data
                heating_requests = bucket_data["heating_requests"]
                if heating_requests:
                    request_values = list(heating_requests.values())
                    avg_heating_request = sum(request_values) / len(request_values)
                    max_heating_request = max(request_values)
                    heating_active = any(r > 0 for r in request_values)

                    # Store with historical timestamp (history_tracker is imported at module level)
                    history_tracker.add_heating_power_snapshot(
                        zone_id=zone.id,
                        avg_heating_request=avg_heating_request,
                        max_heating_request=max_heating_request,
                        heating_active=heating_active,
                        timestamp=bucket_time
                    )

                stored_count += 1

        return stored_count
