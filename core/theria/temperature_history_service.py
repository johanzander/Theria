"""
Temperature History Collection Service

Background service that continuously collects temperature readings for historical tracking.
Runs independently of UI - collects data every minute regardless of whether anyone is viewing.
"""

import asyncio
import logging
import requests
from collections import defaultdict
from datetime import datetime, timedelta

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

                # Read climate entity states for target temps
                target_temps = {}
                current_temps = {}
                for climate_entity in zone.climate_entities:
                    try:
                        state = self.ha_client.get_climate_state(climate_entity)
                        target_temps[climate_entity] = state.get("target_temperature")
                        current_temps[climate_entity] = state.get("current_temperature")
                    except Exception as e:
                        logger.warning(f"Failed to read {climate_entity}: {e}")

                # Store temperature reading for historical tracking
                history_tracker.add_temperature_reading(
                    zone_id=zone.id,
                    current_temp=avg_temp,
                    scheduled_temp=None,  # No scheduler in current implementation
                    target_temps=target_temps,
                    current_temps=current_temps
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
        """Backfill historical data from Home Assistant on startup."""
        # Calculate time range - get last 24 hours of history
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        logger.info(f"Fetching historical data from {start_time} to {end_time}")

        for zone in self.zones:
            try:
                # Fetch historical data for all sensors in the zone
                sensor_histories = {}

                for sensor_entity in zone.temp_sensors:
                    try:
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
        """Fetch historical data for an entity from Home Assistant."""
        try:
            # Format timestamps for HA API (ISO format)
            start_iso = start_time.isoformat()

            # Use HA client to get history
            # Note: We'll need to add this method to HAClient or use direct HTTP
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
        time_buckets = defaultdict(lambda: {"temps": [], "targets": {}, "currents": {}})

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

                        if target_temp is not None:
                            time_buckets[bucket_time]["targets"][entity_id] = float(target_temp)
                        if current_temp is not None:
                            time_buckets[bucket_time]["currents"][entity_id] = float(current_temp)

                except Exception as e:
                    logger.debug(f"Error processing climate history: {e}")

        # Store aggregated data in history tracker
        stored_count = 0
        for bucket_time in sorted(time_buckets.keys()):
            bucket_data = time_buckets[bucket_time]

            # Calculate average temperature for this bucket
            temps = bucket_data["temps"]
            if temps:
                avg_temp = sum(temps) / len(temps)

                # Store in history tracker
                history_tracker.add_temperature_reading(
                    zone_id=zone.id,
                    current_temp=avg_temp,
                    scheduled_temp=None,
                    target_temps=bucket_data["targets"],
                    current_temps=bucket_data["currents"]
                )

                stored_count += 1

        return stored_count
