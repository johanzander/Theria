"""
Thermal Learning Service

Background service that continuously learns thermal characteristics for each zone.
Runs every 15 minutes to collect measurements and update zone models.
"""

import asyncio
import logging
from datetime import datetime, timezone

from .entity_thermal_learner import ClimateEntityThermalLearner, ZoneThermalAggregator
from .ha_client import HAClient
from .history import history_tracker
from .settings import ZoneSettings
from .zone_thermal_learner import ZoneThermalLearner

logger = logging.getLogger(__name__)


class ThermalLearningService:
    """
    Background service for continuous thermal learning.

    Tracks temperature changes per climate entity and learns:
    - Heating rate (how fast entity heats up) across multiple timeframes
    - Cooling rate (how fast entity cools down) across multiple timeframes
    - Aggregates entity data to zone level
    """

    def __init__(
        self,
        ha_client: HAClient,
        zones: list[ZoneSettings],
        learning_interval_minutes: int = 15,
        outdoor_temp_sensor: str = "sensor.theria_outdoor_temp"
    ):
        self.ha_client = ha_client
        self.zones = zones
        self.learning_interval_minutes = learning_interval_minutes
        self.outdoor_temp_sensor = outdoor_temp_sensor
        self.history_service = None  # Will be set later

        # V2: Create learner for each climate entity (not just each zone)
        self.entity_learners: dict[str, ClimateEntityThermalLearner] = {}
        self.zone_aggregators: dict[str, ZoneThermalAggregator] = {}

        for zone in zones:
            zone_entity_learners = []

            for entity_id in zone.climate_entities:
                learner = ClimateEntityThermalLearner(
                    entity_id=entity_id,
                    max_history_hours=168,  # 7 days
                    min_samples_for_confidence=100
                )
                self.entity_learners[entity_id] = learner
                zone_entity_learners.append(learner)

            # Create aggregator for this zone
            self.zone_aggregators[zone.id] = ZoneThermalAggregator(
                zone_id=zone.id,
                entity_learners=zone_entity_learners
            )

        # Keep old zone learners for backward compatibility during migration
        self.learners: dict[str, ZoneThermalLearner] = {}
        for zone in zones:
            self.learners[zone.id] = ZoneThermalLearner(
                zone_id=zone.id,
                learning_rate=0.1,
                measurement_interval_minutes=learning_interval_minutes
            )

        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the thermal learning service."""
        if self._running:
            logger.warning("Thermal learning service already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"🧠 Thermal learning service started for {len(self.learners)} zone(s)")
        logger.info(f"   Learning interval: {self.learning_interval_minutes} minutes")

    async def bootstrap_from_history(self):
        """Bootstrap thermal learning from historical data. Call after history_service is set."""
        if not self.history_service:
            logger.warning("Cannot bootstrap - history service not set")
            return

        await self._bootstrap_from_history()

    async def stop(self):
        """Stop the thermal learning service."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("🧠 Thermal learning service stopped")

    async def _run_loop(self):
        """Main learning loop - collects measurements every interval."""
        logger.info("🧠 Thermal learning loop starting...")

        while self._running:
            try:
                await self._collect_measurements()
            except Exception as e:
                logger.error(f"Error in thermal learning loop: {e}", exc_info=True)

            # Sleep until next interval
            await asyncio.sleep(self.learning_interval_minutes * 60)

    async def _collect_measurements(self):
        """Collect measurements for all climate entities and update learners."""
        timestamp = datetime.now(timezone.utc)

        # Get outdoor temperature
        outdoor_temp = await self._get_outdoor_temp()
        if outdoor_temp is None:
            logger.warning("Failed to get outdoor temperature, skipping measurement")
            return

        logger.debug(f"🌡️  Outdoor temp: {outdoor_temp:.1f}°C")

        # Process each zone
        for zone in self.zones:
            # V1 (old): Zone-level learning for backward compatibility
            old_learner = self.learners[zone.id]

            try:
                # Get zone average indoor temperature
                indoor_temp = await self._get_zone_indoor_temp(zone)
                if indoor_temp is None:
                    logger.warning(f"Zone {zone.id}: Failed to get indoor temp")
                    continue

                # Determine if heating is active
                heating_active = await self._is_zone_heating(zone)

                # Add measurement to old learner
                measurement = old_learner.add_measurement(
                    timestamp=timestamp,
                    indoor_temp=indoor_temp,
                    outdoor_temp=outdoor_temp,
                    heating_active=heating_active
                )

                if measurement:
                    chars = old_learner.get_characteristics()

                    # Store thermal characteristics snapshot for historical tracking
                    history_tracker.add_thermal_snapshot(
                        zone_id=zone.id,
                        heating_rate=chars.heating_rate,
                        heating_rate_confidence=chars.heating_rate_confidence,
                        heating_samples=chars.heating_samples,
                        cooling_rate_base=chars.cooling_rate_base,
                        cooling_rate_confidence=chars.cooling_rate_confidence,
                        cooling_samples=chars.cooling_samples,
                        outdoor_temp_coefficient=chars.outdoor_temp_coefficient
                    )

            except Exception as e:
                logger.error(f"Error learning for zone {zone.id} (old): {e}", exc_info=True)

            # V2 (new): Per-entity learning
            for entity_id in zone.climate_entities:
                try:
                    entity_learner = self.entity_learners[entity_id]

                    # Get temperature for this specific entity
                    entity_temp = await self._get_entity_temp(entity_id)
                    if entity_temp is None:
                        logger.warning(f"Entity {entity_id}: Failed to get temperature")
                        continue

                    # Check if this entity is heating
                    entity_heating = await self._is_entity_heating(entity_id)

                    # Get target temperature
                    entity_target = await self._get_entity_target_temp(entity_id)

                    # Add measurement to entity learner
                    entity_learner.add_measurement(
                        timestamp=timestamp,
                        temperature=entity_temp,
                        outdoor_temp=outdoor_temp,
                        heating_active=entity_heating,
                        target_temp=entity_target
                    )

                    # Log entity characteristics (24h timeframe)
                    chars_24h = entity_learner.get_characteristics("24h")
                    logger.info(
                        f"Entity {entity_id}: "
                        f"Temp {entity_temp:.1f}°C, "
                        f"Heating {'ON' if entity_heating else 'OFF'}, "
                        f"24h rates: Heat {chars_24h.heating_rate:+.2f}°C/h "
                        f"({chars_24h.heating_samples} samples, {chars_24h.heating_confidence:.0%} conf), "
                        f"Cool {chars_24h.cooling_rate:+.2f}°C/h "
                        f"({chars_24h.cooling_samples} samples, {chars_24h.cooling_confidence:.0%} conf)"
                    )

                except Exception as e:
                    logger.error(f"Error learning for entity {entity_id}: {e}", exc_info=True)

            # Log zone aggregate
            try:
                aggregator = self.zone_aggregators[zone.id]
                zone_chars = aggregator.get_aggregate_characteristics("24h", weighted=True)
                logger.info(
                    f"Zone {zone.id} (aggregate): "
                    f"Heat {zone_chars.heating_rate:+.2f}°C/h "
                    f"({zone_chars.heating_confidence:.0%} conf), "
                    f"Cool {zone_chars.cooling_rate:+.2f}°C/h "
                    f"({zone_chars.cooling_confidence:.0%} conf)"
                )
            except Exception as e:
                logger.error(f"Error aggregating zone {zone.id}: {e}", exc_info=True)

    async def _bootstrap_from_history(self):
        """Bootstrap thermal learning from historical data."""
        from .history import history_tracker

        logger.info("🧠 Bootstrapping thermal learning from historical data...")

        # Get 7 days of historical data
        bootstrap_hours = 24 * 7

        # Get outdoor temperature history once (shared across all entities)
        outdoor_history = await self._get_outdoor_temp_history_from_ha(bootstrap_hours)

        for zone in self.zones:
            # V1: Bootstrap old zone-level learner
            learner = self.learners[zone.id]

            try:
                # Get zone temperature history
                zone_history = history_tracker.get_temperature_history(
                    zone_id=zone.id,
                    hours=bootstrap_hours
                )

                if not zone_history or not outdoor_history:
                    logger.warning(f"Zone {zone.id}: No historical data available for bootstrap")
                    continue

                # Process historical readings for zone-level
                heating_count = 0
                cooling_count = 0

                for i in range(1, len(zone_history)):
                    prev_reading = zone_history[i - 1]
                    curr_reading = zone_history[i]

                    # Get timestamps
                    curr_timestamp = self._get_timestamp(curr_reading)
                    if curr_timestamp is None:
                        continue

                    # Get outdoor temp at this time (find closest match)
                    outdoor_temp = self._find_closest_outdoor_temp(
                        curr_timestamp,
                        outdoor_history
                    )

                    if outdoor_temp is None:
                        continue

                    # Calculate indoor temp (average of climate entities)
                    prev_indoor = self._calculate_zone_temp(prev_reading)
                    curr_indoor = self._calculate_zone_temp(curr_reading)

                    if prev_indoor is None or curr_indoor is None:
                        continue

                    # Determine if heating was active (check heating_power_request)
                    heating_requests = self._get_heating_requests(prev_reading)
                    heating_active = any(v > 0 for v in heating_requests.values()) if heating_requests else False

                    # Add measurement to old learner
                    measurement = learner.add_measurement(
                        timestamp=curr_timestamp,
                        indoor_temp=curr_indoor,
                        outdoor_temp=outdoor_temp,
                        heating_active=heating_active
                    )

                    if measurement:
                        if heating_active:
                            heating_count += 1
                        else:
                            cooling_count += 1

                chars = learner.get_characteristics()
                logger.info(
                    f"Zone {zone.id} (V1): Bootstrapped with {heating_count} heating + "
                    f"{cooling_count} cooling samples → "
                    f"Heat: {chars.heating_rate:+.2f}°C/h ({chars.heating_rate_confidence:.0%}), "
                    f"Cool: {chars.cooling_rate_base:+.2f}°C/h ({chars.cooling_rate_confidence:.0%})"
                )

            except Exception as e:
                logger.error(f"Error bootstrapping zone {zone.id} (V1): {e}", exc_info=True)

            # V2: Bootstrap new entity-level learners
            for entity_id in zone.climate_entities:
                try:
                    entity_learner = self.entity_learners[entity_id]

                    # Get zone temperature history (contains all entities)
                    zone_history = history_tracker.get_temperature_history(
                        zone_id=zone.id,
                        hours=bootstrap_hours
                    )

                    if not zone_history or not outdoor_history:
                        continue

                    # NOTE: Resampling historical data to regular intervals causes artifacts
                    # (duplicate values followed by big jumps, resulting in unrealistic rates).
                    # Real-time collection (every 15min) naturally gives regular intervals.
                    # For bootstrap, use original irregular intervals as-is.

                    entity_heating_count = 0
                    entity_cooling_count = 0

                    for i in range(1, len(zone_history)):
                        prev_reading = zone_history[i - 1]
                        curr_reading = zone_history[i]

                        # Get timestamps
                        curr_timestamp = self._get_timestamp(curr_reading)
                        if curr_timestamp is None:
                            continue

                        # Get outdoor temp
                        outdoor_temp = self._find_closest_outdoor_temp(
                            curr_timestamp,
                            outdoor_history
                        )

                        if outdoor_temp is None:
                            continue

                        # Get entity-specific temperature from current_temps dict
                        prev_temps = self._get_current_temps(prev_reading)
                        curr_temps = self._get_current_temps(curr_reading)

                        if not prev_temps or not curr_temps:
                            continue

                        if entity_id not in prev_temps or entity_id not in curr_temps:
                            continue

                        try:
                            prev_entity_temp = float(prev_temps[entity_id])
                            curr_entity_temp = float(curr_temps[entity_id])
                        except (ValueError, TypeError):
                            continue

                        # Get heating request for this entity
                        heating_requests = self._get_heating_requests(prev_reading)
                        entity_heating = (
                            heating_requests.get(entity_id, 0) > 0
                            if heating_requests else False
                        )

                        # Get target temperature for this entity
                        target_temps = self._get_target_temps(curr_reading)
                        entity_target = target_temps.get(entity_id) if target_temps else None
                        if entity_target is not None:
                            try:
                                entity_target = float(entity_target)
                            except (ValueError, TypeError):
                                entity_target = None

                        # Add measurement to entity learner
                        entity_learner.add_measurement(
                            timestamp=curr_timestamp,
                            temperature=curr_entity_temp,
                            outdoor_temp=outdoor_temp,
                            heating_active=entity_heating,
                            target_temp=entity_target
                        )

                        if entity_heating:
                            entity_heating_count += 1
                        else:
                            entity_cooling_count += 1

                    # Log entity bootstrap results (24h timeframe)
                    chars_24h = entity_learner.get_characteristics("24h")
                    logger.info(
                        f"Entity {entity_id} (V2): Bootstrapped with "
                        f"{entity_heating_count} heating + {entity_cooling_count} cooling samples → "
                        f"24h Heat: {chars_24h.heating_rate:+.2f}°C/h ({chars_24h.heating_confidence:.0%}), "
                        f"24h Cool: {chars_24h.cooling_rate:+.2f}°C/h ({chars_24h.cooling_confidence:.0%})"
                    )

                except Exception as e:
                    logger.error(f"Error bootstrapping entity {entity_id} (V2): {e}", exc_info=True)

            # Log zone aggregate after bootstrap
            try:
                aggregator = self.zone_aggregators[zone.id]
                zone_chars = aggregator.get_aggregate_characteristics("24h", weighted=True)
                logger.info(
                    f"Zone {zone.id} (V2 aggregate): "
                    f"Heat {zone_chars.heating_rate:+.2f}°C/h ({zone_chars.heating_confidence:.0%}), "
                    f"Cool {zone_chars.cooling_rate:+.2f}°C/h ({zone_chars.cooling_confidence:.0%})"
                )
            except Exception as e:
                logger.error(f"Error aggregating zone {zone.id} after bootstrap: {e}", exc_info=True)

    def _resample_history_to_intervals(
        self,
        history: list,
        interval_minutes: int = 15
    ) -> list:
        """
        Resample irregular historical data to regular time intervals using bucket aggregation.

        Divides time into fixed buckets and takes the LAST reading in each bucket.
        This preserves step changes (important for step-response detection) while
        creating uniform intervals, similar to InfluxDB's GROUP BY time(15m) LAST.

        Args:
            history: List of historical readings
            interval_minutes: Interval size in minutes (default: 15)

        Returns:
            List of resampled readings at regular intervals
        """
        if not history:
            return []

        from datetime import datetime, timedelta

        # Get time range
        first_ts = self._get_timestamp(history[0])
        last_ts = self._get_timestamp(history[-1])

        if not first_ts or not last_ts:
            return history  # Fallback to original if timestamps missing

        # Sort history by timestamp
        sorted_history = sorted(history, key=lambda r: self._get_timestamp(r) or datetime.min.replace(tzinfo=timezone.utc))

        # Create buckets
        interval_delta = timedelta(minutes=interval_minutes)
        buckets = {}

        # Assign each reading to a bucket
        for reading in sorted_history:
            reading_ts = self._get_timestamp(reading)
            if not reading_ts:
                continue

            # Calculate bucket start time (round down to nearest interval)
            minutes_since_first = (reading_ts - first_ts).total_seconds() / 60
            bucket_index = int(minutes_since_first / interval_minutes)
            bucket_start = first_ts + (interval_delta * bucket_index)

            # Store reading in bucket (will keep overwriting to get LAST reading in bucket)
            buckets[bucket_start] = reading

        # Create resampled list with regular intervals
        resampled = []
        current_time = first_ts
        last_reading = None  # Track last seen reading for forward-fill

        while current_time <= last_ts:
            if current_time in buckets:
                # Use reading from this bucket
                reading = buckets[current_time]
                last_reading = reading
            elif last_reading is not None:
                # Forward-fill: no reading in this bucket, use last known reading
                reading = last_reading
            else:
                # No data yet, skip this bucket
                current_time += interval_delta
                continue

            # Create resampled reading with interval timestamp
            resampled_reading = dict(reading) if isinstance(reading, dict) else reading
            if isinstance(resampled_reading, dict):
                resampled_reading['timestamp'] = current_time.isoformat()
            resampled.append(resampled_reading)

            current_time += interval_delta

        logger.debug(
            f"Resampled history: {len(history)} irregular readings → "
            f"{len(resampled)} regular {interval_minutes}-min intervals ({len(buckets)} buckets with data)"
        )

        return resampled

    async def _get_outdoor_temp_history_from_ha(self, hours: int) -> list[dict]:
        """Get historical outdoor temperature readings from Home Assistant."""
        from datetime import datetime, timedelta
        import requests

        try:
            # Get historical data from HA
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            url = f"{self.ha_client.base_url}/api/history/period/{start_time.isoformat()}"
            params = {
                "filter_entity_id": self.outdoor_temp_sensor,
                "end_time": end_time.isoformat()
            }

            response = requests.get(
                url,
                headers=self.ha_client.headers,
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get outdoor temp history: HTTP {response.status_code}")
                return []

            data = response.json()
            if not data or len(data) == 0:
                return []

            history = data[0]  # First array contains our entity's history

            # Convert to our format
            return [
                {
                    "timestamp": datetime.fromisoformat(reading["last_changed"].replace("Z", "+00:00")),
                    "value": float(reading["state"])
                }
                for reading in history
                if reading.get("state") not in ["unknown", "unavailable", None]
            ]
        except Exception as e:
            logger.error(f"Failed to get outdoor temp history from HA: {e}")
            return []

    def _find_closest_outdoor_temp(self, timestamp, outdoor_history: list[dict]) -> float | None:
        """Find outdoor temperature closest to given timestamp."""
        if not outdoor_history:
            return None

        # Find closest reading (within 30 minutes)
        from datetime import timedelta
        closest = None
        min_diff = None

        for reading in outdoor_history:
            diff = abs((reading["timestamp"] - timestamp).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff = diff
                closest = reading

        # Only use if within 30 minutes
        if min_diff and min_diff < 1800:
            return closest["value"]

        return None

    def _get_timestamp(self, reading):
        """Extract timestamp from reading."""
        from datetime import datetime
        if isinstance(reading, dict):
            ts = reading.get("timestamp")
            if isinstance(ts, str):
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return ts
        return getattr(reading, "timestamp", None)

    def _get_heating_requests(self, reading) -> dict:
        """Extract heating requests from reading."""
        if isinstance(reading, dict):
            return reading.get("heating_requests", {})
        return getattr(reading, "heating_requests", {})

    def _get_current_temps(self, reading) -> dict:
        """Extract current_temps dict from reading."""
        if isinstance(reading, dict):
            return reading.get("current_temps", {})
        return getattr(reading, "current_temps", {}) if hasattr(reading, "current_temps") else {}

    def _get_target_temps(self, reading) -> dict:
        """Extract target_temps dict from reading."""
        if isinstance(reading, dict):
            return reading.get("target_temps", {})
        return getattr(reading, "target_temps", {}) if hasattr(reading, "target_temps") else {}

    def _extract_outdoor_temp(self, reading) -> float | None:
        """Extract outdoor temperature from reading."""
        if isinstance(reading, dict):
            return reading.get("outdoor_temp")
        return getattr(reading, "outdoor_temp", None)

    async def _get_entity_target_temp(self, entity_id: str) -> float | None:
        """Get target/setpoint temperature for a specific climate entity."""
        try:
            state = self.ha_client.get_state(entity_id)
            if state and "attributes" in state:
                target = state["attributes"].get("temperature")
                if target is not None:
                    return float(target)
        except Exception as e:
            logger.debug(f"Failed to get target temp from {entity_id}: {e}")

        return None

    def _calculate_zone_temp(self, reading) -> float | None:
        """Calculate average zone temperature from history reading."""
        # reading is a dict from history_tracker
        if isinstance(reading, dict):
            current_temps = reading.get("current_temps", {})
        else:
            # If it's a model object
            current_temps = getattr(reading, "current_temps", {}) if hasattr(reading, "current_temps") else {}

        if not current_temps:
            return None

        # Use climate entity temperatures
        temps = []
        for entity_id, temp in current_temps.items():
            if entity_id.startswith("climate."):
                try:
                    temps.append(float(temp))
                except (ValueError, TypeError):
                    continue

        if not temps:
            return None

        return sum(temps) / len(temps)

    async def _get_outdoor_temp(self) -> float | None:
        """Get current outdoor temperature."""
        try:
            state = self.ha_client.get_state(self.outdoor_temp_sensor)
            if state:
                return float(state["state"])
        except Exception as e:
            logger.error(f"Failed to get outdoor temp from {self.outdoor_temp_sensor}: {e}")

        return None

    async def _get_zone_indoor_temp(self, zone: ZoneSettings) -> float | None:
        """Get current indoor temperature for zone."""
        # Get temperature from first climate entity
        if not zone.climate_entities:
            logger.warning(f"Zone {zone.id}: No climate entities configured")
            return None

        climate_entity = zone.climate_entities[0]

        try:
            state = self.ha_client.get_state(climate_entity)
            if state:
                # Try current_temperature attribute first
                if "attributes" in state and "current_temperature" in state["attributes"]:
                    return float(state["attributes"]["current_temperature"])
                # Fallback to state
                return float(state["state"])
        except Exception as e:
            logger.error(f"Failed to get indoor temp from {climate_entity}: {e}")

        return None

    async def _is_zone_heating(self, zone: ZoneSettings) -> bool:
        """
        Determine if zone is currently heating.

        Checks heating_power_request attribute if available,
        otherwise falls back to comparing current vs target temp.
        """
        if not zone.climate_entities:
            return False

        climate_entity = zone.climate_entities[0]

        try:
            state = self.ha_client.get_state(climate_entity)
            if not state or "attributes" not in state:
                return False

            attrs = state["attributes"]

            # Try heating_power_request first (most accurate)
            if "heating_power_request" in attrs:
                return float(attrs["heating_power_request"]) > 0

            # Fallback: compare current vs target temp
            current_temp = attrs.get("current_temperature")
            target_temp = attrs.get("temperature")

            if current_temp is not None and target_temp is not None:
                # Assume heating if current is significantly below target
                return float(current_temp) < float(target_temp) - 0.5

        except Exception as e:
            logger.error(f"Failed to check heating status for {climate_entity}: {e}")

        return False

    async def _get_entity_temp(self, entity_id: str) -> float | None:
        """Get current temperature for a specific climate entity."""
        try:
            state = self.ha_client.get_state(entity_id)
            if state:
                # Try current_temperature attribute first
                if "attributes" in state and "current_temperature" in state["attributes"]:
                    return float(state["attributes"]["current_temperature"])
                # Fallback to state
                return float(state["state"])
        except Exception as e:
            logger.error(f"Failed to get temperature from {entity_id}: {e}")

        return None

    async def _is_entity_heating(self, entity_id: str) -> bool:
        """
        Determine if a specific climate entity is currently heating.

        Checks heating_power_request attribute if available,
        otherwise falls back to comparing current vs target temp.
        """
        try:
            state = self.ha_client.get_state(entity_id)
            if not state or "attributes" not in state:
                return False

            attrs = state["attributes"]

            # Try heating_power_request first (most accurate)
            if "heating_power_request" in attrs:
                return float(attrs["heating_power_request"]) > 0

            # Fallback: compare current vs target temp
            current_temp = attrs.get("current_temperature")
            target_temp = attrs.get("temperature")

            if current_temp is not None and target_temp is not None:
                # Assume heating if current is significantly below target
                return float(current_temp) < float(target_temp) - 0.5

        except Exception as e:
            logger.error(f"Failed to check heating status for {entity_id}: {e}")

        return False

    def get_zone_characteristics(self, zone_id: str):
        """Get learned characteristics for a zone."""
        if zone_id not in self.learners:
            return None

        learner = self.learners[zone_id]
        return {
            "characteristics": learner.get_characteristics(),
            "confidence": learner.get_confidence(),
            "recent_measurements": len(learner.recent_measurements)
        }

    def get_all_characteristics(self):
        """Get learned characteristics for all zones."""
        return {
            zone_id: self.get_zone_characteristics(zone_id)
            for zone_id in self.learners
        }
