"""
Thermal Learning Service

Background service that continuously learns thermal characteristics for each zone.
Runs every 15 minutes to collect measurements and update zone models.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from .ha_client import HAClient
from .zone_thermal_learner import ZoneThermalLearner
from .settings import ZoneSettings

logger = logging.getLogger(__name__)


class ThermalLearningService:
    """
    Background service for continuous thermal learning.

    Tracks temperature changes in each zone and learns:
    - Heating rate (how fast zone heats up)
    - Cooling rate (how fast zone cools down)
    - Outdoor temperature effects
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

        # Create learner for each zone
        self.learners: Dict[str, ZoneThermalLearner] = {}
        for zone in zones:
            self.learners[zone.id] = ZoneThermalLearner(
                zone_id=zone.id,
                learning_rate=0.1,  # 10% learning rate
                measurement_interval_minutes=learning_interval_minutes
            )

        self._task: Optional[asyncio.Task] = None
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
        """Collect measurements for all zones and update learners."""
        timestamp = datetime.now()

        # Get outdoor temperature
        outdoor_temp = await self._get_outdoor_temp()
        if outdoor_temp is None:
            logger.warning("Failed to get outdoor temperature, skipping measurement")
            return

        logger.debug(f"🌡️  Outdoor temp: {outdoor_temp:.1f}°C")

        # Process each zone
        for zone in self.zones:
            learner = self.learners[zone.id]

            try:
                # Get indoor temperature from climate entity
                indoor_temp = await self._get_zone_indoor_temp(zone)
                if indoor_temp is None:
                    logger.warning(f"Zone {zone.id}: Failed to get indoor temp")
                    continue

                # Determine if heating is active
                heating_active = await self._is_zone_heating(zone)

                # Add measurement to learner
                measurement = learner.add_measurement(
                    timestamp=timestamp,
                    indoor_temp=indoor_temp,
                    outdoor_temp=outdoor_temp,
                    heating_active=heating_active
                )

                if measurement:
                    chars = learner.get_characteristics()
                    confidence = learner.get_confidence()

                    logger.info(
                        f"Zone {zone.id}: "
                        f"Indoor {indoor_temp:.1f}°C, "
                        f"Heating {'ON' if heating_active else 'OFF'}, "
                        f"Change {measurement.temp_change:+.2f}°C → "
                        f"Heat rate: {chars.heating_rate:+.2f}°C/h "
                        f"(conf {chars.heating_rate_confidence:.0%}), "
                        f"Cool rate: {chars.cooling_rate_base:+.2f}°C/h "
                        f"(conf {chars.cooling_rate_confidence:.0%})"
                    )

            except Exception as e:
                logger.error(f"Error learning for zone {zone.id}: {e}", exc_info=True)

    async def _get_outdoor_temp(self) -> Optional[float]:
        """Get current outdoor temperature."""
        try:
            state = self.ha_client.get_state(self.outdoor_temp_sensor)
            if state:
                return float(state["state"])
        except Exception as e:
            logger.error(f"Failed to get outdoor temp from {self.outdoor_temp_sensor}: {e}")

        return None

    async def _get_zone_indoor_temp(self, zone: ZoneSettings) -> Optional[float]:
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
