"""
Sensor Health Monitoring

Background service that periodically checks every configured sensor
(zone temp sensors/climate entities, heat pump sensors) for two failure
modes:

- Unavailable: HA reports the entity as "unavailable" or "unknown"
- Stale: the entity has a value, but hasn't changed in longer than the
  configured staleness threshold (catches a sensor stuck reporting a
  plausible-looking but outdated reading)

Grouped by physical device (zone id, or "heat_pump") for a compact
dashboard banner, with individual sensor detail available separately.
Visibility only - does not influence any control or scheduling decisions.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .ha_client import HAClient
from .history import history_tracker
from .settings import HeatPumpSettings, ZoneSettings

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_UNAVAILABLE = "unavailable"


@dataclass
class SensorCheck:
    """Result of checking a single sensor entity."""

    entity_id: str
    device_group: str
    device_name: str
    status: str  # STATUS_OK, STATUS_STALE, STATUS_UNAVAILABLE
    message: str
    last_changed: str | None = None


@dataclass
class _Registration:
    entity_id: str
    device_group: str
    device_name: str


def _build_registry(
    zones: list[ZoneSettings], heat_pump_settings: HeatPumpSettings | None
) -> list[_Registration]:
    """Build the list of (entity_id, device_group, device_name) to monitor."""
    registry: list[_Registration] = []
    seen: set[str] = set()

    for zone in zones:
        if not zone.enabled:
            continue
        for entity_id in [*zone.climate_entities, *zone.temp_sensors]:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            registry.append(_Registration(entity_id, zone.id, zone.name))

    if heat_pump_settings and heat_pump_settings.enabled:
        heat_pump_entities = [
            heat_pump_settings.compressor_speed,
            heat_pump_settings.compressor_power,
            heat_pump_settings.mode_switch,
            heat_pump_settings.heat_carrier_forward,
            heat_pump_settings.heat_carrier_return,
            heat_pump_settings.compressor_consumption_total,
            heat_pump_settings.delivered_energy_total,
        ]
        for entity_id in heat_pump_entities:
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            registry.append(_Registration(entity_id, "heat_pump", "Heat Pump"))

    return registry


class HealthMonitorService:
    """Background service for continuous sensor health monitoring."""

    def __init__(
        self,
        ha_client: HAClient,
        zones: list[ZoneSettings],
        heat_pump_settings: HeatPumpSettings | None = None,
        check_interval_seconds: int = 300,
        stale_after_minutes: int = 120,
        startup_grace_seconds: int = 180,
    ):
        self.ha_client = ha_client
        self.check_interval_seconds = check_interval_seconds
        self.stale_after = timedelta(minutes=stale_after_minutes)
        self.startup_grace_seconds = startup_grace_seconds

        self._registry = _build_registry(zones, heat_pump_settings)
        self._task: asyncio.Task | None = None
        self._running = False
        self._started_at: datetime | None = None

        # entity_id -> last known status, so we only log transitions
        self._last_status: dict[str, str] = {}
        # entity_id -> latest SensorCheck, for the status endpoints
        self._latest: dict[str, SensorCheck] = {}
        self._last_check: datetime | None = None

    async def start(self):
        """Start the health monitoring service."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._started_at = datetime.now(UTC)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"🩺 Health monitor started ({len(self._registry)} sensors, "
            f"interval: {self.check_interval_seconds}s)"
        )

    async def stop(self):
        """Stop the health monitoring service."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("🩺 Health monitor stopped")

    async def _run_loop(self):
        """Main check loop - checks all sensors every interval."""
        while self._running:
            try:
                await self.recheck()
            except Exception as e:
                logger.exception(f"Error in health monitor loop: {e}")

            await asyncio.sleep(self.check_interval_seconds)

    async def recheck(self):
        """Check every registered sensor now and update cached state."""
        now = datetime.now(UTC)
        starting_up = (
            self._started_at is not None
            and (now - self._started_at).total_seconds() < self.startup_grace_seconds
        )

        for reg in self._registry:
            check = self._check_sensor(reg, now)
            self._latest[reg.entity_id] = check

            # During the startup grace period, don't record transitions or
            # update _last_status - entities may not have reported yet and
            # that's expected. Leaving _last_status untouched means a sensor
            # that's already broken when the grace period ends is correctly
            # seen as an OK -> bad transition on the first real check,
            # instead of being silently treated as "already known bad".
            if not starting_up:
                self._record_transition(reg, check)
                self._last_status[reg.entity_id] = check.status

        self._last_check = now

    def _check_sensor(self, reg: _Registration, now: datetime) -> SensorCheck:
        try:
            state = self.ha_client.get_state(reg.entity_id)
        except (ValueError, RuntimeError) as e:
            return SensorCheck(
                entity_id=reg.entity_id,
                device_group=reg.device_group,
                device_name=reg.device_name,
                status=STATUS_UNAVAILABLE,
                message=f"Failed to read: {e}",
            )

        value = state.get("state")
        last_changed = state.get("last_changed")

        if value in (None, "unavailable", "unknown"):
            return SensorCheck(
                entity_id=reg.entity_id,
                device_group=reg.device_group,
                device_name=reg.device_name,
                status=STATUS_UNAVAILABLE,
                message=f"Reported state: {value}",
                last_changed=last_changed,
            )

        if last_changed:
            try:
                changed_at = datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
                if now - changed_at > self.stale_after:
                    hours = (now - changed_at).total_seconds() / 3600
                    return SensorCheck(
                        entity_id=reg.entity_id,
                        device_group=reg.device_group,
                        device_name=reg.device_name,
                        status=STATUS_STALE,
                        message=f"No change in {hours:.1f}h",
                        last_changed=last_changed,
                    )
            except ValueError:
                pass

        return SensorCheck(
            entity_id=reg.entity_id,
            device_group=reg.device_group,
            device_name=reg.device_name,
            status=STATUS_OK,
            message="",
            last_changed=last_changed,
        )

    def _record_transition(self, reg: _Registration, check: SensorCheck):
        previous = self._last_status.get(reg.entity_id, STATUS_OK)

        if previous == STATUS_OK and check.status != STATUS_OK:
            history_tracker.add_sensor_failure(
                entity_id=reg.entity_id,
                device_group=reg.device_group,
                status=check.status,
                message=check.message,
            )
            logger.warning(
                f"🩺 Sensor {reg.entity_id} ({reg.device_name}) -> {check.status}: {check.message}"
            )
        elif previous != STATUS_OK and check.status == STATUS_OK:
            history_tracker.add_sensor_recovery(
                entity_id=reg.entity_id,
                device_group=reg.device_group,
                message="Recovered",
            )
            logger.info(f"🩺 Sensor {reg.entity_id} ({reg.device_name}) recovered")

    def get_full_status(self) -> dict:
        """Full per-sensor detail."""
        return {
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "sensors": [
                {
                    "entity_id": c.entity_id,
                    "device_group": c.device_group,
                    "device_name": c.device_name,
                    "status": c.status,
                    "message": c.message,
                    "last_changed": c.last_changed,
                }
                for c in self._latest.values()
            ],
        }

    def get_dashboard_summary(self) -> dict:
        """Lightweight, device-grouped summary for a UI banner.

        Only reports devices with at least one non-OK sensor.
        """
        now = datetime.now(UTC)
        if (
            self._started_at is not None
            and (now - self._started_at).total_seconds() < self.startup_grace_seconds
        ):
            return {"status": "starting_up", "devices": []}

        by_device: dict[str, list[SensorCheck]] = {}
        for check in self._latest.values():
            if check.status == STATUS_OK:
                continue
            by_device.setdefault(check.device_group, []).append(check)

        devices = [
            {
                "device_group": device_group,
                "device_name": checks[0].device_name,
                "issue_count": len(checks),
                "worst_status": (
                    STATUS_UNAVAILABLE
                    if any(c.status == STATUS_UNAVAILABLE for c in checks)
                    else STATUS_STALE
                ),
                "sensors": [
                    {
                        "entity_id": c.entity_id,
                        "status": c.status,
                        "message": c.message,
                    }
                    for c in checks
                ],
            }
            for device_group, checks in by_device.items()
        ]

        return {
            "status": "issues" if devices else "ok",
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "devices": devices,
        }
