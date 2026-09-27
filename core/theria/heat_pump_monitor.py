"""
Heat Pump Monitoring Service

Background service that tracks the IVT AirX heat pump (Husdata H60) state:
instantaneous power, mode (heating vs hot water), carrier temperatures, and
a rolling COP computed from deltas of the cumulative energy counters.

Visibility only - does not influence any control or scheduling decisions.
"""

import asyncio
import logging

from .ha_client import HAClient
from .history import history_tracker
from .settings import HeatPumpSettings

logger = logging.getLogger(__name__)


def _read_float(ha_client: HAClient, entity_id: str | None) -> float | None:
    """Read a sensor's state as a float, returning None if unavailable."""
    if not entity_id:
        return None
    try:
        state = ha_client.get_state(entity_id)
        return float(state["state"])
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.warning(f"Failed to read {entity_id}: {e}")
        return None


def _read_mode(ha_client: HAClient, entity_id: str) -> str:
    """Read switch_valve_1: 'on' = hot water, 'off' = heating."""
    try:
        state = ha_client.get_state(entity_id)
        return "hot_water" if state["state"] == "on" else "heating"
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Failed to read {entity_id}: {e}")
        return "heating"


class HeatPumpMonitorService:
    """Background service for continuous heat pump state collection."""

    def __init__(
        self,
        ha_client: HAClient,
        settings: HeatPumpSettings,
        collection_interval_seconds: int = 60,
    ):
        self.ha_client = ha_client
        self.settings = settings
        self.collection_interval_seconds = collection_interval_seconds

        self._task: asyncio.Task | None = None
        self._running = False

        # Previous cumulative counter readings, for delta-based COP calculation
        self._prev_consumed_total: float | None = None
        self._prev_delivered_total: float | None = None

    async def start(self):
        """Start the heat pump monitoring service."""
        if self._running:
            logger.warning("Heat pump monitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"♨️ Heat pump monitor started (interval: {self.collection_interval_seconds}s)"
        )

    async def stop(self):
        """Stop the heat pump monitoring service."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("♨️ Heat pump monitor stopped")

    async def _run_loop(self):
        """Main collection loop - collects heat pump state every interval."""
        while self._running:
            try:
                await self._collect_snapshot()
            except Exception as e:
                logger.exception(f"Error in heat pump collection loop: {e}")

            await asyncio.sleep(self.collection_interval_seconds)

    async def _collect_snapshot(self):
        """Read current heat pump state and store a snapshot."""
        s = self.settings

        compressor_speed = _read_float(self.ha_client, s.compressor_speed)
        compressor_power = _read_float(self.ha_client, s.compressor_power)
        mode = _read_mode(self.ha_client, s.mode_switch)
        heat_carrier_forward = _read_float(self.ha_client, s.heat_carrier_forward)
        heat_carrier_return = _read_float(self.ha_client, s.heat_carrier_return)
        hotwater_top = _read_float(self.ha_client, s.hotwater_top)
        hotwater_mid = _read_float(self.ha_client, s.hotwater_mid)
        hotwater_setpoint = _read_float(self.ha_client, s.hotwater_setpoint)

        cop = self._compute_rolling_cop()

        history_tracker.add_heat_pump_snapshot(
            compressor_speed=compressor_speed,
            compressor_power=compressor_power,
            mode=mode,
            heat_carrier_forward=heat_carrier_forward,
            heat_carrier_return=heat_carrier_return,
            cop=cop,
            hotwater_top=hotwater_top,
            hotwater_mid=hotwater_mid,
            hotwater_setpoint=hotwater_setpoint,
        )

    def _compute_rolling_cop(self) -> float | None:
        """Compute COP from the change in cumulative energy counters since the last poll.

        Returns None on the first poll (no previous reading), if a counter reset
        (delta < 0), or if there was no compressor consumption in this interval.
        """
        s = self.settings
        consumed_total = _read_float(self.ha_client, s.compressor_consumption_total)
        delivered_total = _read_float(self.ha_client, s.delivered_energy_total)

        cop = None
        if (
            consumed_total is not None
            and delivered_total is not None
            and self._prev_consumed_total is not None
            and self._prev_delivered_total is not None
        ):
            consumed_delta = consumed_total - self._prev_consumed_total
            delivered_delta = delivered_total - self._prev_delivered_total
            if consumed_delta > 0 and delivered_delta >= 0:
                cop = delivered_delta / consumed_delta

        self._prev_consumed_total = consumed_total
        self._prev_delivered_total = delivered_total

        return cop
