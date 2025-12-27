"""
Temperature History Tracking

Simple in-memory history for the last 24 hours.
For production, this should use InfluxDB or similar.
"""

import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class TemperatureReading:
    """A single temperature reading."""

    timestamp: str  # ISO format
    zone_id: str
    current_temp: float  # Zone average
    scheduled_temp: float | None
    target_temps: dict[str, float]  # climate_entity -> target_temp
    current_temps: dict[str, float] | None = None  # climate_entity -> current_temp (actual reading)
    heating_requests: dict[str, float] | None = None  # climate_entity -> heating_power_request %


@dataclass
class ControlEvent:
    """A control action event."""

    timestamp: str  # ISO format
    zone_id: str
    action: str  # "set_temperature", "schedule_change", etc.
    details: str
    temperature: float | None = None


@dataclass
class ThermalCharacteristicsSnapshot:
    """Snapshot of learned thermal characteristics at a point in time."""

    timestamp: str  # ISO format
    zone_id: str
    heating_rate: float  # °C/hour when heating active
    heating_rate_confidence: float  # 0-1
    heating_samples: int
    cooling_rate_base: float  # °C/hour at outdoor temp = 0°C
    cooling_rate_confidence: float  # 0-1
    cooling_samples: int
    outdoor_temp_coefficient: float  # Cooling rate dependency on outdoor temp
    overall_confidence: float  # min(heating_confidence, cooling_confidence)


@dataclass
class HeatingPowerSnapshot:
    """Snapshot of heating power request at a point in time."""

    timestamp: str  # ISO format
    zone_id: str
    avg_heating_request: float  # Average % across climate entities (0-100)
    max_heating_request: float  # Maximum % from any climate entity (0-100)
    heating_active: bool  # Whether any heating was requested


class HistoryTracker:
    """Tracks temperature history and control events."""

    def __init__(self, max_hours: int = 24):
        """Initialize history tracker.

        Args:
            max_hours: How many hours of history to keep
        """
        self.max_hours = max_hours
        self.max_age = timedelta(hours=max_hours)

        # Use deque for efficient append/pop
        self.temperature_readings: deque[TemperatureReading] = deque(maxlen=10000)
        self.control_events: deque[ControlEvent] = deque(maxlen=1000)
        self.thermal_snapshots: deque[ThermalCharacteristicsSnapshot] = deque(maxlen=10080)  # 7 days at 1/min
        self.heating_power_snapshots: deque[HeatingPowerSnapshot] = deque(maxlen=10080)  # 7 days at 1/min

        self.lock = threading.Lock()

    def add_temperature_reading(
        self,
        zone_id: str,
        current_temp: float,
        scheduled_temp: float | None,
        target_temps: dict[str, float],
        current_temps: dict[str, float] | None = None,
        heating_requests: dict[str, float] | None = None,
        timestamp: datetime | None = None,
        skip_cleanup: bool = False
    ):
        """Add a temperature reading.

        Args:
            zone_id: Zone identifier
            current_temp: Current average temperature
            scheduled_temp: Scheduled target temperature
            target_temps: Dict of climate_entity -> target_temp
            current_temps: Dict of climate_entity -> current_temp (actual readings)
            heating_requests: Dict of climate_entity -> heating_power_request % (0-100)
            timestamp: Optional timestamp (defaults to now for real-time readings)
            skip_cleanup: Skip cleanup (use during backfill to preserve historical data)
        """
        # Use provided timestamp for historical backfill, or current time for real-time
        ts = timestamp.isoformat() if timestamp else datetime.now(timezone.utc).isoformat()

        reading = TemperatureReading(
            timestamp=ts,
            zone_id=zone_id,
            current_temp=current_temp,
            scheduled_temp=scheduled_temp,
            target_temps=target_temps,
            current_temps=current_temps or {},
            heating_requests=heating_requests or {}
        )

        with self.lock:
            self.temperature_readings.append(reading)
            if not skip_cleanup:
                self._cleanup_old_data()

    def add_control_event(
        self,
        zone_id: str,
        action: str,
        details: str,
        temperature: float | None = None
    ):
        """Log a control action.

        Args:
            zone_id: Zone identifier
            action: Action type
            details: Human-readable description
            temperature: Temperature set (if applicable)
        """
        event = ControlEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            zone_id=zone_id,
            action=action,
            details=details,
            temperature=temperature
        )

        with self.lock:
            self.control_events.append(event)
            self._cleanup_old_data()

    def get_temperature_history(
        self,
        zone_id: str | None = None,
        hours: int | None = None
    ) -> list[dict]:
        """Get temperature history.

        Args:
            zone_id: Filter by zone (None = all zones)
            hours: How many hours back (None = all available)

        Returns:
            List of temperature readings as dicts
        """
        with self.lock:
            readings = list(self.temperature_readings)

        # Filter by zone
        if zone_id:
            readings = [r for r in readings if r.zone_id == zone_id]

        # Filter by time
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            readings = [
                r for r in readings
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]

        return [asdict(r) for r in readings]

    def get_control_events(
        self,
        zone_id: str | None = None,
        hours: int | None = None
    ) -> list[dict]:
        """Get control events.

        Args:
            zone_id: Filter by zone (None = all zones)
            hours: How many hours back (None = all available)

        Returns:
            List of control events as dicts
        """
        with self.lock:
            events = list(self.control_events)

        # Filter by zone
        if zone_id:
            events = [e for e in events if e.zone_id == zone_id]

        # Filter by time
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            events = [
                e for e in events
                if datetime.fromisoformat(e.timestamp) > cutoff
            ]

        return [asdict(e) for e in events]

    def add_thermal_snapshot(
        self,
        zone_id: str,
        heating_rate: float,
        heating_rate_confidence: float,
        heating_samples: int,
        cooling_rate_base: float,
        cooling_rate_confidence: float,
        cooling_samples: int,
        outdoor_temp_coefficient: float
    ):
        """Add a thermal characteristics snapshot.

        Args:
            zone_id: Zone identifier
            heating_rate: Heating rate (°C/hour)
            heating_rate_confidence: Confidence in heating rate (0-1)
            heating_samples: Number of heating measurements
            cooling_rate_base: Base cooling rate (°C/hour)
            cooling_rate_confidence: Confidence in cooling rate (0-1)
            cooling_samples: Number of cooling measurements
            outdoor_temp_coefficient: Outdoor temperature coefficient
        """
        snapshot = ThermalCharacteristicsSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            zone_id=zone_id,
            heating_rate=heating_rate,
            heating_rate_confidence=heating_rate_confidence,
            heating_samples=heating_samples,
            cooling_rate_base=cooling_rate_base,
            cooling_rate_confidence=cooling_rate_confidence,
            cooling_samples=cooling_samples,
            outdoor_temp_coefficient=outdoor_temp_coefficient,
            overall_confidence=min(heating_rate_confidence, cooling_rate_confidence)
        )

        with self.lock:
            self.thermal_snapshots.append(snapshot)

    def get_thermal_history(
        self,
        zone_id: str | None = None,
        hours: int | None = None
    ) -> list[dict]:
        """Get thermal characteristics history.

        Args:
            zone_id: Filter by zone (None = all zones)
            hours: How many hours back (None = all available)

        Returns:
            List of thermal characteristic snapshots as dicts
        """
        with self.lock:
            snapshots = list(self.thermal_snapshots)

        # Filter by zone
        if zone_id:
            snapshots = [s for s in snapshots if s.zone_id == zone_id]

        # Filter by time
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            snapshots = [
                s for s in snapshots
                if datetime.fromisoformat(s.timestamp) > cutoff
            ]

        return [asdict(s) for s in snapshots]

    def add_heating_power_snapshot(
        self,
        zone_id: str,
        avg_heating_request: float,
        max_heating_request: float,
        heating_active: bool,
        timestamp: datetime | None = None,
        skip_cleanup: bool = False
    ):
        """Add a heating power request snapshot.

        Args:
            zone_id: Zone identifier
            avg_heating_request: Average heating request % (0-100)
            max_heating_request: Maximum heating request % (0-100)
            heating_active: Whether any heating was requested
            timestamp: Optional timestamp (defaults to now for real-time snapshots)
            skip_cleanup: Skip cleanup (use during backfill to preserve historical data)
        """
        # Use provided timestamp for historical backfill, or current time for real-time
        ts = timestamp.isoformat() if timestamp else datetime.now(timezone.utc).isoformat()

        snapshot = HeatingPowerSnapshot(
            timestamp=ts,
            zone_id=zone_id,
            avg_heating_request=avg_heating_request,
            max_heating_request=max_heating_request,
            heating_active=heating_active
        )

        with self.lock:
            self.heating_power_snapshots.append(snapshot)
            if not skip_cleanup:
                self._cleanup_old_data()

    def get_heating_timeline(
        self,
        zone_id: str | None = None,
        hours: int | None = None,
        resolution: str = "raw"
    ) -> list[dict]:
        """Get heating power request timeline.

        Args:
            zone_id: Filter by zone (None = all zones)
            hours: How many hours back (None = all available)
            resolution: "raw", "5m", "15m", "1h" (aggregation resolution)

        Returns:
            List of heating power snapshots as dicts
        """
        with self.lock:
            snapshots = list(self.heating_power_snapshots)

        # Filter by zone
        if zone_id:
            snapshots = [s for s in snapshots if s.zone_id == zone_id]

        # Filter by time
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            snapshots = [
                s for s in snapshots
                if datetime.fromisoformat(s.timestamp) > cutoff
            ]

        # TODO: Implement aggregation for resolution != "raw"
        # For MVP, return raw data
        return [asdict(s) for s in snapshots]

    def _cleanup_old_data(self):
        """Remove data older than max_hours."""
        cutoff = datetime.now(timezone.utc) - self.max_age

        # Clean temperature readings
        while (self.temperature_readings and
               datetime.fromisoformat(self.temperature_readings[0].timestamp) < cutoff):
            self.temperature_readings.popleft()

        # Clean control events
        while (self.control_events and
               datetime.fromisoformat(self.control_events[0].timestamp) < cutoff):
            self.control_events.popleft()

        # Clean thermal snapshots (keep 7 days, don't use max_hours)
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        while (self.thermal_snapshots and
               datetime.fromisoformat(self.thermal_snapshots[0].timestamp) < seven_days_ago):
            self.thermal_snapshots.popleft()

        # Clean heating power snapshots (keep 7 days)
        while (self.heating_power_snapshots and
               datetime.fromisoformat(self.heating_power_snapshots[0].timestamp) < seven_days_ago):
            self.heating_power_snapshots.popleft()


# Global instance
# Increased from 24h to 720h (30 days) to support InfluxDB historical backfill
history_tracker = HistoryTracker(max_hours=720)
