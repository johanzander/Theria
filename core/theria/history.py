"""
Temperature History Tracking

Simple in-memory history for the last 24 hours.
For production, this should use InfluxDB or similar.
"""

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional
import threading


@dataclass
class TemperatureReading:
    """A single temperature reading."""

    timestamp: str  # ISO format
    zone_id: str
    current_temp: float  # Zone average
    scheduled_temp: Optional[float]
    target_temps: dict[str, float]  # climate_entity -> target_temp
    current_temps: dict[str, float] = None  # climate_entity -> current_temp (actual reading)


@dataclass
class ControlEvent:
    """A control action event."""

    timestamp: str  # ISO format
    zone_id: str
    action: str  # "set_temperature", "schedule_change", etc.
    details: str
    temperature: Optional[float] = None


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

        self.lock = threading.Lock()

    def add_temperature_reading(
        self,
        zone_id: str,
        current_temp: float,
        scheduled_temp: Optional[float],
        target_temps: dict[str, float],
        current_temps: Optional[dict[str, float]] = None
    ):
        """Add a temperature reading.

        Args:
            zone_id: Zone identifier
            current_temp: Current average temperature
            scheduled_temp: Scheduled target temperature
            target_temps: Dict of climate_entity -> target_temp
            current_temps: Dict of climate_entity -> current_temp (actual readings)
        """
        reading = TemperatureReading(
            timestamp=datetime.utcnow().isoformat(),
            zone_id=zone_id,
            current_temp=current_temp,
            scheduled_temp=scheduled_temp,
            target_temps=target_temps,
            current_temps=current_temps or {}
        )

        with self.lock:
            self.temperature_readings.append(reading)
            self._cleanup_old_data()

    def add_control_event(
        self,
        zone_id: str,
        action: str,
        details: str,
        temperature: Optional[float] = None
    ):
        """Log a control action.

        Args:
            zone_id: Zone identifier
            action: Action type
            details: Human-readable description
            temperature: Temperature set (if applicable)
        """
        event = ControlEvent(
            timestamp=datetime.utcnow().isoformat(),
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
        zone_id: Optional[str] = None,
        hours: Optional[int] = None
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
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            readings = [
                r for r in readings
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]

        return [asdict(r) for r in readings]

    def get_control_events(
        self,
        zone_id: Optional[str] = None,
        hours: Optional[int] = None
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
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            events = [
                e for e in events
                if datetime.fromisoformat(e.timestamp) > cutoff
            ]

        return [asdict(e) for e in events]

    def _cleanup_old_data(self):
        """Remove data older than max_hours."""
        cutoff = datetime.utcnow() - self.max_age

        # Clean temperature readings
        while (self.temperature_readings and
               datetime.fromisoformat(self.temperature_readings[0].timestamp) < cutoff):
            self.temperature_readings.popleft()

        # Clean control events
        while (self.control_events and
               datetime.fromisoformat(self.control_events[0].timestamp) < cutoff):
            self.control_events.popleft()


# Global instance
history_tracker = HistoryTracker(max_hours=24)
