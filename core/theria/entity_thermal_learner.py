"""
Per-Entity Thermal Learning with Multi-Timeframe Analysis

Learns thermal characteristics individually for each climate entity (radiator/zone)
across multiple timeframes (1h, 6h, 24h, 7d) to provide both short-term responsiveness
and long-term reliability.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    """Get current time as timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# Timeframe definitions in hours
TIMEFRAMES = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}

TimeframeKey = Literal["1h", "6h", "24h", "7d"]


@dataclass
class EntityMeasurement:
    """Single temperature measurement for a climate entity."""

    timestamp: datetime
    temperature: float
    outdoor_temp: float
    heating_active: bool
    target_temp: float | None = None  # Target/setpoint temperature


@dataclass
class TimeframeCharacteristics:
    """Thermal characteristics for a specific timeframe."""

    heating_rate: float = 0.0  # °C/hour when heating
    cooling_rate: float = 0.0  # °C/hour when not heating
    heating_samples: int = 0
    cooling_samples: int = 0
    heating_confidence: float = 0.0  # 0-1
    cooling_confidence: float = 0.0  # 0-1

    @property
    def overall_confidence(self) -> float:
        """Overall confidence weighted by sample count."""
        total_samples = self.heating_samples + self.cooling_samples
        if total_samples == 0:
            return 0.0

        return (
            self.heating_confidence * self.heating_samples
            + self.cooling_confidence * self.cooling_samples
        ) / total_samples


class ClimateEntityThermalLearner:
    """
    Learns thermal characteristics for a single climate entity across multiple timeframes.

    Tracks heating and cooling rates separately for:
    - 1h: Recent behavior, quick feedback
    - 6h: Session average
    - 24h: Daily average (most useful for optimization)
    - 7d: Weekly average (most reliable, accounts for weather variation)
    """

    def __init__(
        self,
        entity_id: str,
        max_history_hours: int = 168,  # 7 days
        min_samples_for_confidence: int = 100,
    ):
        self.entity_id = entity_id
        self.max_history_hours = max_history_hours
        self.min_samples_for_confidence = min_samples_for_confidence

        # Store all measurements (limited to max_history_hours)
        self.measurements: deque[EntityMeasurement] = deque(maxlen=10000)

        # Cache for calculated characteristics per timeframe
        self._characteristics_cache: dict[TimeframeKey, TimeframeCharacteristics] = {}
        self._cache_timestamp: datetime | None = None
        self._cache_validity_seconds = 60  # Recalculate every minute

    def add_measurement(
        self,
        timestamp: datetime,
        temperature: float,
        outdoor_temp: float,
        heating_active: bool,
        target_temp: float | None = None,
    ) -> EntityMeasurement:
        """Add a new measurement and return it."""
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        measurement = EntityMeasurement(
            timestamp=timestamp,
            temperature=temperature,
            outdoor_temp=outdoor_temp,
            heating_active=heating_active,
            target_temp=target_temp,
        )

        self.measurements.append(measurement)

        # Clear cache to force recalculation
        self._cache_timestamp = None

        # Cleanup old measurements
        self._cleanup_old_measurements()

        return measurement

    def _cleanup_old_measurements(self):
        """Remove measurements older than max_history_hours."""
        if not self.measurements:
            return

        cutoff = now_utc() - timedelta(hours=self.max_history_hours)

        # Remove from left (oldest) until we hit the cutoff
        while self.measurements and self.measurements[0].timestamp < cutoff:
            self.measurements.popleft()

    def get_characteristics(
        self, timeframe: TimeframeKey = "24h"
    ) -> TimeframeCharacteristics:
        """
        Get thermal characteristics for specified timeframe.

        Uses caching to avoid recalculating too frequently.
        """
        # Check cache validity
        now = now_utc()
        if (
            self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds()
            < self._cache_validity_seconds
            and timeframe in self._characteristics_cache
        ):
            return self._characteristics_cache[timeframe]

        # Recalculate
        characteristics = self._calculate_characteristics(timeframe)

        # Update cache
        self._characteristics_cache[timeframe] = characteristics
        self._cache_timestamp = now

        return characteristics

    def get_all_timeframes(self) -> dict[TimeframeKey, TimeframeCharacteristics]:
        """Get characteristics for all timeframes."""
        return {tf: self.get_characteristics(tf) for tf in TIMEFRAMES.keys()}

    def _detect_periods(
        self,
        measurements: list[EntityMeasurement],
        min_duration_hours: float = 0.25,
        active_heating_threshold: float = 1.0,
        filter_zero_change: bool = True,
    ) -> list[dict]:
        """
        Detect heating and cooling periods from measurements.

        Uses industry-standard ON/OFF transition detection.

        Args:
            measurements: List of EntityMeasurement objects to analyze
            min_duration_hours: Minimum period duration in hours (default 15 min)
            active_heating_threshold: °C below target to qualify as active heating
            filter_zero_change: Whether to filter out near-zero rate periods

        Returns:
            List of period dictionaries with type, timestamps, temps, and rates
        """
        if len(measurements) < 2:
            return []

        periods = []
        i = 0

        while i < len(measurements):
            if i + 1 < len(measurements):
                curr = measurements[i]
                next_m = measurements[i + 1]

                # ═══════════════════════════════════════════════════════════
                # HEATING PERIOD DETECTION
                # ═══════════════════════════════════════════════════════════
                heating_turned_on = not curr.heating_active and next_m.heating_active
                setpoint_increased_while_heating = (
                    curr.heating_active
                    and next_m.heating_active
                    and curr.target_temp is not None
                    and next_m.target_temp is not None
                    and next_m.target_temp > curr.target_temp + 0.5
                )

                if heating_turned_on or setpoint_increased_while_heating:
                    period_start = next_m
                    period_start_idx = i + 1

                    # Find end of heating period
                    period_end = None
                    period_end_idx = period_start_idx
                    end_reason = None
                    for j in range(period_start_idx + 1, len(measurements)):
                        m = measurements[j]
                        if not m.heating_active:
                            period_end = measurements[j - 1]
                            period_end_idx = j - 1
                            end_reason = "heating_stopped"
                            break

                    if period_end is None and len(measurements) > period_start_idx + 1:
                        period_end = measurements[-1]
                        period_end_idx = len(measurements) - 1
                        end_reason = "ongoing"

                    # Calculate and classify period
                    if period_end and period_end.timestamp > period_start.timestamp:
                        temp_change = period_end.temperature - period_start.temperature
                        duration_hours = (
                            period_end.timestamp - period_start.timestamp
                        ).total_seconds() / 3600

                        # Determine if "active heating" or "maintenance"
                        period_type = "heating"
                        if period_start.target_temp is not None:
                            temp_below_target = (
                                period_start.target_temp - period_start.temperature
                            )
                            if temp_below_target < active_heating_threshold:
                                period_type = "maintenance"

                        if duration_hours >= min_duration_hours:
                            rate = temp_change / duration_hours
                            periods.append(
                                {
                                    "type": period_type,
                                    "entity_id": self.entity_id,
                                    "start_time": period_start.timestamp.isoformat(),
                                    "end_time": period_end.timestamp.isoformat(),
                                    "start_temp": period_start.temperature,
                                    "end_temp": period_end.temperature,
                                    "target_temp": period_start.target_temp,
                                    "temp_change": temp_change,
                                    "duration_hours": duration_hours,
                                    "rate": rate,
                                    "end_reason": end_reason,
                                }
                            )

                    i = period_end_idx + 1
                    continue

                # ═══════════════════════════════════════════════════════════
                # COOLING PERIOD DETECTION
                # ═══════════════════════════════════════════════════════════
                heating_turned_off = curr.heating_active and not next_m.heating_active
                setpoint_decreased_above_target = (
                    curr.target_temp is not None
                    and next_m.target_temp is not None
                    and next_m.target_temp < curr.target_temp - 0.5
                    and next_m.temperature > next_m.target_temp
                )

                if heating_turned_off or setpoint_decreased_above_target:
                    period_start = next_m
                    period_start_idx = i + 1

                    # Find end of cooling period
                    period_end = None
                    period_end_idx = period_start_idx
                    end_reason = None
                    for j in range(period_start_idx + 1, len(measurements)):
                        m = measurements[j]
                        if m.heating_active:
                            period_end = measurements[j - 1]
                            period_end_idx = j - 1
                            end_reason = "heating_started"
                            break

                    if period_end is None and len(measurements) > period_start_idx + 1:
                        period_end = measurements[-1]
                        period_end_idx = len(measurements) - 1
                        end_reason = "ongoing"

                    # Calculate cooling rate
                    if period_end and period_end.timestamp > period_start.timestamp:
                        temp_change = period_end.temperature - period_start.temperature
                        duration_hours = (
                            period_end.timestamp - period_start.timestamp
                        ).total_seconds() / 3600

                        if duration_hours >= min_duration_hours:
                            rate = temp_change / duration_hours
                            # Filter out near-zero rates if requested
                            if not filter_zero_change or abs(rate) > 0.01:
                                periods.append(
                                    {
                                        "type": "cooling",
                                        "entity_id": self.entity_id,
                                        "start_time": period_start.timestamp.isoformat(),
                                        "end_time": period_end.timestamp.isoformat(),
                                        "start_temp": period_start.temperature,
                                        "end_temp": period_end.temperature,
                                        "target_temp": period_start.target_temp,
                                        "temp_change": temp_change,
                                        "duration_hours": duration_hours,
                                        "rate": rate,
                                        "end_reason": end_reason,
                                    }
                                )

                    i = period_end_idx + 1
                    continue

            i += 1

        return periods

    def get_periods(self, hours: int = 24) -> list[dict]:
        """
        Get detailed list of detected heating and cooling periods.

        Uses industry-standard ON/OFF transition detection.
        Returns list of periods with timestamps, temperatures, and rates.
        Useful for visualization.
        """
        cutoff = now_utc() - timedelta(hours=hours)
        recent = [m for m in self.measurements if m.timestamp > cutoff]

        # Use shared period detection method
        return self._detect_periods(
            measurements=recent,
            min_duration_hours=0.25,
            active_heating_threshold=1.0,
            filter_zero_change=True,
        )

    def _calculate_characteristics(
        self, timeframe: TimeframeKey
    ) -> TimeframeCharacteristics:
        """Calculate heating and cooling rates for specific timeframe.

        Uses industry-standard period classification approach based on HVAC best practices:

        Period Classification (based on thermostat deadband/hysteresis):
        ┌─────────────────────┬──────────────────────────┬─────────────────────┐
        │ Period Type         │ Detection Criteria       │ Used for Learning?  │
        ├─────────────────────┼──────────────────────────┼─────────────────────┤
        │ Active Heating      │ heating_ON AND           │ ✓ YES               │
        │                     │ temp < target - 1°C      │ Calculate heating   │
        │                     │ duration > 15 min        │ rate                │
        ├─────────────────────┼──────────────────────────┼─────────────────────┤
        │ Maintenance/Cycling │ heating_ON AND           │ ✗ NO                │
        │                     │ |temp - target| < 0.5°C  │ Too noisy, ignore   │
        ├─────────────────────┼──────────────────────────┼─────────────────────┤
        │ Cooling/Heat Loss   │ heating_OFF AND          │ ✓ YES               │
        │                     │ temp falling             │ Calculate cooling   │
        │                     │ duration > 15 min        │ rate                │
        └─────────────────────┴──────────────────────────┴─────────────────────┘

        This approach:
        1. Detects periods by heating ON/OFF transitions (not setpoint changes)
        2. Filters by temperature differential (active heating vs maintenance)
        3. Requires minimum duration (15 min) to avoid sensor artifacts
        4. Handles rapid setpoint changes correctly (e.g., 17→18→20 or 18→21→20)

        References:
        - Thermostat deadband: 0.5-1°C typical (HVAC industry standard)
        - RC thermal models: System identification using transient periods
        - Filtering maintenance cycles: Standard practice in building energy modeling
        """
        timeframe_hours = TIMEFRAMES[timeframe]
        cutoff = now_utc() - timedelta(hours=timeframe_hours)

        # Filter measurements to timeframe
        recent = [m for m in self.measurements if m.timestamp > cutoff]

        if len(recent) < 2:
            return TimeframeCharacteristics()

        # Use shared period detection method
        detected_periods = self._detect_periods(
            measurements=recent,
            min_duration_hours=0.25,
            active_heating_threshold=1.0,
            filter_zero_change=True,
        )

        # Extract (temp_change, duration) tuples for duration-weighted averaging
        # Filter out "maintenance" periods from heating calculations
        heating_periods = [
            (p["temp_change"], p["duration_hours"])
            for p in detected_periods
            if p["type"] == "heating" and p["temp_change"] > 0
        ]

        cooling_periods = [
            (p["temp_change"], p["duration_hours"])
            for p in detected_periods
            if p["type"] == "cooling"
        ]

        # Calculate statistics using duration-weighted averaging
        # Weighted average = (sum of temp changes) / (sum of durations)
        # This is physically correct: average rate over all time, not average of individual rates

        if heating_periods:
            total_heating_change = sum(tc for tc, _ in heating_periods)
            total_heating_duration = sum(dur for _, dur in heating_periods)
            heating_rate = (
                total_heating_change / total_heating_duration
                if total_heating_duration > 0
                else 0.0
            )
        else:
            heating_rate = 0.0

        if cooling_periods:
            total_cooling_change = sum(tc for tc, _ in cooling_periods)
            total_cooling_duration = sum(dur for _, dur in cooling_periods)
            cooling_rate = (
                total_cooling_change / total_cooling_duration
                if total_cooling_duration > 0
                else 0.0
            )
        else:
            cooling_rate = 0.0

        heating_samples = len(heating_periods)
        cooling_samples = len(cooling_periods)

        # Calculate confidence (0-1 based on sample count)
        heating_confidence = min(heating_samples / self.min_samples_for_confidence, 1.0)
        cooling_confidence = min(cooling_samples / self.min_samples_for_confidence, 1.0)

        return TimeframeCharacteristics(
            heating_rate=heating_rate,
            cooling_rate=cooling_rate,
            heating_samples=heating_samples,
            cooling_samples=cooling_samples,
            heating_confidence=heating_confidence,
            cooling_confidence=cooling_confidence,
        )

    def get_diagnostics(self) -> dict:
        """Get diagnostic information for debugging."""
        all_chars = self.get_all_timeframes()

        return {
            "entity_id": self.entity_id,
            "total_measurements": len(self.measurements),
            "oldest_measurement": (
                self.measurements[0].timestamp.isoformat()
                if self.measurements
                else None
            ),
            "newest_measurement": (
                self.measurements[-1].timestamp.isoformat()
                if self.measurements
                else None
            ),
            "timeframes": {
                tf: {
                    "heating_rate": chars.heating_rate,
                    "cooling_rate": chars.cooling_rate,
                    "heating_samples": chars.heating_samples,
                    "cooling_samples": chars.cooling_samples,
                    "heating_confidence": chars.heating_confidence,
                    "cooling_confidence": chars.cooling_confidence,
                    "overall_confidence": chars.overall_confidence,
                }
                for tf, chars in all_chars.items()
            },
        }


class ZoneThermalAggregator:
    """
    Aggregates thermal characteristics from multiple climate entities into zone-level data.

    Provides both simple averaging and weighted averaging based on confidence.
    """

    def __init__(
        self, zone_id: str, entity_learners: list[ClimateEntityThermalLearner]
    ):
        self.zone_id = zone_id
        self.entity_learners = entity_learners

    def get_aggregate_characteristics(
        self, timeframe: TimeframeKey = "24h", weighted: bool = True
    ) -> TimeframeCharacteristics:
        """
        Calculate zone-level characteristics from entity learners.

        Args:
            timeframe: Which timeframe to aggregate
            weighted: If True, weight by confidence; if False, simple average
        """
        if not self.entity_learners:
            return TimeframeCharacteristics()

        # Get characteristics from all entities
        all_chars = [
            learner.get_characteristics(timeframe) for learner in self.entity_learners
        ]

        if weighted:
            return self._weighted_aggregate(all_chars)
        else:
            return self._simple_aggregate(all_chars)

    def _simple_aggregate(
        self, characteristics: list[TimeframeCharacteristics]
    ) -> TimeframeCharacteristics:
        """Simple average of all entity characteristics."""
        heating_rates = [c.heating_rate for c in characteristics]
        cooling_rates = [c.cooling_rate for c in characteristics]

        return TimeframeCharacteristics(
            heating_rate=float(np.mean(heating_rates)),
            cooling_rate=float(np.mean(cooling_rates)),
            heating_samples=sum(c.heating_samples for c in characteristics),
            cooling_samples=sum(c.cooling_samples for c in characteristics),
            heating_confidence=float(
                np.mean([c.heating_confidence for c in characteristics])
            ),
            cooling_confidence=float(
                np.mean([c.cooling_confidence for c in characteristics])
            ),
        )

    def _weighted_aggregate(
        self, characteristics: list[TimeframeCharacteristics]
    ) -> TimeframeCharacteristics:
        """Weighted average based on confidence."""
        # Heating rate weighted by heating confidence
        heating_weights = [c.heating_confidence for c in characteristics]
        total_heating_weight = sum(heating_weights)

        if total_heating_weight > 0:
            heating_rate = (
                sum(c.heating_rate * c.heating_confidence for c in characteristics)
                / total_heating_weight
            )
        else:
            heating_rate = float(np.mean([c.heating_rate for c in characteristics]))

        # Cooling rate weighted by cooling confidence
        cooling_weights = [c.cooling_confidence for c in characteristics]
        total_cooling_weight = sum(cooling_weights)

        if total_cooling_weight > 0:
            cooling_rate = (
                sum(c.cooling_rate * c.cooling_confidence for c in characteristics)
                / total_cooling_weight
            )
        else:
            cooling_rate = float(np.mean([c.cooling_rate for c in characteristics]))

        return TimeframeCharacteristics(
            heating_rate=heating_rate,
            cooling_rate=cooling_rate,
            heating_samples=sum(c.heating_samples for c in characteristics),
            cooling_samples=sum(c.cooling_samples for c in characteristics),
            heating_confidence=float(
                np.mean([c.heating_confidence for c in characteristics])
            ),
            cooling_confidence=float(
                np.mean([c.cooling_confidence for c in characteristics])
            ),
        )

    def get_comparison(self, timeframe: TimeframeKey = "24h") -> dict:
        """
        Compare thermal performance across entities.

        Returns insights like fastest/slowest heaters.
        """
        if not self.entity_learners:
            return {}

        entity_data = [
            {
                "entity_id": learner.entity_id,
                "characteristics": learner.get_characteristics(timeframe),
            }
            for learner in self.entity_learners
        ]

        # Find fastest and slowest heaters
        heating_sorted = sorted(
            entity_data, key=lambda x: x["characteristics"].heating_rate, reverse=True
        )

        aggregate = self.get_aggregate_characteristics(timeframe, weighted=True)

        return {
            "zone_id": self.zone_id,
            "timeframe": timeframe,
            "entities": [
                {
                    "entity_id": e["entity_id"],
                    "heating_rate": e["characteristics"].heating_rate,
                    "cooling_rate": e["characteristics"].cooling_rate,
                    "heating_confidence": e["characteristics"].heating_confidence,
                    "cooling_confidence": e["characteristics"].cooling_confidence,
                    "heating_samples": e["characteristics"].heating_samples,
                    "cooling_samples": e["characteristics"].cooling_samples,
                }
                for e in entity_data
            ],
            "aggregate": {
                "heating_rate": aggregate.heating_rate,
                "cooling_rate": aggregate.cooling_rate,
                "heating_confidence": aggregate.heating_confidence,
                "cooling_confidence": aggregate.cooling_confidence,
            },
            "insights": {
                "fastest_heater": (
                    heating_sorted[0]["entity_id"] if heating_sorted else None
                ),
                "slowest_heater": (
                    heating_sorted[-1]["entity_id"] if len(heating_sorted) > 1 else None
                ),
                "heating_rate_spread": (
                    heating_sorted[0]["characteristics"].heating_rate
                    - heating_sorted[-1]["characteristics"].heating_rate
                    if len(heating_sorted) > 1
                    else 0.0
                ),
            },
        }
