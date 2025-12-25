"""
Continuous Thermal Learning for Zones

Tracks actual temperature changes to learn each zone's unique thermal characteristics.
This enables accurate prediction for heat capacitor scheduling.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThermalMeasurement:
    """Single thermal response measurement."""
    timestamp: datetime
    indoor_temp: float
    outdoor_temp: float
    heating_active: bool  # Was heating on during this period?
    temp_change: float  # °C change since last measurement
    duration_minutes: float


@dataclass
class ZoneThermalCharacteristics:
    """Learned thermal characteristics for a zone."""

    # Heating rate (°C/hour when heating active)
    heating_rate: float = 0.3  # Initial guess
    heating_rate_confidence: float = 0.0  # 0-1, higher = more samples

    # Cooling rate (°C/hour when heating off, depends on outdoor temp)
    cooling_rate_base: float = -0.6  # Initial guess at outdoor temp = 0°C
    cooling_rate_confidence: float = 0.0

    # Heat loss coefficient (how cooling rate varies with outdoor temp)
    # cooling_rate = cooling_rate_base + outdoor_temp_coefficient * outdoor_temp
    outdoor_temp_coefficient: float = 0.02  # Initial guess

    # Sample statistics
    heating_samples: int = 0
    cooling_samples: int = 0
    last_updated: datetime | None = None

    def predict_temp_change(
        self,
        heating_active: bool,
        outdoor_temp: float,
        duration_minutes: float
    ) -> float:
        """Predict temperature change over duration."""

        if heating_active:
            # Heating rate minus cooling rate
            rate_per_hour = self.heating_rate + self._get_cooling_rate(outdoor_temp)
        else:
            # Just cooling
            rate_per_hour = self._get_cooling_rate(outdoor_temp)

        return rate_per_hour * (duration_minutes / 60.0)

    def _get_cooling_rate(self, outdoor_temp: float) -> float:
        """Calculate cooling rate for given outdoor temperature."""
        return self.cooling_rate_base + self.outdoor_temp_coefficient * outdoor_temp


class ZoneThermalLearner:
    """
    Continuously learns thermal characteristics of a zone.

    Every measurement cycle (15 min):
    1. PREDICT: What temp change do we expect?
    2. MEASURE: What temp change actually happened?
    3. LEARN: Update model based on error
    4. ADAPT: Use improved model for next prediction
    """

    def __init__(
        self,
        zone_id: str,
        learning_rate: float = 0.1,  # How fast to adapt (0.05-0.2)
        measurement_interval_minutes: int = 15
    ):
        self.zone_id = zone_id
        self.learning_rate = learning_rate
        self.measurement_interval_minutes = measurement_interval_minutes

        # Learned characteristics
        self.characteristics = ZoneThermalCharacteristics()

        # Recent measurements (for outlier detection)
        self.recent_measurements: list[ThermalMeasurement] = []
        self.max_recent_measurements = 100

        # Last measurement state
        self.last_timestamp: datetime | None = None
        self.last_indoor_temp: float | None = None
        self.last_outdoor_temp: float | None = None
        self.last_heating_active: bool | None = None

    def add_measurement(
        self,
        timestamp: datetime,
        indoor_temp: float,
        outdoor_temp: float,
        heating_active: bool
    ) -> ThermalMeasurement | None:
        """
        Add new measurement and learn from it.

        Returns the measurement if learning occurred, None if skipped.
        """

        # First measurement - just record state
        if self.last_timestamp is None:
            self._record_state(timestamp, indoor_temp, outdoor_temp, heating_active)
            return None

        # Calculate actual change
        duration_minutes = (timestamp - self.last_timestamp).total_seconds() / 60.0
        temp_change = indoor_temp - self.last_indoor_temp

        # Skip if duration is wrong (missed measurements)
        if abs(duration_minutes - self.measurement_interval_minutes) > 5:
            logger.warning(
                f"Zone {self.zone_id}: Skipping measurement, duration {duration_minutes:.1f}min "
                f"(expected {self.measurement_interval_minutes}min)"
            )
            self._record_state(timestamp, indoor_temp, outdoor_temp, heating_active)
            return None

        # Create measurement
        measurement = ThermalMeasurement(
            timestamp=timestamp,
            indoor_temp=indoor_temp,
            outdoor_temp=outdoor_temp,
            heating_active=self.last_heating_active,  # Heating state during the period
            temp_change=temp_change,
            duration_minutes=duration_minutes
        )

        # Learn from measurement
        self._learn_from_measurement(measurement)

        # Store recent measurements
        self.recent_measurements.append(measurement)
        if len(self.recent_measurements) > self.max_recent_measurements:
            self.recent_measurements.pop(0)

        # Record new state
        self._record_state(timestamp, indoor_temp, outdoor_temp, heating_active)

        return measurement

    def _learn_from_measurement(self, measurement: ThermalMeasurement):
        """Update thermal characteristics based on measurement."""

        # Predict what we expected
        predicted_change = self.characteristics.predict_temp_change(
            heating_active=measurement.heating_active,
            outdoor_temp=measurement.outdoor_temp,
            duration_minutes=measurement.duration_minutes
        )

        # Calculate prediction error
        error = measurement.temp_change - predicted_change

        # Outlier detection (error > 3 standard deviations)
        if abs(error) > 1.0:  # More than 1°C error is suspicious
            logger.warning(
                f"Zone {self.zone_id}: Large prediction error {error:.2f}°C "
                f"(predicted {predicted_change:.2f}, actual {measurement.temp_change:.2f})"
            )
            # Still learn, but with reduced rate
            effective_learning_rate = self.learning_rate * 0.3
        else:
            effective_learning_rate = self.learning_rate

        # Update parameters based on error
        if measurement.heating_active:
            # Update heating rate
            # heating_rate_new = heating_rate_old + learning_rate * error / duration_hours
            duration_hours = measurement.duration_minutes / 60.0
            self.characteristics.heating_rate += effective_learning_rate * (error / duration_hours)
            self.characteristics.heating_samples += 1
            self.characteristics.heating_rate_confidence = min(
                1.0,
                self.characteristics.heating_samples / 100.0
            )
        else:
            # Update cooling rate
            duration_hours = measurement.duration_minutes / 60.0

            # Update base cooling rate
            self.characteristics.cooling_rate_base += effective_learning_rate * (error / duration_hours)

            # Update outdoor temp coefficient (how outdoor temp affects cooling)
            # More sophisticated: could use least squares regression on recent cooling samples

            self.characteristics.cooling_samples += 1
            self.characteristics.cooling_rate_confidence = min(
                1.0,
                self.characteristics.cooling_samples / 100.0
            )

        self.characteristics.last_updated = measurement.timestamp

        logger.info(
            f"Zone {self.zone_id}: Learning update - "
            f"Heating rate: {self.characteristics.heating_rate:.3f}°C/h "
            f"({self.characteristics.heating_samples} samples, "
            f"confidence: {self.characteristics.heating_rate_confidence:.2f}), "
            f"Cooling rate: {self.characteristics.cooling_rate_base:.3f}°C/h "
            f"({self.characteristics.cooling_samples} samples, "
            f"confidence: {self.characteristics.cooling_rate_confidence:.2f})"
        )

    def _record_state(
        self,
        timestamp: datetime,
        indoor_temp: float,
        outdoor_temp: float,
        heating_active: bool
    ):
        """Record current state for next measurement."""
        self.last_timestamp = timestamp
        self.last_indoor_temp = indoor_temp
        self.last_outdoor_temp = outdoor_temp
        self.last_heating_active = heating_active

    def get_characteristics(self) -> ZoneThermalCharacteristics:
        """Get current learned characteristics."""
        return self.characteristics

    def get_confidence(self) -> float:
        """Get overall confidence in learned parameters (0-1)."""
        return min(
            self.characteristics.heating_rate_confidence,
            self.characteristics.cooling_rate_confidence
        )

    def get_recent_measurements(self, limit: int = 100) -> list[ThermalMeasurement]:
        """Get recent thermal measurements.

        Args:
            limit: Maximum number of measurements to return

        Returns:
            List of recent ThermalMeasurement objects (most recent first)
        """
        # Return most recent measurements, limited by count
        measurements = list(self.recent_measurements)
        return measurements[-limit:] if len(measurements) > limit else measurements

    def get_measurements_with_predictions(self, limit: int = 100) -> list[dict]:
        """Get recent measurements with prediction errors.

        Args:
            limit: Maximum number of measurements to return

        Returns:
            List of dicts with measurement data + predicted values + errors
        """
        measurements = self.get_recent_measurements(limit)
        result = []

        for m in measurements:
            # Calculate what the model predicted
            predicted_change = self.characteristics.predict_temp_change(
                heating_active=m.heating_active,
                outdoor_temp=m.outdoor_temp,
                duration_minutes=m.duration_minutes
            )

            # Calculate prediction error
            prediction_error = m.temp_change - predicted_change

            result.append({
                "timestamp": m.timestamp.isoformat(),
                "indoor_temp": m.indoor_temp,
                "outdoor_temp": m.outdoor_temp,
                "heating_active": m.heating_active,
                "temp_change": m.temp_change,
                "duration_minutes": m.duration_minutes,
                "predicted_change": predicted_change,
                "prediction_error": prediction_error
            })

        return result
