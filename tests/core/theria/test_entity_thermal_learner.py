"""
Unit tests for ClimateEntityThermalLearner.

Tests follow Home Assistant best practices:
- Clear, descriptive test names
- Parametrized tests for multiple scenarios
- Fixtures for reusable test data
- Focus on behavior, not implementation
- Edge case coverage
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.theria.entity_thermal_learner import (
    EntityMeasurement,
)


class TestPeriodDetection:
    """Tests for _detect_periods() method."""

    def test_detect_single_heating_period(self, learner, sample_heating_period):
        """Test detection of a single heating period."""
        periods = learner._detect_periods(sample_heating_period)

        assert len(periods) == 1
        period = periods[0]

        # Verify period type
        assert period["type"] == "heating"
        assert period["entity_id"] == learner.entity_id

        # Verify temperature changes
        assert period["start_temp"] == 18.0
        assert period["end_temp"] == 20.0
        assert period["temp_change"] == pytest.approx(2.0, abs=0.01)

        # Verify duration (from minute 15 to 135 = 120 minutes = 2.0 hours)
        assert period["duration_hours"] == pytest.approx(2.0, abs=0.01)

        # Verify rate calculation
        expected_rate = 2.0 / 2.0  # +1.0°C/h
        assert period["rate"] == pytest.approx(expected_rate, abs=0.01)

        # Verify timestamps
        assert period["end_reason"] == "heating_stopped"

    def test_detect_single_cooling_period(self, learner, sample_cooling_period):
        """Test detection of a single cooling period."""
        periods = learner._detect_periods(sample_cooling_period)

        assert len(periods) == 1
        period = periods[0]

        # Verify period type
        assert period["type"] == "cooling"

        # Verify temperature changes
        assert period["start_temp"] == 22.0
        assert period["end_temp"] == 20.0
        assert period["temp_change"] == pytest.approx(-2.0, abs=0.01)

        # Verify duration (from minute 15 to 255 = 240 minutes = 4.0 hours)
        assert period["duration_hours"] == pytest.approx(4.0, abs=0.01)

        # Verify rate calculation
        expected_rate = -2.0 / 4.0  # -0.5°C/h
        assert period["rate"] == pytest.approx(expected_rate, abs=0.01)

    def test_detect_mixed_periods(self, learner, sample_mixed_periods):
        """Test detection of multiple alternating heating/cooling periods."""
        periods = learner._detect_periods(sample_mixed_periods)

        # Should detect 2 heating periods (cooling filtered due to low rate)
        assert len(periods) == 2

        # Verify first period (heating)
        assert periods[0]["type"] == "heating"
        assert periods[0]["temp_change"] == pytest.approx(2.0, abs=0.01)

        # Verify second period (heating)
        assert periods[1]["type"] == "heating"
        assert periods[1]["temp_change"] == pytest.approx(1.5, abs=0.01)

    def test_filter_zero_change_periods(self, learner, zero_change_period):
        """Test that periods with near-zero temperature change are classified as maintenance."""
        periods = learner._detect_periods(zero_change_period, filter_zero_change=True)

        # Zero-change heating period is classified as "maintenance", not filtered
        assert len(periods) == 1
        assert periods[0]["type"] == "maintenance"
        assert abs(periods[0]["rate"]) < 0.01

    def test_include_zero_change_when_disabled(self, learner, zero_change_period):
        """Test that zero-change periods are included when filtering is disabled."""
        periods = learner._detect_periods(zero_change_period, filter_zero_change=False)

        # Should be included when filter is disabled
        assert len(periods) == 1
        assert abs(periods[0]["rate"]) < 0.01  # Near-zero rate

    def test_filter_short_duration_periods(self, learner, short_duration_period):
        """Test that periods shorter than minimum duration are filtered out."""
        # 10 minutes < 15 minutes minimum
        periods = learner._detect_periods(
            short_duration_period, min_duration_hours=0.25
        )

        assert len(periods) == 0

    def test_maintenance_mode_classification(self, learner, maintenance_mode_period):
        """Test that heating near target temperature is classified as 'maintenance'."""
        # Active heating threshold = 1.0°C
        # Temperature 21.5°C, target 22.0°C → only 0.5°C below target
        periods = learner._detect_periods(
            maintenance_mode_period, active_heating_threshold=1.0
        )

        assert len(periods) == 1
        assert periods[0]["type"] == "maintenance"

    def test_no_overlapping_periods(self, learner, sample_mixed_periods):
        """Test that detected periods do not overlap."""
        periods = learner._detect_periods(sample_mixed_periods)

        # Verify no overlaps by checking timestamps
        for i in range(len(periods) - 1):
            end_time_i = datetime.fromisoformat(periods[i]["end_time"])
            start_time_next = datetime.fromisoformat(periods[i + 1]["start_time"])

            # Next period should start after current period ends
            assert (
                start_time_next >= end_time_i
            ), f"Period {i+1} starts before period {i} ends (overlap detected)"

    def test_empty_measurements(self, learner):
        """Test handling of empty measurement list."""
        periods = learner._detect_periods([])
        assert periods == []

    def test_single_measurement(self, learner, create_measurement):
        """Test handling of single measurement (insufficient data)."""
        measurements = [create_measurement(0)]
        periods = learner._detect_periods(measurements)
        assert periods == []

    def test_ongoing_period(self, learner, create_measurement):
        """Test detection of ongoing period (heating still active at end)."""
        measurements = [
            create_measurement(
                0, temperature=18.0, heating_active=False, target_temp=22.0
            ),
            create_measurement(
                15, temperature=18.0, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                30, temperature=18.5, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                45, temperature=19.0, heating_active=True, target_temp=22.0
            ),
            # Heating still active at end (ongoing)
        ]

        periods = learner._detect_periods(measurements)

        assert len(periods) == 1
        assert periods[0]["end_reason"] == "ongoing"
        assert periods[0]["type"] == "heating"


class TestDurationWeightedAveraging:
    """Tests for duration-weighted averaging calculations."""

    def test_weighted_averaging_simple(self, learner, create_measurement):
        """
        Test duration-weighted averaging with simple example.

        Period 1: 1 hour at +2.0°C/h (Δ+2.0°C)
        Period 2: 3 hours at +0.5°C/h (Δ+1.5°C)

        Simple mean: (2.0 + 0.5) / 2 = 1.25°C/h (WRONG)
        Weighted mean: (2.0 + 1.5) / (1 + 3) = 0.875°C/h (CORRECT)
        """
        measurements = [
            # Period 1: Fast heating for 1 hour
            create_measurement(
                0, temperature=18.0, heating_active=False, target_temp=22.0
            ),
            create_measurement(
                15, temperature=18.0, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                75, temperature=20.0, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                90, temperature=20.0, heating_active=False, target_temp=22.0
            ),
            # Period 2: Slow heating for 3 hours
            create_measurement(
                105, temperature=20.0, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                165, temperature=20.5, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                225, temperature=21.0, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                285, temperature=21.5, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                300, temperature=21.5, heating_active=False, target_temp=22.0
            ),
        ]

        # Add measurements to learner
        for m in measurements:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        # Get characteristics
        chars = learner.get_characteristics("24h")

        # Expected weighted average: (2.0 + 1.5) / (1 + 3) = 0.875°C/h
        expected_rate = 3.5 / 4.0
        assert chars.heating_rate == pytest.approx(expected_rate, abs=0.02)
        assert chars.heating_samples == 2

    def test_weighted_averaging_mixed_periods(self, learner, sample_mixed_periods):
        """Test weighted averaging with mixed heating and cooling periods."""
        # Add measurements to learner
        for m in sample_mixed_periods:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        chars = learner.get_characteristics("24h")

        # Expected heating weighted average:
        # Period 1: 1h, +2.0°C → rate = +2.0°C/h
        # Period 2: 2h, +1.5°C → rate = +0.75°C/h
        # Weighted: (2.0 + 1.5) / (1 + 2) = 1.167°C/h
        assert chars.heating_rate == pytest.approx(1.167, abs=0.02)
        assert chars.heating_samples == 2

        # No cooling periods detected in this fixture (cooling rate too low, filtered out)
        assert chars.cooling_samples == 0

    def test_single_period_averaging(self, learner, sample_heating_period):
        """Test that single period averaging works correctly."""
        for m in sample_heating_period:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        chars = learner.get_characteristics("24h")

        # Single period: weighted average = period rate
        # +2.0°C over 2.0h = +1.0°C/h
        expected_rate = 2.0 / 2.0
        assert chars.heating_rate == pytest.approx(expected_rate, abs=0.01)
        assert chars.heating_samples == 1


class TestCharacteristicsCalculation:
    """Tests for get_characteristics() and related methods."""

    def test_confidence_calculation(self, learner, sample_mixed_periods):
        """Test confidence score calculation based on sample count."""
        for m in sample_mixed_periods:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        chars = learner.get_characteristics("24h")

        # Confidence = min(samples / min_samples_for_confidence, 1.0)
        # min_samples_for_confidence = 100 (default)
        # heating_samples = 2 → confidence = 2/100 = 0.02
        # cooling_samples = 0 (no cooling periods in this fixture)
        assert chars.heating_confidence == pytest.approx(0.02, abs=0.001)
        assert chars.cooling_confidence == 0.0

    def test_overall_confidence_weighted(self, learner, sample_mixed_periods):
        """Test overall confidence is weighted by sample counts."""
        for m in sample_mixed_periods:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        chars = learner.get_characteristics("24h")

        # Overall confidence = (heating_conf * heating_samples + cooling_conf * cooling_samples) / total_samples
        # (0.02 * 2 + 0.01 * 1) / 3 = 0.05 / 3 = 0.0167
        expected_overall = (
            chars.heating_confidence * chars.heating_samples
            + chars.cooling_confidence * chars.cooling_samples
        ) / (chars.heating_samples + chars.cooling_samples)

        assert chars.overall_confidence == pytest.approx(expected_overall, abs=0.001)

    def test_no_data_returns_zero_characteristics(self, learner):
        """Test that learner with no data returns zero characteristics."""
        chars = learner.get_characteristics("24h")

        assert chars.heating_rate == 0.0
        assert chars.cooling_rate == 0.0
        assert chars.heating_samples == 0
        assert chars.cooling_samples == 0
        assert chars.heating_confidence == 0.0
        assert chars.cooling_confidence == 0.0
        assert chars.overall_confidence == 0.0

    def test_get_periods_matches_characteristics(self, learner, sample_mixed_periods):
        """Test that get_periods() and get_characteristics() use consistent logic."""
        for m in sample_mixed_periods:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        # Get periods from both methods
        periods_from_get = learner.get_periods(hours=24)
        chars = learner.get_characteristics("24h")

        # Calculate weighted average from periods
        heating_periods = [p for p in periods_from_get if p["type"] == "heating"]
        cooling_periods = [p for p in periods_from_get if p["type"] == "cooling"]

        # Verify sample counts match
        assert len(heating_periods) == chars.heating_samples
        assert len(cooling_periods) == chars.cooling_samples

        # Verify weighted averages match
        if heating_periods:
            total_change = sum(p["temp_change"] for p in heating_periods)
            total_duration = sum(p["duration_hours"] for p in heating_periods)
            expected_rate = total_change / total_duration
            assert chars.heating_rate == pytest.approx(expected_rate, abs=0.001)

        if cooling_periods:
            total_change = sum(p["temp_change"] for p in cooling_periods)
            total_duration = sum(p["duration_hours"] for p in cooling_periods)
            expected_rate = total_change / total_duration
            assert chars.cooling_rate == pytest.approx(expected_rate, abs=0.001)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_temperature_change_in_heating(self, learner, create_measurement):
        """Test that heating periods with negative temp change are filtered out."""
        measurements = [
            create_measurement(
                0, temperature=20.0, heating_active=False, target_temp=22.0
            ),
            create_measurement(
                15, temperature=20.0, heating_active=True, target_temp=22.0
            ),
            create_measurement(
                75, temperature=19.0, heating_active=True, target_temp=22.0
            ),  # Temp decreased!
            create_measurement(
                90, temperature=19.0, heating_active=False, target_temp=22.0
            ),
        ]

        for m in measurements:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        chars = learner.get_characteristics("24h")

        # Heating period with negative change should be filtered
        assert chars.heating_samples == 0
        assert chars.heating_rate == 0.0

    def test_missing_target_temperature(self, learner, create_measurement):
        """Test handling of measurements without target temperature."""
        measurements = [
            create_measurement(
                0, temperature=18.0, heating_active=False, target_temp=None
            ),
            create_measurement(
                15, temperature=18.0, heating_active=True, target_temp=None
            ),
            create_measurement(
                75, temperature=20.0, heating_active=True, target_temp=None
            ),
            create_measurement(
                90, temperature=20.0, heating_active=False, target_temp=None
            ),
        ]

        for m in measurements:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        chars = learner.get_characteristics("24h")

        # Should still detect period, just without maintenance filtering
        assert chars.heating_samples == 1
        assert chars.heating_rate > 0

    def test_timezone_aware_timestamps(self, learner, entity_id):
        """Test that timezone-naive timestamps are handled correctly."""
        # Create timezone-naive timestamp
        naive_time = datetime(2025, 12, 29, 12, 0, 0)  # No tzinfo

        measurement = learner.add_measurement(
            timestamp=naive_time,
            temperature=20.0,
            outdoor_temp=5.0,
            heating_active=False,
            target_temp=21.0,
        )

        # Should be converted to timezone-aware
        assert measurement.timestamp.tzinfo is not None
        assert measurement.timestamp.tzinfo == timezone.utc

    def test_cleanup_old_measurements(self, learner, create_measurement):
        """Test that old measurements are removed based on max_history_hours."""
        # Learner max_history_hours = 168 (7 days)

        # Add old measurements (8 days ago)
        old_time = datetime.now(timezone.utc) - timedelta(hours=200)
        old_measurement = EntityMeasurement(
            timestamp=old_time,
            temperature=20.0,
            outdoor_temp=5.0,
            heating_active=False,
            target_temp=21.0,
        )
        learner.measurements.append(old_measurement)

        # Add recent measurement
        learner.add_measurement(
            timestamp=datetime.now(timezone.utc),
            temperature=21.0,
            outdoor_temp=6.0,
            heating_active=True,
            target_temp=22.0,
        )

        # Old measurement should be cleaned up
        assert len(learner.measurements) == 1
        assert learner.measurements[0].temperature == 21.0


class TestCacheBehavior:
    """Tests for caching behavior of characteristics calculation."""

    def test_cache_validity(self, learner, sample_heating_period):
        """Test that characteristics are cached for performance."""
        for m in sample_heating_period:
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        # First call should populate cache
        chars1 = learner.get_characteristics("24h")
        cache_time1 = learner._cache_timestamp

        # Second call within cache validity period
        chars2 = learner.get_characteristics("24h")
        cache_time2 = learner._cache_timestamp

        # Cache should be reused
        assert cache_time1 == cache_time2
        assert chars1.heating_rate == chars2.heating_rate

    def test_cache_invalidation_on_new_data(self, learner, sample_heating_period):
        """Test that cache is invalidated when new measurements are added."""
        for m in sample_heating_period[:-1]:  # Add all but last
            learner.add_measurement(
                timestamp=m.timestamp,
                temperature=m.temperature,
                outdoor_temp=m.outdoor_temp,
                heating_active=m.heating_active,
                target_temp=m.target_temp,
            )

        learner.get_characteristics("24h")  # Populate cache

        # Add new measurement
        last = sample_heating_period[-1]
        learner.add_measurement(
            timestamp=last.timestamp,
            temperature=last.temperature,
            outdoor_temp=last.outdoor_temp,
            heating_active=last.heating_active,
            target_temp=last.target_temp,
        )

        # Cache should be invalidated
        assert learner._cache_timestamp is None
