"""
System Identification - Learn building thermal parameters from real data.

Analyzes heating/cooling events to estimate:
- Thermal mass (C) in J/K
- Heat loss coefficient (U) in W/K
- Radiator power characteristics
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from scipy.optimize import curve_fit
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThermalEstimate:
    """Estimated thermal parameters."""
    thermal_mass: float  # J/K
    heat_loss_coeff: float  # W/K
    time_constant: float  # seconds
    confidence: float  # 0-1
    samples_used: int
    timestamp: str


class SystemIdentification:
    """Identifies building thermal parameters from real data."""

    def __init__(self, ha_client, zone_id: str, sensor_config: dict = None, radiator_spec: dict = None):
        """Initialize system identification.

        Args:
            ha_client: Home Assistant client (for real-time sensor reading)
            zone_id: Zone identifier
            sensor_config: Sensor configuration mapping:
                {
                    'indoor_temp': 'vantsidan',  # Climate entity or temp sensor
                    'outdoor_temp': 'outdoor',
                    'water_temp': 'radiator_forward'
                }
            radiator_spec: Radiator specifications for power calculation:
                {
                    'count': 2,  # Number of radiators
                    'nominal_power': 1060.0,  # W per radiator at nominal ΔT
                    'nominal_delta_t': 30.0,  # °C
                    'exponent': 1.3  # Power law exponent
                }
        """
        self.ha_client = ha_client
        self.zone_id = zone_id
        self.sensor_config = sensor_config or {}
        self.radiator_spec = radiator_spec or {
            'count': 2,
            'nominal_power': 1060.0,
            'nominal_delta_t': 30.0,
            'exponent': 1.3
        }
        self.latest_estimate: Optional[ThermalEstimate] = None

    def estimate_from_cooldown(
        self,
        indoor_temps: np.ndarray,
        outdoor_temps: np.ndarray,
        timestamps: np.ndarray,
        initial_temp: float
    ) -> Tuple[float, float]:
        """Estimate U and C from cooldown period (no heating).

        When heating is off: dT/dt = -U/C * (T_indoor - T_outdoor)

        Args:
            indoor_temps: Indoor temperature measurements
            outdoor_temps: Outdoor temperature measurements
            timestamps: Unix timestamps
            initial_temp: Starting temperature

        Returns:
            Tuple of (U, C)
        """
        if len(indoor_temps) < 3:
            raise ValueError("Need at least 3 data points")

        # Calculate temperature derivatives (dT/dt)
        dt = np.diff(timestamps)  # seconds
        dT = np.diff(indoor_temps)  # °C
        dT_dt = dT / dt  # °C/s

        # Calculate average temps for each interval
        T_indoor_avg = (indoor_temps[:-1] + indoor_temps[1:]) / 2
        T_outdoor_avg = (outdoor_temps[:-1] + outdoor_temps[1:]) / 2
        delta_T = T_indoor_avg - T_outdoor_avg

        # Model: dT/dt = -(U/C) * (T_indoor - T_outdoor)
        # Linear regression: dT/dt = slope * delta_T
        # slope = -U/C

        # Remove outliers
        mask = np.abs(dT_dt) < 0.01  # Filter extreme values
        if mask.sum() < 2:
            raise ValueError("Too many outliers")

        dT_dt_clean = dT_dt[mask]
        delta_T_clean = delta_T[mask]

        # Linear fit
        slope, intercept = np.polyfit(delta_T_clean, dT_dt_clean, 1)

        U_over_C = -slope  # Should be positive

        if U_over_C <= 0:
            raise ValueError("Invalid slope - heating might be on")

        # Estimate U from known heat loss (~100 W/K for this room)
        # Or use time constant assumption
        # For now, assume U ≈ 100 W/K as baseline
        U_estimated = 100.0  # W/K
        C_estimated = U_estimated / U_over_C

        return U_estimated, C_estimated

    def estimate_from_heating(
        self,
        indoor_temps: np.ndarray,
        outdoor_temps: np.ndarray,
        water_temps: np.ndarray,
        timestamps: np.ndarray,
        heating_request_data: dict = None
    ) -> Tuple[float, float]:
        """Estimate U and C from heating period.

        When heating is on: dT/dt = (P_heating - U*(T_in - T_out)) / C

        Args:
            indoor_temps: Indoor temperature measurements
            outdoor_temps: Outdoor temperature measurements
            water_temps: Water temperature measurements
            timestamps: Unix timestamps
            heating_request_data: Dict mapping climate entity names to [(timestamp, heating_request), ...]

        Returns:
            Tuple of (U, C)
        """
        if len(indoor_temps) < 3:
            raise ValueError("Need at least 3 data points")

        # Calculate derivatives
        dt = np.diff(timestamps)
        dT = np.diff(indoor_temps)
        dT_dt = dT / dt  # °C/s

        # Average values for each interval
        T_indoor_avg = (indoor_temps[:-1] + indoor_temps[1:]) / 2
        T_outdoor_avg = (outdoor_temps[:-1] + outdoor_temps[1:]) / 2
        T_water_avg = (water_temps[:-1] + water_temps[1:]) / 2

        # Estimate radiator power from water temperature using radiator power law
        # P = P_nominal * (ΔT_actual / ΔT_nominal)^n
        delta_T_rad = T_water_avg - T_indoor_avg

        # Get radiator specifications
        P_nominal = self.radiator_spec.get('nominal_power', 1060.0)
        delta_T_nominal = self.radiator_spec.get('nominal_delta_t', 30.0)
        exponent = self.radiator_spec.get('exponent', 1.3)

        # Calculate power for each radiator based on heating requests
        # If heating_request_data is provided, calculate variable power
        # Otherwise, assume all radiators are always on (legacy behavior)
        if heating_request_data:
            logger.info(f"🔥 Calculating variable heating power from {len(heating_request_data)} climate entities")

            # Initialize total power array
            P_heating = np.zeros(len(T_water_avg))

            # For each climate entity, interpolate heating request and add its power
            for entity_name, heating_data in heating_request_data.items():
                if not heating_data:
                    logger.warning(f"   No heating data for {entity_name}, skipping")
                    continue

                # Extract timestamps and values
                heating_times = np.array([ts.timestamp() for ts, _ in heating_data])
                heating_values = np.array([val for _, val in heating_data])

                # Interpolate to match our averaged interval timestamps
                # Average the timestamps for intervals (same as T_indoor_avg, T_water_avg)
                interval_times = (timestamps[:-1] + timestamps[1:]) / 2

                # Interpolate heating request (use nearest neighbor to preserve binary nature)
                heating_interpolated = np.interp(interval_times, heating_times, heating_values)

                # Calculate power for this radiator (only when heating is requested)
                # heating_request is typically 0-100 percentage, or 0/1 binary
                # Normalize to 0-1 range
                heating_active = heating_interpolated / 100.0 if heating_interpolated.max() > 1 else heating_interpolated

                # Add this radiator's contribution
                P_radiator = heating_active * P_nominal * np.power(
                    np.maximum(delta_T_rad, 1.0) / delta_T_nominal,
                    exponent
                )
                P_heating += P_radiator

                logger.info(f"   {entity_name}: {heating_active.mean():.1%} average heating activity")

            logger.info(f"   Total average power: {P_heating.mean():.0f} W (range: {P_heating.min():.0f} - {P_heating.max():.0f} W)")
        else:
            # Legacy: assume all radiators always on (use num_radiators from config)
            num_radiators = self.radiator_spec.get('count', 2)
            P_heating = num_radiators * P_nominal * np.power(
                np.maximum(delta_T_rad, 1.0) / delta_T_nominal,
                exponent
            )
            logger.info(f"⚠️  No heating request data - assuming {num_radiators} radiators always on")

        # Model: C * dT/dt = P_heating - U * (T_indoor - T_outdoor)
        # Rearrange: dT/dt = (1/C) * P_heating - (U/C) * (T_indoor - T_outdoor)

        delta_T_loss = T_indoor_avg - T_outdoor_avg

        # Remove outliers
        mask = (np.abs(dT_dt) < 0.01) & (P_heating > 0)
        if mask.sum() < 3:
            raise ValueError("Insufficient valid data points")

        # Create design matrix for linear regression
        # y = dT/dt
        # X = [P_heating, delta_T_loss]
        # params = [1/C, -U/C]

        X = np.column_stack([P_heating[mask], delta_T_loss[mask]])
        y = dT_dt[mask]

        # Least squares fit
        params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        one_over_C = params[0]
        minus_U_over_C = params[1]

        if one_over_C <= 0:
            raise ValueError("Invalid C estimate")

        C_estimated = 1.0 / one_over_C
        U_estimated = -minus_U_over_C * C_estimated

        if U_estimated <= 0:
            raise ValueError("Invalid U estimate")

        return U_estimated, C_estimated

    def analyze_recent_data(
        self,
        hours_back: int = 24
    ) -> Optional[ThermalEstimate]:
        """Analyze recent data to estimate thermal parameters.

        Uses sensor_config provided during initialization to fetch:
        - indoor_temp: Indoor temperature sensor
        - outdoor_temp: Outdoor temperature sensor
        - water_temp: Radiator/water temperature sensor (optional)
        - heating_power: Actual heating power consumption (optional)

        Args:
            hours_back: How many hours of data to analyze

        Returns:
            ThermalEstimate or None if failed
        """
        if not self.sensor_config:
            logger.warning("No sensor config provided for system identification")
            return None

        try:
            from .influxdb_helper import get_sensor_timeseries

            logger.info(f"Analyzing last {hours_back}h of data for zone {self.zone_id}")

            # Determine time range
            stop_time = datetime.now()
            start_time = stop_time - timedelta(hours=hours_back)

            # Fetch indoor temperature data (support multiple sensors for averaging)
            indoor_sensor = self.sensor_config.get('indoor_temp')
            if not indoor_sensor:
                raise ValueError("indoor_temp sensor not configured")

            # Support multiple indoor sensors (e.g., "vantsidan,klippsidan")
            indoor_sensors = [s.strip() for s in indoor_sensor.split(',')]

            indoor_data_all = []
            for sensor in indoor_sensors:
                # Determine if indoor sensor is a climate entity or regular sensor
                indoor_domain = "climate" if sensor in ['vantsidan', 'klippsidan'] else "sensor"

                indoor_result = get_sensor_timeseries(
                    sensor, start_time, stop_time, domain=indoor_domain
                )
                if indoor_result["status"] != "success":
                    logger.warning(f"Failed to fetch indoor temp from {sensor}: {indoor_result.get('message')}")
                    continue

                data = indoor_result["data"]
                logger.info(f"📥 Fetched {len(data)} indoor temperature points from {sensor}")
                if len(data) > 0:
                    logger.info(f"   Time range: {data[0][0]} to {data[-1][0]}")

                indoor_data_all.append(data)

            if not indoor_data_all:
                raise ValueError("No indoor temperature data available from any sensor")

            # If multiple sensors, average them
            if len(indoor_data_all) == 1:
                indoor_data = indoor_data_all[0]
                logger.info(f"Using single indoor sensor: {len(indoor_data)} points")
            else:
                logger.info(f"Averaging {len(indoor_data_all)} indoor sensors...")
                indoor_data = self._average_timeseries(indoor_data_all)
                logger.info(f"Result: {len(indoor_data)} averaged points")

            if len(indoor_data) < 10:
                raise ValueError(f"Insufficient indoor temp data: {len(indoor_data)} points")

            # Fetch outdoor temperature data
            outdoor_sensor = self.sensor_config.get('outdoor_temp')
            if not outdoor_sensor:
                raise ValueError("outdoor_temp sensor not configured")

            outdoor_result = get_sensor_timeseries(
                outdoor_sensor, start_time, stop_time, domain="sensor"
            )
            if outdoor_result["status"] != "success":
                raise RuntimeError(f"Failed to fetch outdoor temp: {outdoor_result.get('message')}")

            outdoor_data = outdoor_result["data"]
            logger.info(f"📥 Fetched {len(outdoor_data)} outdoor temperature points from InfluxDB")
            if len(outdoor_data) > 0:
                logger.info(f"   Time range: {outdoor_data[0][0]} to {outdoor_data[-1][0]}")

            if len(outdoor_data) < 10:
                raise ValueError(f"Insufficient outdoor temp data: {len(outdoor_data)} points")

            # Try heating analysis first if water temp sensor is available (more accurate)
            water_sensor = self.sensor_config.get('water_temp')
            if water_sensor:
                logger.info(f"🌡️  Water temp sensor configured: {water_sensor}")
                water_result = get_sensor_timeseries(
                    water_sensor, start_time, stop_time, domain="sensor"
                )
                if water_result["status"] == "success":
                    water_data = water_result["data"]
                    logger.info(f"📥 Fetched {len(water_data)} water temperature points from InfluxDB")
                    if len(water_data) > 0:
                        logger.info(f"   Time range: {water_data[0][0]} to {water_data[-1][0]}")

                    # Fetch heating_power_request for both climate entities
                    logger.info("📥 Fetching heating request states from climate entities...")
                    heating_request_data = {}

                    # Get list of indoor sensors (same as for temperature)
                    indoor_sensor = self.sensor_config.get('indoor_temp')
                    if indoor_sensor:
                        indoor_sensors = [s.strip() for s in indoor_sensor.split(',')]

                        for sensor in indoor_sensors:
                            # Only fetch for climate entities (not regular sensors)
                            if sensor in ['vantsidan', 'klippsidan']:
                                heating_result = get_sensor_timeseries(
                                    sensor, start_time, stop_time,
                                    domain="climate",
                                    field_name="heating_power_request"
                                )
                                if heating_result["status"] == "success":
                                    heating_request_data[sensor] = heating_result["data"]
                                    logger.info(f"   {sensor}: {len(heating_result['data'])} heating request points")
                                else:
                                    logger.warning(f"   Failed to fetch heating request for {sensor}: {heating_result.get('message')}")

                    if not heating_request_data:
                        logger.warning("⚠️  No heating request data available - will assume constant heating")

                    logger.info(f"🔗 Aligning {len(indoor_data)} indoor + {len(outdoor_data)} outdoor + {len(water_data)} water temp points...")
                    indoor_temps, outdoor_temps, water_temps, timestamps = self._align_timeseries_3(
                        indoor_data, outdoor_data, water_data
                    )
                    logger.info(f"✅ After alignment: {len(indoor_temps)} matching samples")

                    if len(indoor_temps) >= 10:
                        try:
                            U_estimated, C_estimated = self.estimate_from_heating(
                                indoor_temps=indoor_temps,
                                outdoor_temps=outdoor_temps,
                                water_temps=water_temps,
                                timestamps=timestamps,
                                heating_request_data=heating_request_data
                            )
                            tau = C_estimated / U_estimated

                            estimate = ThermalEstimate(
                                thermal_mass=C_estimated,
                                heat_loss_coeff=U_estimated,
                                time_constant=tau,
                                confidence=0.8,  # Higher confidence with heating data
                                samples_used=len(indoor_temps),
                                timestamp=datetime.utcnow().isoformat()
                            )

                            self.latest_estimate = estimate
                            logger.info(
                                f"System ID (heating): C={C_estimated/1e6:.2f} MJ/K, "
                                f"U={U_estimated:.1f} W/K, τ={tau/3600:.2f}h, "
                                f"samples={len(indoor_temps)}"
                            )

                            return estimate

                        except ValueError as e:
                            logger.warning(f"Heating analysis failed: {e}")
                    else:
                        logger.warning(f"Insufficient aligned data for heating analysis: {len(indoor_temps)} points")
                else:
                    logger.warning(f"Failed to fetch water temp: {water_result.get('message')}")
            else:
                logger.info("No water temp sensor configured, will use cooldown analysis")

            # Fall back to cooldown analysis (uses only indoor/outdoor temps)
            logger.info("Attempting cooldown analysis as fallback...")
            indoor_temps, outdoor_temps, timestamps = self._align_timeseries(indoor_data, outdoor_data)

            if len(indoor_temps) < 10:
                raise ValueError(f"Insufficient aligned data: {len(indoor_temps)} points")

            logger.info(f"Aligned {len(indoor_temps)} data points for cooldown analysis")

            try:
                U_estimated, C_estimated = self.estimate_from_cooldown(
                    indoor_temps=indoor_temps,
                    outdoor_temps=outdoor_temps,
                    timestamps=timestamps,
                    initial_temp=indoor_temps[0]
                )
                tau = C_estimated / U_estimated

                estimate = ThermalEstimate(
                    thermal_mass=C_estimated,
                    heat_loss_coeff=U_estimated,
                    time_constant=tau,
                    confidence=0.7,  # Lower confidence - assumes U=100 W/K
                    samples_used=len(indoor_temps),
                    timestamp=datetime.utcnow().isoformat()
                )

                self.latest_estimate = estimate
                logger.info(
                    f"System ID (cooldown): C={C_estimated/1e6:.2f} MJ/K, "
                    f"U={U_estimated:.1f} W/K, τ={tau/3600:.2f}h, "
                    f"samples={len(indoor_temps)}"
                )

                return estimate

            except ValueError as e:
                logger.error(f"Cooldown analysis failed: {e}")

            # If both failed, return None
            logger.warning("All system identification methods failed")
            return None

        except Exception as e:
            logger.error(f"System identification failed: {e}", exc_info=True)
            return None

    def _average_timeseries(self, timeseries_list):
        """Average multiple time series with different timestamps using interpolation.

        Args:
            timeseries_list: List of [(timestamp, value), ...] for each sensor

        Returns:
            Averaged time series: [(timestamp, avg_value), ...]
        """
        if not timeseries_list:
            return []

        # Use the first sensor's timestamps as reference
        reference_times = [ts for ts, _ in timeseries_list[0]]
        reference_timestamps = np.array([ts.timestamp() for ts in reference_times])

        # Interpolate all other sensors to match reference timestamps
        interpolated_values = []
        for series in timeseries_list:
            times = np.array([ts.timestamp() for ts, _ in series])
            values = np.array([val for _, val in series])
            interp_vals = np.interp(reference_timestamps, times, values)
            interpolated_values.append(interp_vals)

        # Average all interpolated values
        avg_values = np.mean(interpolated_values, axis=0)

        # Return as list of (timestamp, value) tuples
        return list(zip(reference_times, avg_values))

    def _align_timeseries(self, indoor_data, outdoor_data):
        """Align two time series using interpolation.

        Uses indoor timestamps as reference and interpolates outdoor temps.

        Args:
            indoor_data: [(timestamp, value), ...]
            outdoor_data: [(timestamp, value), ...]

        Returns:
            Tuple of (indoor_temps, outdoor_temps, timestamps) as numpy arrays
        """
        if not indoor_data or not outdoor_data:
            return (np.array([]), np.array([]), np.array([]))

        # Use indoor timestamps as reference
        indoor_times = np.array([ts.timestamp() for ts, _ in indoor_data])
        indoor_temps = np.array([val for _, val in indoor_data])

        # Prepare outdoor data for interpolation
        outdoor_times = np.array([ts.timestamp() for ts, _ in outdoor_data])
        outdoor_vals = np.array([val for _, val in outdoor_data])

        # Interpolate outdoor temps to match indoor timestamps
        outdoor_temps = np.interp(indoor_times, outdoor_times, outdoor_vals)

        logger.debug(
            f"Interpolated outdoor temps to {len(indoor_times)} indoor timestamps"
        )

        return (indoor_temps, outdoor_temps, indoor_times)

    def _align_timeseries_3(self, indoor_data, outdoor_data, water_data):
        """Align three time series using interpolation.

        Uses indoor timestamps as reference and interpolates outdoor/water temps.

        Args:
            indoor_data: [(timestamp, value), ...]
            outdoor_data: [(timestamp, value), ...]
            water_data: [(timestamp, value), ...]

        Returns:
            Tuple of (indoor_temps, outdoor_temps, water_temps, timestamps) as numpy arrays
        """
        if not indoor_data or not outdoor_data or not water_data:
            return (np.array([]), np.array([]), np.array([]), np.array([]))

        # Use indoor timestamps as reference (sparsest sensor)
        indoor_times = np.array([ts.timestamp() for ts, _ in indoor_data])
        indoor_temps = np.array([val for _, val in indoor_data])

        # Prepare outdoor data for interpolation
        outdoor_times = np.array([ts.timestamp() for ts, _ in outdoor_data])
        outdoor_vals = np.array([val for _, val in outdoor_data])

        # Prepare water data for interpolation
        water_times = np.array([ts.timestamp() for ts, _ in water_data])
        water_vals = np.array([val for _, val in water_data])

        # Interpolate outdoor and water temps to match indoor timestamps
        outdoor_temps = np.interp(indoor_times, outdoor_times, outdoor_vals)
        water_temps = np.interp(indoor_times, water_times, water_vals)

        logger.debug(
            f"Interpolated outdoor and water temps to {len(indoor_times)} indoor timestamps"
        )

        return (indoor_temps, outdoor_temps, water_temps, indoor_times)

    def get_latest_estimate(self) -> Optional[ThermalEstimate]:
        """Get the most recent parameter estimate."""
        return self.latest_estimate
