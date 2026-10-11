"""
Theria Custom Exceptions

Simple exception hierarchy for error handling.
"""


class TheriaError(Exception):
    """Base exception for Theria."""

    pass


class ConfigurationError(TheriaError):
    """Configuration is invalid."""

    pass


class HAConnectionError(TheriaError):
    """Cannot connect to Home Assistant."""

    pass


class SensorError(TheriaError):
    """Sensor data is unavailable or invalid."""

    pass


class OptimizationError(TheriaError):
    """MPC optimization failed."""

    pass


class PriceDataError(TheriaError):
    """Electricity price data is missing or malformed."""

    pass


class ModelFitError(TheriaError):
    """A model cannot be fitted from the supplied data."""

    pass


class WeatherDataError(TheriaError):
    """Weather forecast data is missing or malformed."""

    pass


class StateStoreError(TheriaError):
    """The persisted plan state cannot be read or written."""

    pass


class SetpointRestoreError(TheriaError):
    """The base room setpoint could not be written back to the heat pump."""

    pass
