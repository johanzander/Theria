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
