"""
Theria Configuration Settings

Minimal configuration structure - will be expanded as needed.
User-facing settings are loaded from config.yaml via Home Assistant add-on.
"""

import re
from dataclasses import dataclass


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclass
class ZoneSettings:
    """Configuration for a single heating zone."""

    id: str
    name: str
    climate_entities: list[str]  # One or more climate entities (radiators) in this zone
    temp_sensors: list[str]  # One or more temperature sensors
    icon: str | None = None
    comfort_target: float = (
        21.0  # Target comfort temperature (°C) - for heat capacitor strategy
    )
    allowed_deviation: float = (
        1.0  # Allowed temperature variation (±°C) - for heat capacitor strategy
    )
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ZoneSettings":
        """Create from dictionary."""
        converted = {_camel_to_snake(k): v for k, v in data.items()}

        # Handle legacy single climate_entity/temp_sensor
        if "climate_entity" in converted:
            converted["climate_entities"] = [converted.pop("climate_entity")]
        if "temp_sensor" in converted:
            converted["temp_sensors"] = [converted.pop("temp_sensor")]

        return cls(**converted)


@dataclass
class HeatPumpSettings:
    """Configuration for heat pump monitoring (IVT AirX / Husdata H60)."""

    compressor_speed: str
    compressor_power: str
    mode_switch: str
    heat_carrier_forward: str
    heat_carrier_return: str
    compressor_consumption_total: str
    compressor_consumption_heating: str
    compressor_consumption_hotwater: str
    delivered_energy_total: str
    delivered_energy_heating: str
    delivered_energy_hotwater: str
    aux_consumption_total: str
    hotwater_setpoint: str | None = None
    hotwater_top: str | None = None
    hotwater_mid: str | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "HeatPumpSettings":
        """Create from dictionary."""
        converted = {_camel_to_snake(k): v for k, v in data.items()}
        return cls(**converted)
