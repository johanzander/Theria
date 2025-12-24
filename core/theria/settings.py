"""
Theria Configuration Settings

Minimal configuration structure - will be expanded as needed.
User-facing settings are loaded from config.yaml via Home Assistant add-on.
"""

from dataclasses import dataclass
import re


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
    comfort_target: float = 21.0  # Target comfort temperature (°C) - for heat capacitor strategy
    allowed_deviation: float = 1.0  # Allowed temperature variation (±°C) - for heat capacitor strategy
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
