"""
Theria Data Models

Minimal data models - will be expanded as features are implemented.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ZoneStatus:
    """Current status of a heating zone."""

    zone_id: str
    timestamp: datetime
    current_temp: float
    target_temp: Optional[float] = None
    hvac_mode: Optional[str] = None


# Placeholder for future models
# Will add as needed:
# - ThermalModel (C, U, eta parameters)
# - MPCTrajectory (optimization results)
# - EnergyData (historical measurements)
# - ComfortProfile (user preferences)
