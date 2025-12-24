"""
Simple thermal building model for MPC optimization.

Uses first-order RC circuit analogy:
- C: Thermal capacitance (J/K) - how much energy to raise temp 1°C
- U: Heat transfer coefficient (W/K) - how fast heat escapes
- P: Heating power (W)

Dynamics: dT/dt = (P_heating - U*(T_indoor - T_outdoor)) / C
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThermalParameters:
    """Building thermal parameters."""

    # Thermal capacitance (J/K) - heat capacity of building
    # Typical: 10-50 MJ/K for residential
    thermal_mass: float = 20e6  # 20 MJ/K (medium-sized building)

    # Heat loss coefficient (W/K) - how fast heat escapes
    # Typical: 100-500 W/K for residential
    heat_loss_coeff: float = 200.0  # W/K

    # Maximum heating power per radiator (W)
    max_heating_power: float = 2000.0  # 2 kW per radiator

    # Time constant (hours) - how fast building responds
    # tau = C / U
    @property
    def time_constant_hours(self) -> float:
        """Time constant in hours."""
        return (self.thermal_mass / self.heat_loss_coeff) / 3600.0


class SimpleThermalModel:
    """Simple first-order thermal building model."""

    def __init__(self, params: ThermalParameters):
        """Initialize thermal model.

        Args:
            params: Thermal parameters for the building
        """
        self.params = params

    def predict_temperature(
        self,
        T_initial: float,
        T_outdoor: np.ndarray,
        P_heating: np.ndarray,
        dt_minutes: float = 15.0
    ) -> np.ndarray:
        """Predict indoor temperature over time.

        Uses forward Euler integration of:
        dT/dt = (P_heating - U*(T_indoor - T_outdoor)) / C

        Args:
            T_initial: Initial indoor temperature (°C)
            T_outdoor: Array of outdoor temperatures (°C) at each timestep
            P_heating: Array of heating power (W) at each timestep
            dt_minutes: Timestep in minutes

        Returns:
            Array of predicted indoor temperatures (°C)
        """
        n_steps = len(T_outdoor)
        T_indoor = np.zeros(n_steps + 1)
        T_indoor[0] = T_initial

        # Convert timestep to seconds
        dt = dt_minutes * 60.0

        # Simulate forward in time
        for i in range(n_steps):
            # Heat gain from radiators
            Q_heating = P_heating[i]

            # Heat loss to outside
            Q_loss = self.params.heat_loss_coeff * (T_indoor[i] - T_outdoor[i])

            # Net heat flow
            Q_net = Q_heating - Q_loss

            # Temperature change: dT = Q_net * dt / C
            dT = (Q_net * dt) / self.params.thermal_mass

            # Update temperature
            T_indoor[i + 1] = T_indoor[i] + dT

        return T_indoor[1:]  # Return predictions (skip initial state)

    def steady_state_temperature(
        self,
        T_outdoor: float,
        P_heating: float
    ) -> float:
        """Calculate steady-state indoor temperature.

        At steady state: dT/dt = 0
        Therefore: P_heating = U * (T_indoor - T_outdoor)
        So: T_indoor = T_outdoor + P_heating / U

        Args:
            T_outdoor: Outdoor temperature (°C)
            P_heating: Heating power (W)

        Returns:
            Steady-state indoor temperature (°C)
        """
        return T_outdoor + (P_heating / self.params.heat_loss_coeff)

    def required_power(
        self,
        T_indoor_target: float,
        T_outdoor: float
    ) -> float:
        """Calculate required heating power to maintain target temperature.

        At steady state: P_heating = U * (T_indoor - T_outdoor)

        Args:
            T_indoor_target: Desired indoor temperature (°C)
            T_outdoor: Outdoor temperature (°C)

        Returns:
            Required heating power (W)
        """
        return self.params.heat_loss_coeff * (T_indoor_target - T_outdoor)
