"""
Simple Home Assistant API Client for Theria

Minimal client for reading sensors and controlling climate entities.
"""

import logging
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


class HAClient:
    """Simple Home Assistant REST API client."""

    def __init__(self, base_url: str, token: str):
        """Initialize HA client.

        Args:
            base_url: Home Assistant URL (e.g., "http://supervisor/core")
            token: Long-lived access token
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # Create a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        # Set default timeout
        self.timeout = 5

    def get_state(self, entity_id: str) -> dict[str, Any]:
        """Get current state of an entity.

        Args:
            entity_id: Entity ID (e.g., "sensor.temperature")

        Returns:
            State dictionary with 'state', 'attributes', etc.

        Raises:
            ValueError: If entity not found
            RuntimeError: If API request fails
        """
        url = f"{self.base_url}/api/states/{entity_id}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Entity not found: {entity_id}") from e
            raise RuntimeError(f"Failed to get state for {entity_id}: {e}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HA API request failed: {e}") from e

    def get_temperature(self, entity_id: str) -> float:
        """Get temperature from sensor or climate entity.

        Handles both:
        - sensor.xyz (temperature in state)
        - climate.xyz (temperature in attributes.current_temperature)

        Args:
            entity_id: Temperature sensor or climate entity (with or without domain prefix)

        Returns:
            Temperature in Celsius

        Raises:
            ValueError: If temperature cannot be read
        """
        # Add domain prefix if not present
        if "." not in entity_id:
            # Guess domain based on typical usage
            # If starts with common climate names, assume climate domain
            if any(
                x in entity_id.lower()
                for x in ["vantsidan", "klippsidan", "thermostat", "climate"]
            ):
                entity_id = f"climate.{entity_id}"
            else:
                entity_id = f"sensor.{entity_id}"

        state = self.get_state(entity_id)

        # For climate entities, use current_temperature attribute
        if entity_id.startswith("climate."):
            try:
                return float(state["attributes"]["current_temperature"])
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Cannot read current_temperature from climate entity {entity_id}: {e}"
                ) from e

        # For sensors, use state value
        try:
            return float(state["state"])
        except (ValueError, KeyError, TypeError) as e:
            # Fallback: try current_temperature attribute anyway
            try:
                return float(state["attributes"]["current_temperature"])
            except (ValueError, KeyError, TypeError):
                raise ValueError(
                    f"Cannot read temperature from {entity_id}: {e}"
                ) from e

    def get_climate_state(self, entity_id: str) -> dict[str, Any]:
        """Get climate entity state.

        Args:
            entity_id: Climate entity ID

        Returns:
            Dictionary with current_temperature, target_temperature, hvac_mode, etc.
        """
        state = self.get_state(entity_id)
        attrs = state.get("attributes", {})

        return {
            "entity_id": entity_id,
            "hvac_mode": state.get("state"),
            "current_temperature": attrs.get("current_temperature"),
            "target_temperature": attrs.get("temperature"),
            "min_temp": attrs.get("min_temp"),
            "max_temp": attrs.get("max_temp"),
        }

    def set_temperature(self, entity_id: str, temperature: float) -> None:
        """Set target temperature for climate entity.

        Args:
            entity_id: Climate entity ID
            temperature: Target temperature in Celsius

        Raises:
            RuntimeError: If service call fails
        """
        url = f"{self.base_url}/api/services/climate/set_temperature"
        data = {
            "entity_id": entity_id,
            "temperature": temperature,
        }

        try:
            logger.debug(f"Calling {url} with data: {data}")
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            logger.info(
                f"Set {entity_id} to {temperature}°C - Response: {response.status_code}"
            )
            logger.debug(f"Response body: {response.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to set temperature for {entity_id}: {e}") from e

    def set_hvac_mode(self, entity_id: str, mode: str) -> None:
        """Set HVAC mode for climate entity.

        Args:
            entity_id: Climate entity ID
            mode: HVAC mode (e.g., "heat", "off", "auto")

        Raises:
            RuntimeError: If service call fails
        """
        url = f"{self.base_url}/api/services/climate/set_hvac_mode"
        data = {
            "entity_id": entity_id,
            "hvac_mode": mode,
        }

        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Set {entity_id} HVAC mode to {mode}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to set HVAC mode for {entity_id}: {e}") from e

    def get_numeric_history(
        self, entity_id: str, start: datetime, stop: datetime
    ) -> list[tuple[datetime, float]]:
        """Numeric state changes of an entity between `start` and `stop` (read only).

        Returns (time, value) for every recorded change; non-numeric states such as
        "unavailable" are left out.

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/api/history/period/{start.isoformat()}"
        params = {
            "filter_entity_id": entity_id,
            "end_time": stop.isoformat(),
            "minimal_response": "",
            "no_attributes": "",
        }
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HA history request failed for {entity_id}: {e}") from e
        points: list[tuple[datetime, float]] = []
        for record in payload[0] if payload else []:
            try:
                value = float(record["state"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append(
                (
                    datetime.fromisoformat(
                        record["last_changed"].replace("Z", "+00:00")
                    ),
                    value,
                )
            )
        return points

    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None:
        """Call a Home Assistant service.

        Raises:
            RuntimeError: If the service call fails
        """
        url = f"{self.base_url}/api/services/{domain}/{service}"
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HA service {domain}.{service} failed: {e}") from e

    def get_service_response(
        self, domain: str, service: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a Home Assistant service that answers with data and return that answer.

        Raises:
            RuntimeError: If the service call fails
        """
        url = f"{self.base_url}/api/services/{domain}/{service}?return_response"
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["service_response"]
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            raise RuntimeError(f"HA service {domain}.{service} failed: {e}") from e


class HANotifier:
    """Home Assistant persistent notifications (the bell in the HA sidebar)."""

    def __init__(self, ha_client: HAClient):
        self._ha = ha_client

    def notify(self, notification_id: str, title: str, message: str) -> None:
        self._ha.call_service(
            "persistent_notification",
            "create",
            {"notification_id": notification_id, "title": title, "message": message},
        )

    def dismiss(self, notification_id: str) -> None:
        self._ha.call_service(
            "persistent_notification", "dismiss", {"notification_id": notification_id}
        )
