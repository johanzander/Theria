"""
Theria API Endpoints
"""

import os
import sys
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

# Add core to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.theria.ha_client import HAClient
from core.theria.history import history_tracker
from core.theria.price_optimizer import PriceOptimizer
from core.theria.settings import ZoneSettings

router = APIRouter()

# Initialize HA client
HA_URL = os.environ.get("HA_URL", "http://supervisor/core")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

ha_client = HAClient(HA_URL, HA_TOKEN) if HA_TOKEN else None

# Load zones from config (in production, this comes from options.json)
# For development, we'll use a simple example
ZONES: list[ZoneSettings] = []

# Thermal learning service (set by app.py during startup)
learning_service = None


def load_zones_from_config():
    """Load zones from Home Assistant add-on options."""
    global ZONES

    # Try to load from options.json (production)
    options_path = "/data/options.json"
    if os.path.exists(options_path):
        import json
        with open(options_path) as f:
            options = json.load(f)
            zone_configs = options.get("zones", [])
            ZONES = [ZoneSettings.from_dict(z) for z in zone_configs]
            logger.info(f"Loaded {len(ZONES)} zones from options.json")
    else:
        # Development: load from config.yaml if available
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
                zone_configs = config.get("options", {}).get("zones", [])
                if zone_configs:
                    ZONES = [ZoneSettings.from_dict(z) for z in zone_configs]
                    logger.info(f"Loaded {len(ZONES)} zones from config.yaml")
                    return

        # Fallback: hardcoded example
        ZONES = [
            ZoneSettings(
                id="butik",
                name="Butik",
                climate_entities=[
                    "climate.vantsidan",
                    "climate.klippsidan"
                ],
                temp_sensors=[
                    "climate.vantsidan",
                    "climate.klippsidan",
                    "sensor.schamponeringstolen_temperature",
                    "sensor.butik_temperature_2"
                ],
                comfort_target=21.0,
                allowed_deviation=1.0,
                enabled=True
            )
        ]
        logger.warning("Using development zone configuration")


# Load zones on module import
load_zones_from_config()

# Initialize price optimizer
price_optimizer = None
if ha_client:
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
                options = config.get("options", {})
                sensors = options.get("sensors", {})
                price_config = options.get("price_optimization", {})

                if price_config.get("enabled"):
                    # Resolve sensor key to entity ID (BESS pattern)
                    sensor_key = price_config.get("price_sensor")
                    if sensor_key and sensor_key in sensors:
                        price_entity = sensors[sensor_key]
                        price_optimizer = PriceOptimizer(
                            ha_client,
                            price_entity=price_entity,
                            expensive_hours=price_config.get("expensive_hours", 4),
                            cheap_hours=price_config.get("cheap_hours", 4),
                            adjustment_degrees=price_config.get("adjustment_degrees", 0.5)
                        )
                        logger.info(f"Price optimizer initialized (sensor: {sensor_key} -> {price_entity})")
    except Exception as e:
        logger.warning(f"Failed to initialize price optimizer in API: {e}")


class SetTemperatureRequest(BaseModel):
    """Request body for setting temperature."""
    temperature: float


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": "Theria",
        "version": "0.1.0",
        "ha_connected": ha_client is not None,
    }


@router.get("/api/zones")
async def get_zones():
    """Get all configured zones."""
    return {
        "zones": [
            {
                "id": zone.id,
                "name": zone.name,
                "climate_entities": zone.climate_entities,
                "temp_sensors": zone.temp_sensors,
                "enabled": zone.enabled,
            }
            for zone in ZONES
        ]
    }


@router.get("/api/zones/{zone_id}/status")
async def get_zone_status(zone_id: str):
    """Get current status of a zone including all temperature readings."""
    if not ha_client:
        raise HTTPException(status_code=503, detail="HA client not initialized")

    # Find zone
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    try:
        # Read all temperature sensors
        temperatures = []
        for sensor in zone.temp_sensors:
            try:
                temp = ha_client.get_temperature(sensor)
                temperatures.append({
                    "entity_id": sensor,
                    "temperature": temp,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to read {sensor}: {e}")
                temperatures.append({
                    "entity_id": sensor,
                    "temperature": None,
                    "error": str(e)
                })

        # Calculate average of valid temperatures
        valid_temps = [t["temperature"] for t in temperatures if t["temperature"] is not None]
        avg_temp = sum(valid_temps) / len(valid_temps) if valid_temps else None

        # Read climate entity states
        climate_states = []
        for climate_entity in zone.climate_entities:
            try:
                state = ha_client.get_climate_state(climate_entity)
                climate_states.append(state)
            except Exception as e:
                logger.warning(f"Failed to read {climate_entity}: {e}")
                climate_states.append({
                    "entity_id": climate_entity,
                    "error": str(e)
                })

        # In observation mode - Theria is learning, not controlling
        # Thermostats are controlled by Netatmo app/hub

        return {
            "zone_id": zone.id,
            "name": zone.name,
            "timestamp": datetime.utcnow().isoformat(),
            "average_temperature": avg_temp,
            "mode": "observation",  # Learning thermal characteristics
            "temperatures": temperatures,
            "climate_states": climate_states,
        }

    except Exception as e:
        logger.error(f"Error getting zone status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/zones/{zone_id}/set_temperature")
async def set_zone_temperature(zone_id: str, request: SetTemperatureRequest):
    """Set target temperature for all radiators in a zone.

    This sets the same target temperature on all climate entities in the zone.
    Each radiator will independently heat until it reaches the target.
    """
    if not ha_client:
        raise HTTPException(status_code=503, detail="HA client not initialized")

    # Find zone
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    if not zone.enabled:
        raise HTTPException(status_code=400, detail=f"Zone is disabled: {zone_id}")

    try:
        results = []
        errors = []

        # Set temperature on all climate entities
        for i, climate_entity in enumerate(zone.climate_entities):
            try:
                # Add small delay between calls to avoid HA race conditions
                if i > 0:
                    time.sleep(0.5)

                ha_client.set_temperature(climate_entity, request.temperature)
                results.append({
                    "entity_id": climate_entity,
                    "status": "success",
                    "target_temperature": request.temperature
                })
            except Exception as e:
                logger.error(f"Failed to set temperature on {climate_entity}: {e}")
                errors.append({
                    "entity_id": climate_entity,
                    "status": "failed",
                    "error": str(e)
                })

        if errors and not results:
            # All failed
            raise HTTPException(
                status_code=500,
                detail=f"Failed to set temperature on all radiators: {errors}"
            )

        # Log control event
        if results:
            history_tracker.add_control_event(
                zone_id=zone.id,
                action="manual_override",
                details=f"Manual temperature set to {request.temperature}°C on {len(results)} radiators",
                temperature=request.temperature
            )

        return {
            "zone_id": zone.id,
            "target_temperature": request.temperature,
            "results": results,
            "errors": errors if errors else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting zone temperature: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "system": "operational",
        "mode": "mvp",
        "zones": len(ZONES),
        "ha_url": HA_URL,
        "ha_connected": ha_client is not None,
    }


@router.get("/api/zones/{zone_id}/history")
async def get_zone_history(zone_id: str, hours: int = 24):
    """Get temperature history for a zone.

    Args:
        zone_id: Zone identifier
        hours: How many hours of history to retrieve (default: 24)

    Returns:
        List of temperature readings with timestamps
    """
    # Verify zone exists
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    history = history_tracker.get_temperature_history(zone_id=zone_id, hours=hours)

    return {
        "zone_id": zone_id,
        "hours": hours,
        "count": len(history),
        "history": history
    }


@router.get("/api/zones/{zone_id}/events")
async def get_zone_events(zone_id: str, hours: int = 24):
    """Get control events for a zone.

    Args:
        zone_id: Zone identifier
        hours: How many hours of events to retrieve (default: 24)

    Returns:
        List of control events with timestamps
    """
    # Verify zone exists
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    events = history_tracker.get_control_events(zone_id=zone_id, hours=hours)

    return {
        "zone_id": zone_id,
        "hours": hours,
        "count": len(events),
        "events": events
    }


@router.get("/api/price/current")
async def get_current_price():
    """Get current electricity price and optimization status."""
    if not price_optimizer:
        return {
            "enabled": False,
            "message": "Price optimization not enabled"
        }

    current_price = price_optimizer.get_current_price()
    price_adjustment = price_optimizer.get_price_adjustment()
    category = price_optimizer.get_current_category()

    # Calculate historical average
    historical_avg = None
    if len(price_optimizer.historical_prices) > 24:
        historical_avg = sum(price_optimizer.historical_prices) / len(price_optimizer.historical_prices)

    return {
        "enabled": True,
        "current_price": current_price,
        "price_adjustment": price_adjustment,
        "category": category,
        "historical_avg_7d": historical_avg,
        "is_expensive_hour": price_adjustment < 0,
        "is_cheap_hour": price_adjustment > 0,
        # Legacy fields (for backward compatibility)
        "expensive_hours": sorted(list(price_optimizer.expensive_hour_set)),
        "cheap_hours": sorted(list(price_optimizer.cheap_hour_set))
    }


@router.get("/api/price/forecast")
async def get_price_forecast(hours: int = 24):
    """Get electricity price forecast.

    Args:
        hours: How many hours ahead to forecast (default: 24)

    Returns:
        List of price forecast entries with categories
    """
    if not price_optimizer:
        return {
            "enabled": False,
            "forecast": []
        }

    forecast = price_optimizer.get_price_forecast_24h()

    return {
        "enabled": True,
        "count": len(forecast),
        "forecast": [
            {
                "timestamp": entry["timestamp"].isoformat(),
                "hour": entry["hour"],
                "price": entry["price"],
                "category": price_optimizer.hour_categories.get(entry["hour"], "NORMAL"),
                # Legacy fields (for backward compatibility)
                "is_expensive": entry["hour"] in price_optimizer.expensive_hour_set,
                "is_cheap": entry["hour"] in price_optimizer.cheap_hour_set
            }
            for entry in forecast[:hours]
        ]
    }


@router.get("/api/thermal/characteristics")
async def get_thermal_characteristics(zone_id: str = None):
    """Get learned thermal characteristics for zones.

    Args:
        zone_id: Optional zone filter (returns all zones if not specified)

    Returns:
        Learned thermal characteristics (heating/cooling rates, confidence)
    """
    if learning_service is None:
        raise HTTPException(status_code=503, detail="Thermal learning service not available")

    if zone_id:
        chars = learning_service.get_zone_characteristics(zone_id)
        if chars is None:
            raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

        # Convert dataclass to dict for JSON response
        characteristics = chars["characteristics"]
        return {
            "zone_id": zone_id,
            "heating_rate": characteristics.heating_rate,
            "heating_rate_confidence": characteristics.heating_rate_confidence,
            "heating_samples": characteristics.heating_samples,
            "cooling_rate_base": characteristics.cooling_rate_base,
            "cooling_rate_confidence": characteristics.cooling_rate_confidence,
            "cooling_samples": characteristics.cooling_samples,
            "outdoor_temp_coefficient": characteristics.outdoor_temp_coefficient,
            "last_updated": characteristics.last_updated.isoformat() if characteristics.last_updated else None,
            "overall_confidence": chars["confidence"],
            "recent_measurements": chars["recent_measurements"]
        }
    else:
        # Return all zones
        all_chars = learning_service.get_all_characteristics()
        result = {}

        for zid, chars in all_chars.items():
            if chars:
                characteristics = chars["characteristics"]
                result[zid] = {
                    "heating_rate": characteristics.heating_rate,
                    "heating_rate_confidence": characteristics.heating_rate_confidence,
                    "heating_samples": characteristics.heating_samples,
                    "cooling_rate_base": characteristics.cooling_rate_base,
                    "cooling_rate_confidence": characteristics.cooling_rate_confidence,
                    "cooling_samples": characteristics.cooling_samples,
                    "outdoor_temp_coefficient": characteristics.outdoor_temp_coefficient,
                    "last_updated": characteristics.last_updated.isoformat() if characteristics.last_updated else None,
                    "overall_confidence": chars["confidence"],
                    "recent_measurements": chars["recent_measurements"]
                }

        return result
