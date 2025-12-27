"""
Theria API Endpoints
"""

import os
import sys
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
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

        # Read climate entity states with heating_power_request
        climate_states = []
        total_heating_request = 0
        heating_request_count = 0

        for climate_entity in zone.climate_entities:
            try:
                state = ha_client.get_climate_state(climate_entity)

                # Get full state to extract heating_power_request attribute
                full_state = ha_client.get_state(climate_entity)
                heating_power_request = full_state.get("attributes", {}).get("heating_power_request")

                # Add heating_power_request to climate state
                if heating_power_request is not None:
                    state["heating_power_request"] = heating_power_request
                    total_heating_request += float(heating_power_request)
                    heating_request_count += 1

                climate_states.append(state)
            except Exception as e:
                logger.warning(f"Failed to read {climate_entity}: {e}")
                climate_states.append({
                    "entity_id": climate_entity,
                    "error": str(e)
                })

        # Calculate average heating power request (if available)
        avg_heating_request = None
        max_heating_request = 0.0
        if heating_request_count > 0:
            avg_heating_request = total_heating_request / heating_request_count
            # Find max heating request
            for state in climate_states:
                if "heating_power_request" in state:
                    max_heating_request = max(max_heating_request, state["heating_power_request"])

        # Determine heating status based on heating_power_request
        is_heating = avg_heating_request > 0 if avg_heating_request is not None else None

        # Note: Temperature and heating power history is collected by the background
        # TemperatureHistoryService, not by this API endpoint. This ensures data
        # collection happens continuously regardless of UI access.

        # In observation mode - Theria is learning, not controlling
        # Thermostats are controlled by Netatmo app/hub

        return {
            "zone_id": zone.id,
            "name": zone.name,
            "timestamp": datetime.utcnow().isoformat(),
            "average_temperature": avg_temp,
            "average_heating_request": avg_heating_request,  # NEW: Average heating power request (0-100)
            "is_heating": is_heating,  # NEW: Whether zone is currently heating
            "mode": "observation",  # Learning thermal characteristics
            "temperatures": temperatures,
            "climate_states": climate_states,
        }

    except Exception as e:
        logger.error(f"Error getting zone status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
async def get_zone_history(
    zone_id: str,
    hours: int = None,
    from_ms: int = Query(None, alias="from"),  # Unix timestamp in ms (Grafana-style)
    to: int = None,                             # Unix timestamp in ms
    start_date: str = None,  # Legacy support
    end_date: str = None     # Legacy support
):
    """Get temperature history for a zone.

    Args:
        zone_id: Zone identifier
        hours: How many hours of history to retrieve (default: 24)
        from_ms: Start time as Unix timestamp in milliseconds (preferred)
        to: End time as Unix timestamp in milliseconds (preferred)
        start_date: Start date in YYYY-MM-DD format (legacy)
        end_date: End date in YYYY-MM-DD format (legacy)

    Returns:
        List of temperature readings with timestamps
    """
    from datetime import datetime

    # Verify zone exists
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    # INDUSTRY STANDARD: Accept Unix timestamps (like Grafana/Kibana)
    if from_ms is not None and to is not None:
        # Unix timestamps in milliseconds - convert to UTC timezone-aware datetime
        start_dt = datetime.fromtimestamp(from_ms / 1000.0, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(to / 1000.0, tz=timezone.utc)
        # Calculate hours from start_dt to NOW (not just the range duration)
        # This ensures we fetch enough historical data
        now_utc = datetime.now(timezone.utc)
        hours = int((now_utc - start_dt).total_seconds() / 3600) + 2
    elif start_date and end_date:
        # Legacy format: Date strings (YYYY-MM-DD) - assume UTC
        start_dt = datetime.fromisoformat(start_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        hours = int((end_dt - start_dt).total_seconds() / 3600) + 2
    elif hours is None:
        hours = 24
        start_dt = None
        end_dt = None
    else:
        start_dt = None
        end_dt = None

    history = history_tracker.get_temperature_history(zone_id=zone_id, hours=hours)

    # Filter to date range if specified (compare timezone-aware datetimes)
    if start_dt and end_dt:
        filtered_history = []
        for h in history:
            ts_str = h["timestamp"]
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            try:
                ts = datetime.fromisoformat(ts_str)  # Keep timezone-aware
                if start_dt <= ts < end_dt:
                    filtered_history.append(h)
            except ValueError:
                continue
        history = filtered_history

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
        "expensive_hours": sorted(price_optimizer.expensive_hour_set),
        "cheap_hours": sorted(price_optimizer.cheap_hour_set)
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
async def get_thermal_characteristics(zone_id: str | None = None):
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


@router.get("/api/thermal/characteristics/history")
async def get_thermal_characteristics_history(zone_id: str | None = None, hours: int = 24):
    """Get historical thermal characteristics snapshots.

    Args:
        zone_id: Optional zone filter (None = all zones)
        hours: How many hours of history to retrieve (default: 24, max: 168 for 7d)

    Returns:
        Thermal characteristics history with snapshots over time
    """
    # Limit hours to 7 days max
    hours = min(hours, 168)

    snapshots = history_tracker.get_thermal_history(zone_id=zone_id, hours=hours)

    return {
        "zone_id": zone_id,
        "hours": hours,
        "count": len(snapshots),
        "snapshots": snapshots
    }


@router.get("/api/thermal/measurements")
async def get_thermal_measurements(zone_id: str, hours: int = 24, limit: int = 100):
    """Get recent thermal measurements with predictions.

    Args:
        zone_id: Zone identifier (required)
        hours: How many hours back (default: 24, max: 168 for 7d)
        limit: Maximum number of measurements (default: 100, max: 1000)

    Returns:
        Recent measurements with predicted values and errors
    """
    if not learning_service:
        raise HTTPException(status_code=503, detail="Thermal learning service not available")

    # Verify zone exists
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    # Limit parameters
    hours = min(hours, 168)
    limit = min(limit, 1000)

    # Get measurements from learner
    if zone_id not in learning_service.learners:
        return {
            "zone_id": zone_id,
            "measurements": [],
            "message": "No measurements available yet"
        }

    learner = learning_service.learners[zone_id]
    measurements = learner.get_measurements_with_predictions(limit=limit)

    # Filter by time range
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    measurements = [
        m for m in measurements
        if datetime.fromisoformat(m["timestamp"]) > cutoff
    ]

    return {
        "zone_id": zone_id,
        "hours": hours,
        "count": len(measurements),
        "measurements": measurements
    }


@router.get("/api/zones/{zone_id}/heating_timeline")
async def get_heating_timeline(
    zone_id: str,
    hours: int = None,
    resolution: str = "raw",
    from_ms: int = Query(None, alias="from"),  # Unix timestamp in ms (Grafana-style)
    to: int = None,                             # Unix timestamp in ms
    start_date: str = None,  # Legacy support
    end_date: str = None     # Legacy support
):
    """Get heating power request timeline.

    Args:
        zone_id: Zone identifier
        hours: How many hours back (default: 24, max: 168)
        resolution: "raw", "5m", "15m", "1h" (default: "raw")
        from_ms: Start time as Unix timestamp in milliseconds (preferred)
        to: End time as Unix timestamp in milliseconds (preferred)
        start_date: Start date in YYYY-MM-DD format (legacy)
        end_date: End date in YYYY-MM-DD format (legacy)

    Returns:
        Heating power timeline with avg/max request and status
    """
    from datetime import datetime

    # Verify zone exists
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    # INDUSTRY STANDARD: Accept Unix timestamps (like Grafana/Kibana)
    if from_ms is not None and to is not None:
        # Unix timestamps in milliseconds - convert to UTC timezone-aware datetime
        start_dt = datetime.fromtimestamp(from_ms / 1000.0, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(to / 1000.0, tz=timezone.utc)
        # Calculate hours from start_dt to NOW (not just the range duration)
        # This ensures we fetch enough historical data
        now_utc = datetime.now(timezone.utc)
        hours = int((now_utc - start_dt).total_seconds() / 3600) + 2
    elif start_date and end_date:
        # Legacy format: Date strings (YYYY-MM-DD) - assume UTC
        start_dt = datetime.fromisoformat(start_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        hours = int((end_dt - start_dt).total_seconds() / 3600) + 2
    elif hours is None:
        hours = 24
        start_dt = None
        end_dt = None
    else:
        start_dt = None
        end_dt = None

    # Limit hours
    hours = min(hours, 744)  # Increased limit for month view

    timeline = history_tracker.get_heating_timeline(
        zone_id=zone_id,
        hours=hours,
        resolution=resolution
    )

    # Filter to date range if specified (compare timezone-aware datetimes)
    if start_dt and end_dt:
        filtered_timeline = []
        for item in timeline:
            ts_str = item["timestamp"]
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1] + '+00:00'
            try:
                ts = datetime.fromisoformat(ts_str)  # Keep timezone-aware
                if start_dt <= ts < end_dt:
                    filtered_timeline.append(item)
            except ValueError:
                continue
        timeline = filtered_timeline

    return {
        "zone_id": zone_id,
        "hours": hours,
        "resolution": resolution,
        "count": len(timeline),
        "timeline": timeline
    }


@router.get("/api/thermal/entity/{entity_id}")
async def get_entity_thermal(entity_id: str, timeframe: str = "24h"):
    """Get thermal characteristics for a specific climate entity.

    Args:
        entity_id: Climate entity ID (e.g., "climate.vantsidan")
        timeframe: Timeframe for rates: "1h", "6h", "24h", "7d" (default: "24h")

    Returns:
        Per-entity thermal characteristics across all timeframes
    """
    if not learning_service:
        raise HTTPException(status_code=503, detail="Thermal learning service not available")

    # Find entity in learning service
    if entity_id not in learning_service.entity_learners:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found in thermal learning")

    learner = learning_service.entity_learners[entity_id]

    # Get all timeframes
    all_timeframes = learner.get_all_timeframes()

    # Build response with all timeframes
    result = {
        "entity_id": entity_id,
        "timeframes": {}
    }

    for tf, chars in all_timeframes.items():
        result["timeframes"][tf] = {
            "heating_rate": chars.heating_rate,
            "cooling_rate": chars.cooling_rate,
            "heating_samples": chars.heating_samples,
            "cooling_samples": chars.cooling_samples,
            "heating_confidence": chars.heating_confidence,
            "cooling_confidence": chars.cooling_confidence,
            "overall_confidence": chars.overall_confidence
        }

    return result


@router.get("/api/thermal/periods/{zone_id}")
async def get_thermal_periods(zone_id: str, hours: int = 24):
    """Get detected heating and cooling periods for a zone.

    Args:
        zone_id: Zone ID (e.g., "butik")
        hours: Hours of history to retrieve (default: 24)

    Returns:
        List of detected periods with timestamps, temperatures, and rates
    """
    if not learning_service:
        raise HTTPException(status_code=503, detail="Thermal learning service not available")

    # Get all entity learners for this zone
    zone_periods = []

    for entity_id, learner in learning_service.entity_learners.items():
        # Check if this entity belongs to the zone (simple check based on entity_id)
        # TODO: Use proper zone mapping from settings
        periods = learner.get_periods(hours=hours)
        zone_periods.extend(periods)

    # Sort by start time
    zone_periods.sort(key=lambda p: p["start_time"])

    return {
        "zone_id": zone_id,
        "hours": hours,
        "periods": zone_periods
    }


@router.get("/api/thermal/zone/{zone_id}/comparison")
async def get_zone_entity_comparison(zone_id: str, timeframe: str = "24h"):
    """Compare thermal performance across a zone's climate entities.

    Args:
        zone_id: Zone identifier
        timeframe: Timeframe to compare: "1h", "6h", "24h", "7d" (default: "24h")

    Returns:
        Comparison of all entities in the zone with insights
    """
    if not learning_service:
        raise HTTPException(status_code=503, detail="Thermal learning service not available")

    # Verify zone exists
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    # Get aggregator for this zone
    if zone_id not in learning_service.zone_aggregators:
        raise HTTPException(status_code=404, detail=f"No thermal data for zone {zone_id}")

    aggregator = learning_service.zone_aggregators[zone_id]

    # Validate timeframe
    valid_timeframes = ["1h", "6h", "24h", "7d"]
    if timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
        )

    # Get comparison data
    comparison = aggregator.get_comparison(timeframe)

    return comparison


@router.get("/api/sensor/history")
async def get_sensor_history(entity_id: str, hours: int = 24):
    """Get historical data for any sensor or binary_sensor.

    Args:
        entity_id: Full entity ID (e.g., 'sensor.heating_setpoint', 'binary_sensor.switch_valve_1')
        hours: How many hours back (default: 24)

    Returns:
        Historical state changes with timestamps and values
    """
    from datetime import datetime, timedelta
    import requests

    # Limit hours
    hours = min(hours, 168)

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    # Fetch from HA history API
    try:
        url = f"{ha_client.base_url}/api/history/period/{start_time.isoformat()}"
        params = {
            "filter_entity_id": entity_id,
            "end_time": end_time.isoformat()
        }

        response = requests.get(
            url,
            headers=ha_client.headers,
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                history = data[0]  # First array contains our entity's history

                # Format the data
                formatted_history = []
                for state_change in history:
                    timestamp = state_change.get("last_changed") or state_change.get("last_updated")
                    state = state_change.get("state")

                    if timestamp and state not in ("unknown", "unavailable", None):
                        formatted_history.append({
                            "timestamp": timestamp,
                            "state": state
                        })

                return {
                    "entity_id": entity_id,
                    "hours": hours,
                    "count": len(formatted_history),
                    "history": formatted_history
                }

        return {
            "entity_id": entity_id,
            "hours": hours,
            "count": 0,
            "history": []
        }

    except Exception as e:
        logger.error(f"Failed to fetch sensor history for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch sensor history: {str(e)}")


# =============================================================================
# UNIFIED CHART DATA ENDPOINT (Phase 1)
# =============================================================================

def _build_chartjs_datasets(history: list, zone: ZoneSettings) -> list:
    """
    Convert temperature history to Chart.js datasets.

    Returns format: [{label, data: [{x, y}], borderColor, ...}, ...]
    Eliminates frontend transformation - backend returns ready-to-render format.
    """
    datasets = []

    # Create dataset for each climate entity
    for entity_id in zone.climate_entities:
        # Collect data points for each series
        current_temps = []
        target_temps = []
        heating_data = []

        for reading in history:
            ts = reading["timestamp"]

            # Current temperature
            if reading.get("current_temps", {}).get(entity_id) is not None:
                current_temps.append({"x": ts, "y": reading["current_temps"][entity_id]})

            # Target temperature
            if reading.get("target_temps", {}).get(entity_id) is not None:
                target_temps.append({"x": ts, "y": reading["target_temps"][entity_id]})

            # Heating request percentage
            if reading.get("heating_requests", {}).get(entity_id) is not None:
                heating_data.append({"x": ts, "y": reading["heating_requests"][entity_id]})

        entity_name = entity_id.replace('climate.', '')

        # Current temperature dataset
        datasets.append({
            "label": f"{entity_name} - Current",
            "data": current_temps,
            "borderColor": "#4fd1c5",
            "borderWidth": 2,
            "tension": 0.1,
            "pointRadius": 0,
            "yAxisID": "y"
        })

        # Target temperature dataset
        datasets.append({
            "label": f"{entity_name} - Target",
            "data": target_temps,
            "borderColor": "#4fd1c5",
            "borderWidth": 1.5,
            "borderDash": [5, 5],
            "stepped": "before",  # Step function for setpoint changes
            "pointRadius": 0,
            "yAxisID": "y"
        })

        # Heating percentage dataset
        datasets.append({
            "label": f"{entity_name} - Heating %",
            "data": heating_data,
            "backgroundColor": "rgba(239, 68, 68, 0.2)",
            "borderColor": "rgba(239, 68, 68, 0.5)",
            "borderWidth": 1,
            "fill": True,
            "stepped": "before",  # Step function - heating is binary/percentage, not smooth
            "tension": 0,  # No curve interpolation
            "pointRadius": 0,
            "yAxisID": "y1"
        })

    return datasets


def _build_chartjs_annotations(periods: list) -> dict:
    """
    Convert thermal periods to Chart.js annotations.

    Returns format: {period_0: {type: 'box', ...}, period_label_0: {...}, ...}
    Pre-builds Chart.js annotation objects on backend - zero frontend work.

    Note: Includes entity_id in annotation metadata for frontend filtering.
    """
    annotations = {}

    for idx, period in enumerate(periods):
        is_heating = period["type"] == "heating"
        # Much more subtle opacity for background shading
        bg_color = "rgba(239, 68, 68, 0.08)" if is_heating else "rgba(59, 130, 246, 0.08)"
        border_color = "rgba(239, 68, 68, 0.6)" if is_heating else "rgba(59, 130, 246, 0.6)"

        # Extract entity name for filtering (e.g., "climate.vantsidan" -> "vantsidan")
        entity_name = period["entity_id"].replace("climate.", "")

        # Background shading for period - subtle overlay in temperature range
        annotations[f"period_{entity_name}_{idx}"] = {
            "type": "box",
            "xMin": period["start_time"],
            "xMax": period["end_time"],
            "yMin": 15,  # Temperature range instead of 0
            "yMax": 25,  # Temperature range instead of 30
            "backgroundColor": bg_color,
            "borderWidth": 0,
            "xScaleID": "x",
            "yScaleID": "y",
            "entity_id": period["entity_id"]  # Add for frontend filtering
        }

        # Rate label
        rate_text = f"{period['rate']:+.2f}°C/h"

        # Calculate midpoint for label placement
        from datetime import datetime
        start = datetime.fromisoformat(period["start_time"].replace('Z', '+00:00'))
        end = datetime.fromisoformat(period["end_time"].replace('Z', '+00:00'))
        mid_timestamp = start + (end - start) / 2

        annotations[f"period_label_{entity_name}_{idx}"] = {
            "type": "label",
            "xValue": mid_timestamp.isoformat(),
            "yValue": 22,
            "backgroundColor": border_color,
            "content": [rate_text],
            "font": {"size": 10},
            "color": "#fff",
            "padding": 4,
            "borderRadius": 4,
            "position": "start",
            "xScaleID": "x",
            "yScaleID": "y",
            "entity_id": period["entity_id"]  # Add for frontend filtering
        }

    return annotations


@router.get("/api/zones/{zone_id}/chart_data")
async def get_zone_chart_data(
    zone_id: str,
    hours: int = None,
    from_ms: int = Query(None, alias="from"),  # Unix timestamp in ms (Grafana-style)
    to: int = None,                             # Unix timestamp in ms
    start_date: str = None,  # Legacy support
    end_date: str = None     # Legacy support
):
    """
    Unified endpoint that returns EVERYTHING a Chart.js chart needs.

    Returns complete chart configuration:
    - datasets: Chart.js-ready data in [{x, y}] format
    - annotations: Chart.js annotation objects (heating/cooling periods)
    - scales: Chart.js scale configuration

    Frontend can render directly without any transformation!
    Eliminates N+1 queries - ONE call gets all data.

    Args:
        zone_id: Zone identifier (e.g., "butik")
        hours: Hours of history to retrieve (default: 24)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)

    Returns:
        Complete Chart.js chart configuration
    """
    from datetime import datetime, timedelta

    # Find zone
    zone = next((z for z in ZONES if z.id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    try:
        # INDUSTRY STANDARD: Accept Unix timestamps (like Grafana/Kibana)
        if from_ms is not None and to is not None:
            # Unix timestamps in milliseconds - convert to UTC timezone-aware datetime
            start_dt = datetime.fromtimestamp(from_ms / 1000.0, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(to / 1000.0, tz=timezone.utc)
            # Calculate hours from start_dt to NOW (not just the range duration)
            # This ensures we fetch enough historical data
            now_utc = datetime.now(timezone.utc)
            hours = int((now_utc - start_dt).total_seconds() / 3600) + 2
        elif start_date and end_date:
            # Legacy format: Date strings (YYYY-MM-DD) - assume UTC
            start_dt = datetime.fromisoformat(start_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            hours = int((now_utc - start_dt).total_seconds() / 3600) + 2
        elif hours is None:
            hours = 24  # Default
            start_dt = None
            end_dt = None
        else:
            start_dt = None
            end_dt = None

        # Step 1: Get temperature history (reuse existing code)
        history = history_tracker.get_temperature_history(zone_id=zone_id, hours=hours)

        # Filter history to exact date range if specified (compare timezone-aware datetimes)
        if start_dt and end_dt:
            filtered_history = []
            for h in history:
                # Parse timestamp (handle both 'Z' and '+00:00' timezone formats)
                ts_str = h["timestamp"]
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                try:
                    ts = datetime.fromisoformat(ts_str)  # Keep timezone-aware
                    # Use >= and < for half-open interval [start_dt, end_dt)
                    if start_dt <= ts < end_dt:
                        filtered_history.append(h)
                except ValueError:
                    continue  # Skip malformed timestamps
            history = filtered_history

        # Step 2: Get thermal periods from entity learners
        periods = []
        if learning_service:
            for entity_id in zone.climate_entities:
                if entity_id in learning_service.entity_learners:
                    learner = learning_service.entity_learners[entity_id]
                    entity_periods = learner.get_periods(hours=hours)
                    periods.extend(entity_periods)

        # Step 3: Transform to Chart.js format
        datasets = _build_chartjs_datasets(history, zone)
        annotations = _build_chartjs_annotations(periods)

        # Build x-axis scale configuration
        x_scale = {
            "type": "time",
            "time": {
                "unit": "hour" if hours <= 48 else "day",
                "displayFormats": {
                    "hour": "HH:mm",
                    "day": "MMM dd"
                },
                "tooltipFormat": "PPpp"
            },
            "ticks": {
                "color": "#a0aec0",
                "maxRotation": 0,
                "minRotation": 0,
                "maxTicksLimit": 12
            },
            "grid": {"display": False}
        }

        # Add explicit time bounds when a specific date range is selected
        # This ensures the x-axis shows the full requested range even if data is sparse
        if start_dt and end_dt:
            x_scale["min"] = start_dt.isoformat()
            x_scale["max"] = end_dt.isoformat()

        return {
            "zone_id": zone_id,
            "time_range_hours": hours,
            "datasets": datasets,
            "annotations": annotations,
            "scales": {
                "x": x_scale,
                "y": {
                    "type": "linear",
                    "position": "left",
                    "title": {
                        "display": True,
                        "text": "Temperature (°C)",
                        "color": "#a0aec0"
                    },
                    "suggestedMin": 15,
                    "suggestedMax": 25,
                    "grid": {"color": "rgba(255, 255, 255, 0.1)"},
                    "ticks": {"color": "#a0aec0"}
                },
                "y1": {
                    "type": "linear",
                    "position": "right",
                    "title": {
                        "display": True,
                        "text": "Heating Request (%)",
                        "color": "#a0aec0"
                    },
                    "min": 0,
                    "max": 100,
                    "grid": {"display": False},
                    "ticks": {
                        "color": "#a0aec0"
                    }
                }
            },
            "metadata": {
                "source": "history_tracker",
                "data_points": len(history),
                "period_count": len(periods)
            }
        }

    except Exception as e:
        logger.error(f"Failed to build chart data for {zone_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build chart data: {str(e)}")
