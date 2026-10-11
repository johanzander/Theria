"""
Mock Home Assistant server for local Theria development.

Serves synthetic states for every entity referenced in config.yaml (zones,
heat pump, price sensor, outdoor temp) so Theria can run end-to-end without
a real Home Assistant instance. Mirrors the shape of bess-manager's mock_ha
server, scoped down to what Theria actually needs (no scenario replay, no
websocket, no faketime).

One sensor is deliberately seeded as stale (last_changed 3h in the past,
frozen) to demonstrate the sensor health monitor's banner without waiting.

Usage:
    python scripts/mock_ha/server.py
    (then run Theria's backend with HA_URL=http://localhost:8123, HA_TOKEN=mock_token)
"""

import asyncio
import os
import random
from datetime import UTC, datetime, timedelta

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")

app = FastAPI(title="Mock Home Assistant")

_background_tasks: set[asyncio.Task] = set()

# entity_id -> {"state": str, "attributes": dict, "last_changed": datetime}
_states: dict[str, dict] = {}

# Sensor deliberately frozen to demonstrate the stale-sensor health check.
STALE_ENTITY_ID = "sensor.butik_temperature_2"


def _now() -> datetime:
    return datetime.now(UTC)


def _set_state(
    entity_id: str, state, attributes: dict | None = None, freeze: bool = False
):
    existing = _states.get(entity_id)
    if freeze and existing:
        # Keep the original last_changed - simulates a stuck sensor.
        last_changed = existing["last_changed"]
    else:
        last_changed = _now()

    _states[entity_id] = {
        "state": str(state),
        "attributes": attributes or {},
        "last_changed": last_changed,
    }


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _generate_price_forecast() -> list[dict]:
    """Generate a synthetic 24h Nordpool-style price forecast (SEK/kWh)."""
    today = datetime.now().date()
    forecast = []
    for day_offset in (0, 1):
        day = today + timedelta(days=day_offset)
        for hour in range(24):
            # Cheap at night, expensive morning/evening peaks.
            base = 0.5
            if 6 <= hour <= 9 or 17 <= hour <= 20:
                base = 2.2
            elif 0 <= hour <= 5:
                base = 0.2
            price = round(base + random.uniform(-0.15, 0.15), 3)
            forecast.append(
                {
                    "start": datetime(
                        day.year, day.month, day.day, hour, tzinfo=UTC
                    ).isoformat(),
                    "value": price,
                }
            )
    return forecast


def _seed_states(config: dict):
    options = config.get("options", {})

    # Price + outdoor sensor
    sensors = options.get("sensors", {})
    price_entity = sensors.get("nordpool_price")
    if price_entity:
        forecast = _generate_price_forecast()
        _set_state(
            price_entity,
            forecast[0]["value"],
            {"raw_today": forecast[:24], "raw_tomorrow": forecast[24:]},
        )

    outdoor_entity = sensors.get("outdoor_temp")
    if outdoor_entity:
        _set_state(outdoor_entity, 4.5)

    # Zones
    for zone in options.get("zones", []):
        target = zone.get("comfort_target", 21.0)
        for entity_id in zone.get("climate_entities", []):
            _set_state(
                entity_id,
                "heat",
                {
                    "current_temperature": round(target - 0.3, 1),
                    "temperature": target,
                    "heating_power_request": 35,
                    "min_temp": 10,
                    "max_temp": 28,
                },
            )
        for entity_id in zone.get("temp_sensors", []):
            if entity_id in _states:
                continue  # already seeded as a climate entity above
            freeze = entity_id == STALE_ENTITY_ID
            value = round(target - random.uniform(0, 0.8), 1)
            _set_state(entity_id, value, freeze=freeze)
            if freeze:
                # Back-date so it's already >2h stale on first health check.
                _states[entity_id]["last_changed"] = _now() - timedelta(hours=3)

    # Heat pump
    hp = options.get("heat_pump", {})
    if hp.get("enabled"):
        _set_state(hp["compressor_speed"], 45)
        _set_state(hp["compressor_power"], 2.1)
        _set_state(hp["mode_switch"], "off")  # off = heating, on = hot water
        _set_state(hp["heat_carrier_forward"], 38.5)
        _set_state(hp["heat_carrier_return"], 32.0)
        _set_state(hp["compressor_consumption_total"], 12000.0)
        _set_state(hp["compressor_consumption_heating"], 9500.0)
        _set_state(hp["compressor_consumption_hotwater"], 2500.0)
        _set_state(hp["delivered_energy_total"], 36000.0)
        _set_state(hp["delivered_energy_heating"], 28500.0)
        _set_state(hp["delivered_energy_hotwater"], 7500.0)
        _set_state(hp["aux_consumption_total"], 800.0)
        if hp.get("hotwater_setpoint"):
            _set_state(hp["hotwater_setpoint"], 55.0)
        if hp.get("hotwater_top"):
            _set_state(hp["hotwater_top"], 52.0)
        if hp.get("hotwater_mid"):
            _set_state(hp["hotwater_mid"], 47.0)


async def _drift_loop(config: dict):
    """Periodically nudge sensor values so the dashboard looks alive."""
    hp = config.get("options", {}).get("heat_pump", {})
    zones = config.get("options", {}).get("zones", [])

    while True:
        await asyncio.sleep(10)

        # Zone temps: small random walk toward target.
        for zone in zones:
            target = zone.get("comfort_target", 21.0)
            for entity_id in [
                *zone.get("climate_entities", []),
                *zone.get("temp_sensors", []),
            ]:
                if entity_id == STALE_ENTITY_ID or entity_id not in _states:
                    continue
                current = _states[entity_id]
                if entity_id.startswith("climate."):
                    temp = float(current["attributes"]["current_temperature"])
                    temp += random.uniform(-0.05, 0.05) + (target - temp) * 0.05
                    current["attributes"]["current_temperature"] = round(temp, 2)
                    current["last_changed"] = _now()
                else:
                    temp = float(current["state"])
                    temp += random.uniform(-0.05, 0.05) + (target - temp) * 0.05
                    _set_state(entity_id, round(temp, 2))

        # Heat pump: accumulate energy counters, occasionally switch to hot water.
        if hp.get("enabled") and hp["mode_switch"] in _states:
            heating = _states[hp["mode_switch"]]["state"] == "off"
            if random.random() < 0.05:
                heating = not heating
                _set_state(hp["mode_switch"], "off" if heating else "on")

            power_kw = (
                round(random.uniform(1.8, 2.8), 2)
                if heating
                else round(random.uniform(0.5, 1.0), 2)
            )
            _set_state(hp["compressor_power"], power_kw)
            _set_state(hp["compressor_speed"], round(power_kw / 3.0 * 100, 0))

            consumed_delta = power_kw * (10 / 3600)  # kWh over this 10s tick
            cop = 3.2 if heating else 2.5
            delivered_delta = consumed_delta * cop

            total = (
                float(_states[hp["compressor_consumption_total"]]["state"])
                + consumed_delta
            )
            _set_state(hp["compressor_consumption_total"], round(total, 4))
            delivered_total = (
                float(_states[hp["delivered_energy_total"]]["state"]) + delivered_delta
            )
            _set_state(hp["delivered_energy_total"], round(delivered_total, 4))

            key = (
                "compressor_consumption_heating"
                if heating
                else "compressor_consumption_hotwater"
            )
            _set_state(
                hp[key], round(float(_states[hp[key]]["state"]) + consumed_delta, 4)
            )
            key = "delivered_energy_heating" if heating else "delivered_energy_hotwater"
            _set_state(
                hp[key], round(float(_states[hp[key]]["state"]) + delivered_delta, 4)
            )


@app.on_event("startup")
async def startup():
    config = _load_config()
    _seed_states(config)
    task = asyncio.create_task(_drift_loop(config))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    print(
        f"Mock HA seeded with {len(_states)} entities. Stale demo entity: {STALE_ENTITY_ID}"
    )


def _state_response(entity_id: str) -> dict:
    entry = _states[entity_id]
    ts = entry["last_changed"].isoformat()
    return {
        "entity_id": entity_id,
        "state": entry["state"],
        "attributes": entry["attributes"],
        "last_changed": ts,
        "last_updated": ts,
    }


@app.get("/api/states")
async def get_all_states():
    return [_state_response(e) for e in _states]


@app.get("/api/states/{entity_id}")
async def get_state(entity_id: str):
    if entity_id not in _states:
        return JSONResponse(status_code=404, content={"message": "Entity not found"})
    return _state_response(entity_id)


@app.post("/api/services/{domain}/{service}")
async def call_service(domain: str, service: str):
    print(f"[mock-ha] service call: {domain}.{service}")
    return []


@app.post("/mock/toggle_unavailable/{entity_id}")
async def toggle_unavailable(entity_id: str):
    """Manually flip an entity to 'unavailable' to test the health monitor."""
    if entity_id not in _states:
        return JSONResponse(status_code=404, content={"message": "Entity not found"})
    entry = _states[entity_id]
    if entry["state"] == "unavailable":
        return JSONResponse(
            status_code=400,
            content={
                "message": "Already unavailable - restart the mock server to reset"
            },
        )
    entry["state"] = "unavailable"
    entry["last_changed"] = _now()
    return {"entity_id": entity_id, "state": "unavailable"}


@app.get("/")
async def root():
    return {"message": "Mock Home Assistant for Theria dev", "entities": len(_states)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)
