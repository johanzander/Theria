# Theria - Smart Heating Optimization

## Project Overview

Theria is a Home Assistant add-on that optimizes heating costs using electricity price-based control. It monitors Nordpool day-ahead prices and learns thermal characteristics of heating zones to minimize energy costs while maintaining comfort.

**Current Status:** MVP development - basic thermal learning and simple price optimization implemented.

## Core Strategy

**Heat Capacitor Approach** (inspired by PowerSaver):

- Pre-heat zones during cheap electricity hours (store thermal energy)
- Coast during expensive hours (use stored energy)
- Stay within user-defined comfort boundaries (e.g., 21°C ± 1°C)

**Key Principle:** Use your building's thermal mass as a battery to shift energy consumption to cheaper hours.

## Architecture

### What's Implemented ✅

1. **ThermalLearningService** - Continuously learns thermal characteristics:

   - Heating rate (°C/hour when heating active)
   - Cooling rate (°C/hour when heating off)
   - Runs every 15 minutes, tracks measurements

2. **Price plan** (`core/theria/plan/`) - the price feed is HA's Nord Pool integration (`nordpool.get_prices_for_date`),
   the one price source for the plan, the charts and the measured days; the plan moves heating by changing the pump setpoint

3. **Zone Configuration** - Multi-zone support:

   - Each zone has climate_entities, temp_sensors
   - comfort_target and allowed_deviation defined (not used yet)

4. **History Tracking** - Temperature and control event logging

5. **FastAPI Backend** - REST API with endpoints:

   - `/api/zones/{id}/status` - Zone temperature readings
   - `/api/price/current` - Current price and category
   - `/api/price/forecast` - 24h price forecast
   - `/api/thermal/characteristics` - Learned thermal rates

### What's Planned (Architecture Designed, Not Implemented) 📋

1. **Full Heat Capacitor Strategy**:

   - 96-period scheduling (24h at 15-min resolution)
   - Forward-looking optimization (not just reactive)
   - Buy-sell pair matching (cheap before expensive)
   - Use comfort_target ± allowed_deviation for temperature ranges

2. **Zone-Specific Schedules**:

   - Each zone creates its own optimal heating schedule
   - Different thermal characteristics (radiators vs floor heating)
   - Independent savings tracking per zone

3. **Thermal Model Integration**:

   - Use learned characteristics for temperature prediction
   - Calculate time needed to heat/cool zones
   - Optimize pre-heat timing based on thermal response

## Key Design Decisions

1. **No MPC Yet** - Current implementation uses simple price-based adjustments, not full Model Predictive Control
2. **Learning First** - System learns thermal characteristics before optimization
3. **Home Assistant Integration** - Deployed as HA add-on, not standalone
4. **Multi-Zone from Day 1** - Architecture supports multiple zones with shared heating source

## Configuration

### Zone Settings (config.yaml)

```yaml
zones:
  - id: "butik"
    name: "Butik"
    climate_entities:
      - "climate.vantsidan"
      - "climate.klippsidan"
    temp_sensors:
      - "climate.vantsidan"
      - "climate.klippsidan"
    comfort_target: 21.0  # Target comfort temperature
    allowed_deviation: 1.0  # Allowed variation (±°C)
    enabled: true
```

## Important Files

### Core Application

- `backend/app.py` - FastAPI application, lifespan management
- `backend/api.py` - REST API endpoints
- `core/theria/settings.py` - Configuration models (ZoneSettings)
- `core/theria/ha_client.py` - Home Assistant API client

### Optimization & Learning

- `core/theria/plan/price_feed.py` - Nord Pool prices from Home Assistant (plan, charts, measured days)
- `core/theria/thermal_learning_service.py` - Background learning service
- `core/theria/zone_thermal_learner.py` - Per-zone thermal learning
- `core/theria/history.py` - Event history tracking

### Configuration & Documentation

- `config.yaml` - HA add-on configuration
- `README.md` - Project overview, heat capacitor concept
- `architecture.md` - Detailed architecture (96-period scheduling design)
- `SYSTEM_SETUP.md` - Reference implementation (Johan's house)

### Infrastructure

- `Dockerfile` - HA add-on container
- `build.json` - Multi-architecture build config
- `package-addon.sh` - Local packaging script

## Development Workflow

### Start Dev Server

```bash
./dev-start.sh  # Starts FastAPI on port 8081 with hot reload
```

### Test

```bash
./test-server.sh  # Import tests and endpoint checks
```

### Environment

- Python 3.11+ with `.venv`
- Requires `.env` file with HA_URL and HA_TOKEN
- Uses watchdog for auto-reload

## Reference System

**Johan's House (Sweden):**

- IVT AirX heat pump (Husdata H60 integration)
- 3 zones: Boutique (radiators), First Floor (floor heating), Guest House (radiators)
- Nordpool SE4 pricing
- Outdoor sensor: `sensor.outdoor`
- System identification shows: Boutique has 2.33 MJ/K thermal mass, 51.7 W/K heat loss

## Current Simplification vs Design

**Current:** Simple reactive price-based adjustment (±0.5°C based on current hour)

**Design:** Forward-looking heat capacitor with:

- 24-hour schedule creation when prices published
- Multiple pre-heat/coast cycles per day
- Thermal model predictions
- Savings calculation vs baseline

**Next Step:** Implement full heat capacitor scheduler using learned thermal characteristics.

## Code Style & Quality Standards

**See [CODING_STANDARDS.md](../CODING_STANDARDS.md) for complete coding guidelines.**

Key highlights:

- **Pre-commit**: Zero tolerance for IDE errors/warnings
- **Error handling**: Never use string matching on exception messages
- **Testing**: Focus on behavior verification, not implementation details
- **Git commits**: Never commit without approval, no AI attribution in messages
- **Code quality**: DRY enforcement, no hasattr/fallbacks, deterministic design

## Important Notes

1. **comfort_target and allowed_deviation** - Configured but not yet used in optimization logic
2. **Thermal characteristics** - Being learned continuously but not yet used for predictions
3. **Port 8081** - Consistent across all configs (Dockerfile, config.yaml, docker-compose)
4. **No frontend** - Backend-only API, no UI yet
5. **Observation mode** - System is learning, not actively controlling thermostats yet
