# Theria Copilot Instructions

## Project Overview

Theria is a **smart heating optimization system** that uses price-based heat capacitor strategy to minimize electricity costs while maintaining comfort. It runs as a **Home Assistant add-on**, monitors dynamic electricity prices (Nordpool), and automatically controls multi-zone heating systems by pre-heating during cheap off-peak hours.

**Current Status**: Functional MVP - FastAPI backend with price optimization, thermal learning, history tracking, and basic web UI operational.

## Architecture & Key Components

### Service Architecture (Current Implementation)

The system currently implements:

- **Price Optimizer**: Monitors Nordpool electricity prices and adjusts heating based on expensive/cheap hours
- **Thermal Learning Service**: Continuously learns zone thermal characteristics (heating/cooling rates, outdoor temp coefficient)
- **Zone Controllers**: Multi-zone support with independent temperature control per zone
- **History Tracker**: Tracks temperature history and control events for analytics
- **REST API**: FastAPI endpoints for zone status, temperature control, price info, and thermal characteristics
- **Web UI**: Basic monitoring dashboard (static/index.html)

### Current Implementation

```
Theria/
├── backend/
│   ├── app.py                    # FastAPI application with lifespan management
│   ├── api.py                    # REST API endpoints (10+ endpoints)
│   ├── log_config.py             # Loguru logging configuration
│   └── static/index.html         # Web UI dashboard
├── core/theria/
│   ├── ha_client.py              # Home Assistant REST API client wrapper
│   ├── ha_api_controller.py      # Legacy controller (battery manager pattern)
│   ├── price_optimizer.py        # Nordpool price-based optimization
│   ├── thermal_learning_service.py # Continuous thermal characteristic learning
│   ├── zone_thermal_learner.py   # Per-zone thermal model learning
│   ├── thermal_model.py          # Thermal physics models
│   ├── history.py                # Temperature & event tracking
│   ├── settings.py               # Configuration dataclasses
│   └── models.py                 # Data models (ZoneStatus, etc.)
├── config.yaml                   # Home Assistant add-on configuration
└── docker-compose.yml            # Development environment
```

## Critical Development Context

### Home Assistant Integration

- **Primary deployment**: Home Assistant Supervisor add-on (not standalone Python app)
- **API communication**: Uses `HomeAssistantAPIController` for REST API calls to HA Core
- **Ingress support**: FastAPI app must respect `INGRESS_PREFIX` environment variable
- **Sensor access**: All temperature sensors, climate entities, and price data come through HA API
- **Authentication**: Uses long-lived access token from HA (`HA_TOKEN` env var)

### Development Workflow

**Start development server**:

```bash
./dev-start.sh  # Starts FastAPI on port 8081 with hot reload
```

**Test server**:

```bash
./test-server.sh  # Runs import tests and endpoint checks
```

**Environment setup**:

- Uses Python 3.10+ with venv (standard virtual environment in `venv/`)
- Set `PYTHONPATH` to include both project root and `backend/` directory
- Required env vars: `HA_URL`, `HA_TOKEN` (defaults in dev-start.sh)

### Code Style & Standards

**See [CODING_STANDARDS.md](../CODING_STANDARDS.md) for complete coding guidelines.**

**Quick Reference:**

- **Formatting**: Black (line length: 88)
- **Linting**: Ruff with pycodestyle, flake8-bugbear, pyupgrade rules
- **Type checking**: mypy strict mode (all functions must have type annotations)
- **Import order**: isort with known-first-party: ["core", "app"]

See [pyproject.toml](../pyproject.toml) for complete tool configuration.

### Configuration Pattern

**User-facing config**: `config.yaml` (Home Assistant add-on options)
**Python settings**: [core/theria/settings.py](../core/theria/settings.py) with `@dataclass` patterns

- `SystemSettings.from_ha_config()` loads from config.yaml
- Use `_camel_to_snake()` converter for HA camelCase → Python snake_case

### Data Models

**Current implementation**:

- `ZoneStatus` - Current zone state (temp, target, HVAC mode)
- `ThermalCharacteristics` - Learned thermal parameters (heating/cooling rates, confidence)
- `TemperatureReading` - Historical temperature data point
- `ControlEvent` - System control action tracking

**Pattern**: Use `@dataclass` with type hints, avoid Pydantic unless FastAPI requires it

## Key Files Reference

| File                                                                                  | Purpose                                                          |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [backend/app.py](../backend/app.py)                                                   | FastAPI application with lifespan, global exception handling     |
| [backend/api.py](../backend/api.py)                                                   | REST API endpoints (10+ endpoints for zones, price, thermal)     |
| [backend/static/index.html](../backend/static/index.html)                             | Web UI dashboard for monitoring                                  |
| [core/theria/ha_client.py](../core/theria/ha_client.py)                               | Home Assistant API client (get/set temperature, climate state)   |
| [core/theria/price_optimizer.py](../core/theria/price_optimizer.py)                   | Nordpool price tracking and temperature adjustment logic         |
| [core/theria/thermal_learning_service.py](../core/theria/thermal_learning_service.py) | Background service for continuous thermal learning               |
| [core/theria/zone_thermal_learner.py](../core/theria/zone_thermal_learner.py)         | Per-zone thermal characteristic learning                         |
| [core/theria/history.py](../core/theria/history.py)                                   | Temperature and control event tracking                           |
| [core/theria/settings.py](../core/theria/settings.py)                                 | Configuration dataclasses (ZoneSettings)                         |
| [config.yaml](../config.yaml)                                                         | HA add-on configuration schema with zones and price optimization |
| [architecture.md](../architecture.md)                                                 | High-level architecture diagrams and design decisions            |

## Domain-Specific Knowledge

### Control Strategy

- **Heat Capacitor Strategy**: Simple price-based control (inspired by PowerSaver BESS approach)
  - During **cheap hours**: Pre-heat by raising target temp (e.g., 21°C → 21.5°C)
  - During **expensive hours**: Coast by lowering target temp (e.g., 21°C → 20.5°C)
  - **No complex solvers required** - simple rule-based logic
- **Thermal Learning**: Continuous learning of zone characteristics
  - Heating rate (°C/hour when heating active)
  - Cooling rate base (°C/hour passive cooling)
  - Outdoor temperature coefficient (impact of outdoor temp on cooling)
  - Confidence tracking for each parameter
- **Multi-Zone Support**: Each zone operates independently with unique thermal characteristics

### Energy Optimization

- **Price-Based Scheduling**: System identifies most expensive/cheapest hours from Nordpool day-ahead prices
- **Temperature Adjustments**: Configurable adjustment (default ±0.5°C) during price periods
- **Zone Independence**: Each zone can have different comfort targets and deviation limits
- **Observation Mode**: Currently learning thermal behavior before enabling active control

## Development Phases

**Phase 0** (✅ Complete): Foundation & FastAPI setup
**Phase 1** (✅ Complete): Core infrastructure (HA integration, zone management)
**Phase 2** (✅ Complete): Price optimization (Nordpool integration, heat capacitor logic)
**Phase 3** (✅ Complete): Thermal learning (continuous characteristic learning)
**Phase 4** (🚧 Current): Observation & refinement (gathering data, improving accuracy)
**Phase 5** (Planned): Active control (automated temperature adjustments)
**Phase 6** (Planned): Analytics dashboard & savings tracking
**Phase 7** (Planned): Production polish & HA add-on packaging

## Common Pitfalls

1. **Don't use subprocess or bash for HA API calls** - Use `HomeAssistantAPIController` methods
2. **Always include type hints** - mypy strict mode enforces this
3. **Respect INGRESS_PREFIX** - FastAPI `root_path` must be set for HA ingress
4. **Don't create tests/ directory yet** - MVP focuses on working implementation first
5. **Use loguru, not logging** - Already configured in log_config.py (imported in app.py)

## When Adding Features

1. **Check architecture.md** - Does it align with heat capacitor strategy?
2. **Add model to models.py** - Use `@dataclass` pattern with type hints
3. **Add endpoint to api.py** - Follow existing `@router.get()` pattern (currently 10+ endpoints)
4. **Update config.yaml schema** - If user-configurable, add to options/schema
5. **Document in docstrings** - All public functions need type hints + docstrings
6. **Consider thermal learning** - Will the feature need thermal characteristic data?
7. **Test with real HA** - Use ./dev-start.sh with HA_URL and HA_TOKEN env vars

## Available API Endpoints

The system provides the following REST API endpoints:

**System Status:**

- `GET /api/health` - Health check
- `GET /api/status` - System status and configuration

**Zone Management:**

- `GET /api/zones` - List all configured zones
- `GET /api/zones/{zone_id}/status` - Current zone status with all temperature readings
- `POST /api/zones/{zone_id}/set_temperature` - Manually set zone temperature
- `GET /api/zones/{zone_id}/history` - Temperature history (default 24h)
- `GET /api/zones/{zone_id}/events` - Control event history

**Price Optimization:**

- `GET /api/price/current` - Current electricity price and optimization status
- `GET /api/price/forecast` - 24-hour price forecast with categories

**Thermal Learning:**

- `GET /api/thermal/characteristics` - Learned thermal characteristics for zones

## Testing Strategy

**Current**: Manual testing with real Home Assistant instance

- Use `./dev-start.sh` to start development server
- Set `HA_URL` and `HA_TOKEN` environment variables
- Monitor logs with loguru output
- Web UI available at http://localhost:8081/static/index.html

**Future**: pytest with >80% coverage

- Test fixtures in `tests/conftest.py`
- Mock HA API calls with `pytest-asyncio`
- Separate unit/integration/E2E test directories
