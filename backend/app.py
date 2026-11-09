"""
Theria Backend Application

Minimal FastAPI application - will be expanded as features are added.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

# Add core to Python path (log_config reads where the data directory is)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import API router
import heat_pump_api
import log_config  # noqa: F401
import plan_api
import radiator_api
import sensor_api
from api import DATA_DIR, ZONES, ha_client
from api import router as api_router
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from core.theria.exceptions import TheriaError
from core.theria.ha_client import HANotifier
from core.theria.health_monitor import HealthMonitorService
from core.theria.heat_pump_monitor import HeatPumpMonitorService
from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import InfluxHistorySource
from core.theria.plan.energy_forecast import (
    EnergyForecaster,
    EnergyForecastStore,
    load_energy_history,
)
from core.theria.plan.event_log import EventLog
from core.theria.plan.exec_trace import ExecTraceStore
from core.theria.plan.executor import PlanExecutor, SafetyMonitor
from core.theria.plan.forecast_check import (
    ForecastChecker,
    ForecastCheckStore,
    load_measured,
)
from core.theria.plan.lk_executor import LKExecutor
from core.theria.plan.outcomes import (
    DayOutcome,
    OutcomeInputs,
    day_outcomes,
    load_outcome_inputs,
)
from core.theria.plan.parameter_log import ParameterLog
from core.theria.plan.plan_record import PlanRecordStore
from core.theria.plan.plan_run import PlanRunStore
from core.theria.plan.price_feed import NordpoolFeed
from core.theria.plan.service import PlanService
from core.theria.plan.settings import load_plan_settings
from core.theria.plan.settings_manager import SettingsManager
from core.theria.plan.state_store import PlanStateStore
from core.theria.plan.weather import OpenMeteoClient
from core.theria.radiators.service import RadiatorService
from core.theria.radiators.settings import load_radiator_settings
from core.theria.radiators.store import PrefsStore
from core.theria.sensor_guard import SensorGuard, load_guard_settings
from core.theria.settings import HeatPumpSettings
from core.theria.temperature_history_service import TemperatureHistoryService
from core.theria.thermal_learning_service import ThermalLearningService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Theria starting")
    if ha_client and ha_client.test_mode:
        logger.warning("Test mode: reads Home Assistant, writes nothing to it")

    # Log registered routes
    routes = [
        f"{getattr(route, 'path', '?')} - {getattr(route, 'methods', ['MOUNT'])}"
        for route in app.routes
    ]
    logger.info(f"Registered routes: {routes}")

    # Start heat pump monitor if enabled
    heat_pump_service = None
    heat_pump_settings = None
    if ha_client:
        try:
            import yaml

            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    heat_pump_config = config.get("options", {}).get("heat_pump", {})

                    if heat_pump_config.get("enabled"):
                        heat_pump_settings = HeatPumpSettings.from_dict(
                            heat_pump_config
                        )
                        heat_pump_service = HeatPumpMonitorService(
                            ha_client,
                            heat_pump_settings,
                            collection_interval_seconds=60,
                        )
                        await heat_pump_service.start()
                        logger.info("♨️ Heat pump monitoring enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize heat pump monitor: {e}")

    # Start sensor health monitor if HA client is available
    health_monitor_service = None
    if ha_client and ZONES:
        health_monitor_service = HealthMonitorService(
            ha_client,
            ZONES,
            heat_pump_settings=heat_pump_settings,
            check_interval_seconds=300,
            stale_after_minutes=120,
        )
        await health_monitor_service.start()

        import api

        api.health_monitor_service = health_monitor_service

    heat_pump_api.ha_client = ha_client

    # Start thermal learning service if HA client is available
    learning_service = None
    history_service = None
    if ha_client and ZONES:
        learning_service = ThermalLearningService(
            ha_client,
            ZONES,
            learning_interval_minutes=1,  # 1 min for optimal learning resolution
            outdoor_temp_sensor="sensor.outdoor",  # H60 outdoor temp sensor
        )
        await learning_service.start()
        logger.info(f"🧠 Thermal learning enabled for {len(ZONES)} zone(s)")

        # Start temperature history collection service
        history_service = TemperatureHistoryService(
            ha_client, ZONES, collection_interval_seconds=60  # Collect every minute
        )
        await history_service.start()

        # Connect history service to learning service and bootstrap
        learning_service.history_service = history_service
        await learning_service.bootstrap_from_history()

        # Make learning service available to API
        import api

        api.learning_service = learning_service
    else:
        logger.warning("⚠️ Thermal learning disabled (no HA client or zones)")

    # Comfort sensor guard: battery, silence and agreement checks; feeds the plan
    sensor_guard = None
    if ha_client:
        try:
            guard_settings = load_guard_settings(
                os.path.join(DATA_DIR, "options.json"),
                os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
            )
            if guard_settings:
                sensor_guard = SensorGuard(
                    ha_client, guard_settings, notifier=HANotifier(ha_client)
                )
                sensor_api.guard = sensor_guard
                await sensor_guard.start()
                logger.info(
                    f"🔋 Comfort sensor guard started ({len(guard_settings.sensors)} sensors)"
                )
        except TheriaError as e:
            logger.error(f"Comfort sensor guard not started: {e}")

    # Start the price-driven plan service (writes to HA only in active mode, which needs
    # explicit confirmation through the API)
    plan_service = None
    radiator_service = None
    if ha_client:
        try:
            plan_settings = load_plan_settings(
                os.path.join(DATA_DIR, "options.json"),
                os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
            )
            if plan_settings:
                overrides_path = os.path.join(DATA_DIR, "plan_overrides.json")
                settings_manager = SettingsManager(plan_settings, overrides_path)
                plan_settings = settings_manager.current
                weather_client = OpenMeteoClient(
                    plan_settings.latitude, plan_settings.longitude
                )
                state_path = os.path.join(DATA_DIR, "plan_state.json")
                parameter_path = os.path.join(DATA_DIR, "plan_parameters.jsonl")
                event_path = os.path.join(DATA_DIR, "plan_events.jsonl")
                record_path = os.path.join(DATA_DIR, "plan_record.jsonl")
                runs_path = os.path.join(DATA_DIR, "plan_runs.jsonl")
                trace_path = os.path.join(DATA_DIR, "exec_trace.jsonl")
                check_path = os.path.join(DATA_DIR, "forecast_check.jsonl")
                history_source = InfluxHistorySource()
                built: list[PlanService] = (
                    []
                )  # the loader reads the fitted pump from it

                plan_record = PlanRecordStore(record_path)
                prices = NordpoolFeed(
                    ha_client,
                    plan_settings.price_config_entry,
                    plan_settings.price_area,
                    plan_settings.timezone,
                )

                def actuals_loader(start: datetime, stop: datetime) -> OutcomeInputs:
                    return load_outcome_inputs(
                        history_source,
                        plan_settings,
                        start,
                        stop,
                        prices.history(start, stop),
                    )

                def outcome_loader(start: datetime, stop: datetime) -> list[DayOutcome]:
                    return day_outcomes(
                        actuals_loader(start, stop),
                        built[0].pump_params,
                        plan_settings.rts_base_c,
                        lambda moment: not plan_settings.in_night(moment),
                        plan_settings.comfort_starts_at,
                    )

                plan_runs = PlanRunStore(runs_path, datetime.now(UTC))
                forecast_check_store = ForecastCheckStore(check_path)
                plan_api.forecast_check_store = forecast_check_store
                forecast_check = ForecastChecker(
                    plan_runs,
                    lambda start, stop: load_measured(
                        history_source,
                        plan_settings,
                        weather_client.history,
                        start,
                        stop,
                    ),
                    forecast_check_store,
                    lambda: datetime.now(UTC),
                )

                energy_forecaster = EnergyForecaster(
                    lambda start, stop: load_energy_history(
                        history_source,
                        plan_settings,
                        weather_client.history,
                        start,
                        stop,
                    ),
                    weather_client.fetch,
                    EnergyForecastStore(os.path.join(DATA_DIR, "energy_forecast.jsonl")),
                    plan_settings.timezone,
                    lambda: datetime.now(UTC),
                    lambda hour: built[0].planned_setpoint(hour),
                )
                plan_api.energy_forecaster = energy_forecaster

                plan_service = PlanService(
                    ha_client,
                    plan_settings,
                    weather_client,
                    calibrator=Calibrator(
                        history_source, weather_client, plan_settings
                    ),
                    state_store=PlanStateStore(state_path),
                    sensor_guard=sensor_guard,
                    executor=PlanExecutor(ha_client, plan_settings),
                    lk_executor=LKExecutor(ha_client, plan_settings),
                    safety=SafetyMonitor(ha_client, plan_settings),
                    notifier=HANotifier(ha_client),
                    outcome_loader=outcome_loader,
                    parameter_log=ParameterLog(parameter_path),
                    event_log=EventLog(event_path),
                    plan_record=plan_record,
                    plan_runs=plan_runs,
                    exec_trace=ExecTraceStore(trace_path, datetime.now(UTC)),
                    log_dir=os.path.join(DATA_DIR, "logs"),
                    settings_manager=settings_manager,
                    actuals_loader=actuals_loader,
                    price_periods=prices.periods,
                    price_history=prices.history,
                    forecast_check=forecast_check,
                    energy_forecast=energy_forecaster,
                )
                built.append(plan_service)
                plan_api.service = plan_service
                await plan_service.start()
                logger.info(
                    f"📈 Price plan service started (mode: {plan_settings.mode})"
                )

                # Salon and guest house radiators: shadow only, shares the plan's price feed
                radiator_settings = load_radiator_settings(
                    os.path.join(DATA_DIR, "options.json"),
                    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
                )
                if radiator_settings:
                    radiator_prefs_path = os.path.join(DATA_DIR, "radiator_prefs.json")
                    radiator_events_path = os.path.join(
                        DATA_DIR, "radiator_events.jsonl"
                    )
                    radiator_service = RadiatorService(
                        ha_client,
                        radiator_settings,
                        PrefsStore(radiator_settings.defaults, radiator_prefs_path),
                        SensorGuard(ha_client, radiator_settings.salon_sensor_guard),
                        prices.periods,
                        event_log=EventLog(radiator_events_path),
                    )
                    radiator_api.service = radiator_service
                    plan_service.use_salon_status(radiator_service.salon_status)
                    await radiator_service.start()
                    logger.info("🔥 Radiator plan service started (shadow)")
        except TheriaError as e:
            logger.error(f"Price plan service not started: {e}")

    yield

    # Shutdown
    logger.info("Theria shutting down")
    if radiator_service:
        await radiator_service.stop()
    if plan_service:
        await plan_service.stop()
    if sensor_guard:
        await sensor_guard.stop()
    if learning_service:
        await learning_service.stop()
    if history_service:
        await history_service.stop()
    if heat_pump_service:
        await heat_pump_service.stop()
    if health_monitor_service:
        await health_monitor_service.stop()


# Create FastAPI application
app = FastAPI(
    title="Theria API",
    description="Smart heating optimization using electricity price-based heat capacitor strategy",
    version="0.1.0",
    lifespan=lifespan,
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions gracefully."""
    import traceback

    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Request path: {request.url.path}")
    logger.error(f"Stack trace:\n{tb_str}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "message": "Internal server error",
        },
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)
app.include_router(plan_api.router)
app.include_router(heat_pump_api.router)
app.include_router(radiator_api.router)
app.include_router(sensor_api.router)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
plan_ui_dir = os.path.join(static_dir, "plan")
if os.path.exists(plan_ui_dir):
    # Built React page (frontend/); must be mounted before the general /static mount
    app.mount(
        "/static/plan", StaticFiles(directory=plan_ui_dir, html=True), name="plan-ui"
    )
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static files from {static_dir}")


@app.middleware("http")
async def no_cache_for_html(request: Request, call_next):
    """HTML is never cached: a new deploy has to show up at once (the js and css files have
    content-hashed names and can be cached as long as anyone likes)."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _html_page(request: Request, path: str, base_suffix: str = "") -> HTMLResponse:
    """Serve an HTML file with a <base> tag so that relative links work behind HA ingress."""
    ingress_path = request.headers.get("X-Ingress-Path", "")
    with open(path) as f:
        html_content = f.read()
    base_tag = f'<base href="{ingress_path}/{base_suffix}">'
    if ingress_path or base_suffix:
        html_content = html_content.replace("<head>", f"<head>\n    {base_tag}", 1)
    return HTMLResponse(content=html_content)


@app.get("/")
async def root(request: Request):
    """Start page: the React app (built from frontend/ into static/plan)."""
    index = os.path.join(plan_ui_dir, "index.html")
    if not os.path.exists(index):
        return HTMLResponse(
            status_code=503,
            content="The frontend is not built: run `npm run build` in frontend/.",
        )
    # The built page uses relative asset and API urls, resolved against static/plan/
    return _html_page(request, index, "static/plan/")


@app.get("/legacy")
async def legacy_dashboard(request: Request):
    """The old dashboard, kept for reference."""
    return _html_page(request, os.path.join(static_dir, "index.html"))


@app.get("/zone-insights")
@app.get("/zone-insights.html")
async def zone_insights(request: Request):
    """Zone insights page (old UI)."""
    return _html_page(request, os.path.join(static_dir, "zone-insights.html"))


@app.get("/system-insights")
@app.get("/system-insights.html")
async def system_insights(request: Request):
    """System insights page (old UI)."""
    return _html_page(request, os.path.join(static_dir, "system-insights.html"))


# For development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
