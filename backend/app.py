"""
Theria Backend Application

Minimal FastAPI application - will be expanded as features are added.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import log_config  # noqa: F401
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Add core to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import API router
import plan_api
import sensor_api
from api import ZONES, ha_client
from api import router as api_router

from core.theria.exceptions import TheriaError
from core.theria.ha_client import HANotifier
from core.theria.health_monitor import HealthMonitorService
from core.theria.heat_pump_monitor import HeatPumpMonitorService
from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import InfluxHistorySource
from core.theria.plan.executor import PlanExecutor, SafetyMonitor
from core.theria.plan.lk_executor import LKExecutor
from core.theria.plan.outcomes import DayOutcome, day_outcomes, load_outcome_inputs
from core.theria.plan.service import PlanService
from core.theria.plan.settings import load_plan_settings
from core.theria.plan.state_store import PlanStateStore
from core.theria.plan.weather import OpenMeteoClient
from core.theria.price_optimizer import PriceOptimizer
from core.theria.sensor_guard import SensorGuard, load_guard_settings
from core.theria.settings import HeatPumpSettings
from core.theria.temperature_history_service import TemperatureHistoryService
from core.theria.thermal_learning_service import ThermalLearningService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Theria starting")

    # Log registered routes
    routes = [
        f"{getattr(route, 'path', '?')} - {getattr(route, 'methods', ['MOUNT'])}"
        for route in app.routes
    ]
    logger.info(f"Registered routes: {routes}")

    # Initialize price optimizer if enabled
    # Note: Price optimizer is also initialized in api.py for endpoint access
    if ha_client:
        try:
            # Load price config from options (in production) or config.yaml (dev)
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
                        if not sensor_key:
                            raise ValueError(
                                "price_sensor not configured in price_optimization"
                            )

                        price_entity = sensors.get(sensor_key)
                        if not price_entity:
                            raise ValueError(
                                f"Sensor key '{sensor_key}' not found in sensors config"
                            )

                        PriceOptimizer(
                            ha_client,
                            price_entity=price_entity,
                            expensive_hours=price_config.get("expensive_hours", 4),
                            cheap_hours=price_config.get("cheap_hours", 4),
                            adjustment_degrees=price_config.get(
                                "adjustment_degrees", 0.5
                            ),
                        )
                        logger.info(
                            f"💰 Price optimization enabled (sensor: {sensor_key} -> {price_entity})"
                        )
        except Exception as e:
            logger.warning(f"Failed to initialize price optimizer: {e}")

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
                "/data/options.json",
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
    if ha_client:
        try:
            plan_settings = load_plan_settings(
                "/data/options.json",
                os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
            )
            if plan_settings:
                weather_client = OpenMeteoClient(
                    plan_settings.latitude, plan_settings.longitude
                )
                state_path = (
                    "/data/plan_state.json"
                    if os.path.isdir("/data")
                    else os.path.join(
                        os.path.dirname(__file__), "..", ".plan_state.json"
                    )
                )
                history_source = InfluxHistorySource()
                built: list[PlanService] = (
                    []
                )  # the loader reads the fitted pump from it

                def outcome_loader(start: datetime, stop: datetime) -> list[DayOutcome]:
                    return day_outcomes(
                        load_outcome_inputs(history_source, plan_settings, start, stop),
                        built[0].pump_params,
                        plan_settings.rts_base_c,
                    )

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
                )
                built.append(plan_service)
                plan_api.service = plan_service
                await plan_service.start()
                logger.info(
                    f"📈 Price plan service started (mode: {plan_settings.mode})"
                )
        except TheriaError as e:
            logger.error(f"Price plan service not started: {e}")

    yield

    # Shutdown
    logger.info("Theria shutting down")
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
