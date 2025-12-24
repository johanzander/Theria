"""
Theria Backend Application

Minimal FastAPI application - will be expanded as features are added.
"""

import os
import sys
from contextlib import asynccontextmanager

import log_config  # noqa: F401
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Add core to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import API router
from api import router as api_router, ha_client, ZONES
from core.theria.thermal_learning_service import ThermalLearningService
from core.theria.price_optimizer import PriceOptimizer

# Get ingress prefix for Home Assistant integration
INGRESS_PREFIX = os.environ.get("INGRESS_PREFIX", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown."""
    # Startup
    logger.info(f"Theria starting with ingress prefix: {INGRESS_PREFIX}")

    # Log registered routes
    routes = [
        f"{getattr(route, 'path', '?')} - {getattr(route, 'methods', ['MOUNT'])}"
        for route in app.routes
    ]
    logger.info(f"Registered routes: {routes}")

    # Initialize price optimizer if enabled
    price_optimizer = None
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
                            raise ValueError("price_sensor not configured in price_optimization")

                        price_entity = sensors.get(sensor_key)
                        if not price_entity:
                            raise ValueError(f"Sensor key '{sensor_key}' not found in sensors config")

                        price_optimizer = PriceOptimizer(
                            ha_client,
                            price_entity=price_entity,
                            expensive_hours=price_config.get("expensive_hours", 4),
                            cheap_hours=price_config.get("cheap_hours", 4),
                            adjustment_degrees=price_config.get("adjustment_degrees", 0.5)
                        )
                        logger.info(f"💰 Price optimization enabled (sensor: {sensor_key} -> {price_entity})")
        except Exception as e:
            logger.warning(f"Failed to initialize price optimizer: {e}")

    # Start thermal learning service if HA client is available
    learning_service = None
    if ha_client and ZONES:
        learning_service = ThermalLearningService(
            ha_client,
            ZONES,
            learning_interval_minutes=1,  # 1 min for optimal learning resolution
            outdoor_temp_sensor="sensor.outdoor"  # H60 outdoor temp sensor
        )
        await learning_service.start()
        logger.info(f"🧠 Thermal learning enabled for {len(ZONES)} zone(s)")

        # Make learning service available to API
        import api
        api.learning_service = learning_service
    else:
        logger.warning("⚠️ Thermal learning disabled (no HA client or zones)")

    yield

    # Shutdown
    logger.info("Theria shutting down")
    if learning_service:
        await learning_service.stop()


# Create FastAPI application
app = FastAPI(
    title="Theria API",
    description="Smart heating optimization using electricity price-based heat capacitor strategy",
    version="0.1.0",
    root_path=INGRESS_PREFIX,
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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static files from {static_dir}")


@app.get("/")
async def root():
    """Root endpoint - redirect to UI."""
    return RedirectResponse(url="/static/index.html")


# For development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
