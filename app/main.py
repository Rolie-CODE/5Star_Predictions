"""
main.py
-------
FastAPI application entry point for 5Star Predictions.

Startup sequence:
  1. Initialize DB (apply Alembic migrations)
  2. Load ML models from disk
  3. Start APScheduler background jobs
  4. Register API routers

On shutdown:
  1. Shut down APScheduler gracefully
  2. Close the API-Sports HTTP client
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes import games, predictions
from app.api.schemas import HealthResponse
from app.core.config import settings
from app.core.scheduler import create_scheduler
from app.db.session import AsyncSessionLocal, init_db
from app.ingestion.api_sports import api_sports_client
from app.ml.predict import load_models, model_loader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=== 5Star Predictions API starting up ===")

    # 1. Apply database migrations
    await init_db()
    logger.info("Database initialized.")

    # 2. Load ML models (non-fatal — app still starts without models)
    load_models()
    if model_loader.is_loaded:
        logger.info("ML models loaded successfully.")
    else:
        logger.warning(
            "ML models NOT loaded. Run `python -m app.ml.train` to generate model files, "
            "then restart the server."
        )

    # 3. Start background scheduler
    scheduler = create_scheduler()
    scheduler.start()
    logger.info("APScheduler started with %d jobs.", len(scheduler.get_jobs()))

    yield   # ← app is live and serving requests here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("=== Shutting down... ===")
    scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped.")
    await api_sports_client.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="5Star Predictions API",
    description=(
        "Basketball prediction engine. Provides Home/Away winner predictions, "
        "Full-Time and Half-Time Over/Under predictions with trust percentages, "
        "and top daily picks — all powered by XGBoost trained on NBA data."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — set CORS_ORIGINS to your frontend domain(s) in production.
# The API uses no cookies, so credentials are never allowed.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(games.router, prefix="/games")
app.include_router(predictions.router, prefix="/predictions")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Returns API health status.
    - `status`: 'ok' when the database is reachable and models are loaded,
      otherwise 'degraded'.  Always HTTP 200 so a deploy without models
      still passes Render's health check.
    """
    db_ok = False
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        logger.exception("Health check: database unreachable.")

    return HealthResponse(
        status="ok" if db_ok and model_loader.is_loaded else "degraded",
        environment=settings.APP_ENV,
        models_loaded=model_loader.is_loaded,
        db_connected=db_ok,
    )
