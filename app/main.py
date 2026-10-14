"""
main.py
-------
FastAPI application entry point for 5Star Predictions.

Startup sequence:
  1. Initialize DB (create tables if they don't exist)
  2. Load ML models from disk
  3. Start APScheduler background jobs
  4. Register API routers

On shutdown:
  1. Shut down APScheduler gracefully
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.scheduler import create_scheduler
from app.db.session import init_db
from app.ml.predict import load_models, model_loader
from app.api.routes import games, predictions

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

    # 1. Create DB tables
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
    logger.info("APScheduler started with 3 scheduled jobs.")

    yield   # ← app is live and serving requests here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("=== Shutting down... ===")
    scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped.")


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
# CORS (permissive for now — tighten when you have a frontend domain)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(games.router,       prefix="/games",       tags=["Games"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Returns API health status.
    - `models_loaded`: True if XGBoost model files are loaded and ready.
    - `status`: Always 'ok' if the server is reachable.
    """
    from app.db.session import AsyncSessionLocal
    db_ok = False
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(__import__("sqlalchemy").text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "environment": settings.APP_ENV,
        "models_loaded": model_loader.is_loaded,
        "db_connected": db_ok,
    }
