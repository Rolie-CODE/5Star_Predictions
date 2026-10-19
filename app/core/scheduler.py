"""
scheduler.py
------------
APScheduler background job definitions for 5Star Predictions.

Jobs:
  - fetch_and_predict  : Runs every day at 07:00 local time.
                         Syncs today's games from API-Sports, then runs ML predictions.
  - fetch_results      : Runs every 2 hours to update finished game scores.
  - retrain_model      : Runs on the 1st of every month at 03:00 to retrain XGBoost models.
"""

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db.session import AsyncSessionLocal
from app.ingestion.data_pipeline import sync_todays_games, sync_all_team_stats
from app.services.prediction_service import run_predictions_for_today

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheduled job functions
# ---------------------------------------------------------------------------

async def job_fetch_and_predict() -> None:
    """Sync today's games + odds, then run ML predictions for all of them."""
    logger.info("[Scheduler] Starting daily fetch + predict job...")
    async with AsyncSessionLocal() as db:
        try:
            games_synced = await sync_todays_games(db)
            logger.info(f"[Scheduler] Synced {games_synced} games from API-Sports.")

            await sync_all_team_stats(db)
            logger.info("[Scheduler] Team stats updated.")

            predictions_made = await run_predictions_for_today(db)
            logger.info(f"[Scheduler] Generated {predictions_made} predictions.")
        except Exception as exc:
            logger.exception(f"[Scheduler] fetch_and_predict job failed: {exc}")


async def job_fetch_results() -> None:
    """Refresh scores for games that may have finished since last run."""
    logger.info("[Scheduler] Refreshing game results...")
    async with AsyncSessionLocal() as db:
        try:
            synced = await sync_todays_games(db)
            logger.info(f"[Scheduler] Refreshed {synced} game records.")
        except Exception as exc:
            logger.exception(f"[Scheduler] fetch_results job failed: {exc}")


async def job_retrain_model() -> None:
    """Monthly retrain — imports and runs the training script in-process."""
    logger.info("[Scheduler] Starting monthly model retraining...")
    try:
        # Import here to avoid circular imports at module load time
        import importlib
        train_module = importlib.import_module("app.ml.train")
        await train_module.run_training()          # train.py must expose run_training()
        from app.ml.predict import load_models
        load_models()
        logger.info("[Scheduler] Model retraining complete.")
    except Exception as exc:
        logger.exception(f"[Scheduler] Retrain job failed: {exc}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def create_scheduler() -> AsyncIOScheduler:
    """Build and return a configured APScheduler instance (not yet started)."""
    scheduler = AsyncIOScheduler()

    # Daily at 07:00 — fetch games + predict
    scheduler.add_job(
        job_fetch_and_predict,
        trigger=CronTrigger(hour=7, minute=0),
        id="fetch_and_predict",
        name="Daily fetch + predict",
        replace_existing=True,
        misfire_grace_time=3600,   # allow up to 1h late start
    )

    # Every 2 hours — refresh scores (only meaningful on game days, harmless otherwise)
    scheduler.add_job(
        job_fetch_results,
        trigger=CronTrigger(minute=0, hour="*/2"),
        id="fetch_results",
        name="Bi-hourly score refresh",
        replace_existing=True,
        misfire_grace_time=1800,
    )

    # 1st of every month at 03:00 — retrain
    scheduler.add_job(
        job_retrain_model,
        trigger=CronTrigger(day=1, hour=3, minute=0),
        id="retrain_model",
        name="Monthly model retrain",
        replace_existing=True,
        misfire_grace_time=7200,
    )

    return scheduler
