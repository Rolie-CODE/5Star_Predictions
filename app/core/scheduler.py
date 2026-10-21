"""
scheduler.py
------------
APScheduler background job definitions for 5Star Predictions.

All times are in ``settings.LEAGUE_TIMEZONE`` (US Eastern by default).

Jobs:
  - catch_up           : Runs once at startup.  Fills in anything missed while
                         the server was asleep (Render's free plan sleeps).
  - fetch_and_predict  : 10:00 and 17:00.  Syncs today's games + lines, team
                         stats, then runs ML predictions.  The second run
                         picks up lines that move during the day.
  - fetch_results      : Every 2 hours from 19:00 to 03:00.  Updates scores
                         (yesterday + today) and team stats.

Models are not retrained here: the training data lives only on the
developer's machine and Render's free disk is wiped on restart.  Retrain
locally with ``python -m app.ml.train`` and commit app/ml/models/.

API-Sports budget on a 12-game day: about 2 × 14 + 5 × 2 ≈ 40 requests,
well inside the Free plan's 100/day.
"""

import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select

from app.core.config import settings
from app.db.models import Game
from app.db.session import AsyncSessionLocal
from app.ingestion.data_pipeline import backfill_season, recompute_team_stats, sync_recent_results
from app.ml.predict import model_loader
from app.services.prediction_service import (
    refresh_todays_predictions,
    todays_games_missing_predictions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheduled job functions
# ---------------------------------------------------------------------------

async def job_fetch_and_predict() -> None:
    """Sync today's games + lines and team stats, then run ML predictions."""
    logger.info("[Scheduler] Starting fetch + predict job...")
    async with AsyncSessionLocal() as db:
        try:
            predictions_made = await refresh_todays_predictions(db)
            logger.info("[Scheduler] Generated %d predictions.", predictions_made)
        except Exception:
            logger.exception("[Scheduler] fetch_and_predict job failed.")


async def job_fetch_results() -> None:
    """Refresh scores for games that may have finished since last run."""
    logger.info("[Scheduler] Refreshing game results...")
    async with AsyncSessionLocal() as db:
        try:
            synced = await sync_recent_results(db)
            await recompute_team_stats(db)
            logger.info("[Scheduler] Refreshed %d game records.", synced)
        except Exception:
            logger.exception("[Scheduler] fetch_results job failed.")


async def job_catch_up() -> None:
    """
    Startup job: backfill the season if the DB has no results yet, then run
    the full pipeline if any of today's games still lack a prediction.
    """
    async with AsyncSessionLocal() as db:
        try:
            finished = await db.scalar(
                select(func.count(Game.id)).where(
                    Game.season == settings.NBA_SEASON, Game.status == "finished"
                )
            )
            if not finished:
                # One request; succeeds on paid plans only.
                await backfill_season(db)

            missing = await todays_games_missing_predictions(db)
            if model_loader.is_loaded and (missing is None or missing > 0):
                # The full refresh also syncs yesterday's results.
                count = await refresh_todays_predictions(db)
                logger.info("[Scheduler] Catch-up generated %d predictions.", count)
            else:
                await sync_recent_results(db)
                await recompute_team_stats(db)
        except Exception:
            logger.exception("[Scheduler] catch_up job failed.")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def create_scheduler() -> AsyncIOScheduler:
    """Build and return a configured APScheduler instance (not yet started)."""
    tz = settings.league_tz
    scheduler = AsyncIOScheduler(timezone=tz)

    scheduler.add_job(
        job_catch_up,
        trigger="date",
        run_date=datetime.now(tz),
        id="catch_up",
        name="Startup catch-up",
        replace_existing=True,
    )

    scheduler.add_job(
        job_fetch_and_predict,
        trigger=CronTrigger(hour="10,17", minute=0, timezone=tz),
        id="fetch_and_predict",
        name="Fetch + predict",
        replace_existing=True,
        misfire_grace_time=3600,   # allow up to 1h late start
        coalesce=True,
    )

    scheduler.add_job(
        job_fetch_results,
        trigger=CronTrigger(hour="19,21,23,1,3", minute=0, timezone=tz),
        id="fetch_results",
        name="Evening score refresh",
        replace_existing=True,
        misfire_grace_time=1800,
        coalesce=True,
    )

    return scheduler
