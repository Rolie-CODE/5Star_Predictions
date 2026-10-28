"""
app/services/prediction_service.py
-----------------------------------
Business-logic layer that orchestrates the full prediction pipeline:

  1. Load today's (or upcoming) scheduled games from the database.
  2. Build a feature vector for each game via `feature_engineering`.
  3. Pass features through the trained ML models via `prediction_engine`.
  4. Persist (upsert) results as `Prediction` rows in the database.
  5. Expose read helpers that return structured dicts matching the API schemas.

All public functions are async and expect an `AsyncSession` from SQLAlchemy.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.models import Game, Prediction
from app.ingestion.data_pipeline import recompute_team_stats, sync_games
from app.ml import features as feature_engineering
from app.ml.predict import prediction_engine

logger = logging.getLogger(__name__)

# Serialises full refreshes (scheduler job vs. manual POST /refresh).
_refresh_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; every stored value is UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _today_bounds() -> tuple[datetime, datetime]:
    """
    Return the UTC instants bounding today's league-local calendar day.

    An NBA "today" runs midnight to midnight US Eastern, so a 10:30pm ET
    tip-off (02:30 UTC the next day) still belongs to today's slate.
    """
    tz = settings.league_tz
    today = datetime.now(tz).date()
    start = datetime.combine(today, time.min, tzinfo=tz)
    end = datetime.combine(today + timedelta(days=1), time.min, tzinfo=tz)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _prediction_window() -> tuple[datetime, datetime]:
    """
    UTC instants bounding today's and tomorrow's league-local slates.

    Predicting tomorrow too keeps the rolling 24-hour views (upcoming games,
    top picks) filled after today's last tip-off.
    """
    start, end = _today_bounds()
    return start, end + timedelta(days=1)


def _ou_block(pick: str | None, line: float | None, trust: float | None) -> dict | None:
    if pick is None or trust is None:
        return None
    return {"pick": pick, "line": line, "trust": trust}


def _game_to_dict(game: Game, prediction: Prediction | None) -> dict:
    """
    Convert a Game ORM row (+ optional Prediction row) into a plain dict
    that matches the `GameResponse` Pydantic schema.
    """
    pred_block: dict | None = None

    if prediction is not None:
        pred_block = {
            "winner": {
                "pick": prediction.winner_pick,
                "trust": prediction.winner_trust,
            },
            "full_time_ou": _ou_block(
                prediction.ft_ou_pick, game.ou_line, prediction.ft_ou_trust
            ),
            "half_time_ou": _ou_block(
                prediction.ht_ou_pick, game.ht_ou_line, prediction.ht_ou_trust
            ),
        }

    return {
        "game_id": game.id,
        "api_sports_id": game.api_sports_id,
        "home_team": game.home_team.name,
        "away_team": game.away_team.name,
        "tip_off": _as_utc(game.date),
        "status": game.status,
        "prediction": pred_block,
    }


async def _fetch_games_with_predictions(
    db: AsyncSession,
    *,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Shared helper: load all games whose tip-off falls within [start, end),
    eagerly loading related teams and their prediction row.
    """
    stmt = (
        select(Game)
        .where(Game.date >= start, Game.date < end)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
            selectinload(Game.prediction),
        )
        .order_by(Game.date.asc())
    )
    result = await db.execute(stmt)
    games: list[Game] = list(result.scalars().all())

    return [_game_to_dict(g, getattr(g, "prediction", None)) for g in games]


# ---------------------------------------------------------------------------
# Core prediction pipeline
# ---------------------------------------------------------------------------


async def run_predictions_for_today(db: AsyncSession) -> int:
    """
    Run the full prediction pipeline for every scheduled game today and
    tomorrow.

    Steps
    -----
    1. Query all games on today's and tomorrow's league-local slates that
       haven't started.
    2. Build a feature vector for each game.
    3. Run the ML models to get winner / FT O/U / HT O/U predictions.
    4. Upsert a `Prediction` row in the database.

    Returns
    -------
    int
        The number of games for which a prediction was successfully generated
        and saved.
    """
    start, end = _prediction_window()

    # -- 1. Load today's and tomorrow's scheduled games -------------------
    # Games past tip-off are skipped even if their status is stale, so each
    # stored prediction is the last pre-game one and can be graded fairly.
    stmt = (
        select(Game)
        .where(
            Game.date >= max(start, _utc_now()),
            Game.date < end,
            Game.status == "scheduled",
        )
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
        )
        .order_by(Game.date.asc())
    )
    result = await db.execute(stmt)
    games: list[Game] = list(result.scalars().all())

    logger.info(
        "run_predictions_for_today: found %d scheduled game(s) for %s",
        len(games),
        start.date(),
    )

    count = 0
    for game in games:
        try:
            # -- 2. Build feature vector ----------------------------------
            features = await feature_engineering.build_feature_vector(db, game)

            # -- 3. Run ML models -----------------------------------------
            result_payload: dict = prediction_engine.predict(features)
            # Expected keys in result_payload:
            #   winner_pick, winner_trust, ft_ou_pick, ft_ou_trust,
            #   ht_ou_pick, ht_ou_trust, model_version

            # -- 4. Upsert Prediction row ----------------------------------
            existing_result = await db.execute(
                select(Prediction).where(Prediction.game_id == game.id)
            )
            prediction: Prediction | None = existing_result.scalar_one_or_none()
            if prediction is None:
                prediction = Prediction(game_id=game.id)
                db.add(prediction)
            for key, value in result_payload.items():
                setattr(prediction, key, value)
            prediction.created_at = _utc_now()

            await db.flush()  # propagate within the transaction
            count += 1
            logger.debug(
                "Prediction saved for game %d (%s vs %s): winner=%s (%.1f%%)",
                game.id,
                game.home_team.name,
                game.away_team.name,
                result_payload["winner_pick"],
                result_payload["winner_trust"],
            )

        except Exception:
            logger.exception(
                "Unexpected error generating prediction for game %d.", game.id
            )
            # Continue processing remaining games rather than aborting.

    await db.commit()
    logger.info("run_predictions_for_today: committed %d prediction(s).", count)
    return count


async def refresh_todays_predictions(db: AsyncSession) -> int:
    """
    Full pipeline: final scores for yesterday, today's and tomorrow's
    fixtures and lines, team stats, then predictions.  Returns the number
    of games predicted.
    """
    async with _refresh_lock:
        today = settings.league_today()
        await sync_games(db, [today - timedelta(days=1)], fetch_odds=False)
        await sync_games(db, [today, today + timedelta(days=1)])
        await recompute_team_stats(db)
        return await run_predictions_for_today(db)


async def todays_games_missing_predictions(db: AsyncSession) -> int | None:
    """
    Count today's and tomorrow's not-yet-started games that have no
    prediction row.

    Returns ``None`` when either day has no games stored at all (its
    fixtures have not been synced yet, or it is an off day).
    """
    start, end = _prediction_window()
    tomorrow_start = end - timedelta(days=1)
    result = await db.execute(
        select(Game.date, Game.status, Prediction.id)
        .outerjoin(Prediction, Prediction.game_id == Game.id)
        .where(Game.date >= start, Game.date < end)
    )
    rows = result.all()
    if not any(_as_utc(d) < tomorrow_start for d, _, _ in rows) or not any(
        _as_utc(d) >= tomorrow_start for d, _, _ in rows
    ):
        return None
    return sum(1 for _, status, prediction_id in rows if status == "scheduled" and prediction_id is None)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


async def get_game_with_prediction(
    db: AsyncSession,
    game_id: int,
) -> dict | None:
    """
    Fetch a single game by its internal primary key, joined with its teams
    and any existing prediction.

    Returns
    -------
    dict | None
        A dict matching the `GameResponse` schema, or ``None`` if the game
        does not exist.
    """
    stmt = (
        select(Game)
        .where(Game.id == game_id)
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
            selectinload(Game.prediction),
        )
    )
    result = await db.execute(stmt)
    game: Game | None = result.scalar_one_or_none()

    if game is None:
        return None

    return _game_to_dict(game, getattr(game, "prediction", None))


async def get_todays_games_with_predictions(db: AsyncSession) -> list[dict]:
    """
    Return all games scheduled for today (midnight→midnight UTC) with their
    predictions joined.

    Returns
    -------
    list[dict]
        A list of dicts each matching the `GameResponse` schema, ordered by
        tip-off time ascending.
    """
    start, end = _today_bounds()
    return await _fetch_games_with_predictions(db, start=start, end=end)


async def get_upcoming_games_with_predictions(db: AsyncSession) -> list[dict]:
    """
    Return all games with a tip-off between now and 24 hours from now.

    Returns
    -------
    list[dict]
        A list of dicts each matching the `GameResponse` schema, ordered by
        tip-off time ascending.
    """
    now = _utc_now()
    end = now + timedelta(hours=24)
    return await _fetch_games_with_predictions(db, start=now, end=end)


async def get_top_predictions(
    db: AsyncSession,
    prediction_type: str,
    limit: int,
) -> list[dict]:
    """
    Return the top-N predictions for a given prediction category, ranked by
    trust score descending.

    Parameters
    ----------
    db:
        Active async database session.
    prediction_type:
        One of ``'winner'``, ``'ft_ou'``, or ``'ht_ou'``.
    limit:
        Maximum number of items to return (caller is responsible for capping
        this at 20).

    Returns
    -------
    list[dict]
        A list of dicts each matching the `TopPickItem` schema.

    Raises
    ------
    ValueError
        If ``prediction_type`` is not one of the accepted values.
    """
    # Map prediction_type → (pick column, trust column, game line attr name)
    # NOTE: O/U lines live on Game (ou_line, ht_ou_line), NOT on Prediction.
    # We store the Game attribute name as a string and look it up with getattr.
    _type_map: dict[str, tuple] = {
        "winner": (
            Prediction.winner_pick,
            Prediction.winner_trust,
            None,           # no O/U line for winner picks
        ),
        "ft_ou": (
            Prediction.ft_ou_pick,
            Prediction.ft_ou_trust,
            "ou_line",      # Game.ou_line
        ),
        "ht_ou": (
            Prediction.ht_ou_pick,
            Prediction.ht_ou_trust,
            "ht_ou_line",   # Game.ht_ou_line
        ),
    }

    if prediction_type not in _type_map:
        raise ValueError(
            f"Invalid prediction_type '{prediction_type}'. "
            f"Must be one of: {list(_type_map.keys())}"
        )

    pick_col, trust_col, game_line_attr = _type_map[prediction_type]

    # Only consider games that tip off within the next 24 hours so the
    # top-picks list stays actionable.
    now = _utc_now()
    end = now + timedelta(hours=24)

    stmt = (
        select(Game, Prediction)
        .join(Prediction, Prediction.game_id == Game.id)
        .where(Game.date >= now, Game.date < end, pick_col.is_not(None))
        .options(
            selectinload(Game.home_team),
            selectinload(Game.away_team),
        )
        .order_by(trust_col.desc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    rows = result.all()  # list of (Game, Prediction) tuples

    items: list[dict] = []
    for game, prediction in rows:
        pick = getattr(prediction, pick_col.key)
        trust = getattr(prediction, trust_col.key)
        line = getattr(game, game_line_attr) if game_line_attr is not None else None

        items.append(
            {
                "game_id": game.id,
                "home_team": game.home_team.name,
                "away_team": game.away_team.name,
                "tip_off": _as_utc(game.date),
                "pick": pick,
                "trust": trust,
                "line": line,
            }
        )

    return items


# Profit on a winning 1-unit bet at -110; a loss costs 1 unit.
_WIN_PAYOUT = 100 / 110


def _pick_record(results: list[bool], *, with_roi: bool) -> dict:
    graded, correct = len(results), sum(results)
    return {
        "graded": graded,
        "correct": correct,
        "accuracy": round(correct / graded * 100, 1) if graded else None,
        "roi": (
            round((correct * _WIN_PAYOUT - (graded - correct)) / graded * 100, 1)
            if with_roi and graded
            else None
        ),
    }


async def get_performance(
    db: AsyncSession,
    season: str,
    model_version: str | None = None,
) -> dict:
    """
    Grade stored predictions against final scores for *season*.

    Winner picks count when the game has a final score; Over/Under picks
    when it also has a line (pushes are not graded).  ``roi`` is profit per
    bet in percent at standard -110 prices.
    """
    stmt = (
        select(Game, Prediction)
        .join(Prediction, Prediction.game_id == Game.id)
        .where(
            Game.season == season,
            Game.status == "finished",
            Game.home_score.is_not(None),
            Game.away_score.is_not(None),
        )
    )
    if model_version is not None:
        stmt = stmt.where(Prediction.model_version == model_version)
    rows = (await db.execute(stmt)).all()

    winner: list[bool] = []
    ft_ou: list[bool] = []
    for game, prediction in rows:
        if game.home_score != game.away_score:
            home_won = game.home_score > game.away_score
            winner.append((prediction.winner_pick == "Home") == home_won)
        total = game.home_score + game.away_score
        if prediction.ft_ou_pick is not None and game.ou_line is not None and total != game.ou_line:
            ft_ou.append((prediction.ft_ou_pick == "Over") == (total > game.ou_line))

    return {
        "season": season,
        "model_version": model_version,
        "winner": _pick_record(winner, with_roi=False),
        "full_time_ou": _pick_record(ft_ou, with_roi=True),
    }
