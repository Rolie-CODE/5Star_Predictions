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

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# ---------------------------------------------------------------------------
# Internal project imports
# ---------------------------------------------------------------------------
# These modules must exist in the project.  Adjust import paths if your
# package layout differs.
from app.db.models import Game, Prediction, Team  # ORM models
from app.ml import features as feature_engineering  # ML helpers
from app.ml.predict import prediction_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=timezone.utc)


def _today_bounds() -> tuple[datetime, datetime]:
    """Return (start_of_today_UTC, start_of_tomorrow_UTC) as aware datetimes."""
    now = _utc_now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


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
            "full_time_ou": {
                "pick": prediction.ft_ou_pick,
                "line": game.ou_line,
                "trust": prediction.ft_ou_trust,
            },
            "half_time_ou": {
                "pick": prediction.ht_ou_pick,
                "line": game.ht_ou_line,
                "trust": prediction.ht_ou_trust,
            },
        }

    return {
        "game_id": game.id,
        "api_sports_id": game.api_sports_id,
        "home_team": game.home_team.name,
        "away_team": game.away_team.name,
        "tip_off": game.date,
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
    Run the full prediction pipeline for every scheduled game today.

    Steps
    -----
    1. Query all games for today whose status indicates they haven't started
       yet ('NS', 'scheduled').
    2. Build a feature vector for each game.
    3. Run the ML models to get winner / FT O/U / HT O/U predictions.
    4. Upsert a `Prediction` row in the database.

    Returns
    -------
    int
        The number of games for which a prediction was successfully generated
        and saved.
    """
    start, end = _today_bounds()
    scheduled_statuses = {"NS", "scheduled"}

    # -- 1. Load today's scheduled games ----------------------------------
    stmt = (
        select(Game)
        .where(
            Game.date >= start,
            Game.date < end,
            Game.status.in_(scheduled_statuses),
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
            if features is None:
                logger.warning(
                    "Skipping game %d (%s vs %s) — insufficient feature data.",
                    game.id,
                    game.home_team.name,
                    game.away_team.name,
                )
                continue

            # -- 3. Run ML models -----------------------------------------
            result_payload: dict = prediction_engine.predict(features)
            # Expected keys in result_payload:
            #   winner_pick, winner_trust, ft_ou_pick, ft_ou_trust,
            #   ht_ou_pick, ht_ou_trust, model_version

            # -- 4. Upsert Prediction row ----------------------------------
            existing_stmt = select(Prediction).where(
                Prediction.game_id == game.id
            )
            existing_result = await db.execute(existing_stmt)
            existing: Prediction | None = existing_result.scalar_one_or_none()

            if existing is not None:
                # Update fields on the existing row
                await db.execute(
                    update(Prediction)
                    .where(Prediction.game_id == game.id)
                    .values(
                        winner_pick=result_payload["winner_pick"],
                        winner_trust=result_payload["winner_trust"],
                        ft_ou_pick=result_payload["ft_ou_pick"],
                        ft_ou_trust=result_payload["ft_ou_trust"],
                        ht_ou_pick=result_payload["ht_ou_pick"],
                        ht_ou_trust=result_payload["ht_ou_trust"],
                        model_version=result_payload["model_version"],
                    )
                )
            else:
                new_prediction = Prediction(
                    game_id=game.id,
                    winner_pick=result_payload["winner_pick"],
                    winner_trust=result_payload["winner_trust"],
                    ft_ou_pick=result_payload["ft_ou_pick"],
                    ft_ou_trust=result_payload["ft_ou_trust"],
                    ht_ou_pick=result_payload["ht_ou_pick"],
                    ht_ou_trust=result_payload["ht_ou_trust"],
                    model_version=result_payload["model_version"],
                    created_at=_utc_now(),
                )
                db.add(new_prediction)

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
        .where(Game.date >= now, Game.date < end)
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
                "tip_off": game.date,
                "pick": pick,
                "trust": trust,
                "line": line,
            }
        )

    return items
