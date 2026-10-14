"""
app/api/routes/games.py
-----------------------
FastAPI router exposing game-centric read endpoints.

All routes delegate data retrieval to `prediction_service` and return
responses validated against the `GameResponse` Pydantic schema.

Prefix: /games
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import GameResponse
from app.db.session import get_db  # async session dependency
from app.services import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Games"])

# Convenience type alias for the DB dependency
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# GET /games/today
# ---------------------------------------------------------------------------


@router.get(
    "/today",
    response_model=list[GameResponse],
    summary="Today's games with predictions",
    description=(
        "Returns all games scheduled for today (midnight → midnight UTC) "
        "along with their model predictions, if available."
    ),
)
async def get_today_games(db: DBSession) -> list[GameResponse]:
    """
    Fetch every game on today's slate, joined with their predictions.

    Games are ordered by tip-off time ascending.  A game whose prediction
    has not yet been computed will have `prediction: null` in the response.
    """
    try:
        game_dicts = await prediction_service.get_todays_games_with_predictions(db)
    except Exception:
        logger.exception("Error fetching today's games.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve today's games.",
        )

    return [GameResponse.model_validate(g) for g in game_dicts]


# ---------------------------------------------------------------------------
# GET /games/upcoming
# ---------------------------------------------------------------------------


@router.get(
    "/upcoming",
    response_model=list[GameResponse],
    summary="Upcoming games (next 24 hours) with predictions",
    description=(
        "Returns all games whose tip-off falls within the next 24 hours from "
        "the time of the request, along with their model predictions."
    ),
)
async def get_upcoming_games(db: DBSession) -> list[GameResponse]:
    """
    Fetch upcoming games (now → now+24 h), joined with their predictions.

    Useful for a rolling 'next day' view regardless of calendar boundaries.
    Games are ordered by tip-off time ascending.
    """
    try:
        game_dicts = await prediction_service.get_upcoming_games_with_predictions(db)
    except Exception:
        logger.exception("Error fetching upcoming games.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve upcoming games.",
        )

    return [GameResponse.model_validate(g) for g in game_dicts]


# ---------------------------------------------------------------------------
# GET /games/{game_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{game_id}",
    response_model=GameResponse,
    summary="Single game with prediction",
    description="Returns a single game record by its internal ID, with prediction if available.",
    responses={
        404: {"description": "Game not found."},
    },
)
async def get_game(game_id: int, db: DBSession) -> GameResponse:
    """
    Fetch a game by its internal primary key.

    Returns 404 if no game with that ID exists in the database.
    """
    try:
        game_dict = await prediction_service.get_game_with_prediction(db, game_id)
    except Exception:
        logger.exception("Error fetching game %d.", game_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve game.",
        )

    if game_dict is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with ID {game_id} was not found.",
        )

    return GameResponse.model_validate(game_dict)
