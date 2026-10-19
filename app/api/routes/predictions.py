"""
app/api/routes/predictions.py
------------------------------
FastAPI router for prediction-specific endpoints.

  GET  /predictions/top     — top-N picks by trust, for a given prediction type
  POST /predictions/refresh — triggers the full pipeline for today's games

Prefix: /predictions
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import TopPickItem, TopPicksResponse
from app.db.session import get_db  # async session dependency
from app.ingestion.data_pipeline import sync_all_team_stats, sync_todays_games
from app.services import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])

# Convenience type alias for the DB dependency
DBSession = Annotated[AsyncSession, Depends(get_db)]

# Valid values for the `type` query parameter
_VALID_PREDICTION_TYPES = {"winner", "ft_ou", "ht_ou"}
_MAX_LIMIT = 20


# ---------------------------------------------------------------------------
# GET /predictions/top
# ---------------------------------------------------------------------------


@router.get(
    "/top",
    response_model=TopPicksResponse,
    summary="Top picks by trust score",
    description=(
        "Returns the top-N model picks for the specified prediction category, "
        "ranked by confidence (trust) descending.  Only upcoming games "
        "(within the next 24 hours) are considered."
    ),
)
async def get_top_predictions(
    db: DBSession,
    type: Annotated[
        str,
        Query(
            alias="type",
            description=(
                "Prediction category to rank. "
                "One of: 'winner', 'ft_ou' (full-time O/U), 'ht_ou' (half-time O/U)."
            ),
            examples=["winner"],
        ),
    ] = "winner",
    limit: Annotated[
        int,
        Query(
            description=f"Number of picks to return (1–{_MAX_LIMIT}).",
            ge=1,
            le=_MAX_LIMIT,
        ),
    ] = 5,
) -> TopPicksResponse:
    """
    Top-N predictions endpoint.

    Validates the ``type`` query parameter and delegates to the service layer.
    The result is wrapped in a ``TopPicksResponse`` envelope.
    """
    # Validate prediction type (belt-and-suspenders — Pydantic won't catch
    # arbitrary string values in a Query param without a custom validator).
    if type not in _VALID_PREDICTION_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid prediction type '{type}'. "
                f"Must be one of: {sorted(_VALID_PREDICTION_TYPES)}."
            ),
        )

    try:
        picks_dicts = await prediction_service.get_top_predictions(
            db, prediction_type=type, limit=limit
        )
    except ValueError as exc:
        # Service layer may also raise ValueError for bad types (defensive)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Error fetching top predictions (type=%s).", type)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve top predictions.",
        )

    top_picks = [TopPickItem.model_validate(p) for p in picks_dicts]

    return TopPicksResponse(prediction_type=type, top_picks=top_picks)


# ---------------------------------------------------------------------------
# POST /predictions/refresh
# ---------------------------------------------------------------------------


@router.post(
    "/refresh",
    summary="Refresh predictions for today's games",
    description=(
        "Triggers the full ML prediction pipeline for every scheduled game today. "
        "Predictions are upserted into the database.  Returns the number of games "
        "that were successfully predicted."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Pipeline completed successfully.",
            "content": {
                "application/json": {
                    "example": {"predictions_made": 8}
                }
            },
        }
    },
)
async def refresh_predictions(db: DBSession) -> dict:
    """
    Kick off the prediction pipeline for today's scheduled games.

    This is an intentionally synchronous (from the client's perspective)
    endpoint — the pipeline runs inline and the response is returned only
    after all predictions have been committed.  For large slates consider
    delegating to a background task queue instead.

    Returns
    -------
    dict
        ``{"predictions_made": <int>}``
    """
    try:
        await sync_todays_games(db)
        await sync_all_team_stats(db)
        count = await prediction_service.run_predictions_for_today(db)
    except Exception:
        logger.exception("Error running prediction pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction pipeline failed. Check server logs for details.",
        )

    logger.info("Prediction refresh completed: %d game(s) predicted.", count)
    return {"predictions_made": count}
