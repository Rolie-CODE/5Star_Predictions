"""
app/api/routes/predictions.py
------------------------------
FastAPI router for prediction-specific endpoints.

  GET  /predictions/top         — top-N picks by trust, for a given prediction type
  GET  /predictions/performance — stored picks graded against final scores
  POST /predictions/refresh     — triggers the full pipeline for today's games

Prefix: /predictions
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import PerformanceResponse, TopPickItem, TopPicksResponse
from app.core.config import settings
from app.db.session import get_db  # async session dependency
from app.services import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])

# Convenience type alias for the DB dependency
DBSession = Annotated[AsyncSession, Depends(get_db)]

# Valid values for the `type` query parameter
_VALID_PREDICTION_TYPES = {"winner", "ft_ou", "ht_ou"}
_MAX_LIMIT = 20


def require_admin_key(
    x_api_key: Annotated[str | None, Header(description="Value of ADMIN_API_KEY.")] = None,
) -> None:
    """Reject the request unless X-API-Key matches settings.ADMIN_API_KEY."""
    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoints are disabled: ADMIN_API_KEY is not configured.",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


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
    ] = settings.TOP_PREDICTIONS_LIMIT,
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
# GET /predictions/performance
# ---------------------------------------------------------------------------


@router.get(
    "/performance",
    response_model=PerformanceResponse,
    summary="Live accuracy of stored predictions",
    description=(
        "Grades every stored prediction for a season against final scores. "
        "Over/Under picks also report profit per bet at -110 prices."
    ),
)
async def get_performance(
    db: DBSession,
    season: Annotated[
        str | None,
        Query(description="Season to grade, e.g. '2026-2027'. Defaults to the current season."),
    ] = None,
    model_version: Annotated[
        str | None,
        Query(description="Only grade predictions from this model version, e.g. '3.0.0'."),
    ] = None,
) -> PerformanceResponse:
    try:
        result = await prediction_service.get_performance(
            db, season=season or settings.NBA_SEASON, model_version=model_version
        )
    except Exception:
        logger.exception("Error grading predictions.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to grade predictions.",
        )
    return PerformanceResponse.model_validate(result)


# ---------------------------------------------------------------------------
# POST /predictions/refresh
# ---------------------------------------------------------------------------


@router.post(
    "/refresh",
    summary="Refresh predictions for today's games",
    description=(
        "Triggers the full ML prediction pipeline for every scheduled game today. "
        "Predictions are upserted into the database.  Returns the number of games "
        "that were successfully predicted.  Requires the X-API-Key header."
    ),
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin_key)],
    responses={
        200: {
            "description": "Pipeline completed successfully.",
            "content": {
                "application/json": {
                    "example": {"predictions_made": 8}
                }
            },
        },
        401: {"description": "Missing or invalid X-API-Key header."},
        403: {"description": "ADMIN_API_KEY is not configured on the server."},
    },
)
async def refresh_predictions(db: DBSession) -> dict:
    """
    Kick off the prediction pipeline for today's scheduled games.

    This is an intentionally synchronous (from the client's perspective)
    endpoint — the pipeline runs inline and the response is returned only
    after all predictions have been committed.  With the Free-plan rate
    limit (10 requests/minute) a full slate takes one to two minutes.

    Returns
    -------
    dict
        ``{"predictions_made": <int>}``
    """
    try:
        count = await prediction_service.refresh_todays_predictions(db)
    except Exception:
        logger.exception("Error running prediction pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction pipeline failed. Check server logs for details.",
        )

    logger.info("Prediction refresh completed: %d game(s) predicted.", count)
    return {"predictions_made": count}
