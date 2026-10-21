"""
app/api/schemas.py
------------------
Pydantic v2 response schemas for the 5Star Predictions API.

All models use `model_config = ConfigDict(from_attributes=True)` so they
can be constructed directly from SQLAlchemy ORM instances as well as plain
dicts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Prediction sub-schemas
# ---------------------------------------------------------------------------


class WinnerPrediction(BaseModel):
    """Predicted winner for a game."""

    model_config = ConfigDict(from_attributes=True)

    pick: str = Field(
        ...,
        description="Predicted winning side: 'Home' or 'Away'.",
        examples=["Home", "Away"],
    )
    trust: float = Field(
        ...,
        description="Calibrated probability of the pick, as a percentage (50–100).",
        examples=[73.4],
        ge=0.0,
        le=100.0,
    )


class OUPrediction(BaseModel):
    """Over/Under prediction for a game (full-time or half-time)."""

    model_config = ConfigDict(from_attributes=True)

    pick: str = Field(
        ...,
        description="Predicted outcome: 'Over' or 'Under'.",
        examples=["Over", "Under"],
    )
    line: float | None = Field(
        default=None,
        description="The sportsbook total line this prediction is relative to.",
        examples=[219.5],
    )
    trust: float = Field(
        ...,
        description="Calibrated probability of the pick, as a percentage (50–100).",
        examples=[61.2],
        ge=0.0,
        le=100.0,
    )


class GamePrediction(BaseModel):
    """Composite prediction block attached to a single game."""

    model_config = ConfigDict(from_attributes=True)

    winner: WinnerPrediction = Field(..., description="Winner prediction.")
    full_time_ou: OUPrediction | None = Field(
        default=None,
        description="Full-time Over/Under prediction, or null when no sportsbook line was available.",
    )
    half_time_ou: OUPrediction | None = Field(
        default=None,
        description="Half-time Over/Under prediction, or null while no validated half-time model exists.",
    )


# ---------------------------------------------------------------------------
# Top-level game response schemas
# ---------------------------------------------------------------------------


class GameResponse(BaseModel):
    """Full game record, optionally enriched with a prediction."""

    model_config = ConfigDict(from_attributes=True)

    game_id: int = Field(..., description="Internal primary-key of the game.")
    api_sports_id: int = Field(
        ..., description="Upstream API-Sports fixture ID."
    )
    home_team: str = Field(..., description="Display name of the home team.")
    away_team: str = Field(..., description="Display name of the away team.")
    tip_off: datetime = Field(
        ..., description="Scheduled tip-off time (UTC)."
    )
    status: str = Field(
        ...,
        description="Game status: 'scheduled', 'live', 'finished', 'postponed' or 'cancelled'.",
    )
    prediction: GamePrediction | None = Field(
        default=None,
        description="Model prediction, or null if not yet computed.",
    )


# ---------------------------------------------------------------------------
# Top-picks schemas
# ---------------------------------------------------------------------------


class TopPickItem(BaseModel):
    """A single high-confidence pick returned in the top-picks endpoint."""

    model_config = ConfigDict(from_attributes=True)

    game_id: int = Field(..., description="Internal primary-key of the game.")
    home_team: str = Field(..., description="Display name of the home team.")
    away_team: str = Field(..., description="Display name of the away team.")
    tip_off: datetime = Field(..., description="Scheduled tip-off time (UTC).")
    pick: str = Field(
        ...,
        description="Predicted outcome (e.g. 'Home', 'Away', 'Over', 'Under').",
    )
    trust: float = Field(
        ...,
        description="Calibrated probability of the pick, as a percentage (50–100).",
        ge=0.0,
        le=100.0,
    )
    line: float | None = Field(
        default=None,
        description="Sportsbook O/U line — only present for Over/Under picks.",
    )


class TopPicksResponse(BaseModel):
    """Envelope returned by the /predictions/top endpoint."""

    model_config = ConfigDict(from_attributes=True)

    prediction_type: str = Field(
        ...,
        description="The prediction category: 'winner', 'ft_ou', or 'ht_ou'.",
        examples=["winner", "ft_ou", "ht_ou"],
    )
    top_picks: list[TopPickItem] = Field(
        ..., description="Ordered list of top picks, highest trust first."
    )


# ---------------------------------------------------------------------------
# Health check schema
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Returned by the /health endpoint."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(
        ...,
        description="Overall service health: 'ok' or 'degraded'.",
        examples=["ok", "degraded"],
    )
    environment: str = Field(..., description="Value of APP_ENV.")
    models_loaded: bool = Field(
        ..., description="True if ML models are loaded and ready."
    )
    db_connected: bool = Field(
        ..., description="True if the database connection is healthy."
    )
