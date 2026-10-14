"""
app/core/config.py
------------------
Central configuration for the 5Star Predictions application.

Uses pydantic-settings so every field can be overridden via a `.env` file
or real environment variables (env-vars take precedence over the file).

Usage:
    from app.core.config import settings
    print(settings.API_SPORTS_KEY)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    APP_ENV: str = "development"
    """Runtime environment label ('development', 'staging', 'production')."""

    # ------------------------------------------------------------------ #
    # API-Sports credentials & endpoints
    # ------------------------------------------------------------------ #
    API_SPORTS_KEY: str = ""
    """Secret API key for api-sports.io — set this in your .env file."""

    API_SPORTS_BASE_URL: str = "https://v1.basketball.api-sports.io"
    """Base URL for the API-Sports Basketball v1 endpoint."""

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite+aiosqlite:///./5star.db"
    """
    Async SQLAlchemy database URL.
    • SQLite  (default) : sqlite+aiosqlite:///./5star.db
    • PostgreSQL        : postgresql+asyncpg://user:pass@host/dbname
    """

    # ------------------------------------------------------------------ #
    # NBA / League parameters
    # ------------------------------------------------------------------ #
    NBA_LEAGUE_ID: int = 12
    """API-Sports internal ID for the NBA league."""

    NBA_SEASON: str = "2025-2026"
    """Active NBA season string as expected by the API (e.g. '2025-2026')."""

    # ------------------------------------------------------------------ #
    # Prediction display
    # ------------------------------------------------------------------ #
    TOP_PREDICTIONS_LIMIT: int = 5
    """Maximum number of top predictions to surface in the /top endpoint."""

    # ------------------------------------------------------------------ #
    # Pydantic-settings config
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        # Look for a .env file starting from the project root
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra keys present in the .env file instead of raising
        extra="ignore",
        # Case-insensitive env-var matching (MY_VAR == my_var)
        case_sensitive=False,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of instantiating
# Settings yourself so the .env file is only parsed once.
# ---------------------------------------------------------------------------
settings: Settings = Settings()
