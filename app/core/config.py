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

from datetime import date, datetime
from zoneinfo import ZoneInfo

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def season_for_date(day: date) -> str:
    """
    Return the NBA season string (e.g. '2026-2027') that *day* belongs to.

    A new season is considered to start on 1 September, which covers
    training camp and preseason games.
    """
    start_year = day.year if day.month >= 9 else day.year - 1
    return f"{start_year}-{start_year + 1}"


def normalize_database_url(url: str) -> str:
    """
    Force PostgreSQL URLs onto the asyncpg driver.

    Render (and most hosts) hand out ``postgres://`` or ``postgresql://``
    URLs, which SQLAlchemy would map to the synchronous psycopg2 driver.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    APP_ENV: str = "development"
    """Runtime environment label ('development', 'staging', 'production')."""

    ADMIN_API_KEY: str = ""
    """
    Secret required in the ``X-API-Key`` header for admin endpoints such as
    POST /predictions/refresh.  When empty, admin endpoints are disabled.
    """

    CORS_ORIGINS: str = "*"
    """Comma-separated list of allowed browser origins ('*' for any)."""

    # ------------------------------------------------------------------ #
    # API-Sports credentials & endpoints
    # ------------------------------------------------------------------ #
    API_SPORTS_KEY: str = ""
    """Secret API key for api-sports.io — set this in your .env file."""

    API_SPORTS_BASE_URL: str = "https://v1.basketball.api-sports.io"
    """Base URL for the API-Sports Basketball v1 endpoint."""

    API_SPORTS_MAX_REQUESTS_PER_MINUTE: int = 10
    """Client-side throttle (the Free plan allows 10/min). 0 disables it."""

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite+aiosqlite:///./5star.db"
    """
    Async SQLAlchemy database URL.
    • SQLite  (default) : sqlite+aiosqlite:///./5star.db
    • PostgreSQL        : postgresql://user:pass@host/dbname
      (rewritten to postgresql+asyncpg:// automatically)
    """

    # ------------------------------------------------------------------ #
    # NBA / League parameters
    # ------------------------------------------------------------------ #
    NBA_LEAGUE_ID: int = 12
    """API-Sports internal ID for the NBA league."""

    NBA_SEASON: str = ""
    """
    Active NBA season string as expected by the API (e.g. '2026-2027').
    Leave empty to derive it from today's date in LEAGUE_TIMEZONE.
    """

    LEAGUE_TIMEZONE: str = "America/New_York"
    """
    Timezone that defines a game "day".  NBA slates are scheduled in US
    Eastern time; using UTC would split one evening's games across two days.
    """

    # ------------------------------------------------------------------ #
    # Prediction display
    # ------------------------------------------------------------------ #
    TOP_PREDICTIONS_LIMIT: int = 5
    """Default number of top predictions returned by the /top endpoint."""

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

    @field_validator("DATABASE_URL")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        return normalize_database_url(value)

    @model_validator(mode="after")
    def _default_season(self) -> "Settings":
        if not self.NBA_SEASON:
            self.NBA_SEASON = season_for_date(self.league_today())
        return self

    @property
    def league_tz(self) -> ZoneInfo:
        return ZoneInfo(self.LEAGUE_TIMEZONE)

    def league_today(self) -> date:
        """Today's date in the league timezone."""
        return datetime.now(self.league_tz).date()

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of instantiating
# Settings yourself so the .env file is only parsed once.
# ---------------------------------------------------------------------------
settings: Settings = Settings()
