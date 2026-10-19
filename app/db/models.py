"""
app/db/models.py
----------------
SQLAlchemy ORM models for the 5Star Predictions project.

Models
------
  Team        — NBA franchise information sourced from API-Sports.
  Game        — Individual matchup with scores and O/U lines.
  TeamStats   — Season-aggregate statistics per team.
  Prediction  — Model-generated picks and confidence scores for a game.

All timestamps default to the current UTC time at the moment of row
insertion.  Foreign-key relationships are declared bidirectionally so
SQLAlchemy can eager-load related objects when needed.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared base class for all ORM models."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
class Team(Base):
    """
    Represents an NBA franchise.

    Populated / refreshed by the data-ingestion service from the
    API-Sports /teams endpoint.
    """

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    api_sports_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True,
        comment="Unique team identifier assigned by API-Sports.",
    )

    name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        comment="Full franchise name, e.g. 'Los Angeles Lakers'.",
    )

    abbreviation: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        comment="Short ticker-style code, e.g. 'LAL'.",
    )

    conference: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="'East' or 'West'.",
    )

    division: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
        comment="Division name, e.g. 'Pacific'.",
    )

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    home_games: Mapped[list["Game"]] = relationship(
        "Game",
        foreign_keys="Game.home_team_id",
        back_populates="home_team",
        lazy="select",
    )

    away_games: Mapped[list["Game"]] = relationship(
        "Game",
        foreign_keys="Game.away_team_id",
        back_populates="away_team",
        lazy="select",
    )

    stats: Mapped[list["TeamStats"]] = relationship(
        "TeamStats",
        back_populates="team",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Team id={self.id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game(Base):
    """
    A single NBA game (past, live, or upcoming).

    Score fields are ``None`` until the game is completed.
    ``ou_line`` and ``ht_ou_line`` are populated from a sportsbook odds feed.
    """

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    api_sports_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True,
        comment="Unique game identifier assigned by API-Sports.",
    )

    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Scheduled tip-off time (UTC).",
    )

    # ------------------------------------------------------------------ #
    # Team foreign keys
    # ------------------------------------------------------------------ #
    home_team_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("teams.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    away_team_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("teams.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------ #
    # Game state
    # ------------------------------------------------------------------ #
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="scheduled",
        comment="One of: 'scheduled', 'live', 'finished'.",
    )

    season: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Season string, e.g. '2025-2026'.",
    )

    # ------------------------------------------------------------------ #
    # Scores
    # ------------------------------------------------------------------ #
    home_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Final home team score (populated after game ends).",
    )

    away_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Final away team score (populated after game ends).",
    )

    home_halftime_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Home team score at the end of the first half.",
    )

    away_halftime_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Away team score at the end of the first half.",
    )

    # ------------------------------------------------------------------ #
    # O/U lines
    # ------------------------------------------------------------------ #
    ou_line: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Sportsbook full-game over/under points total.",
    )

    ht_ou_line: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Sportsbook first-half over/under points total.",
    )

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    home_team: Mapped["Team"] = relationship(
        "Team",
        foreign_keys=[home_team_id],
        back_populates="home_games",
        lazy="joined",
    )

    away_team: Mapped["Team"] = relationship(
        "Team",
        foreign_keys=[away_team_id],
        back_populates="away_games",
        lazy="joined",
    )

    prediction: Mapped["Prediction | None"] = relationship(
        "Prediction",
        back_populates="game",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Game id={self.id} "
            f"home={self.home_team_id} "
            f"away={self.away_team_id} "
            f"date={self.date.date()}>"
        )


# ---------------------------------------------------------------------------
# TeamStats
# ---------------------------------------------------------------------------
class TeamStats(Base):
    """
    Season-level aggregate statistics for a single team.

    A unique row per (team_id, season) pair is enforced at the DB level.
    The ingestion service upserts these rows whenever it fetches fresh data.
    """

    __tablename__ = "team_stats"

    __table_args__ = (
        UniqueConstraint("team_id", "season", name="uq_team_stats_team_season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    team_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    season: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Season string matching Game.season, e.g. '2025-2026'.",
    )

    # ------------------------------------------------------------------ #
    # Win / loss record
    # ------------------------------------------------------------------ #
    games_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ------------------------------------------------------------------ #
    # Scoring averages
    # ------------------------------------------------------------------ #
    ppg: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Points per game (offensive average).",
    )

    oppg: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Opponent points per game allowed (defensive average).",
    )

    # ------------------------------------------------------------------ #
    # Home / away splits
    # ------------------------------------------------------------------ #
    home_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    home_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ------------------------------------------------------------------ #
    # Shooting percentages (optional — may not always be available)
    # ------------------------------------------------------------------ #
    fg_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Field-goal percentage (0–100 scale).",
    )

    three_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Three-point percentage (0–100 scale).",
    )

    ft_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Free-throw percentage (0–100 scale).",
    )

    # ------------------------------------------------------------------ #
    # Audit
    # ------------------------------------------------------------------ #
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="Timestamp of the last ingestion update (UTC).",
    )

    @property
    def win_pct(self) -> float:
        """Season win rate as a fraction in [0, 1]."""
        return self.wins / self.games_played if self.games_played else 0.0

    @property
    def home_win_pct(self) -> float:
        """Home win rate as a fraction in [0, 1]."""
        total = self.home_wins + self.home_losses
        return self.home_wins / total if total else 0.0

    @property
    def away_win_pct(self) -> float:
        """Away win rate as a fraction in [0, 1]."""
        total = self.away_wins + self.away_losses
        return self.away_wins / total if total else 0.0

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    team: Mapped["Team"] = relationship(
        "Team",
        back_populates="stats",
        lazy="joined",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TeamStats team_id={self.team_id} "
            f"season={self.season!r} "
            f"record={self.wins}-{self.losses}>"
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class Prediction(Base):
    """
    Model-generated predictions for a single game.

    One prediction row per game (enforced by the unique constraint on
    ``game_id``). Trust values are confidence percentages in [0, 100].
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    game_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="One-to-one link back to the game being predicted.",
    )

    # ------------------------------------------------------------------ #
    # Winner pick
    # ------------------------------------------------------------------ #
    winner_pick: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="Predicted winner side: 'Home' or 'Away'.",
    )

    winner_trust: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Model confidence for winner pick, in [0, 100].",
    )

    # ------------------------------------------------------------------ #
    # Full-time over/under pick
    # ------------------------------------------------------------------ #
    ft_ou_pick: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="Full-game O/U pick: 'Over' or 'Under'.",
    )

    ft_ou_trust: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Model confidence for the full-game O/U pick, in [0, 100].",
    )

    # ------------------------------------------------------------------ #
    # Half-time over/under pick
    # ------------------------------------------------------------------ #
    ht_ou_pick: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="First-half O/U pick: 'Over' or 'Under'.",
    )

    ht_ou_trust: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Model confidence for the first-half O/U pick, in [0, 100].",
    )

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    model_version: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        comment="Identifier for the model/algorithm version that created this row.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="Timestamp when the prediction was generated (UTC).",
    )

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    game: Mapped["Game"] = relationship(
        "Game",
        back_populates="prediction",
        lazy="joined",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Prediction id={self.id} "
            f"game_id={self.game_id} "
            f"winner={self.winner_pick!r} "
            f"ft_ou={self.ft_ou_pick!r} "
            f"ht_ou={self.ht_ou_pick!r}>"
        )
