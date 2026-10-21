"""
app/ml/features.py
------------------
Feature engineering pipeline for 5Star Predictions.

Training (train.py) and live inference both build features through
``TeamForm`` and ``features_from_forms`` so the two can never drift apart.
The column order defined in FEATURE_COLUMNS **must** remain stable —
changing it requires retraining all models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Game, TeamStats

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------
# The order here is canonical.  Every downstream consumer (train.py,
# predict.py) must iterate over this list to build model inputs.

FEATURE_COLUMNS: list[str] = [
    "home_ppg",
    "home_oppg",
    "home_win_pct",
    "home_home_win_pct",
    "away_ppg",
    "away_oppg",
    "away_win_pct",
    "away_away_win_pct",
    "ppg_diff",
    "oppg_diff",
    "home_net_rating",
    "away_net_rating",
    "net_rating_diff",
    "win_pct_diff",
    "expected_total",
    "ou_line",
    "ht_ou_line",
]

# Head-specific inputs match what the corresponding historical sources contain.
WINNER_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS[:15]
FT_OU_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS[:16]
HT_OU_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS

# Values used for a team that has not played yet this season.
BASELINE_PPG = 110.0
BASELINE_WIN_PCT = 0.5


# ---------------------------------------------------------------------------
# Regular-season windows
# ---------------------------------------------------------------------------
# Team form only counts regular-season games so preseason and playoff
# results never leak into the stats.  Dates are in US Eastern time.

REGULAR_SEASON_WINDOWS: dict[str, tuple[date, date]] = {
    "2022-2023": (date(2022, 10, 18), date(2023, 4, 9)),
    "2023-2024": (date(2023, 10, 24), date(2024, 4, 14)),
    "2024-2025": (date(2024, 10, 22), date(2025, 4, 13)),
    "2025-2026": (date(2025, 10, 21), date(2026, 4, 12)),
    # End date is an estimate until the league publishes the full schedule.
    "2026-2027": (date(2026, 10, 20), date(2027, 4, 11)),
}


def is_regular_season(season: str, day: date) -> bool:
    """Return True if *day* (league-local date) is in *season*'s regular season."""
    window = REGULAR_SEASON_WINDOWS.get(season)
    if window is None:
        # Unknown season: approximate with mid-October to mid-April.
        start_year = int(season[:4])
        window = (date(start_year, 10, 15), date(start_year + 1, 4, 16))
    return window[0] <= day <= window[1]


# ---------------------------------------------------------------------------
# Team form
# ---------------------------------------------------------------------------


@dataclass
class TeamForm:
    """Season-to-date record for one team, built from completed games."""

    games: int = 0
    wins: int = 0
    points_for: int = 0
    points_against: int = 0
    home_games: int = 0
    home_wins: int = 0
    away_games: int = 0
    away_wins: int = 0

    def add_game(self, points_for: int, points_against: int, is_home: bool) -> None:
        won = points_for > points_against
        self.games += 1
        self.wins += won
        self.points_for += points_for
        self.points_against += points_against
        if is_home:
            self.home_games += 1
            self.home_wins += won
        else:
            self.away_games += 1
            self.away_wins += won

    @property
    def ppg(self) -> float:
        return self.points_for / self.games if self.games else BASELINE_PPG

    @property
    def oppg(self) -> float:
        return self.points_against / self.games if self.games else BASELINE_PPG

    @property
    def win_pct(self) -> float:
        return self.wins / self.games if self.games else BASELINE_WIN_PCT

    @property
    def home_win_pct(self) -> float:
        return self.home_wins / self.home_games if self.home_games else BASELINE_WIN_PCT

    @property
    def away_win_pct(self) -> float:
        return self.away_wins / self.away_games if self.away_games else BASELINE_WIN_PCT

    @classmethod
    def from_stats(cls, stats: TeamStats | None) -> TeamForm:
        """Rebuild a form from a persisted TeamStats row (None → no games yet)."""
        if stats is None or not stats.games_played:
            return cls()
        return cls(
            games=stats.games_played,
            wins=stats.wins,
            points_for=round(stats.ppg * stats.games_played),
            points_against=round(stats.oppg * stats.games_played),
            home_games=stats.home_wins + stats.home_losses,
            home_wins=stats.home_wins,
            away_games=stats.away_wins + stats.away_losses,
            away_wins=stats.away_wins,
        )


def features_from_forms(
    home: TeamForm,
    away: TeamForm,
    ou_line: float | None,
    ht_ou_line: float | None,
) -> dict:
    """
    Build the flat feature dict for a matchup.

    Missing sportsbook lines stay ``None``; the O/U heads are only run when
    their line is present, because the models never saw games without one.
    """
    home_net_rating = home.ppg - home.oppg
    away_net_rating = away.ppg - away.oppg
    return {
        "home_ppg": home.ppg,
        "home_oppg": home.oppg,
        "home_win_pct": home.win_pct,
        "home_home_win_pct": home.home_win_pct,
        "away_ppg": away.ppg,
        "away_oppg": away.oppg,
        "away_win_pct": away.win_pct,
        "away_away_win_pct": away.away_win_pct,
        "ppg_diff": home.ppg - away.ppg,
        "oppg_diff": home.oppg - away.oppg,
        "home_net_rating": home_net_rating,
        "away_net_rating": away_net_rating,
        "net_rating_diff": home_net_rating - away_net_rating,
        "win_pct_diff": home.win_pct - away.win_pct,
        # Baseline expected combined score — used as a naive O/U reference.
        "expected_total": home.ppg + away.ppg,
        "ou_line": ou_line,
        "ht_ou_line": ht_ou_line,
    }


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

async def _fetch_team_stats(
    db: AsyncSession,
    team_id: int,
    season: str,
) -> TeamStats | None:
    """Return the TeamStats row for *team_id* in *season*, or None."""
    result = await db.execute(
        select(TeamStats).where(
            TeamStats.team_id == team_id,
            TeamStats.season == season,
        )
    )
    return result.scalars().first()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_feature_vector(db: AsyncSession, game: Game) -> dict:
    """
    Build a feature dictionary for *game*.

    A team without a TeamStats row yet (no completed games this season)
    gets the same baseline values the training pipeline uses for a team's
    first game.

    Returns
    -------
    dict
        Flat feature dictionary whose keys exactly match ``FEATURE_COLUMNS``.
    """
    home_stats = await _fetch_team_stats(db, game.home_team_id, game.season)
    away_stats = await _fetch_team_stats(db, game.away_team_id, game.season)

    return features_from_forms(
        TeamForm.from_stats(home_stats),
        TeamForm.from_stats(away_stats),
        game.ou_line,
        game.ht_ou_line,
    )
