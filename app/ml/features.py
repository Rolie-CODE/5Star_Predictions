"""
app/ml/features.py
------------------
Feature engineering pipeline for 5Star Predictions.

Training (train.py) and live inference both build features by replaying
finished games through ``LeagueHistory`` and asking it what was known before
each game, so the two can never drift apart.  Live inference starts from the
state training ended in (``carryover.json``) and replays the current season
from the database on top of it.

The column order defined in FEATURE_COLUMNS **must** remain stable —
changing it requires retraining all models.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.models import Game, TeamStats

logger = logging.getLogger(__name__)

CARRYOVER_PATH = Path(__file__).resolve().parent / "models" / "carryover.json"

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------
# The order here is canonical.  Every downstream consumer (train.py,
# predict.py) must iterate over these lists to build model inputs.  Scoring
# features are relative to the league average, because raw totals drift
# from era to era (about 210 points a game in 2018, 230 in 2026) and tree
# models cannot extrapolate past the values they were trained on.

WINNER_FEATURE_COLUMNS: list[str] = [
    # Team strength
    "elo_diff",
    "home_net_rating",
    "away_net_rating",
    "net_rating_diff",
    "home_win_pct",
    "away_win_pct",
    "home_home_win_pct",
    "away_away_win_pct",
    "home_recent_margin",
    "away_recent_margin",
    "home_games",
    "away_games",
    # Schedule
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
]

FT_OU_FEATURE_COLUMNS: list[str] = WINNER_FEATURE_COLUMNS + [
    "home_total_vs_league",
    "away_total_vs_league",
    "ou_line_vs_league",
    "expected_total_vs_line",
]

# No validated half-time model exists; kept so a future one has a slot.
HT_OU_FEATURE_COLUMNS: list[str] = FT_OU_FEATURE_COLUMNS + ["ht_ou_line"]

FEATURE_COLUMNS: list[str] = HT_OU_FEATURE_COLUMNS

# Values used for a team that has not played yet this season.
BASELINE_PPG = 110.0
BASELINE_WIN_PCT = 0.5

# Elo: average rating, home-court bonus in rating points, update size, and
# the share of a rating kept per season (the rest regresses to the average).
ELO_START = 1500.0
ELO_HOME_ADVANTAGE = 100.0
ELO_K = 20.0
ELO_CARRYOVER = 0.75
# Scale each update by the margin of victory (see ``Elo.update``).
ELO_MARGIN_OF_VICTORY = True

# Days of rest counted at most; a team's first game of a season counts as
# fully rested.
REST_CAP = 4
# Weight of the latest game in a team's recent point margin (~last 10 games).
RECENT_WEIGHT = 0.2
# Games played is capped: past this, more games add little reliability.
GAMES_CAP = 20
# Early in a season the league scoring average is blended with last
# season's, as if this many of last season's games were included.
LEAGUE_AVG_PRIOR_GAMES = 100


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


def season_year(season: str) -> int:
    """'2017-18' and '2017-2018' both start in 2017."""
    return int(season[:4])


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


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


class Elo:
    """Team ratings that carry over from season to season."""

    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self.seasons: dict[str, int] = {}

    def rating(self, team: str, season: int) -> float:
        """*team*'s rating in *season*, regressed once per season since it last played."""
        rating = self.ratings.get(team, ELO_START)
        gap = season - self.seasons.get(team, season)
        if gap > 0:
            rating = ELO_START + ELO_CARRYOVER**gap * (rating - ELO_START)
        self.ratings[team] = rating
        self.seasons[team] = season
        return rating

    def home_win_prob(self, season: int, home: str, away: str) -> float:
        diff = self.rating(home, season) + ELO_HOME_ADVANTAGE - self.rating(away, season)
        return 1 / (1 + 10 ** (-diff / 400))

    def update(self, season: int, home: str, away: str, home_margin: int) -> None:
        """Move both ratings after a game the home team won by *home_margin* points."""
        home_won = home_margin > 0
        shift = ELO_K * (home_won - self.home_win_prob(season, home, away))
        if ELO_MARGIN_OF_VICTORY:
            # FiveThirtyEight's NBA multiplier: bigger wins move ratings more,
            # damped when the favourite wins, which is expected anyway.
            winner_edge = (
                self.rating(home, season) + ELO_HOME_ADVANTAGE - self.rating(away, season)
            ) * (1 if home_won else -1)
            shift *= (abs(home_margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_edge)
        self.ratings[home] += shift
        self.ratings[away] -= shift


# ---------------------------------------------------------------------------
# League history
# ---------------------------------------------------------------------------


class LeagueHistory:
    """
    Replays games in chronological order and answers "what was known about
    this matchup before tip-off".  Call ``features`` for a game, then
    ``add_result`` once it is final.
    """

    def __init__(self) -> None:
        self.elo = Elo()
        self._forms: dict[tuple[int, str], TeamForm] = defaultdict(TeamForm)
        self._recent: dict[tuple[int, str], float] = defaultdict(float)
        self._last_day: dict[tuple[int, str], date] = {}
        # Per season: [sum of combined scores, games].
        self._totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        # Season averages carried over from training, by season.
        self._carried_avg_totals: dict[int, float] = {}

    # -- state --------------------------------------------------------------

    def league_avg_total(self, season: int) -> float:
        """Average combined score so far, blended early on with last season's."""
        earlier = [s for s, (_, n) in self._totals.items() if s < season and n]
        earlier += [s for s in self._carried_avg_totals if s < season]
        if earlier:
            prior = self._season_avg(max(earlier))
        else:
            prior = 2 * BASELINE_PPG
        total, games = self._totals.get(season, (0, 0))
        return (prior * LEAGUE_AVG_PRIOR_GAMES + total) / (LEAGUE_AVG_PRIOR_GAMES + games)

    def _season_avg(self, season: int) -> float:
        total, games = self._totals.get(season, (0, 0))
        return total / games if games else self._carried_avg_totals[season]

    def rest_days(self, season: int, day: date, team: str) -> int:
        last = self._last_day.get((season, team))
        return REST_CAP if last is None else min((day - last).days, REST_CAP)

    # -- features -----------------------------------------------------------

    def features(
        self,
        season: int,
        day: date,
        home: str,
        away: str,
        ou_line: float | None,
        ht_ou_line: float | None = None,
        home_market_prob: float | None = None,
    ) -> dict:
        """
        Feature dict for a game on *day* (league-local date).

        Besides the model columns it carries the raw ``ou_line`` and
        ``ht_ou_line`` (``None`` when missing; the O/U heads are only run
        with a line), the sportsbook ``home_market_prob`` (served instead of
        the winner model when present) and ``elo_home_prob``, used as a
        backtest baseline.
        """
        home_form, away_form = self._forms[(season, home)], self._forms[(season, away)]
        avg_total = self.league_avg_total(season)

        def game_total(form: TeamForm) -> float:
            # A team's typical combined score; the league's before it plays.
            return form.ppg + form.oppg if form.games else avg_total

        home_net = home_form.ppg - home_form.oppg if home_form.games else 0.0
        away_net = away_form.ppg - away_form.oppg if away_form.games else 0.0
        home_rest = self.rest_days(season, day, home)
        away_rest = self.rest_days(season, day, away)
        elo_diff = self.elo.rating(home, season) - self.elo.rating(away, season)
        expected_total = (game_total(home_form) + game_total(away_form)) / 2

        return {
            "elo_diff": elo_diff,
            "home_net_rating": home_net,
            "away_net_rating": away_net,
            "net_rating_diff": home_net - away_net,
            "home_win_pct": home_form.win_pct,
            "away_win_pct": away_form.win_pct,
            "home_home_win_pct": home_form.home_win_pct,
            "away_away_win_pct": away_form.away_win_pct,
            "home_recent_margin": self._recent[(season, home)],
            "away_recent_margin": self._recent[(season, away)],
            "home_games": min(home_form.games, GAMES_CAP),
            "away_games": min(away_form.games, GAMES_CAP),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_b2b": int(home_rest == 1),
            "away_b2b": int(away_rest == 1),
            "home_total_vs_league": game_total(home_form) - avg_total,
            "away_total_vs_league": game_total(away_form) - avg_total,
            "ou_line_vs_league": None if ou_line is None else ou_line - avg_total,
            "expected_total_vs_line": None if ou_line is None else expected_total - ou_line,
            "ou_line": ou_line,
            "ht_ou_line": ht_ou_line,
            "home_market_prob": home_market_prob,
            "elo_home_prob": 1 / (1 + 10 ** (-(elo_diff + ELO_HOME_ADVANTAGE) / 400)),
        }

    # -- updates ------------------------------------------------------------

    def note_game(self, season: int, day: date, team: str) -> None:
        """Record that *team* played on *day* (counts for rest; no result)."""
        self._last_day[(season, team)] = day

    def add_result(
        self, season: int, day: date, home: str, away: str, home_score: int, away_score: int
    ) -> None:
        """Fold a final score into every team-level state."""
        self.elo.update(season, home, away, home_score - away_score)
        self._forms[(season, home)].add_game(home_score, away_score, is_home=True)
        self._forms[(season, away)].add_game(away_score, home_score, is_home=False)
        for team, margin in ((home, home_score - away_score), (away, away_score - home_score)):
            key = (season, team)
            self._recent[key] = RECENT_WEIGHT * margin + (1 - RECENT_WEIGHT) * self._recent[key]
            self._last_day[key] = day
        totals = self._totals[season]
        totals[0] += home_score + away_score
        totals[1] += 1

    # -- carryover ------------------------------------------------------------

    def carryover(self, team_ids: dict[str, str] | None = None) -> dict:
        """
        The state a new season starts from: Elo ratings (re-keyed through
        *team_ids* when given; unmapped teams are dropped) and each season's
        league scoring average.
        """
        def key(team: str) -> str | None:
            return team if team_ids is None else team_ids.get(team)

        return {
            "elo": {key(t): r for t, r in self.elo.ratings.items() if key(t) is not None},
            "elo_seasons": {
                key(t): s for t, s in self.elo.seasons.items() if key(t) is not None
            },
            "avg_totals": {str(s): self._season_avg(s) for s, (_, n) in self._totals.items() if n},
        }

    @classmethod
    def from_carryover(cls, state: dict | None) -> LeagueHistory:
        history = cls()
        if state:
            history.elo.ratings = {t: float(r) for t, r in state.get("elo", {}).items()}
            history.elo.seasons = {t: int(s) for t, s in state.get("elo_seasons", {}).items()}
            history._carried_avg_totals = {
                int(s): float(v) for s, v in state.get("avg_totals", {}).items()
            }
        return history


# ---------------------------------------------------------------------------
# Live inference
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_carryover() -> dict | None:
    """The training carryover (keyed by API-Sports team id), or None if missing."""
    if not CARRYOVER_PATH.exists():
        logger.warning("%s not found; Elo ratings start from scratch.", CARRYOVER_PATH)
        return None
    return json.loads(CARRYOVER_PATH.read_text(encoding="utf-8"))


def _as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; every stored value is UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _league_date(value: datetime) -> date:
    return _as_utc(value).astimezone(settings.league_tz).date()


async def build_feature_vectors(db: AsyncSession, games: list[Game]) -> list[dict]:
    """
    Build a feature dict for each of *games*, in the same order.

    Every stored game of the same seasons that tipped off earlier is
    replayed first: finished regular-season games update the state, and
    other games already played count toward rest.
    """
    if not games:
        return []
    seasons = sorted({game.season for game in games})
    result = await db.execute(
        select(Game)
        .where(Game.season.in_(seasons))
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .order_by(Game.date.asc())
    )
    history_games = [
        g for g in result.scalars().all()
        if g.status not in {"postponed", "cancelled"}
        and is_regular_season(g.season, _league_date(g.date))
    ]

    carryover = load_carryover()
    if carryover:
        # Skip exhibitions (All-Star sides are unknown to the carryover),
        # exactly as training drops them.
        known = set(carryover["elo"])
        history_games = [
            g for g in history_games
            if str(g.home_team.api_sports_id) in known and str(g.away_team.api_sports_id) in known
        ]
    history = LeagueHistory.from_carryover(carryover)
    features_by_id: dict[int, dict] = {}
    replayed = 0
    for game in sorted(games, key=lambda g: _as_utc(g.date)):
        tip_off = _as_utc(game.date)
        while replayed < len(history_games) and _as_utc(history_games[replayed].date) < tip_off:
            _replay(history, history_games[replayed])
            replayed += 1
        features_by_id[game.id] = history.features(
            season_year(game.season),
            _league_date(game.date),
            str(game.home_team.api_sports_id),
            str(game.away_team.api_sports_id),
            game.ou_line,
            game.ht_ou_line,
            game.home_win_prob,
        )
    return [features_by_id[game.id] for game in games]


def _replay(history: LeagueHistory, game: Game) -> None:
    season, day = season_year(game.season), _league_date(game.date)
    home, away = str(game.home_team.api_sports_id), str(game.away_team.api_sports_id)
    if game.status == "finished" and game.home_score is not None and game.away_score is not None:
        history.add_result(season, day, home, away, game.home_score, game.away_score)
    else:
        # Started or still to be scored: it still counts as a game played.
        history.note_game(season, day, home)
        history.note_game(season, day, away)


async def build_feature_vector(db: AsyncSession, game: Game) -> dict:
    """Feature dict for a single game (see ``build_feature_vectors``)."""
    return (await build_feature_vectors(db, [game]))[0]
