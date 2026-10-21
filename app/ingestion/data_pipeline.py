"""
app/ingestion/data_pipeline.py
-------------------------------
Orchestrates fetching data from API-Sports, transforming the raw JSON into
ORM model instances, and persisting (upserting) them into the database.

Public API
----------
    sync_todays_games(db)      → fetch + upsert today's games & their odds
    sync_recent_results(db)    → refresh yesterday's and today's scores (no odds)
    backfill_season(db)        → fetch every game of the season (paid plans)
    recompute_team_stats(db)   → rebuild TeamStats from finished games in the DB

Design notes
------------
* "Today" is the league-local date (``settings.LEAGUE_TIMEZONE``), so one
  evening's slate is never split across two UTC days.
* Team stats are computed from our own finished-game rows with the same
  ``TeamForm`` logic the training pipeline uses, instead of the API-Sports
  /statistics endpoint (unavailable for current seasons on the Free plan,
  and 30 requests per refresh).
* All DB writes use a "select then insert/update" upsert pattern to stay
  compatible with both SQLite and PostgreSQL.
* Each game is committed on its own, so one bad game only rolls back
  itself, never the games processed before it.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Game, Team, TeamStats
from app.ingestion.api_sports import api_sports_client
from app.ml.features import TeamForm, is_regular_season

logger = logging.getLogger(__name__)

# API-Sports bet ids / names (from the /bets endpoint).
_FT_OU_BET = (4, "Over/Under")
_HT_OU_BET = (5, "Over/Under 1st Half")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return a timezone-aware UTC datetime for audit timestamps."""
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; every stored value is UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _league_date(value: datetime) -> date:
    """Calendar date of *value* in the league timezone."""
    return _as_utc(value).astimezone(settings.league_tz).date()


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Coerce *value* to int, returning *default* on failure."""
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce *value* to float, returning *default* on failure."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_game_status(raw_status: dict | str | None) -> str:
    """
    Normalise the API-Sports game status to one of:
    ``'scheduled'``, ``'live'``, ``'finished'``, ``'postponed'`` or
    ``'cancelled'``.

    API-Sports returns a nested dict ``{"long": "...", "short": "..."}``
    for the ``status`` field.
    """
    if isinstance(raw_status, dict):
        short = (raw_status.get("short") or "").upper()
    elif isinstance(raw_status, str):
        short = raw_status.upper()
    else:
        return "scheduled"

    if short in {"FT", "AOT", "AWD"}:
        return "finished"
    if short in {"Q1", "Q2", "Q3", "Q4", "HT", "OT", "BT", "LIVE"}:
        return "live"
    if short in {"POST", "SUSP"}:
        return "postponed"
    if short in {"CANC", "ABD"}:
        return "cancelled"
    return "scheduled"


def _main_line(values: list[dict]) -> float | None:
    """
    Return one bookmaker's main total from its list of Over/Under values.

    Each value looks like ``{"value": "Over 220.5", "odd": "1.90"}``.  The
    main line is the one whose Over and Under prices are closest to each
    other; alternate lines are priced far from even.
    """
    prices: dict[float, dict[str, float]] = defaultdict(dict)
    for entry in values:
        side, _, number = str(entry.get("value", "")).partition(" ")
        line = _safe_float(number)
        odd = _safe_float(entry.get("odd"))
        if side in {"Over", "Under"} and line is not None and odd is not None:
            prices[line][side] = odd

    paired = {line: p for line, p in prices.items() if len(p) == 2}
    if not paired:
        return None
    return min(paired, key=lambda line: abs(paired[line]["Over"] - paired[line]["Under"]))


def _extract_line(odds_response: list[dict], bet: tuple[int, str]) -> float | None:
    """
    Consensus total for *bet* across every bookmaker in an odds payload.

    Takes each bookmaker's main line and returns the median, mirroring how
    the training data's Kaggle totals were built.
    """
    bet_id, bet_name = bet
    lines: list[float] = []
    try:
        for entry in odds_response:
            for bookmaker in entry.get("bookmakers", []):
                for b in bookmaker.get("bets", []):
                    if b.get("id") == bet_id or b.get("name") == bet_name:
                        line = _main_line(b.get("values", []))
                        if line is not None:
                            lines.append(line)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to extract %r line from odds payload.", bet_name)
        return None
    return float(statistics.median(lines)) if lines else None


def _parse_game(raw: dict) -> dict | None:
    """Turn a raw API-Sports game object into Game column values (or None)."""
    api_game_id = _safe_int(raw.get("id"))
    if api_game_id is None:
        logger.warning("Skipping game with missing id: %s", raw)
        return None

    raw_date = raw.get("date")
    try:
        # e.g. "2026-10-21T19:00:00-04:00" — stored as UTC.
        tip_off = datetime.fromisoformat(raw_date)
    except (TypeError, ValueError):
        logger.warning("Skipping game %d: bad date %r.", api_game_id, raw_date)
        return None
    if tip_off.tzinfo is None:
        tip_off = tip_off.replace(tzinfo=timezone.utc)
    tip_off = tip_off.astimezone(timezone.utc)

    teams_block: dict = raw.get("teams") or {}
    home_block: dict = teams_block.get("home") or {}
    away_block: dict = teams_block.get("away") or {}
    home_api_id = _safe_int(home_block.get("id"))
    away_api_id = _safe_int(away_block.get("id"))
    if home_api_id is None or away_api_id is None:
        logger.warning("Skipping game %d: missing team ids.", api_game_id)
        return None

    scores_block: dict = raw.get("scores") or {}
    home_scores: dict = scores_block.get("home") or {}
    away_scores: dict = scores_block.get("away") or {}

    # Halftime = quarter_1 + quarter_2
    home_q1 = _safe_int(home_scores.get("quarter_1"), 0) or 0
    home_q2 = _safe_int(home_scores.get("quarter_2"), 0) or 0
    away_q1 = _safe_int(away_scores.get("quarter_1"), 0) or 0
    away_q2 = _safe_int(away_scores.get("quarter_2"), 0) or 0

    return {
        "api_sports_id": api_game_id,
        "date": tip_off,
        "home_api_id": home_api_id,
        "home_name": home_block.get("name") or "Unknown",
        "away_api_id": away_api_id,
        "away_name": away_block.get("name") or "Unknown",
        "status": _parse_game_status(raw.get("status")),
        "season": str((raw.get("league") or {}).get("season") or settings.NBA_SEASON),
        "home_score": _safe_int(home_scores.get("total")),
        "away_score": _safe_int(away_scores.get("total")),
        "home_halftime_score": (home_q1 + home_q2) if (home_q1 or home_q2) else None,
        "away_halftime_score": (away_q1 + away_q2) if (away_q1 or away_q2) else None,
    }


# ---------------------------------------------------------------------------
# Upserts
# ---------------------------------------------------------------------------


async def _upsert_team(db: AsyncSession, api_id: int, name: str) -> Team:
    """
    Insert a Team row if it doesn't exist, or return the existing one.

    The name is refreshed on existing rows in case a franchise is renamed;
    other metadata (abbreviation, conference, division) is left alone.
    """
    result = await db.execute(select(Team).where(Team.api_sports_id == api_id))
    team: Team | None = result.scalar_one_or_none()

    if team is None:
        team = Team(api_sports_id=api_id, name=name)
        db.add(team)
        await db.flush()  # populate team.id without committing
        logger.debug("Inserted new Team: api_sports_id=%d, name=%r", api_id, name)
    elif team.name != name:
        team.name = name
        logger.debug("Updated Team name: api_sports_id=%d → %r", api_id, name)

    return team


async def _upsert_game(db: AsyncSession, parsed: dict) -> Game:
    """Insert or refresh a Game row from ``_parse_game`` output."""
    home_team = await _upsert_team(db, parsed["home_api_id"], parsed["home_name"])
    away_team = await _upsert_team(db, parsed["away_api_id"], parsed["away_name"])

    result = await db.execute(
        select(Game).where(Game.api_sports_id == parsed["api_sports_id"])
    )
    game: Game | None = result.scalar_one_or_none()
    if game is None:
        game = Game(
            api_sports_id=parsed["api_sports_id"],
            home_team_id=home_team.id,
            away_team_id=away_team.id,
        )
        db.add(game)

    for column in (
        "date",
        "status",
        "season",
        "home_score",
        "away_score",
        "home_halftime_score",
        "away_halftime_score",
    ):
        setattr(game, column, parsed[column])

    await db.flush()
    return game


async def _update_lines(db: AsyncSession, game: Game) -> None:
    """Fetch odds for *game* and store any lines found (never erases one)."""
    odds_response = await api_sports_client.get_game_odds(game_id=game.api_sports_id)
    ou_line = _extract_line(odds_response, _FT_OU_BET)
    ht_ou_line = _extract_line(odds_response, _HT_OU_BET)
    if ou_line is not None:
        game.ou_line = ou_line
    if ht_ou_line is not None:
        game.ht_ou_line = ht_ou_line
    logger.debug(
        "Odds for game %d: ou_line=%s, ht_ou_line=%s",
        game.api_sports_id,
        ou_line,
        ht_ou_line,
    )


async def _store_games(
    db: AsyncSession,
    raw_games: list[dict],
    *,
    fetch_odds: bool,
) -> int:
    """Upsert *raw_games*, committing each one separately."""
    count = 0
    for raw in raw_games:
        parsed = _parse_game(raw)
        if parsed is None:
            continue
        try:
            game = await _upsert_game(db, parsed)
            # Odds only matter before tip-off; skip the request otherwise.
            if fetch_odds and game.status == "scheduled":
                try:
                    await _update_lines(db, game)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to update odds for game %d; continuing.",
                        parsed["api_sports_id"],
                    )
            await db.commit()
            count += 1
        except Exception:  # noqa: BLE001
            logger.exception(
                "Unexpected error processing game %s; skipping.", parsed["api_sports_id"]
            )
            # Only this game's uncommitted changes are discarded.
            await db.rollback()
    return count


# ---------------------------------------------------------------------------
# Public: game syncs
# ---------------------------------------------------------------------------


async def sync_games(
    db: AsyncSession,
    days: list[date],
    *,
    fetch_odds: bool = True,
) -> int:
    """
    Fetch and upsert every NBA game on each league-local date in *days*.

    Returns
    -------
    int
        Number of games successfully upserted.
    """
    count = 0
    for day in days:
        raw_games = await api_sports_client.get_games_on_date(day, settings.NBA_LEAGUE_ID)
        count += await _store_games(db, raw_games, fetch_odds=fetch_odds)
    logger.info("sync_games(%s): committed %d game(s).", days, count)
    return count


async def sync_todays_games(db: AsyncSession, *, fetch_odds: bool = True) -> int:
    """Fetch today's slate (league-local date), including sportsbook lines."""
    return await sync_games(db, [settings.league_today()], fetch_odds=fetch_odds)


async def sync_recent_results(db: AsyncSession) -> int:
    """
    Refresh scores for yesterday's and today's games (no odds requests).

    Yesterday is included so late games that finish after midnight still
    get their final scores.
    """
    today = settings.league_today()
    return await sync_games(db, [today - timedelta(days=1), today], fetch_odds=False)


async def backfill_season(db: AsyncSession) -> int:
    """
    Load every game of the current season in one request.

    Needs a paid API-Sports plan for the current season; on the Free plan
    it logs the plan error and returns 0, and the database fills up
    day by day through ``sync_recent_results`` instead.
    """
    raw_games = await api_sports_client.get_season_games(
        settings.NBA_LEAGUE_ID, settings.NBA_SEASON
    )
    count = await _store_games(db, raw_games, fetch_odds=False)
    logger.info("backfill_season: committed %d game(s).", count)
    return count


# ---------------------------------------------------------------------------
# Public: team stats
# ---------------------------------------------------------------------------


async def recompute_team_stats(db: AsyncSession, season: str | None = None) -> int:
    """
    Rebuild every team's TeamStats row for *season* from finished games.

    Only regular-season games count, exactly as in training.

    Returns
    -------
    int
        Number of TeamStats rows written.
    """
    season = season or settings.NBA_SEASON
    result = await db.execute(
        select(Game).where(Game.season == season).order_by(Game.date.asc())
    )
    games: list[Game] = list(result.scalars().all())

    forms: dict[int, TeamForm] = defaultdict(TeamForm)
    for game in games:
        # Touch both teams so every team in the season gets a row.
        home, away = forms[game.home_team_id], forms[game.away_team_id]
        if (
            game.status != "finished"
            or game.home_score is None
            or game.away_score is None
            or not is_regular_season(season, _league_date(game.date))
        ):
            continue
        home.add_game(game.home_score, game.away_score, is_home=True)
        away.add_game(game.away_score, game.home_score, is_home=False)

    existing_result = await db.execute(select(TeamStats).where(TeamStats.season == season))
    existing = {ts.team_id: ts for ts in existing_result.scalars().all()}

    for team_id, form in forms.items():
        ts = existing.get(team_id)
        if ts is None:
            ts = TeamStats(team_id=team_id, season=season)
            db.add(ts)
        ts.games_played = form.games
        ts.wins = form.wins
        ts.losses = form.games - form.wins
        ts.ppg = form.points_for / form.games if form.games else 0.0
        ts.oppg = form.points_against / form.games if form.games else 0.0
        ts.home_wins = form.home_wins
        ts.home_losses = form.home_games - form.home_wins
        ts.away_wins = form.away_wins
        ts.away_losses = form.away_games - form.away_wins
        ts.updated_at = _utc_now()

    await db.commit()
    logger.info("recompute_team_stats(%s): %d team(s) updated.", season, len(forms))
    return len(forms)
