"""
app/ingestion/data_pipeline.py
-------------------------------
Orchestrates fetching data from API-Sports, transforming the raw JSON into
ORM model instances, and persisting (upserting) them into the database.

Public API
----------
    sync_todays_games(db)      → fetch + upsert today's games & their odds
    sync_team_stats(db, tid)   → fetch + upsert stats for one team
    sync_all_team_stats(db)    → iterate all teams in DB, call sync_team_stats

Design notes
------------
* All DB writes use a "select then insert/update" upsert pattern to stay
  compatible with both SQLite and PostgreSQL without relying on
  dialect-specific ON CONFLICT clauses.
* Every public function commits or flushes within its own transaction
  boundary so partial success is possible (e.g. one bad game won't roll
  back the rest).
* The functions accept an ``AsyncSession`` so they compose naturally with
  the FastAPI ``get_db`` dependency, APScheduler jobs, or CLI scripts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Game, Team, TeamStats
from app.ingestion.api_sports import api_sports_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return a timezone-aware UTC datetime for audit timestamps."""
    return datetime.now(timezone.utc)


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
    ``'scheduled'``, ``'live'``, or ``'finished'``.

    API-Sports returns a nested dict ``{"long": "...", "short": "..."}``
    for the ``status`` field.
    """
    if isinstance(raw_status, dict):
        short = raw_status.get("short", "").upper()
    elif isinstance(raw_status, str):
        short = raw_status.upper()
    else:
        return "scheduled"

    if short in {"FT", "AOT", "CANC"}:
        return "finished"
    if short in {"Q1", "Q2", "Q3", "Q4", "HT", "OT", "BT", "LIVE"}:
        return "live"
    return "scheduled"


def _extract_ou_line(odds_response: list[dict]) -> float | None:
    """
    Parse the full-game Over/Under line from an API-Sports odds response.

    Strategy
    --------
    1. Locate the first bookmaker entry (already filtered to Bet365 by the
       client, so index 0 is fine).
    2. Find the bet with ``name == 'Over/Under'``.
    3. Collect all numeric values from the ``values`` list.
    4. Return the value closest to the midpoint of the range — this is
       generally the "main" line rather than an alternate.

    Returns ``None`` if the line cannot be determined.
    """
    if not odds_response:
        return None

    try:
        bookmaker_entry = odds_response[0]
        bookmakers: list[dict] = bookmaker_entry.get("bookmakers", [])
        if not bookmakers:
            return None

        bets: list[dict] = bookmakers[0].get("bets", [])
        ou_bet: dict | None = next(
            (b for b in bets if b.get("name") == "Over/Under"), None
        )
        if ou_bet is None:
            logger.debug("No 'Over/Under' bet found in odds payload.")
            return None

        raw_values: list[dict] = ou_bet.get("values", [])
        # Each value looks like {"value": "Over 220.5", "odd": "1.90"}
        lines: list[float] = []
        for entry in raw_values:
            label: str = str(entry.get("value", ""))
            # Strip "Over " / "Under " prefix and parse the numeric part
            numeric_part = (
                label.replace("Over", "")
                .replace("Under", "")
                .strip()
            )
            val = _safe_float(numeric_part)
            if val is not None:
                lines.append(val)

        if not lines:
            return None

        # Deduplicate and pick the value closest to the midpoint
        unique_lines = sorted(set(lines))
        mid = (unique_lines[0] + unique_lines[-1]) / 2.0
        closest = min(unique_lines, key=lambda x: abs(x - mid))
        return closest

    except Exception:  # noqa: BLE001
        logger.exception("Failed to extract O/U line from odds payload.")
        return None


def _extract_ht_ou_line(odds_response: list[dict]) -> float | None:
    """
    Parse the first-half Over/Under line from the same odds payload.

    Tries the bet named ``'Over/Under (1st Half)'`` or ``'1H Over/Under'``.
    Falls back to ``None`` if neither is present.
    """
    if not odds_response:
        return None

    _ht_bet_names = {"Over/Under (1st Half)", "1H Over/Under", "1st Half Over/Under"}

    try:
        bookmaker_entry = odds_response[0]
        bookmakers: list[dict] = bookmaker_entry.get("bookmakers", [])
        if not bookmakers:
            return None

        bets: list[dict] = bookmakers[0].get("bets", [])
        ht_bet: dict | None = next(
            (b for b in bets if b.get("name") in _ht_bet_names), None
        )
        if ht_bet is None:
            logger.debug("No first-half O/U bet found in odds payload.")
            return None

        raw_values: list[dict] = ht_bet.get("values", [])
        lines: list[float] = []
        for entry in raw_values:
            label: str = str(entry.get("value", ""))
            numeric_part = (
                label.replace("Over", "")
                .replace("Under", "")
                .strip()
            )
            val = _safe_float(numeric_part)
            if val is not None:
                lines.append(val)

        if not lines:
            return None

        unique_lines = sorted(set(lines))
        mid = (unique_lines[0] + unique_lines[-1]) / 2.0
        return min(unique_lines, key=lambda x: abs(x - mid))

    except Exception:  # noqa: BLE001
        logger.exception("Failed to extract first-half O/U line from odds payload.")
        return None


# ---------------------------------------------------------------------------
# Team upsert
# ---------------------------------------------------------------------------


async def _upsert_team(db: AsyncSession, api_id: int, name: str) -> Team:
    """
    Insert a Team row if it doesn't exist, or return the existing one.

    We only set ``name`` on insert; we do not overwrite existing team
    metadata (abbreviation, conference, division) because those may have
    been enriched from other sources.

    Parameters
    ----------
    db:
        Active async database session.
    api_id:
        API-Sports team identifier.
    name:
        Team name from the API.

    Returns
    -------
    Team
        The persisted (or already-existing) ORM instance.
    """
    stmt = select(Team).where(Team.api_sports_id == api_id)
    result = await db.execute(stmt)
    team: Team | None = result.scalar_one_or_none()

    if team is None:
        team = Team(api_sports_id=api_id, name=name)
        db.add(team)
        await db.flush()  # populate team.id without committing
        logger.debug("Inserted new Team: api_sports_id=%d, name=%r", api_id, name)
    else:
        # Always keep the name fresh in case the franchise was renamed.
        if team.name != name:
            team.name = name
            logger.debug(
                "Updated Team name: api_sports_id=%d → %r", api_id, name
            )

    return team


# ---------------------------------------------------------------------------
# Public: sync_todays_games
# ---------------------------------------------------------------------------


async def sync_todays_games(db: AsyncSession) -> int:
    """
    Fetch today's games from API-Sports and upsert them into the database.

    For each game the function also:
    * Upserts the home and away ``Team`` rows.
    * Fetches Bet365 odds and stores the full-game and first-half O/U lines
      on the ``Game`` row.

    Parameters
    ----------
    db:
        Active async database session.  The caller is responsible for the
        outer transaction; this function calls ``flush()`` after each game
        but does **not** ``commit()``.  Call ``await db.commit()`` after
        the function returns if you are outside a ``get_db`` context.

    Returns
    -------
    int
        Number of games successfully upserted (teams + game rows).
    """
    logger.info(
        "sync_todays_games: starting sync for league=%d, season=%s",
        settings.NBA_LEAGUE_ID,
        settings.NBA_SEASON,
    )

    raw_games: list[dict] = await api_sports_client.get_todays_games(
        league_id=settings.NBA_LEAGUE_ID,
        season=settings.NBA_SEASON,
    )

    if not raw_games:
        logger.info("sync_todays_games: no games returned from API.")
        return 0

    count = 0
    for raw in raw_games:
        try:
            # ----------------------------------------------------------------
            # 1. Parse core game fields
            # ----------------------------------------------------------------
            api_game_id: int | None = _safe_int(raw.get("id"))
            if api_game_id is None:
                logger.warning("Skipping game with missing id: %s", raw)
                continue

            # Date / tip-off
            raw_date = raw.get("date")
            if raw_date is None:
                logger.warning("Skipping game %d: missing date.", api_game_id)
                continue

            if isinstance(raw_date, str):
                # API returns ISO-8601 strings such as "2025-11-01T00:30:00+00:00"
                try:
                    tip_off: datetime = datetime.fromisoformat(raw_date)
                    if tip_off.tzinfo is None:
                        tip_off = tip_off.replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.warning(
                        "Could not parse date %r for game %d.", raw_date, api_game_id
                    )
                    continue
            else:
                logger.warning(
                    "Unexpected date type %s for game %d.", type(raw_date), api_game_id
                )
                continue

            # Teams
            teams_block: dict = raw.get("teams", {})
            home_block: dict = teams_block.get("home", {})
            away_block: dict = teams_block.get("away", {})

            home_api_id = _safe_int(home_block.get("id"))
            home_name: str = home_block.get("name", "Unknown")
            away_api_id = _safe_int(away_block.get("id"))
            away_name: str = away_block.get("name", "Unknown")

            if home_api_id is None or away_api_id is None:
                logger.warning("Skipping game %d: missing team ids.", api_game_id)
                continue

            # Scores
            scores_block: dict = raw.get("scores", {})
            home_scores: dict = scores_block.get("home", {})
            away_scores: dict = scores_block.get("away", {})

            home_total = _safe_int(home_scores.get("total"))
            away_total = _safe_int(away_scores.get("total"))

            # Halftime = quarter_1 + quarter_2
            home_q1 = _safe_int(home_scores.get("quarter_1"), 0) or 0
            home_q2 = _safe_int(home_scores.get("quarter_2"), 0) or 0
            away_q1 = _safe_int(away_scores.get("quarter_1"), 0) or 0
            away_q2 = _safe_int(away_scores.get("quarter_2"), 0) or 0
            home_ht = (home_q1 + home_q2) if (home_q1 or home_q2) else None
            away_ht = (away_q1 + away_q2) if (away_q1 or away_q2) else None

            # Status
            status_str = _parse_game_status(raw.get("status"))

            # Season (from API or fall back to settings)
            season_raw = raw.get("league", {}).get("season") or settings.NBA_SEASON
            season_str = str(season_raw)

            # ----------------------------------------------------------------
            # 2. Upsert teams
            # ----------------------------------------------------------------
            home_team = await _upsert_team(db, home_api_id, home_name)
            away_team = await _upsert_team(db, away_api_id, away_name)

            # ----------------------------------------------------------------
            # 3. Upsert game row
            # ----------------------------------------------------------------
            stmt = select(Game).where(Game.api_sports_id == api_game_id)
            result = await db.execute(stmt)
            game: Game | None = result.scalar_one_or_none()

            if game is None:
                game = Game(
                    api_sports_id=api_game_id,
                    date=tip_off,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    status=status_str,
                    season=season_str,
                    home_score=home_total,
                    away_score=away_total,
                    home_halftime_score=home_ht,
                    away_halftime_score=away_ht,
                )
                db.add(game)
                logger.debug(
                    "Inserted Game api_sports_id=%d (%s vs %s).",
                    api_game_id,
                    home_name,
                    away_name,
                )
            else:
                # Refresh mutable fields (scores, status may have changed)
                await db.execute(
                    update(Game)
                    .where(Game.api_sports_id == api_game_id)
                    .values(
                        date=tip_off,
                        status=status_str,
                        home_score=home_total,
                        away_score=away_total,
                        home_halftime_score=home_ht,
                        away_halftime_score=away_ht,
                    )
                )
                logger.debug(
                    "Updated Game api_sports_id=%d (%s vs %s).",
                    api_game_id,
                    home_name,
                    away_name,
                )

            # Flush so game.id is populated for newly inserted rows
            await db.flush()

            # ----------------------------------------------------------------
            # 4. Fetch odds and update O/U lines
            # ----------------------------------------------------------------
            try:
                odds_response = await api_sports_client.get_game_odds(
                    game_id=api_game_id
                )
                ou_line = _extract_ou_line(odds_response)
                ht_ou_line = _extract_ht_ou_line(odds_response)

                if ou_line is not None or ht_ou_line is not None:
                    await db.execute(
                        update(Game)
                        .where(Game.api_sports_id == api_game_id)
                        .values(
                            ou_line=ou_line,
                            ht_ou_line=ht_ou_line,
                        )
                    )
                    await db.flush()
                    logger.debug(
                        "Updated odds for game %d: ou_line=%s, ht_ou_line=%s",
                        api_game_id,
                        ou_line,
                        ht_ou_line,
                    )
                else:
                    logger.debug(
                        "No O/U line available for game %d.", api_game_id
                    )

            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to update odds for game %d; continuing.", api_game_id
                )

            count += 1

        except Exception:  # noqa: BLE001
            logger.exception(
                "Unexpected error processing raw game %s; skipping.", raw.get("id")
            )
            # Roll back the current partial game so we don't leave the
            # session in a broken state, then continue with the next game.
            await db.rollback()
            # Re-establish a clean state for subsequent games
            continue

    await db.commit()
    logger.info(
        "sync_todays_games: committed %d game(s) to the database.", count
    )
    return count


# ---------------------------------------------------------------------------
# Public: sync_team_stats
# ---------------------------------------------------------------------------


async def sync_team_stats(db: AsyncSession, team_api_id: int) -> bool:
    """
    Fetch and upsert season statistics for a single team.

    Parameters
    ----------
    db:
        Active async database session.
    team_api_id:
        The ``Team.api_sports_id`` value (API-Sports integer ID).

    Returns
    -------
    bool
        ``True`` if the upsert succeeded, ``False`` otherwise.
    """
    logger.debug(
        "sync_team_stats: fetching stats for api_sports_id=%d", team_api_id
    )

    # ------------------------------------------------------------------
    # 1. Fetch from API-Sports
    # ------------------------------------------------------------------
    raw_stats: dict | None = await api_sports_client.get_team_statistics(
        team_id=team_api_id,
        league_id=settings.NBA_LEAGUE_ID,
        season=settings.NBA_SEASON,
    )
    if raw_stats is None:
        logger.warning(
            "sync_team_stats: no stats for api_sports_id=%d.", team_api_id
        )
        return False

    try:
        # ------------------------------------------------------------------
        # 2. Parse the API-Sports statistics payload
        #
        # Expected structure (abbreviated):
        # {
        #   "games": {
        #     "played": {"home": 20, "away": 18, "all": 38},
        #     "wins":   {"home": {"total": 15, ...}, "away": {"total": 10, ...}, "all": {"total": 25, ...}},
        #     "loses":  {"home": {"total": 5,  ...}, "away": {"total": 8,  ...}, "all": {"total": 13, ...}}
        #   },
        #   "points": {
        #     "for":     {"average": {"home": "115.2", "away": "109.8", "all": "112.6"}},
        #     "against": {"average": {"home": "108.3", "away": "111.5", "all": "109.8"}}
        #   }
        # }
        # ------------------------------------------------------------------
        games_block: dict = raw_stats.get("games", {})
        wins_block: dict = games_block.get("wins", {})
        loses_block: dict = games_block.get("loses", {})

        games_played = _safe_int(
            games_block.get("played", {}).get("all"), default=0
        ) or 0

        wins_all = _safe_int(
            wins_block.get("all", {}).get("total"), default=0
        ) or 0
        losses_all = _safe_int(
            loses_block.get("all", {}).get("total"), default=0
        ) or 0
        home_wins = _safe_int(
            wins_block.get("home", {}).get("total"), default=0
        ) or 0
        home_losses = _safe_int(
            loses_block.get("home", {}).get("total"), default=0
        ) or 0
        away_wins = _safe_int(
            wins_block.get("away", {}).get("total"), default=0
        ) or 0
        away_losses = _safe_int(
            loses_block.get("away", {}).get("total"), default=0
        ) or 0

        points_block: dict = raw_stats.get("points", {})
        ppg = _safe_float(
            points_block.get("for", {}).get("average", {}).get("all"), default=0.0
        ) or 0.0
        oppg = _safe_float(
            points_block.get("against", {}).get("average", {}).get("all"), default=0.0
        ) or 0.0

        # Shooting percentages — API-Sports field names vary; guard with get()
        fg_pct = _safe_float(
            raw_stats.get("field_goals", {})
            .get("percentage", {})
            .get("all")
        )
        three_pct = _safe_float(
            raw_stats.get("three_points", {})
            .get("percentage", {})
            .get("all")
        )
        ft_pct = _safe_float(
            raw_stats.get("free_throws", {})
            .get("percentage", {})
            .get("all")
        )

        # ------------------------------------------------------------------
        # 3. Resolve our internal Team PK
        # ------------------------------------------------------------------
        team_stmt = select(Team).where(Team.api_sports_id == team_api_id)
        team_result = await db.execute(team_stmt)
        team: Team | None = team_result.scalar_one_or_none()

        if team is None:
            logger.warning(
                "sync_team_stats: Team with api_sports_id=%d not in DB; skipping.",
                team_api_id,
            )
            return False

        # ------------------------------------------------------------------
        # 4. Upsert TeamStats row
        # ------------------------------------------------------------------
        ts_stmt = select(TeamStats).where(
            TeamStats.team_id == team.id,
            TeamStats.season == settings.NBA_SEASON,
        )
        ts_result = await db.execute(ts_stmt)
        ts: TeamStats | None = ts_result.scalar_one_or_none()

        if ts is None:
            ts = TeamStats(
                team_id=team.id,
                season=settings.NBA_SEASON,
                games_played=games_played,
                wins=wins_all,
                losses=losses_all,
                ppg=ppg,
                oppg=oppg,
                home_wins=home_wins,
                home_losses=home_losses,
                away_wins=away_wins,
                away_losses=away_losses,
                fg_pct=fg_pct,
                three_pct=three_pct,
                ft_pct=ft_pct,
                updated_at=_utc_now(),
            )
            db.add(ts)
            logger.debug(
                "Inserted TeamStats for team_id=%d, season=%s.",
                team.id,
                settings.NBA_SEASON,
            )
        else:
            await db.execute(
                update(TeamStats)
                .where(
                    TeamStats.team_id == team.id,
                    TeamStats.season == settings.NBA_SEASON,
                )
                .values(
                    games_played=games_played,
                    wins=wins_all,
                    losses=losses_all,
                    ppg=ppg,
                    oppg=oppg,
                    home_wins=home_wins,
                    home_losses=home_losses,
                    away_wins=away_wins,
                    away_losses=away_losses,
                    fg_pct=fg_pct,
                    three_pct=three_pct,
                    ft_pct=ft_pct,
                    updated_at=_utc_now(),
                )
            )
            logger.debug(
                "Updated TeamStats for team_id=%d, season=%s.",
                team.id,
                settings.NBA_SEASON,
            )

        await db.flush()
        return True

    except Exception:  # noqa: BLE001
        logger.exception(
            "sync_team_stats: failed for api_sports_id=%d.", team_api_id
        )
        return False


# ---------------------------------------------------------------------------
# Public: sync_all_team_stats
# ---------------------------------------------------------------------------


async def sync_all_team_stats(db: AsyncSession) -> int:
    """
    Refresh season statistics for every team currently in the database.

    Teams are queried from the ``teams`` table, so any team that has been
    inserted by ``sync_todays_games`` (or by a prior sync) will have its
    stats updated.

    Parameters
    ----------
    db:
        Active async database session.

    Returns
    -------
    int
        Number of teams whose stats were successfully upserted.
    """
    logger.info("sync_all_team_stats: querying all teams from DB.")

    teams_result = await db.execute(select(Team))
    all_teams: list[Team] = list(teams_result.scalars().all())

    if not all_teams:
        logger.info("sync_all_team_stats: no teams in database; nothing to sync.")
        return 0

    logger.info(
        "sync_all_team_stats: syncing stats for %d team(s).", len(all_teams)
    )

    success_count = 0
    for team in all_teams:
        try:
            ok = await sync_team_stats(db, team_api_id=team.api_sports_id)
            if ok:
                success_count += 1
        except Exception:  # noqa: BLE001
            logger.exception(
                "sync_all_team_stats: unhandled error for team api_sports_id=%d; continuing.",
                team.api_sports_id,
            )

    await db.commit()
    logger.info(
        "sync_all_team_stats: finished — %d / %d team(s) updated.",
        success_count,
        len(all_teams),
    )
    return success_count

