"""
app/ingestion/api_sports.py
---------------------------
Async HTTP client wrapping the API-Sports Basketball v1 API.

All public methods return parsed Python objects (list[dict] or dict | None).
They never raise on API-level errors — failures are logged and an empty
result is returned so the ingestion pipeline can continue gracefully.

Usage
-----
    from app.ingestion.api_sports import api_sports_client

    games = await api_sports_client.get_todays_games(league_id=12, season="2025-2026")

The module exports a module-level singleton ``api_sports_client`` so the
``httpx.AsyncClient`` connection pool is shared across all callers.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bet365 bookmaker ID on API-Sports
_BET365_BOOKMAKER_ID: int = 6

# Default timeout (seconds) for all API-Sports requests
_REQUEST_TIMEOUT: float = 20.0


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------


class ApiSportsClient:
    """
    Thin async wrapper around the API-Sports Basketball v1 REST API.

    The underlying ``httpx.AsyncClient`` is created lazily on the first
    request and reused for all subsequent calls within the same process
    lifetime, giving us connection-pool reuse without holding a client
    open during import time.
    """

    def __init__(self) -> None:
        # Lazily initialised in _client()
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        """Return (or create) the shared async HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=settings.API_SPORTS_BASE_URL,
                headers={
                    "x-apisports-key": settings.API_SPORTS_KEY,
                    "Accept": "application/json",
                },
                timeout=_REQUEST_TIMEOUT,
            )
        return self._http

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict | None:
        """
        Execute a GET request against *endpoint* with *params*.

        Returns
        -------
        dict | None
            The parsed JSON body on success, or ``None`` if the request
            failed or the body could not be decoded.
        """
        url = endpoint  # base_url handles the host
        try:
            response = await self._client().get(url, params=params)
            response.raise_for_status()
            data: dict = response.json()

            # API-Sports signals server-side errors via the `errors` field
            # even when HTTP 200 is returned.
            errors = data.get("errors")
            if errors:
                logger.warning(
                    "API-Sports returned errors for %s (params=%s): %s",
                    endpoint,
                    params,
                    errors,
                )
                return None

            return data

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP %s error fetching %s (params=%s): %s",
                exc.response.status_code,
                endpoint,
                params,
                exc,
            )
        except httpx.RequestError as exc:
            logger.error(
                "Network error fetching %s (params=%s): %s",
                endpoint,
                params,
                exc,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Unexpected error fetching %s (params=%s)", endpoint, params
            )

        return None

    @staticmethod
    def _response_list(data: dict | None) -> list[dict]:
        """
        Extract the ``response`` array from an API-Sports payload.

        Returns an empty list when *data* is ``None`` or the key is absent /
        not a list.
        """
        if data is None:
            return []
        resp = data.get("response", [])
        return resp if isinstance(resp, list) else []

    @staticmethod
    def _response_object(data: dict | None) -> dict | None:
        """
        Extract the ``response`` object from an API-Sports payload where
        the response is a single dict (not a list).

        Returns ``None`` when unavailable.
        """
        if data is None:
            return None
        resp = data.get("response")
        return resp if isinstance(resp, dict) else None

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def get_todays_games(
        self,
        league_id: int,
        season: str,
    ) -> list[dict]:
        """
        Fetch all games scheduled for **today** (calendar date, UTC).

        Parameters
        ----------
        league_id:
            API-Sports internal league identifier (e.g. 12 for NBA).
        season:
            Season string such as ``'2025-2026'``.

        Returns
        -------
        list[dict]
            Raw game objects from the API ``response`` array.  Empty list on
            any error.
        """
        today_str = date.today().isoformat()  # YYYY-MM-DD
        logger.debug(
            "Fetching today's games: league=%d, season=%s, date=%s",
            league_id,
            season,
            today_str,
        )
        data = await self._get(
            "/games",
            params={"date": today_str, "league": league_id, "season": season},
        )
        games = self._response_list(data)
        logger.info("get_todays_games returned %d game(s).", len(games))
        return games

    async def get_upcoming_games(
        self,
        league_id: int,
        season: str,
    ) -> list[dict]:
        """
        Fetch all games in the upcoming 24-hour window.

        The window is built by combining today's and tomorrow's schedules so
        we never miss a game that tips off around midnight UTC.  Duplicates
        (games that appear in both responses) are deduplicated by ``id``.

        Parameters
        ----------
        league_id:
            API-Sports internal league identifier.
        season:
            Season string such as ``'2025-2026'``.

        Returns
        -------
        list[dict]
            Deduplicated raw game objects.  Empty list on error.
        """
        today_str = date.today().isoformat()
        tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

        logger.debug(
            "Fetching upcoming games: league=%d, season=%s, today=%s, tomorrow=%s",
            league_id,
            season,
            today_str,
            tomorrow_str,
        )

        today_data, tomorrow_data = (
            await self._get(
                "/games",
                params={"date": today_str, "league": league_id, "season": season},
            ),
            await self._get(
                "/games",
                params={"date": tomorrow_str, "league": league_id, "season": season},
            ),
        )

        today_games = self._response_list(today_data)
        tomorrow_games = self._response_list(tomorrow_data)

        # Deduplicate by API-Sports game id
        seen: set[int] = set()
        combined: list[dict] = []
        for g in today_games + tomorrow_games:
            gid = g.get("id")
            if gid not in seen:
                seen.add(gid)
                combined.append(g)

        logger.info("get_upcoming_games returned %d unique game(s).", len(combined))
        return combined

    async def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: str,
    ) -> dict | None:
        """
        Fetch season-level statistics for a single team.

        Parameters
        ----------
        team_id:
            API-Sports team identifier.
        league_id:
            API-Sports league identifier.
        season:
            Season string such as ``'2025-2026'``.

        Returns
        -------
        dict | None
            The ``response`` object from the API, or ``None`` on error.
        """
        logger.debug(
            "Fetching team stats: team_id=%d, league=%d, season=%s",
            team_id,
            league_id,
            season,
        )
        data = await self._get(
            "/teams/statistics",
            params={"team": team_id, "league": league_id, "season": season},
        )
        stats = self._response_object(data)
        if stats is None:
            logger.warning(
                "No statistics returned for team_id=%d, league=%d, season=%s.",
                team_id,
                league_id,
                season,
            )
        return stats

    async def get_game_odds(self, game_id: int) -> list[dict]:
        """
        Fetch odds for a single game from Bet365 (bookmaker id = 6).

        Parameters
        ----------
        game_id:
            API-Sports game identifier.

        Returns
        -------
        list[dict]
            Raw odds objects from the ``response`` array.  Empty list on error.
        """
        logger.debug("Fetching odds: game_id=%d, bookmaker=%d", game_id, _BET365_BOOKMAKER_ID)
        data = await self._get(
            "/odds",
            params={"game": game_id, "bookmaker": _BET365_BOOKMAKER_ID},
        )
        odds = self._response_list(data)
        logger.debug("get_game_odds returned %d entry/entries for game %d.", len(odds), game_id)
        return odds

    async def get_h2h(self, team1_id: int, team2_id: int) -> list[dict]:
        """
        Fetch the head-to-head history between two teams.

        Parameters
        ----------
        team1_id:
            API-Sports identifier for the first team.
        team2_id:
            API-Sports identifier for the second team.

        Returns
        -------
        list[dict]
            Raw game objects from the ``response`` array.  Empty list on error.
        """
        h2h_param = f"{team1_id}-{team2_id}"
        logger.debug("Fetching H2H: %s", h2h_param)
        data = await self._get("/games", params={"h2h": h2h_param})
        games = self._response_list(data)
        logger.info(
            "get_h2h (%s) returned %d game(s).", h2h_param, len(games)
        )
        return games

    async def get_standings(self, league_id: int, season: str) -> list[dict]:
        """
        Fetch league standings for a given season.

        Parameters
        ----------
        league_id:
            API-Sports league identifier.
        season:
            Season string such as ``'2025-2026'``.

        Returns
        -------
        list[dict]
            Raw standings objects from the ``response`` array.  Empty list on
            error.
        """
        logger.debug(
            "Fetching standings: league=%d, season=%s", league_id, season
        )
        data = await self._get(
            "/standings",
            params={"league": league_id, "season": season},
        )
        standings = self._response_list(data)
        logger.info(
            "get_standings (league=%d, season=%s) returned %d entry/entries.",
            league_id,
            season,
            len(standings),
        )
        return standings

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client and release pooled connections."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            logger.debug("ApiSportsClient HTTP client closed.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared client instance.  Import and use this directly — do NOT
#: instantiate ``ApiSportsClient`` yourself, or you will get a separate
#: connection pool per import site.
api_sports_client: ApiSportsClient = ApiSportsClient()

