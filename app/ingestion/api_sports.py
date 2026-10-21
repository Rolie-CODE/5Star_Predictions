"""
app/ingestion/api_sports.py
---------------------------
Async HTTP client wrapping the API-Sports Basketball v1 API.

All public methods return parsed Python objects (list[dict]).  They never
raise on API-level errors — failures are logged and an empty result is
returned so the ingestion pipeline can continue gracefully.

Usage
-----
    from app.ingestion.api_sports import api_sports_client

    games = await api_sports_client.get_games_on_date(date(2026, 10, 21))

The module exports a module-level singleton ``api_sports_client`` so the
``httpx.AsyncClient`` connection pool (and the rate limiter) is shared
across all callers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
        # Monotonic timestamps of recent requests, for the per-minute limit.
        self._recent_requests: list[float] = []
        self._rate_lock = asyncio.Lock()

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

    async def _wait_for_rate_limit(self) -> None:
        """Sleep until another request fits in the per-minute budget."""
        limit = settings.API_SPORTS_MAX_REQUESTS_PER_MINUTE
        if limit <= 0:
            return
        async with self._rate_lock:
            while True:
                now = time.monotonic()
                self._recent_requests = [t for t in self._recent_requests if now - t < 60]
                if len(self._recent_requests) < limit:
                    self._recent_requests.append(now)
                    return
                await asyncio.sleep(60 - (now - self._recent_requests[0]) + 0.1)

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict | None:
        """
        Execute a GET request against *endpoint* with *params*.

        Returns
        -------
        dict | None
            The parsed JSON body on success, or ``None`` if the request
            failed or the body could not be decoded.
        """
        await self._wait_for_rate_limit()
        try:
            response = await self._client().get(endpoint, params=params)
            response.raise_for_status()
            data: dict = response.json()

            # API-Sports signals server-side errors via the `errors` field
            # even when HTTP 200 is returned (e.g. plan restrictions).
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

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def get_games_on_date(self, day: date, league_id: int) -> list[dict]:
        """
        Fetch all *league_id* games on *day* (a league-local calendar date).

        The request deliberately omits ``league``/``season``: the Free plan
        rejects current seasons when they are passed, but allows a date-only
        query (for yesterday → tomorrow).  Results are filtered client-side.
        The ``timezone`` parameter makes the date boundaries follow
        ``settings.LEAGUE_TIMEZONE`` instead of UTC.
        """
        data = await self._get(
            "/games",
            params={"date": day.isoformat(), "timezone": settings.LEAGUE_TIMEZONE},
        )
        games = [
            g for g in self._response_list(data)
            if (g.get("league") or {}).get("id") == league_id
        ]
        logger.info("get_games_on_date(%s) returned %d game(s).", day, len(games))
        return games

    async def get_season_games(self, league_id: int, season: str) -> list[dict]:
        """
        Fetch every game of a season in a single request.

        Paid plans only for the current season; on the Free plan this
        returns an empty list (the plan error is logged).
        """
        data = await self._get(
            "/games",
            params={
                "league": league_id,
                "season": season,
                "timezone": settings.LEAGUE_TIMEZONE,
            },
        )
        games = self._response_list(data)
        logger.info(
            "get_season_games(league=%d, season=%s) returned %d game(s).",
            league_id,
            season,
            len(games),
        )
        return games

    async def get_game_odds(self, game_id: int) -> list[dict]:
        """
        Fetch odds from every bookmaker for a single game.

        Returns the raw ``response`` array (one entry per game, each with a
        ``bookmakers`` list).  Empty list on error.
        """
        data = await self._get("/odds", params={"game": game_id})
        odds = self._response_list(data)
        logger.debug("get_game_odds returned %d entry/entries for game %d.", len(odds), game_id)
        return odds

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
