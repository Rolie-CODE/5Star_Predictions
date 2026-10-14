"""
app/ml/features.py
------------------
Feature engineering pipeline for 5Star Predictions.

Given a Game record and its associated TeamStats, this module produces
a flat feature dictionary that is consumed by both the training pipeline
and the live inference engine.  The column order defined in FEATURE_COLUMNS
**must** remain stable — changing it requires retraining all models.
"""

from __future__ import annotations

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

async def build_feature_vector(
    db: AsyncSession,
    game: Game,
) -> dict | None:
    """
    Build a feature dictionary for *game*.

    Returns ``None`` if TeamStats are missing for either team, which means
    the game cannot be used for training or inference.

    Parameters
    ----------
    db:
        An active async SQLAlchemy session.
    game:
        The Game ORM instance to featurise.  Must have ``home_team_id``,
        ``away_team_id``, ``season``, ``ou_line``, and ``ht_ou_line``
        attributes.

    Returns
    -------
    dict | None
        Flat feature dictionary whose keys exactly match ``FEATURE_COLUMNS``,
        or ``None`` when stats data is unavailable.
    """
    # ------------------------------------------------------------------
    # 1. Load team stats for the current season
    # ------------------------------------------------------------------
    home_stats = await _fetch_team_stats(db, game.home_team_id, game.season)
    away_stats = await _fetch_team_stats(db, game.away_team_id, game.season)

    if home_stats is None or away_stats is None:
        # Cannot build a meaningful feature vector without both stat rows.
        return None

    # ------------------------------------------------------------------
    # 2. Extract raw per-team values
    # ------------------------------------------------------------------
    home_ppg: float = float(home_stats.ppg)
    home_oppg: float = float(home_stats.oppg)
    home_win_pct: float = float(home_stats.win_pct)
    home_home_win_pct: float = float(home_stats.home_win_pct)

    away_ppg: float = float(away_stats.ppg)
    away_oppg: float = float(away_stats.oppg)
    away_win_pct: float = float(away_stats.win_pct)
    away_away_win_pct: float = float(away_stats.away_win_pct)

    # ------------------------------------------------------------------
    # 3. Derived features
    # ------------------------------------------------------------------
    home_net_rating: float = home_ppg - home_oppg
    away_net_rating: float = away_ppg - away_oppg

    ppg_diff: float = home_ppg - away_ppg
    oppg_diff: float = home_oppg - away_oppg
    net_rating_diff: float = home_net_rating - away_net_rating
    win_pct_diff: float = home_win_pct - away_win_pct

    # Baseline expected combined score — used as a naive O/U reference.
    expected_total: float = home_ppg + away_ppg

    # O/U lines — default to 0.0 when not yet set by the sportsbook.
    ou_line: float = float(game.ou_line) if game.ou_line is not None else 0.0
    ht_ou_line: float = float(game.ht_ou_line) if game.ht_ou_line is not None else 0.0

    # ------------------------------------------------------------------
    # 4. Assemble ordered feature dict
    # ------------------------------------------------------------------
    features: dict = {
        "home_ppg": home_ppg,
        "home_oppg": home_oppg,
        "home_win_pct": home_win_pct,
        "home_home_win_pct": home_home_win_pct,
        "away_ppg": away_ppg,
        "away_oppg": away_oppg,
        "away_win_pct": away_win_pct,
        "away_away_win_pct": away_away_win_pct,
        "ppg_diff": ppg_diff,
        "oppg_diff": oppg_diff,
        "home_net_rating": home_net_rating,
        "away_net_rating": away_net_rating,
        "net_rating_diff": net_rating_diff,
        "win_pct_diff": win_pct_diff,
        "expected_total": expected_total,
        "ou_line": ou_line,
        "ht_ou_line": ht_ou_line,
    }

    # Sanity-check: ensure every canonical column is present.
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise RuntimeError(
            f"build_feature_vector: missing keys {missing}. "
            "Update FEATURE_COLUMNS or the feature dict to stay in sync."
        )

    return features
