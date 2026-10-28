"""Train NBA prediction models from verified historical game data.

Run from the project root with ``python -m app.ml.train``.  The winner model
uses Kaggle, API-Sports and the app's own finished games.  The full-game
totals model uses every game with an observed sportsbook total.  No
synthetic fallback is used.  Pass ``--no-db`` to train from the CSV files
only (used at build time, before a database exists).

Each model is fitted on the oldest 70% of games, calibrated on the next
15%, and scored on the newest 15%, all in chronological order.

Kaggle's historical totals file has full-game lines only, so no halftime
model is trained here; the API serves no half-time picks until one is
validated.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from app.core.config import settings
from app.ml.features import (
    FT_OU_FEATURE_COLUMNS,
    WINNER_FEATURE_COLUMNS,
    TeamForm,
    features_from_forms,
    is_regular_season,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "app" / "ml" / "models"
KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle_nba"
API_SPORTS_FILE = PROJECT_ROOT / "data" / "api_sports_nba_historical_2022_2025.csv"

RNG_SEED = 42
MIN_TRAINING_ROWS = 100
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15

WINNER_FEATURES = WINNER_FEATURE_COLUMNS
FT_OU_FEATURES = FT_OU_FEATURE_COLUMNS

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RNG_SEED,
    "verbosity": 0,
}


@dataclass
class GameRecord:
    game_id: str
    date: datetime
    season: str
    home_id: str
    away_id: str
    home_score: int
    away_score: int
    ou_line: float | None = None


def _league_date(value: datetime):
    """League-local calendar date of a naive-UTC or aware datetime."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(settings.league_tz).date()


def _read_totals() -> dict[str, float]:
    path = KAGGLE_DIR / "nba_betting_totals.csv"
    if not path.exists():
        raise FileNotFoundError(f"Kaggle totals file is missing: {path}")

    by_game: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                # Usually total1 == total2; average both sides for the few
                # records where the source has a small discrepancy.
                by_game[row["game_id"]].append(
                    (float(row["total1"]) + float(row["total2"])) / 2
                )
            except (KeyError, TypeError, ValueError):
                continue
    return {game_id: float(statistics.median(lines)) for game_id, lines in by_game.items()}


def _read_kaggle_games(totals: dict[str, float]) -> list[GameRecord]:
    path = KAGGLE_DIR / "nba_games_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Kaggle game file is missing: {path}")

    by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("season_type") != "Regular Season" or not row.get("game_date"):
                continue
            by_game[row["game_id"]].append(row)

    records: list[GameRecord] = []
    for game_id, rows in by_game.items():
        if len(rows) != 2:
            continue
        home = next((row for row in rows if row.get("is_home", "").lower() == "t"), None)
        away = next((row for row in rows if row.get("is_home", "").lower() == "f"), None)
        if home is None or away is None:
            continue
        try:
            records.append(
                GameRecord(
                    game_id=f"kaggle:{game_id}",
                    date=datetime.fromisoformat(home["game_date"]),
                    season=home.get("season") or "unknown",
                    home_id=home["team_id"],
                    away_id=away["team_id"],
                    home_score=int(home["pts"]),
                    away_score=int(away["pts"]),
                    ou_line=totals.get(game_id),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return records


def _read_api_sports_games() -> list[GameRecord]:
    if not API_SPORTS_FILE.exists():
        raise FileNotFoundError(f"API-Sports historical file is missing: {API_SPORTS_FILE}")

    records: list[GameRecord] = []
    with API_SPORTS_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in {"FT", "AOT"}:
                continue
            try:
                tip_off = datetime.fromisoformat(row["date"])
                # The file includes preseason and playoff games; keep the
                # regular season only, matching the Kaggle source.
                if not is_regular_season(row["season"], _league_date(tip_off)):
                    continue
                records.append(
                    GameRecord(
                        game_id=f"api:{row['game_id']}",
                        # Naive UTC so they sort alongside Kaggle's
                        # date-only values without mixing aware/naive objects.
                        date=tip_off.astimezone(timezone.utc).replace(tzinfo=None),
                        season=row["season"],
                        home_id=row["home_team_id"],
                        away_id=row["away_team_id"],
                        home_score=int(row["home_score"]),
                        away_score=int(row["away_score"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return records


async def _read_db_games() -> list[GameRecord]:
    """Finished regular-season games collected by the app since deployment."""
    from sqlalchemy import select

    from app.db.models import Game
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Game).where(Game.status == "finished"))
        games = list(result.unique().scalars().all())

    records: list[GameRecord] = []
    for game in games:
        if game.home_score is None or game.away_score is None:
            continue
        if not is_regular_season(game.season, _league_date(game.date)):
            continue
        records.append(
            GameRecord(
                # Same id space as the API-Sports CSV, so duplicates merge.
                game_id=f"api:{game.api_sports_id}",
                date=_as_naive_utc(game.date),
                season=game.season,
                home_id=str(game.home_team.api_sports_id),
                away_id=str(game.away_team.api_sports_id),
                home_score=game.home_score,
                away_score=game.away_score,
                ou_line=game.ou_line,
            )
        )
    return records


def _as_naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _features_for_records(
    records: list[GameRecord],
) -> tuple[list[GameRecord], list[list[float]], list[int], list[GameRecord], list[list[float]], list[int]]:
    """
    Build leak-safe pregame rolling stats and head-specific samples.

    Returns (winner records, X, y, totals records, X, y), each in
    chronological order.
    """
    forms: dict[tuple[str, str], TeamForm] = defaultdict(TeamForm)
    winner_rows: list[GameRecord] = []
    winner_x: list[list[float]] = []
    winner_y: list[int] = []
    totals_rows: list[GameRecord] = []
    totals_x: list[list[float]] = []
    totals_y: list[int] = []

    for game in sorted(records, key=lambda item: (item.date, item.season, item.home_id)):
        home = forms[(game.season, game.home_id)]
        away = forms[(game.season, game.away_id)]
        values = features_from_forms(home, away, game.ou_line, None)

        winner_rows.append(game)
        winner_x.append([float(values[name]) for name in WINNER_FEATURES])
        winner_y.append(int(game.home_score > game.away_score))

        total = game.home_score + game.away_score
        if game.ou_line is not None and total != game.ou_line:
            totals_rows.append(game)
            totals_x.append([float(values[name]) for name in FT_OU_FEATURES])
            totals_y.append(int(total > game.ou_line))

        # Update form only after the features were taken (no leakage).
        home.add_game(game.home_score, game.away_score, is_home=True)
        away.add_game(game.away_score, game.home_score, is_home=False)

    return winner_rows, winner_x, winner_y, totals_rows, totals_x, totals_y


def fit_calibrated(
    X_fit: np.ndarray, y_fit: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray
) -> tuple[XGBClassifier, IsotonicRegression]:
    """Fit the classifier, then an isotonic map from its raw to calibrated probability."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_fit, y_fit, verbose=False)
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(model.predict_proba(X_cal)[:, 1], y_cal)
    return model, calibrator


def _fit_temporal(name: str, X: list[list[float]], y: list[int]) -> tuple[XGBClassifier, dict]:
    """
    Fit on the oldest 70%, calibrate on the next 15%, score on the newest 15%.

    Returns the model and a metadata dict including the calibration map.
    """
    n = len(y)
    if n < MIN_TRAINING_ROWS:
        raise RuntimeError(f"Only {n} real rows are available for {name}; need {MIN_TRAINING_ROWS}.")
    train_end = int(n * TRAIN_FRACTION)
    calib_end = int(n * (TRAIN_FRACTION + CALIBRATION_FRACTION))
    splits = (y[:train_end], y[train_end:calib_end], y[calib_end:])
    if train_end < MIN_TRAINING_ROWS or any(len(set(part)) < 2 for part in splits):
        raise RuntimeError(f"Not enough examples from both classes for chronological {name} evaluation.")

    X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y)
    model, calibrator = fit_calibrated(
        X_arr[:train_end], y_arr[:train_end],
        X_arr[train_end:calib_end], y_arr[train_end:calib_end],
    )

    test_y = y_arr[calib_end:]
    test_p = calibrator.predict(model.predict_proba(X_arr[calib_end:])[:, 1])
    accuracy = float(accuracy_score(test_y, test_p >= 0.5))
    brier = float(brier_score_loss(test_y, test_p))
    print(
        f"[train] {name}: {n} real samples, chronological test accuracy {accuracy:.3f}, "
        f"Brier {brier:.4f}"
    )
    return model, {
        "samples": n,
        "holdout_method": "chronological: fit oldest 70%, calibrate next 15%, test newest 15%",
        "holdout_accuracy": round(accuracy * 100, 2),
        "holdout_brier": round(brier, 4),
        "calibration": {
            "x": [round(float(v), 6) for v in calibrator.X_thresholds_],
            "y": [round(float(v), 6) for v in calibrator.y_thresholds_],
        },
    }


def _save_model(model: XGBClassifier, filename: str) -> None:
    model.save_model(str(MODELS_DIR / filename))
    print(f"[train] Saved {MODELS_DIR / filename}")


def load_records(
    db_games: list[GameRecord] | None = None,
) -> tuple[list[GameRecord], list[GameRecord]]:
    """
    Return (Kaggle games, API-Sports games).  Each source keeps its own team
    ids, so build features for them separately.
    """
    print("[train] Loading real Kaggle and API-Sports game results")
    totals = _read_totals()
    kaggle_games = _read_kaggle_games(totals)

    # API-Sports CSV and app DB share the same game ids; the DB copy wins.
    api_by_id = {game.game_id: game for game in _read_api_sports_games()}
    csv_count = len(api_by_id)
    api_by_id.update({game.game_id: game for game in (db_games or [])})
    api_games = list(api_by_id.values())
    print(
        f"[train] Kaggle: {len(kaggle_games)} regular-season results, "
        f"{sum(game.ou_line is not None for game in kaggle_games)} with full-game totals; "
        f"API-Sports CSV: {csv_count} regular-season results; "
        f"app database: {len(db_games or [])} finished games"
    )
    return kaggle_games, api_games


def train_models(db_games: list[GameRecord] | None = None) -> None:
    """CPU-bound training; call via ``asyncio.to_thread`` from async code."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_games, api_games = load_records(db_games)

    # Build each source independently so a team's historical form never
    # crosses datasets or season boundaries.
    k_rows, k_wx, k_wy, k_trows, k_tx, k_ty = _features_for_records(kaggle_games)
    a_rows, a_wx, a_wy, a_trows, a_tx, a_ty = _features_for_records(api_games)

    def _chronological(rows, xs, ys):
        merged = sorted(zip(rows, xs, ys), key=lambda row: (row[0].date, row[0].season))
        return [row[1] for row in merged], [row[2] for row in merged]

    winner_x, winner_y = _chronological(k_rows + a_rows, k_wx + a_wx, k_wy + a_wy)
    ft_x, ft_y = _chronological(k_trows + a_trows, k_tx + a_tx, k_ty + a_ty)

    winner_model, winner_meta = _fit_temporal("winner_model", winner_x, winner_y)
    ft_model, ft_meta = _fit_temporal("ft_ou_model", ft_x, ft_y)
    _save_model(winner_model, "winner_model.json")
    _save_model(ft_model, "ft_ou_model.json")

    metadata = {
        "version": "3.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "real_historical",
        "models": {
            "winner_model": {
                "data_source": "Kaggle NBA results + API-Sports 2022-2025 + app database",
                "feature_columns": WINNER_FEATURES,
                **winner_meta,
            },
            "ft_ou_model": {
                "data_source": "Kaggle median sportsbook totals + app database consensus lines",
                "feature_columns": FT_OU_FEATURES,
                **ft_meta,
            },
            "ht_ou_model": {
                "data_source": "none — no verified historical first-half lines available",
                "validated": False,
            },
        },
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[train] Updated winner and full-game totals models with real data.")
    print("[train] No half-time model: half-time picks stay disabled.")


async def read_db_games_or_none() -> list[GameRecord]:
    """App database games, or none if the database can't be read."""
    try:
        return await _read_db_games()
    except Exception:  # noqa: BLE001
        logger.exception("Could not read finished games from the database; using CSVs only.")
        return []


async def run_training(use_db: bool = True) -> None:
    """Entry point used by the monthly scheduler and the CLI."""
    db_games = await read_db_games_or_none() if use_db else []
    # Training takes minutes of CPU; keep it off the event loop.
    await asyncio.to_thread(train_models, db_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-db", action="store_true", help="train from the CSV files only")
    args = parser.parse_args()
    asyncio.run(run_training(use_db=not args.no_db))
