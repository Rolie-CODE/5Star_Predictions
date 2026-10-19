"""Train NBA prediction models from verified historical game data.

Run from the project root with ``python -m app.ml.train``.  The winner model
uses Kaggle and API-Sports results.  The full-game totals model uses Kaggle
games with an observed sportsbook total.  No synthetic fallback is used.

Kaggle's historical totals file has full-game lines only, so this script keeps
the existing halftime model untouched and marks it as legacy/unvalidated in
metadata.  It must not be presented as a model trained on real halftime lines.
"""

from __future__ import annotations

import asyncio
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from app.ml.features import (
    FEATURE_COLUMNS,
    FT_OU_FEATURE_COLUMNS,
    WINNER_FEATURE_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "app" / "ml" / "models"
KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle_nba"
API_SPORTS_FILE = PROJECT_ROOT / "data" / "api_sports_nba_historical_2022_2025.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
MIN_TRAINING_ROWS = 100
BASELINE_PPG = 110.0

# The live feature builder returns all 17 values.  Each model only consumes
# features that were available in its corresponding real historical source.
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
    date: datetime
    season: str
    home_id: str
    away_id: str
    home_score: int
    away_score: int
    ou_line: float | None = None


@dataclass
class TeamForm:
    games: int = 0
    wins: int = 0
    points_for: int = 0
    points_against: int = 0
    home_games: int = 0
    home_wins: int = 0
    away_games: int = 0
    away_wins: int = 0


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
            if row.get("status") not in {"FT", "AOT", "POST"}:
                continue
            try:
                records.append(
                    GameRecord(
                        # Normalize timestamps so they sort alongside Kaggle's
                        # date-only values without mixing aware/naive objects.
                        date=datetime.fromisoformat(row["date"]).replace(tzinfo=None),
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


def _features_for_records(
    records: list[GameRecord],
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    """Build leak-safe pregame rolling stats and head-specific samples."""
    forms: dict[tuple[str, str], TeamForm] = defaultdict(TeamForm)
    winner_x: list[list[float]] = []
    winner_y: list[int] = []
    totals_x: list[list[float]] = []
    totals_y: list[int] = []

    for game in sorted(records, key=lambda item: (item.date, item.season, item.home_id)):
        home = forms[(game.season, game.home_id)]
        away = forms[(game.season, game.away_id)]

        home_ppg = home.points_for / home.games if home.games else BASELINE_PPG
        home_oppg = home.points_against / home.games if home.games else BASELINE_PPG
        away_ppg = away.points_for / away.games if away.games else BASELINE_PPG
        away_oppg = away.points_against / away.games if away.games else BASELINE_PPG
        home_win_pct = home.wins / home.games if home.games else 0.5
        home_home_win_pct = home.home_wins / home.home_games if home.home_games else 0.5
        away_win_pct = away.wins / away.games if away.games else 0.5
        away_away_win_pct = away.away_wins / away.away_games if away.away_games else 0.5

        expected_total = home_ppg + away_ppg
        values = {
            "home_ppg": home_ppg,
            "home_oppg": home_oppg,
            "home_win_pct": home_win_pct,
            "home_home_win_pct": home_home_win_pct,
            "away_ppg": away_ppg,
            "away_oppg": away_oppg,
            "away_win_pct": away_win_pct,
            "away_away_win_pct": away_away_win_pct,
            "ppg_diff": home_ppg - away_ppg,
            "oppg_diff": home_oppg - away_oppg,
            "home_net_rating": home_ppg - home_oppg,
            "away_net_rating": away_ppg - away_oppg,
            "net_rating_diff": (home_ppg - home_oppg) - (away_ppg - away_oppg),
            "win_pct_diff": home_win_pct - away_win_pct,
            "expected_total": expected_total,
            "ou_line": game.ou_line if game.ou_line is not None else 0.0,
            "ht_ou_line": 0.0,
        }
        winner_x.append([float(values[name]) for name in WINNER_FEATURES])
        winner_y.append(int(game.home_score > game.away_score))

        if game.ou_line is not None and game.home_score + game.away_score != game.ou_line:
            totals_x.append([float(values[name]) for name in FT_OU_FEATURES])
            totals_y.append(int(game.home_score + game.away_score > game.ou_line))

        home.games += 1
        away.games += 1
        home.points_for += game.home_score
        home.points_against += game.away_score
        away.points_for += game.away_score
        away.points_against += game.home_score
        if game.home_score > game.away_score:
            home.wins += 1
            home.home_wins += 1
        else:
            away.wins += 1
            away.away_wins += 1
        home.home_games += 1
        away.away_games += 1

    return winner_x, winner_y, totals_x, totals_y


def _fit_temporal(name: str, X: list[list[float]], y: list[int]) -> tuple[XGBClassifier, float]:
    if len(y) < MIN_TRAINING_ROWS:
        raise RuntimeError(f"Only {len(y)} real rows are available for {name}; need {MIN_TRAINING_ROWS}.")
    split = int(len(y) * 0.8)
    if split < MIN_TRAINING_ROWS or len(set(y[:split])) < 2 or len(set(y[split:])) < 2:
        raise RuntimeError(f"Not enough examples from both classes for chronological {name} evaluation.")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(np.asarray(X[:split]), np.asarray(y[:split]), verbose=False)
    accuracy = float(accuracy_score(y[split:], model.predict(np.asarray(X[split:]))))
    print(f"[train] {name}: {len(y)} real samples, chronological holdout accuracy {accuracy:.3f}")
    return model, accuracy


def _save_model(model: XGBClassifier, filename: str) -> None:
    model.save_model(str(MODELS_DIR / filename))
    print(f"[train] Saved {MODELS_DIR / filename}")


async def run_training() -> None:
    """Entry point used by the monthly scheduler."""
    await train_real_data()


async def train_real_data() -> None:
    print("[train] Loading real Kaggle and API-Sports game results")
    totals = _read_totals()
    kaggle_games = sorted(_read_kaggle_games(totals), key=lambda item: (item.date, item.season))
    api_games = sorted(_read_api_sports_games(), key=lambda item: (item.date, item.season))
    print(
        f"[train] Kaggle: {len(kaggle_games)} regular-season results, "
        f"{sum(game.ou_line is not None for game in kaggle_games)} with full-game totals; "
        f"API-Sports: {len(api_games)} completed results"
    )

    # Build each source independently so a team's historical form never
    # crosses datasets or season boundaries.
    kaggle_wx, kaggle_wy, ft_x, ft_y = _features_for_records(kaggle_games)
    api_wx, api_wy, _, _ = _features_for_records(api_games)

    winner_pairs = sorted(
        zip(kaggle_games, kaggle_wx, kaggle_wy), key=lambda row: (row[0].date, row[0].season)
    ) + sorted(zip(api_games, api_wx, api_wy), key=lambda row: (row[0].date, row[0].season))
    winner_pairs.sort(key=lambda row: (row[0].date, row[0].season))
    winner_x = [row[1] for row in winner_pairs]
    winner_y = [row[2] for row in winner_pairs]

    winner_model, winner_accuracy = _fit_temporal("winner_model", winner_x, winner_y)
    ft_model, ft_accuracy = _fit_temporal("ft_ou_model", ft_x, ft_y)
    _save_model(winner_model, "winner_model.json")
    _save_model(ft_model, "ft_ou_model.json")

    metadata = {
        "version": "2.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "real_historical",
        "models": {
            "winner_model": {
                "data_source": "Kaggle NBA game results + API-Sports 2022-2025",
                "samples": len(winner_y),
                "feature_columns": WINNER_FEATURES,
                "holdout_method": "chronological last 20%",
                "holdout_accuracy": round(winner_accuracy * 100, 2),
            },
            "ft_ou_model": {
                "data_source": "Kaggle median sportsbook full-game totals",
                "samples": len(ft_y),
                "feature_columns": FT_OU_FEATURES,
                "holdout_method": "chronological last 20%",
                "holdout_accuracy": round(ft_accuracy * 100, 2),
            },
            "ht_ou_model": {
                "data_source": "legacy synthetic model; no verified historical first-half lines available",
                "validated": False,
                "feature_columns": FEATURE_COLUMNS,
            },
        },
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[train] Updated winner and full-game totals models with real data.")
    print("[train] Halftime totals model remains legacy and unvalidated: no first-half lines in Kaggle data.")


if __name__ == "__main__":
    asyncio.run(train_real_data())
