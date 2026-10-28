"""Train NBA prediction models from verified historical game data.

Run from the project root with ``python -m app.ml.train``.  The winner model
uses Kaggle, the NBA Database (seasons Kaggle lacks, up to 2021-22),
API-Sports and the app's own finished games.  The full-game
totals model uses every game with an observed sportsbook total.  No
synthetic fallback is used.  Pass ``--no-db`` to train from the CSV files
only (used at build time, before a database exists).

All sources are replayed as one history (API-Sports teams are matched to
NBA.com ids by name), so Elo ratings carry over between seasons.  The state
after the last game is saved to ``carryover.json`` for live inference.

Each model is fitted on the oldest 70% of games, calibrated on the next
15%, and scored on the newest 15%, all in chronological order.

The sportsbook money line is not a model input: in the backtest neither a
model given it as a feature nor a blend beat the no-vig money line alone,
so the app serves the market's probability when a game has a line.

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
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from app.core.config import settings
from app.ml.features import (
    CARRYOVER_PATH,
    FT_OU_FEATURE_COLUMNS,
    WINNER_FEATURE_COLUMNS,
    LeagueHistory,
    is_regular_season,
    season_year,
)
from app.ml.predict import Calibrator, logit

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "app" / "ml" / "models"
KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle_nba"
API_SPORTS_FILE = PROJECT_ROOT / "data" / "api_sports_nba_historical_2022_2025.csv"
# Optional: the "NBA Basketball Database" (wyattowalsh/basketball) game table
# and the Kaggle money lines.  Both use the same NBA.com game ids as Kaggle.
NBA_DB_GAMES_FILE = KAGGLE_DIR / "csv" / "game.csv"
MONEY_LINE_FILE = KAGGLE_DIR / "nba_betting_money_line.csv"
# API-Sports covers seasons from this start year on; NBA Database games from
# these seasons are skipped so no game is counted twice.
API_SPORTS_FIRST_SEASON = 2022
# API-Sports teams with fewer games than this in a season are exhibition sides.
MIN_TEAM_SEASON_GAMES = 10

RNG_SEED = 42
MIN_TRAINING_ROWS = 100
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
# Seasons (start year) before this are left out of training; None keeps all.
FIRST_TRAINING_SEASON: int | None = None

WINNER_FEATURES = WINNER_FEATURE_COLUMNS
FT_OU_FEATURES = FT_OU_FEATURE_COLUMNS

# Shallower trees with more, smaller steps: in the walk-forward backtest this
# beat the earlier 300 x depth 4 x 0.05 on accuracy and Brier score.
XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.02,
    "min_child_weight": 20,
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
    # No-vig sportsbook home-win probability (median across books), if known.
    home_market_prob: float | None = None


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


def _read_nba_db_games() -> list[GameRecord]:
    """Regular-season games from the NBA Database before API-Sports' seasons."""
    if not NBA_DB_GAMES_FILE.exists():
        print(f"[train] {NBA_DB_GAMES_FILE} not found; skipping the NBA Database")
        return []

    records: list[GameRecord] = []
    with NBA_DB_GAMES_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("season_type") != "Regular Season":
                continue
            try:
                # season_id "22018" -> the 2018-19 season, in Kaggle's format.
                start_year = int(row["season_id"][1:])
                if start_year >= API_SPORTS_FIRST_SEASON:
                    continue
                records.append(
                    GameRecord(
                        game_id=f"kaggle:{row['game_id']}",
                        date=datetime.fromisoformat(row["game_date"]),
                        season=f"{start_year}-{str(start_year + 1)[-2:]}",
                        home_id=row["team_id_home"],
                        away_id=row["team_id_away"],
                        home_score=int(float(row["pts_home"])),
                        away_score=int(float(row["pts_away"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return records


def _implied_prob(american: float) -> float:
    """Win probability implied by American odds, vig included."""
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def _read_money_lines() -> dict[str, tuple[str, float]]:
    """
    Map NBA.com game id -> (team id, that team's no-vig win probability),
    the median across bookmakers.
    """
    if not MONEY_LINE_FILE.exists():
        print(f"[train] {MONEY_LINE_FILE} not found; no sportsbook win probabilities")
        return {}

    by_game: dict[str, tuple[str, list[float]]] = {}
    with MONEY_LINE_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                team_p = _implied_prob(float(row["price1"]))
                other_p = _implied_prob(float(row["price2"]))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            team_id, probs = by_game.setdefault(row["game_id"], (row["team_id"], []))
            if row["team_id"] != team_id:
                team_p, other_p = other_p, team_p
            probs.append(team_p / (team_p + other_p))
    return {
        game_id: (team_id, float(statistics.median(probs)))
        for game_id, (team_id, probs) in by_game.items()
    }


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
                home_market_prob=game.home_win_prob,
            )
        )
    return records


def _as_naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _record_day(game: GameRecord) -> date:
    """League-local date: Kaggle-format dates already are; API-Sports ones are UTC."""
    return game.date.date() if game.game_id.startswith("kaggle:") else _league_date(game.date)


def pregame_features(
    records: list[GameRecord],
) -> tuple[list[tuple[GameRecord, dict]], LeagueHistory]:
    """
    Leak-safe feature dict for every game, in chronological order, and the
    history state after the last game (the live carryover).
    """
    history = LeagueHistory()
    pairs: list[tuple[GameRecord, dict]] = []
    for game in sorted(records, key=lambda item: (item.date, item.season, item.home_id)):
        season, day = season_year(game.season), _record_day(game)
        pairs.append((game, history.features(season, day, game.home_id, game.away_id, game.ou_line)))
        # Update only after the features were taken (no leakage).
        history.add_result(season, day, game.home_id, game.away_id, game.home_score, game.away_score)
    return pairs, history


def winner_samples(pairs: list[tuple[GameRecord, dict]]) -> tuple[list[list[float]], list[int]]:
    X = [[float(values[name]) for name in WINNER_FEATURES] for _, values in pairs]
    y = [int(game.home_score > game.away_score) for game, _ in pairs]
    return X, y


def totals_pairs(pairs: list[tuple[GameRecord, dict]]) -> list[tuple[GameRecord, dict]]:
    """Games with a sportsbook total that did not land exactly on it."""
    return [
        (game, values) for game, values in pairs
        if game.ou_line is not None and game.home_score + game.away_score != game.ou_line
    ]


def totals_samples(pairs: list[tuple[GameRecord, dict]]) -> tuple[list[list[float]], list[int]]:
    """Expects ``totals_pairs`` output."""
    X = [[float(values[name]) for name in FT_OU_FEATURES] for _, values in pairs]
    y = [int(game.home_score + game.away_score > game.ou_line) for game, _ in pairs]
    return X, y


def fit_sigmoid_calibration(raw_p: np.ndarray, y: np.ndarray) -> Calibrator:
    """
    Platt scaling of raw probabilities.  In the backtest it beat isotonic
    regression on Brier score and log loss, and being smooth it never maps
    different matchups onto one flat step (the source of "50.0%" picks).
    """
    lr = LogisticRegression(C=1e6).fit(np.asarray(logit(raw_p)).reshape(-1, 1), y)
    return Calibrator(
        {"method": "sigmoid", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
    )


def fit_calibrated(
    X_fit: np.ndarray, y_fit: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray
) -> tuple[XGBClassifier, Calibrator]:
    """Fit the classifier, then a map from its raw to calibrated probability."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_fit, y_fit, verbose=False)
    return model, fit_sigmoid_calibration(model.predict_proba(X_cal)[:, 1], y_cal)


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
        "calibration": calibrator.spec,
    }


def _save_model(model: XGBClassifier, filename: str) -> None:
    model.save_model(str(MODELS_DIR / filename))
    print(f"[train] Saved {MODELS_DIR / filename}")


def _api_to_nba_team_ids() -> dict[str, str]:
    """
    API-Sports team id -> NBA.com team id, matched by current team name, so
    one team's history runs unbroken across both sources.  API-Sports teams
    without a match get an id of their own ("api:<id>").
    """
    api_names: dict[str, str] = {}
    with API_SPORTS_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            api_names[row["home_team_id"]] = row["home_team"]
            api_names[row["away_team_id"]] = row["away_team"]

    nba_ids: dict[str, str] = {}
    if NBA_DB_GAMES_FILE.exists():
        with NBA_DB_GAMES_FILE.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                # Later rows win, so current franchise names map to their ids.
                nba_ids[row["team_name_home"]] = row["team_id_home"]
                nba_ids[row["team_name_away"]] = row["team_id_away"]
    return {api_id: nba_ids.get(name, f"api:{api_id}") for api_id, name in api_names.items()}


def load_records(
    db_games: list[GameRecord] | None = None,
    first_season: int | None = None,
) -> tuple[list[GameRecord], dict[str, str]]:
    """
    Every game from every source with NBA.com team ids where known, plus the
    map from those ids back to API-Sports ids (used for the live carryover).
    Kaggle seasons that start before *first_season* are dropped.
    """
    print("[train] Loading real Kaggle, NBA Database and API-Sports game results")
    totals = _read_totals()
    # Same NBA.com game ids: keep each Kaggle game, add the ones it lacks.
    kaggle_by_id = {game.game_id: game for game in _read_kaggle_games(totals)}
    kaggle_count = len(kaggle_by_id)
    for game in _read_nba_db_games():
        kaggle_by_id.setdefault(game.game_id, game)
    nba_db_added = len(kaggle_by_id) - kaggle_count

    money_lines = _read_money_lines()
    for game in kaggle_by_id.values():
        line = money_lines.get(game.game_id.removeprefix("kaggle:"))
        if line is not None:
            team_id, team_p = line
            game.home_market_prob = team_p if team_id == game.home_id else 1 - team_p

    kaggle_games = [
        game for game in kaggle_by_id.values()
        if first_season is None or int(game.season[:4]) >= first_season
    ]

    # API-Sports CSV and app DB share the same game ids; the DB copy wins.
    api_by_id = {game.game_id: game for game in _read_api_sports_games()}
    csv_count = len(api_by_id)
    api_by_id.update({game.game_id: game for game in (db_games or [])})
    api_games = list(api_by_id.values())

    team_ids = _api_to_nba_team_ids()
    for game in api_games:
        # The app database can hold a team the CSV never saw.
        game.home_id = team_ids.setdefault(game.home_id, f"api:{game.home_id}")
        game.away_id = team_ids.setdefault(game.away_id, f"api:{game.away_id}")
    unmatched = sorted(api for api, team in team_ids.items() if team.startswith("api:"))

    # The live server replays the current season on top of the carryover.
    current = season_year(settings.NBA_SEASON)
    before = len(kaggle_games) + len(api_games)
    kaggle_games = [game for game in kaggle_games if season_year(game.season) < current]
    api_games = [game for game in api_games if season_year(game.season) < current]
    current_season_games = before - len(kaggle_games) - len(api_games)

    # API-Sports lists All-Star exhibitions under the NBA during the regular
    # season; their teams play a game or two, real teams about 82.
    games_played: dict[tuple[str, str], int] = defaultdict(int)
    for game in api_games:
        games_played[(game.season, game.home_id)] += 1
        games_played[(game.season, game.away_id)] += 1
    api_count = len(api_games)
    api_games = [
        game for game in api_games
        if games_played[(game.season, game.home_id)] >= MIN_TEAM_SEASON_GAMES
        and games_played[(game.season, game.away_id)] >= MIN_TEAM_SEASON_GAMES
    ]
    exhibitions = api_count - len(api_games)
    records = kaggle_games + api_games
    print(
        f"[train] Kaggle + NBA Database: {len(kaggle_games)} regular-season results "
        f"({nba_db_added} from the NBA Database), "
        f"{sum(game.ou_line is not None for game in kaggle_games)} with full-game totals, "
        f"{sum(game.home_market_prob is not None for game in kaggle_games)} with money lines; "
        f"API-Sports CSV: {csv_count} regular-season results; "
        f"app database: {len(db_games or [])} finished games; "
        f"API-Sports teams without an NBA.com match: {unmatched or 'none'}; "
        f"left out: {exhibitions} exhibition and {current_season_games} current-season games"
    )
    nba_to_api = {team: api for api, team in team_ids.items()}
    return records, nba_to_api


def train_models(db_games: list[GameRecord] | None = None) -> None:
    """CPU-bound training; call via ``asyncio.to_thread`` from async code."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    records, nba_to_api = load_records(db_games, FIRST_TRAINING_SEASON)
    pairs, history = pregame_features(records)

    winner_x, winner_y = winner_samples(pairs)
    ft_x, ft_y = totals_samples(totals_pairs(pairs))

    winner_model, winner_meta = _fit_temporal("winner_model", winner_x, winner_y)
    ft_model, ft_meta = _fit_temporal("ft_ou_model", ft_x, ft_y)
    _save_model(winner_model, "winner_model.json")
    _save_model(ft_model, "ft_ou_model.json")

    # Live inference continues this history in API-Sports team ids.
    CARRYOVER_PATH.write_text(
        json.dumps(history.carryover(nba_to_api), indent=2), encoding="utf-8"
    )
    print(f"[train] Saved {CARRYOVER_PATH}")

    metadata = {
        "version": "5.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "real_historical",
        "models": {
            "winner_model": {
                "data_source": "Kaggle + NBA Database results + API-Sports 2022-2025 + app database",
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
