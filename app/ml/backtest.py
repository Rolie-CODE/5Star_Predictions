"""Season-by-season (walk-forward) backtest of the prediction models.

Run from the project root with ``python -m app.ml.backtest``.

For every test season the models are rebuilt exactly as ``train.py`` builds
them, using only what was known before that season: fitted on all earlier
seasons except the latest ones, calibrated on those latest seasons, then
scored on the test season.  Results are compared with simple baselines:

* Winner: "always pick the home team" and a basic Elo rating.
* Full-game totals: "always Over" / "always Under", and profit per bet at
  standard -110 prices (52.4% is break-even).

No sportsbook-favourite baseline yet: the training data has no money lines.

Results are printed and written to ``app/ml/models/backtest.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import brier_score_loss

from app.ml import train

FIRST_TEST_SEASON = 2010
# Calibrate on the latest seasons before the test season, adding seasons
# until there are at least this many games (the 2018-19 file is partial).
MIN_CALIBRATION_ROWS = 1000
# Skip test seasons too small to say anything (partial files, the current season).
MIN_TEST_ROWS = 200

# Elo: 1500 average, home-court bonus in rating points, and the share of a
# rating kept over the summer (the rest regresses to the average).
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_CARRYOVER = 0.75

# Profit on a winning 1-unit bet at -110; a loss costs 1 unit.
WIN_PAYOUT = 100 / 110
TRUST_THRESHOLDS = (0.50, 0.55, 0.60)

OUTPUT_FILE = train.MODELS_DIR / "backtest.json"


def season_year(season: str) -> int:
    """'2017-18' and '2017-2018' both start in 2017."""
    return int(season[:4])


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------


class Elo:
    """Pre-game home-win probabilities from a basic Elo rating."""

    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}
        self._seasons: dict[str, str] = {}

    def _rating(self, team: str, season: str) -> float:
        rating = self._ratings.get(team, ELO_START)
        if self._seasons.get(team, season) != season:
            rating = ELO_START + ELO_CARRYOVER * (rating - ELO_START)
        self._seasons[team] = season
        self._ratings[team] = rating
        return rating

    def home_win_prob(self, season: str, home: str, away: str) -> float:
        diff = self._rating(home, season) + ELO_HOME_ADVANTAGE - self._rating(away, season)
        return 1 / (1 + 10 ** (-diff / 400))

    def update(self, season: str, home: str, away: str, home_won: bool) -> None:
        shift = ELO_K * (home_won - self.home_win_prob(season, home, away))
        self._ratings[home] += shift
        self._ratings[away] -= shift


def elo_home_probs(records: list[train.GameRecord]) -> dict[str, float]:
    """Pre-game Elo home-win probability for every game, by game id."""
    elo = Elo()
    probs: dict[str, float] = {}
    for game in sorted(records, key=lambda item: (item.date, item.season, item.home_id)):
        probs[game.game_id] = elo.home_win_prob(game.season, game.home_id, game.away_id)
        elo.update(game.season, game.home_id, game.away_id, game.home_score > game.away_score)
    return probs


# ---------------------------------------------------------------------------
# Samples and walk-forward fits
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    season: int
    x: list[float]
    y: int
    elo_p: float | None = None


def build_samples(records: list[train.GameRecord]) -> tuple[list[Sample], list[Sample]]:
    """(winner samples, totals samples) for one source, with Elo baselines."""
    rows, wx, wy, trows, tx, ty = train._features_for_records(records)
    elo = elo_home_probs(records)
    winner = [
        Sample(season_year(r.season), x, y, elo[r.game_id]) for r, x, y in zip(rows, wx, wy)
    ]
    totals = [Sample(season_year(r.season), x, y) for r, x, y in zip(trows, tx, ty)]
    return winner, totals


def _calibration_seasons(seasons: list[int], test: int, counts: dict[int, int]) -> list[int]:
    """Latest seasons before *test* with at least MIN_CALIBRATION_ROWS games."""
    chosen: list[int] = []
    for season in reversed([s for s in seasons if s < test]):
        chosen.append(season)
        if sum(counts[s] for s in chosen) >= MIN_CALIBRATION_ROWS:
            break
    return chosen


def walk_forward(samples: list[Sample], first_season: int):
    """
    Yield (test season, test samples, calibrated probabilities) for every
    season from *first_season* on, each fitted only on earlier seasons.
    """
    seasons = sorted({s.season for s in samples})
    counts = {season: sum(s.season == season for s in samples) for season in seasons}
    for test in seasons:
        if test < first_season or counts[test] < MIN_TEST_ROWS:
            continue
        cal_seasons = _calibration_seasons(seasons, test, counts)
        fit = [s for s in samples if s.season < min(cal_seasons, default=test)]
        cal = [s for s in samples if s.season in cal_seasons]
        held_out = [s for s in samples if s.season == test]
        if len(fit) < train.MIN_TRAINING_ROWS or len({s.y for s in cal}) < 2:
            continue
        model, calibrator = train.fit_calibrated(
            np.asarray([s.x for s in fit], dtype=float), np.asarray([s.y for s in fit]),
            np.asarray([s.x for s in cal], dtype=float), np.asarray([s.y for s in cal]),
        )
        raw = model.predict_proba(np.asarray([s.x for s in held_out], dtype=float))[:, 1]
        yield test, held_out, calibrator.predict(raw)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == y))


def betting_record(y: np.ndarray, p: np.ndarray, min_trust: float) -> dict:
    """Bets, hit rate and profit per bet when betting picks with trust >= *min_trust*."""
    trust = np.maximum(p, 1 - p)
    picked = trust >= min_trust
    bets = int(picked.sum())
    wins = int(((p >= 0.5) == y)[picked].sum())
    return {
        "min_trust": min_trust,
        "bets": bets,
        "hit_rate": round(wins / bets, 4) if bets else None,
        "roi": round((wins * WIN_PAYOUT - (bets - wins)) / bets, 4) if bets else None,
    }


def winner_metrics(held_out: list[Sample], p: np.ndarray) -> dict:
    y = np.asarray([s.y for s in held_out])
    elo_p = np.asarray([s.elo_p for s in held_out])
    return {
        "games": len(y),
        "model_accuracy": round(accuracy(y, p), 4),
        "model_brier": round(float(brier_score_loss(y, p)), 4),
        "always_home_accuracy": round(float(y.mean()), 4),
        "elo_accuracy": round(accuracy(y, elo_p), 4),
        "elo_brier": round(float(brier_score_loss(y, elo_p)), 4),
    }


def totals_metrics(held_out: list[Sample], p: np.ndarray) -> dict:
    y = np.asarray([s.y for s in held_out])
    over = np.ones(len(y))
    return {
        "games": len(y),
        "model_accuracy": round(accuracy(y, p), 4),
        "model_brier": round(float(brier_score_loss(y, p)), 4),
        "always_over_accuracy": round(float(y.mean()), 4),
        "always_over_roi": betting_record(y, over, 0.5)["roi"],
        "always_under_roi": betting_record(y, 1 - over, 0.5)["roi"],
        "bets": [betting_record(y, p, t) for t in TRUST_THRESHOLDS],
    }


def _evaluate(samples: list[Sample], first_season: int, metrics) -> dict:
    """Per-season metrics plus the same metrics pooled over all test seasons."""
    by_season: dict[str, dict] = {}
    pooled_samples: list[Sample] = []
    pooled_p: list[np.ndarray] = []
    for test, held_out, p in walk_forward(samples, first_season):
        by_season[str(test)] = metrics(held_out, p)
        pooled_samples += held_out
        pooled_p.append(p)
    pooled = metrics(pooled_samples, np.concatenate(pooled_p)) if pooled_p else None
    return {"seasons": by_season, "all_test_seasons": pooled}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "   -  " if value is None else f"{value * 100:5.1f}%"


def _print_winner(result: dict) -> None:
    print("\nWinner model (accuracy; Brier, lower is better)")
    print(f"{'season':>8} {'games':>6} {'model':>7} {'home':>7} {'elo':>7} {'brier':>7} {'elo br':>7}")
    rows = list(result["seasons"].items()) + [("all", result["all_test_seasons"])]
    for season, m in rows:
        if m is None:
            continue
        print(
            f"{season:>8} {m['games']:>6} {_pct(m['model_accuracy'])} "
            f"{_pct(m['always_home_accuracy'])} {_pct(m['elo_accuracy'])} "
            f"{m['model_brier']:>7.4f} {m['elo_brier']:>7.4f}"
        )


def _print_totals(result: dict) -> None:
    print("\nFull-game totals model (profit per 1-unit bet at -110; break-even hit rate 52.4%)")
    header = "".join(f"  >={int(t * 100)}% bets/hit/roi" for t in TRUST_THRESHOLDS)
    print(f"{'season':>8} {'games':>6} {'model':>7} {'over':>7}{header}")
    rows = list(result["seasons"].items()) + [("all", result["all_test_seasons"])]
    for season, m in rows:
        if m is None:
            continue
        bets = "".join(
            f"  {b['bets']:>5} {_pct(b['hit_rate'])} {_pct(b['roi'])}" for b in m["bets"]
        )
        print(
            f"{season:>8} {m['games']:>6} {_pct(m['model_accuracy'])} "
            f"{_pct(m['always_over_accuracy'])}{bets}"
        )


def run_backtest(
    db_games: list[train.GameRecord],
    first_season: int,
    first_train_season: int | None = train.FIRST_TRAINING_SEASON,
) -> dict:
    """CPU-bound; builds every per-season model and returns the report."""
    kaggle_games, api_games = train.load_records(db_games, first_train_season)
    winner: list[Sample] = []
    totals: list[Sample] = []
    for records in (kaggle_games, api_games):
        w, t = build_samples(records)
        winner += w
        totals += t

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "walk-forward by season: fit on earlier seasons, calibrate on the latest "
            f"seasons before the test season (>= {MIN_CALIBRATION_ROWS} games), test on the season"
        ),
        "first_test_season": first_season,
        "first_training_season": first_train_season,
        "winner_model": _evaluate(winner, first_season, winner_metrics),
        "ft_ou_model": _evaluate(totals, first_season, totals_metrics),
    }
    _print_winner(report["winner_model"])
    _print_totals(report["ft_ou_model"])
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[backtest] Wrote {OUTPUT_FILE}")
    return report


async def main(use_db: bool, first_season: int, first_train_season: int | None) -> None:
    db_games = await train.read_db_games_or_none() if use_db else []
    await asyncio.to_thread(run_backtest, db_games, first_season, first_train_season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-db", action="store_true", help="use the CSV files only")
    parser.add_argument(
        "--first-season", type=int, default=FIRST_TEST_SEASON,
        help=f"first season (start year) to test on, default {FIRST_TEST_SEASON}",
    )
    parser.add_argument(
        "--first-train-season", type=int, default=train.FIRST_TRAINING_SEASON,
        help="drop seasons (start year) before this from the data, default "
        f"{train.FIRST_TRAINING_SEASON}",
    )
    args = parser.parse_args()
    asyncio.run(main(not args.no_db, args.first_season, args.first_train_season))
