"""
app/ml/train.py
---------------
Training script for 5Star Predictions XGBoost models.

Run from the project root with:
    python -m app.ml.train

The script will:
  1. Attempt to load historical Game + Prediction records from the database.
  2. Fall back to synthetic NBA-realistic data when fewer than 100 finished
     games exist (useful during initial development / CI).
  3. Train three XGBClassifier models:
       - winner_model   : predicts home win (1) vs away win (0)
       - ft_ou_model    : predicts full-time Over (1) vs Under (0)
       - ht_ou_model    : predicts halftime Over (1) vs Under (0)
  4. Save each model to app/ml/models/<name>.json (XGBoost native format).
  5. Write app/ml/models/metadata.json with accuracy stats and version info.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Path helpers — allow `python -m app.ml.train` from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/5Star_Predictions
MODELS_DIR = PROJECT_ROOT / "app" / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to sys.path so internal imports resolve correctly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.features import FEATURE_COLUMNS  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Reproducibility seed
# ---------------------------------------------------------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Minimum real-game records needed before we use the DB data
# ---------------------------------------------------------------------------
MIN_REAL_GAMES = 100

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters (conservative defaults; tune as data grows)
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RNG_SEED,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate *n_samples* rows of synthetic NBA-realistic training data.

    Returns
    -------
    X : np.ndarray, shape (n_samples, len(FEATURE_COLUMNS))
    y_winner : np.ndarray, shape (n_samples,)  — 1=home win, 0=away win
    y_ft_ou  : np.ndarray, shape (n_samples,)  — 1=FT over, 0=FT under
    y_ht_ou  : np.ndarray, shape (n_samples,)  — 1=HT over, 0=HT under
    """
    rng = np.random.default_rng(RNG_SEED)

    # --- raw team stats ------------------------------------------------
    home_ppg = rng.uniform(100, 125, n_samples)
    home_oppg = rng.uniform(100, 125, n_samples)
    home_win_pct = rng.uniform(0.2, 0.8, n_samples)
    home_home_win_pct = np.clip(home_win_pct + rng.uniform(0.0, 0.15, n_samples), 0.0, 1.0)

    away_ppg = rng.uniform(100, 125, n_samples)
    away_oppg = rng.uniform(100, 125, n_samples)
    away_win_pct = rng.uniform(0.2, 0.8, n_samples)
    away_away_win_pct = np.clip(away_win_pct - rng.uniform(0.0, 0.15, n_samples), 0.0, 1.0)

    # --- derived features ----------------------------------------------
    home_net_rating = home_ppg - home_oppg
    away_net_rating = away_ppg - away_oppg

    ppg_diff = home_ppg - away_ppg
    oppg_diff = home_oppg - away_oppg
    net_rating_diff = home_net_rating - away_net_rating
    win_pct_diff = home_win_pct - away_win_pct
    expected_total = home_ppg + away_ppg

    # O/U lines: set close to expected total with small noise
    ou_noise = rng.normal(0, 3, n_samples)
    ou_line = expected_total + ou_noise
    ht_ou_noise = rng.normal(0, 2, n_samples)
    ht_ou_line = (expected_total / 2) + ht_ou_noise  # rough halftime estimate

    # --- assemble feature matrix in FEATURE_COLUMNS order --------------
    col_data = {
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
    X = np.column_stack([col_data[c] for c in FEATURE_COLUMNS])

    # --- labels --------------------------------------------------------
    # Winner: home team wins ~58% of the time; net rating + 2 home advantage
    home_advantage = 2.0
    noise = rng.normal(0, 5, n_samples)
    y_winner = ((home_net_rating + home_advantage) > (away_net_rating + noise)).astype(int)

    # Full-time O/U: actual total drawn from normal around expected_total
    actual_total = rng.normal(expected_total, 15, n_samples)
    y_ft_ou = (actual_total > ou_line).astype(int)

    # Halftime O/U: actual HT total roughly half of FT total with some spread
    actual_ht_total = rng.normal(expected_total / 2, 8, n_samples)
    y_ht_ou = (actual_ht_total > ht_ou_line).astype(int)

    return X, y_winner, y_ft_ou, y_ht_ou


# ---------------------------------------------------------------------------
# DB data loader
# ---------------------------------------------------------------------------

async def _load_db_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Attempt to load training data from the application database.

    Returns the same (X, y_winner, y_ft_ou, y_ht_ou) tuple as the synthetic
    generator, or ``None`` when there are insufficient finished games.
    """
    try:
        from sqlalchemy import select
        from app.db.session import AsyncSessionLocal
        from app.db.models import Game, TeamStats, Prediction
    except ImportError as exc:
        print(f"[train] DB import failed ({exc}); will use synthetic data.")
        return None

    async with AsyncSessionLocal() as db:
        # Fetch all finished games with both scores recorded
        result = await db.execute(
            select(Game).where(
                Game.status == "finished",
                Game.home_score.is_not(None),
                Game.away_score.is_not(None),
            )
        )
        games: list[Game] = list(result.scalars().all())

    if len(games) < MIN_REAL_GAMES:
        print(
            f"[train] Only {len(games)} finished games found "
            f"(need {MIN_REAL_GAMES}). Falling back to synthetic data."
        )
        return None

    print(f"[train] Loaded {len(games)} finished games from DB.")

    from app.ml.features import build_feature_vector

    rows: list[list[float]] = []
    y_winner_list: list[int] = []
    y_ft_ou_list: list[int] = []
    y_ht_ou_list: list[int] = []

    async with AsyncSessionLocal() as db:
        for game in games:
            # O/U targets require sportsbook lines and actual halftime scores.
            # Missing values must not be mislabeled as an Under result.
            if any(
                value is None
                for value in (
                    game.ou_line,
                    game.ht_ou_line,
                    game.home_halftime_score,
                    game.away_halftime_score,
                )
            ):
                continue

            fv = await build_feature_vector(db, game)
            if fv is None:
                continue  # missing stats — skip

            row = [fv[c] for c in FEATURE_COLUMNS]
            rows.append(row)

            # Winner label
            y_winner_list.append(1 if game.home_score > game.away_score else 0)

            # Full-time O/U label
            actual_total = game.home_score + game.away_score
            y_ft_ou_list.append(1 if actual_total > game.ou_line else 0)

            # Halftime O/U label
            ht_total = game.home_halftime_score + game.away_halftime_score
            y_ht_ou_list.append(1 if ht_total > game.ht_ou_line else 0)

    if len(rows) < MIN_REAL_GAMES:
        print(
            f"[train] Only {len(rows)} usable feature vectors after filtering. "
            "Falling back to synthetic data."
        )
        return None

    X = np.array(rows, dtype=float)
    return (
        X,
        np.array(y_winner_list, dtype=int),
        np.array(y_ft_ou_list, dtype=int),
        np.array(y_ht_ou_list, dtype=int),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_classifier(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[XGBClassifier, float]:
    """Train a single XGBClassifier and return (model, test_accuracy)."""
    print(f"[train] Training {name} …")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[train] {name} accuracy: {acc:.4f} ({acc * 100:.1f}%)")
    return model, acc


def _save_model(model: XGBClassifier, filename: str) -> None:
    """Save model to the models directory using XGBoost's native JSON format."""
    path = MODELS_DIR / filename
    model.save_model(str(path))
    print(f"[train] Saved {path}")


def _save_metadata(
    winner_acc: float,
    ft_ou_acc: float,
    ht_ou_acc: float,
    source: str,
) -> None:
    """Write training metadata to models/metadata.json."""
    meta = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "feature_columns": FEATURE_COLUMNS,
        "winner_accuracy": round(winner_acc * 100, 2),
        "ft_ou_accuracy": round(ft_ou_acc * 100, 2),
        "ht_ou_accuracy": round(ht_ou_acc * 100, 2),
    }
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

async def main() -> None:
    print("[train] 5Star Predictions — Model Training")
    print(f"[train] Models will be written to: {MODELS_DIR}")

    # ------------------------------------------------------------------
    # 1. Acquire data
    # ------------------------------------------------------------------
    data = await _load_db_data()
    if data is None:
        print("[train] Generating synthetic training data …")
        data = _generate_synthetic_data(n_samples=2000)
        data_source = "synthetic"
    else:
        data_source = "database"

    X, y_winner, y_ft_ou, y_ht_ou = data
    print(f"[train] Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Train/test split (80 / 20)
    # ------------------------------------------------------------------
    X_tr, X_te, yw_tr, yw_te = train_test_split(
        X, y_winner, test_size=0.2, random_state=RNG_SEED, stratify=y_winner
    )
    _, _, yf_tr, yf_te = train_test_split(
        X, y_ft_ou, test_size=0.2, random_state=RNG_SEED, stratify=y_ft_ou
    )
    _, _, yh_tr, yh_te = train_test_split(
        X, y_ht_ou, test_size=0.2, random_state=RNG_SEED, stratify=y_ht_ou
    )

    # ------------------------------------------------------------------
    # 3. Train all three models
    # ------------------------------------------------------------------
    winner_model, winner_acc = _train_classifier(
        "winner_model", X_tr, X_te, yw_tr, yw_te
    )
    ft_ou_model, ft_ou_acc = _train_classifier(
        "ft_ou_model", X_tr, X_te, yf_tr, yf_te
    )
    ht_ou_model, ht_ou_acc = _train_classifier(
        "ht_ou_model", X_tr, X_te, yh_tr, yh_te
    )

    # ------------------------------------------------------------------
    # 4. Persist models and metadata
    # ------------------------------------------------------------------
    _save_model(winner_model, "winner_model.json")
    _save_model(ft_ou_model, "ft_ou_model.json")
    _save_model(ht_ou_model, "ht_ou_model.json")
    _save_metadata(winner_acc, ft_ou_acc, ht_ou_acc, data_source)

    print("\n[train] Training complete.")
    print(f"  Winner model accuracy : {winner_acc * 100:.1f}%")
    print(f"  FT O/U model accuracy : {ft_ou_acc * 100:.1f}%")
    print(f"  HT O/U model accuracy : {ht_ou_acc * 100:.1f}%")


async def run_training() -> None:
    """Entry point used by the monthly scheduler."""
    await main()


if __name__ == "__main__":
    asyncio.run(main())
