from datetime import datetime

import numpy as np
import pytest

from app.ml import train
from app.ml.features import (
    BASELINE_PPG,
    FT_OU_FEATURE_COLUMNS,
    WINNER_FEATURE_COLUMNS,
    TeamForm,
    features_from_forms,
)
from app.ml.predict import Calibrator, PredictionEngine


def _record(day, home, away, home_score, away_score, line=None):
    return train.GameRecord(
        game_id=f"t:{day}{home}", date=datetime(2024, 11, day), season="2024-2025",
        home_id=home, away_id=away, home_score=home_score, away_score=away_score, ou_line=line,
    )


def test_training_features_use_only_earlier_games():
    records = [_record(2, "A", "B", 100, 90, 200.0), _record(1, "A", "C", 130, 120)]
    rows, wx, wy, trows, tx, ty = train._features_for_records(records)
    assert [r.date.day for r in rows] == [1, 2]
    home_ppg = WINNER_FEATURE_COLUMNS.index("home_ppg")
    assert wx[0][home_ppg] == BASELINE_PPG  # first game: no history yet
    assert wx[1][home_ppg] == 130           # only the day-1 game is known
    assert wy == [1, 1]
    # Only the game with a line becomes a totals sample: 190 < 200 -> Under.
    assert len(tx) == 1 and ty == [0]
    assert tx[0][FT_OU_FEATURE_COLUMNS.index("ou_line")] == 200.0


@pytest.mark.skipif(
    not train.API_SPORTS_FILE.exists(), reason="training data is kept locally, not in git"
)
def test_api_sports_csv_excludes_preseason_and_postponed():
    records = train._read_api_sports_games()
    ids = {r.game_id for r in records}
    assert "api:326118" not in ids  # 2022-09-30 preseason game in Tokyo
    assert all(r.date >= datetime(2022, 10, 18) for r in records)
    # About 1,230 regular-season games per season.
    assert 3 * 1150 < len(records) < 3 * 1240


def test_calibrator_interpolates_and_defaults_to_identity():
    cal = Calibrator({"x": [0.2, 0.8], "y": [0.4, 0.6]})
    assert cal(0.5) == pytest.approx(0.5)
    assert cal(0.95) == pytest.approx(0.6)
    assert Calibrator(None)(0.73) == 0.73


class _ConstModel:
    def __init__(self, p1):
        self.p1 = p1

    def predict_proba(self, X):
        return np.array([[1 - self.p1, self.p1]])


class _FakeLoader:
    is_loaded = True
    model_version = "test"
    winner_model = _ConstModel(0.3)
    ft_ou_model = _ConstModel(0.9)
    ht_ou_model = None

    def calibrator(self, name):
        return Calibrator(None)


def _features(ou_line):
    return features_from_forms(TeamForm(), TeamForm(), ou_line, None)


def test_prediction_skips_totals_without_a_line():
    result = PredictionEngine(_FakeLoader()).predict(_features(None))
    assert (result["winner_pick"], result["winner_trust"]) == ("Away", 70.0)
    assert result["ft_ou_pick"] is None and result["ft_ou_trust"] is None
    assert result["ht_ou_pick"] is None


def test_prediction_with_line():
    result = PredictionEngine(_FakeLoader()).predict(_features(225.5))
    assert (result["ft_ou_pick"], result["ft_ou_trust"]) == ("Over", 90.0)
    assert result["ht_ou_pick"] is None  # no validated half-time model
