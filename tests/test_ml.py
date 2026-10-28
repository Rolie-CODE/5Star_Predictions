from datetime import datetime

import numpy as np
import pytest

from app.ml import train
from app.ml.features import (
    FT_OU_FEATURE_COLUMNS,
    REST_CAP,
    WINNER_FEATURE_COLUMNS,
    LeagueHistory,
)
from app.ml.predict import Calibrator, PredictionEngine


def _record(day, home, away, home_score, away_score, line=None):
    return train.GameRecord(
        game_id=f"t:{day}{home}", date=datetime(2024, 11, day), season="2024-2025",
        home_id=home, away_id=away, home_score=home_score, away_score=away_score, ou_line=line,
    )


def test_training_features_use_only_earlier_games():
    records = [_record(2, "A", "B", 100, 90, 200.0), _record(1, "A", "C", 130, 120)]
    pairs, history = train.pregame_features(records)
    assert [game.date.day for game, _ in pairs] == [1, 2]
    wx, wy = train.winner_samples(pairs)
    net = WINNER_FEATURE_COLUMNS.index("home_net_rating")
    rest = WINNER_FEATURE_COLUMNS.index("home_rest_days")
    assert wx[0][net] == 0 and wx[0][rest] == REST_CAP  # first game: no history yet
    assert wx[1][net] == 10 and wx[1][rest] == 1        # only the day-1 game is known
    assert wy == [1, 1]
    # Only the game with a line becomes a totals sample: 190 < 200 -> Under.
    tx, ty = train.totals_samples(train.totals_pairs(pairs))
    assert len(tx) == 1 and ty == [0]
    assert len(tx[0]) == len(FT_OU_FEATURE_COLUMNS)
    # The final state includes both games (it becomes the live carryover).
    assert history.carryover()["avg_totals"] == {"2024": 220.0}


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
    return LeagueHistory().features(2024, datetime(2024, 11, 1).date(), "A", "B", ou_line)


def test_prediction_skips_totals_without_a_line():
    result = PredictionEngine(_FakeLoader()).predict(_features(None))
    assert (result["winner_pick"], result["winner_trust"]) == ("Away", 70.0)
    assert result["ft_ou_pick"] is None and result["ft_ou_trust"] is None
    assert result["ht_ou_pick"] is None


def test_market_line_sets_the_pick_and_the_model_pick_is_kept():
    values = _features(None)
    values["home_market_prob"] = 0.62
    result = PredictionEngine(_FakeLoader()).predict(values)
    assert (result["winner_pick"], result["winner_trust"], result["winner_source"]) == (
        "Home", 62.0, "market",
    )
    assert (result["model_winner_pick"], result["model_winner_trust"]) == ("Away", 70.0)


def test_without_a_market_line_the_model_pick_is_served():
    result = PredictionEngine(_FakeLoader()).predict(_features(None))
    assert result["winner_source"] == "model"
    assert (result["winner_pick"], result["winner_trust"]) == ("Away", 70.0)


def test_prediction_with_line():
    result = PredictionEngine(_FakeLoader()).predict(_features(225.5))
    assert (result["ft_ou_pick"], result["ft_ou_trust"]) == ("Over", 90.0)
    assert result["ht_ou_pick"] is None  # no validated half-time model


def test_money_lines_are_no_vig_and_oriented_per_team(tmp_path, monkeypatch):
    path = tmp_path / "money_line.csv"
    path.write_text(
        "game_id,book_name,book_id,team_id,a_team_id,price1,price2\n"
        "001,A,1,10,20,-200,170\n"
        # Same game, the other book lists the teams the other way round.
        "001,B,2,20,10,180,-220\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "MONEY_LINE_FILE", path)
    team_id, p = train._read_money_lines()["001"]
    book_a = (200 / 300) / (200 / 300 + 100 / 270)
    book_b = (220 / 320) / (220 / 320 + 100 / 280)
    assert team_id == "10"
    assert p == pytest.approx((book_a + book_b) / 2)


def test_nba_db_games_stop_where_api_sports_starts(tmp_path, monkeypatch):
    path = tmp_path / "game.csv"
    path.write_text(
        "season_id,team_id_home,game_id,game_date,pts_home,team_id_away,pts_away,season_type\n"
        "22019,1,0021900001,2019-10-22 00:00:00,110.0,2,100.0,Regular Season\n"
        "42019,1,0041900001,2020-08-17 00:00:00,110.0,2,100.0,Playoffs\n"
        "22022,1,0022200001,2022-10-18 00:00:00,110.0,2,100.0,Regular Season\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "NBA_DB_GAMES_FILE", path)
    (game,) = train._read_nba_db_games()
    assert (game.game_id, game.season, game.home_score) == ("kaggle:0021900001", "2019-20", 110)
