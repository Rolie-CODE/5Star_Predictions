from datetime import date

from app.db.models import TeamStats
from app.ml.features import (
    BASELINE_PPG,
    FEATURE_COLUMNS,
    TeamForm,
    features_from_forms,
    is_regular_season,
)


def test_new_team_gets_training_baseline():
    form = TeamForm()
    assert form.ppg == BASELINE_PPG and form.win_pct == 0.5 and form.home_win_pct == 0.5


def test_form_round_trips_through_team_stats():
    form = TeamForm()
    form.add_game(120, 100, is_home=True)
    form.add_game(95, 105, is_home=False)
    stats = TeamStats(
        games_played=form.games, wins=form.wins, losses=form.games - form.wins,
        ppg=form.points_for / form.games, oppg=form.points_against / form.games,
        home_wins=form.home_wins, home_losses=form.home_games - form.home_wins,
        away_wins=form.away_wins, away_losses=form.away_games - form.away_wins,
    )
    assert TeamForm.from_stats(stats) == form
    assert TeamForm.from_stats(None) == TeamForm()


def test_feature_dict_has_every_column_and_keeps_missing_lines_none():
    features = features_from_forms(TeamForm(), TeamForm(), None, None)
    assert list(features) == FEATURE_COLUMNS
    assert features["ou_line"] is None and features["expected_total"] == 2 * BASELINE_PPG


def test_regular_season_window():
    assert not is_regular_season("2022-2023", date(2022, 9, 30))  # Tokyo preseason
    assert is_regular_season("2022-2023", date(2022, 10, 18))
    assert not is_regular_season("2022-2023", date(2023, 4, 15))  # playoffs
    assert is_regular_season("2030-2031", date(2030, 12, 1))  # unknown season fallback
