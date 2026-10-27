from datetime import date, datetime, timedelta, timezone

import pytest

from app.db.models import Game, Team, TeamStats
from app.ml import features
from app.ml.features import (
    BASELINE_PPG,
    ELO_START,
    FEATURE_COLUMNS,
    LeagueHistory,
    TeamForm,
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
    values = LeagueHistory().features(2024, date(2024, 10, 22), "A", "B", None)
    assert set(FEATURE_COLUMNS) <= set(values)
    assert values["ou_line"] is None and values["ou_line_vs_league"] is None
    # Nothing known yet: even teams, fully rested, home edge from Elo only.
    assert values["elo_diff"] == 0 and values["home_rest_days"] == features.REST_CAP
    assert values["elo_home_prob"] == pytest.approx(0.64, abs=0.01)


def test_rest_recent_form_and_league_average():
    history = LeagueHistory()
    history.add_result(2024, date(2024, 11, 1), "A", "B", 120, 100)
    values = history.features(2024, date(2024, 11, 2), "A", "C", 230.0)
    assert values["home_rest_days"] == 1 and values["home_b2b"] == 1
    assert values["away_rest_days"] == features.REST_CAP and values["away_b2b"] == 0
    assert values["home_recent_margin"] == pytest.approx(features.RECENT_WEIGHT * 20)
    assert values["home_net_rating"] == 20 and values["away_net_rating"] == 0
    # One 220-point game is blended with the default prior of 2 * 110.
    assert history.league_avg_total(2024) == pytest.approx(220)
    assert values["ou_line_vs_league"] == pytest.approx(10)


def test_elo_carries_over_and_regresses_per_missed_season():
    history = LeagueHistory()
    for day in range(1, 11):
        history.add_result(2022, date(2022, 11, day), "A", "B", 120, 100)
    end_of_season = history.elo.ratings["A"] - ELO_START
    assert end_of_season > 0
    restored = LeagueHistory.from_carryover(history.carryover())
    one_season = restored.elo.rating("A", 2023) - ELO_START
    assert one_season == pytest.approx(features.ELO_CARRYOVER * end_of_season)
    # Season-to-date stats do not carry over; Elo does.
    values = restored.features(2023, date(2023, 10, 25), "A", "B", None)
    assert values["home_net_rating"] == 0 and values["elo_diff"] > 0

    skipped = LeagueHistory.from_carryover(history.carryover())
    two_seasons = skipped.elo.rating("A", 2024) - ELO_START
    assert two_seasons == pytest.approx(features.ELO_CARRYOVER**2 * end_of_season)


def test_bigger_wins_move_elo_more():
    close, blowout = LeagueHistory(), LeagueHistory()
    close.add_result(2024, date(2024, 11, 1), "A", "B", 101, 100)
    blowout.add_result(2024, date(2024, 11, 1), "A", "B", 130, 100)
    assert blowout.elo.ratings["A"] > close.elo.ratings["A"] > ELO_START


def test_carryover_remaps_team_ids():
    history = LeagueHistory()
    history.add_result(2024, date(2024, 11, 1), "nba-1", "nba-2", 110, 100)
    state = history.carryover({"nba-1": "132", "nba-2": "133"})
    assert set(state["elo"]) == {"132", "133"}
    assert state["avg_totals"] == {"2024": 210.0}


def test_live_features_match_a_training_replay(run, session_factory, monkeypatch):
    monkeypatch.setattr(features, "load_carryover", lambda: None)
    start = datetime(2026, 10, 21, 23, 30, tzinfo=timezone.utc)

    async def scenario():
        async with session_factory() as db:
            a, b, c = (Team(api_sports_id=i, name=n) for i, n in ((1, "A"), (2, "B"), (3, "C")))
            db.add_all([a, b, c])
            await db.flush()

            def game(api_id, day, home, away, hs=None, as_=None, status="finished", line=None):
                return Game(
                    api_sports_id=api_id, date=start + timedelta(days=day),
                    home_team_id=home.id, away_team_id=away.id, status=status,
                    season="2026-2027", home_score=hs, away_score=as_, ou_line=line,
                )

            target = game(13, 3, a, c, status="scheduled", line=231.5)
            db.add_all([
                game(10, 0, a, b, 120, 100),
                game(11, 1, b, c, 99, 104),
                game(12, 2, c, a, status="live"),     # started, no score yet
                target,
            ])
            await db.commit()
            (live,) = await features.build_feature_vectors(db, [target])

        # The same games replayed the way training does it.
        history = LeagueHistory()
        history.add_result(2026, date(2026, 10, 21), "1", "2", 120, 100)
        history.add_result(2026, date(2026, 10, 22), "2", "3", 99, 104)
        history.note_game(2026, date(2026, 10, 23), "3")
        history.note_game(2026, date(2026, 10, 23), "1")
        expected = history.features(2026, date(2026, 10, 24), "1", "3", 231.5)
        assert live == expected
        assert live["home_rest_days"] == 1  # the live game counts as played

    run(scenario())


def test_regular_season_window():
    assert not is_regular_season("2022-2023", date(2022, 9, 30))  # Tokyo preseason
    assert is_regular_season("2022-2023", date(2022, 10, 18))
    assert not is_regular_season("2022-2023", date(2023, 4, 15))  # playoffs
    assert is_regular_season("2030-2031", date(2030, 12, 1))  # unknown season fallback
