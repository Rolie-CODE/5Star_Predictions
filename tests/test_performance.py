from datetime import datetime, timedelta, timezone

import pytest

from app.db.models import Game, Prediction, Team
from app.services import prediction_service


def _game(api_id, home, away, home_score, away_score, line, status="finished", date=None):
    return Game(
        api_sports_id=api_id, date=date or datetime(2026, 10, 22, 23, 30, tzinfo=timezone.utc),
        home_team_id=home.id, away_team_id=away.id, status=status, season="2026-2027",
        home_score=home_score, away_score=away_score, ou_line=line,
    )


def _prediction(game, winner, ou, version="3.0.0"):
    return Prediction(
        game_id=game.id, winner_pick=winner, winner_trust=60.0,
        ft_ou_pick=ou, ft_ou_trust=55.0 if ou else None, model_version=version,
    )


def test_performance_grades_finished_games_only(run, session_factory):
    async def scenario():
        async with session_factory() as db:
            a, b = Team(api_sports_id=1, name="A"), Team(api_sports_id=2, name="B")
            db.add_all([a, b])
            await db.flush()
            games = [
                _game(10, a, b, 120, 100, 210.5),   # Home right, Over right
                _game(11, a, b, 100, 105, 210.5),   # Home wrong, Under right
                _game(12, a, b, 110, 100, 210.0),   # Away wrong, push: not graded
                _game(13, a, b, None, None, 220.5, status="scheduled"),
            ]
            db.add_all(games)
            await db.flush()
            db.add_all([
                _prediction(games[0], "Home", "Over"),
                _prediction(games[1], "Home", "Under"),
                _prediction(games[2], "Away", "Over", version="2.0.0"),
                _prediction(games[3], "Home", "Over"),
            ])
            await db.commit()

            result = await prediction_service.get_performance(db, "2026-2027")
            assert result["winner"] == {"graded": 3, "correct": 1, "accuracy": 33.3, "roi": None}
            ou = result["full_time_ou"]
            assert (ou["graded"], ou["correct"], ou["accuracy"]) == (2, 2, 100.0)
            assert ou["roi"] == pytest.approx(90.9)

            v3 = await prediction_service.get_performance(db, "2026-2027", "3.0.0")
            assert v3["winner"]["graded"] == 2

    run(scenario())


def test_started_games_keep_their_pregame_prediction(run, session_factory, monkeypatch):
    monkeypatch.setattr(
        prediction_service.prediction_engine, "predict",
        lambda features: {"winner_pick": "Home", "winner_trust": 60.0, "model_version": "x"},
    )

    async def scenario():
        async with session_factory() as db:
            a, b = Team(api_sports_id=1, name="A"), Team(api_sports_id=2, name="B")
            db.add_all([a, b])
            await db.flush()
            now = datetime.now(timezone.utc)
            # Status is still "scheduled" because results have not synced yet.
            started = _game(10, a, b, None, None, None, "scheduled", now - timedelta(minutes=5))
            upcoming = _game(11, a, b, None, None, None, "scheduled", now + timedelta(minutes=5))
            db.add_all([started, upcoming])
            await db.commit()

            assert await prediction_service.run_predictions_for_today(db) == 1

    run(scenario())
