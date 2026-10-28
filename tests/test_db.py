from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect, select

from app.db.models import Game, Team, TeamStats
from app.ingestion.data_pipeline import recompute_team_stats


def _game(api_id, day, home, away, home_score, away_score, status="finished"):
    return Game(
        api_sports_id=api_id, date=datetime(2026, 10, day, 23, 30, tzinfo=timezone.utc),
        home_team_id=home.id, away_team_id=away.id, status=status, season="2026-2027",
        home_score=home_score, away_score=away_score,
    )


def test_recompute_team_stats_counts_regular_season_finals_only(run, session_factory):
    async def scenario():
        async with session_factory() as db:
            a, b = Team(api_sports_id=1, name="A"), Team(api_sports_id=2, name="B")
            db.add_all([a, b])
            await db.flush()
            db.add_all([
                _game(10, 10, a, b, 99, 88),          # preseason: ignored
                _game(11, 21, a, b, 120, 100),
                _game(12, 23, b, a, 110, 105),
                _game(13, 25, a, b, None, None, status="scheduled"),
            ])
            await db.commit()

            assert await recompute_team_stats(db, "2026-2027") == 2
            rows = {ts.team_id: ts for ts in (await db.execute(select(TeamStats))).scalars()}
            ts_a = rows[a.id]
            assert (ts_a.games_played, ts_a.wins, ts_a.losses) == (2, 1, 1)
            assert ts_a.ppg == (120 + 105) / 2 and ts_a.oppg == (100 + 110) / 2
            assert (ts_a.home_wins, ts_a.home_losses, ts_a.away_wins, ts_a.away_losses) == (1, 0, 0, 1)

    run(scenario())


def test_migrations_build_the_model_schema(tmp_path, monkeypatch):
    from alembic import command

    from app.core.config import settings
    from app.db.session import _alembic_config

    db_path = tmp_path / "migrated.db"
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    command.upgrade(_alembic_config(), "head")

    columns = {
        c["name"]: c
        for c in inspect(create_engine(f"sqlite:///{db_path}")).get_columns("predictions")
    }
    assert columns["ft_ou_pick"]["nullable"] and columns["ht_ou_trust"]["nullable"]
    assert not columns["winner_pick"]["nullable"]
    assert {"winner_source", "model_winner_pick", "model_winner_trust"} <= set(columns)
    games = {c["name"] for c in inspect(create_engine(f"sqlite:///{db_path}")).get_columns("games")}
    assert "home_win_prob" in games
