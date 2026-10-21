from datetime import date

from app.core.config import normalize_database_url, season_for_date


def test_season_rolls_over_in_september():
    assert season_for_date(date(2026, 8, 31)) == "2025-2026"
    assert season_for_date(date(2026, 9, 1)) == "2026-2027"
    assert season_for_date(date(2027, 3, 15)) == "2026-2027"


def test_postgres_urls_use_asyncpg():
    assert normalize_database_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert normalize_database_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert normalize_database_url("postgresql+asyncpg://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert normalize_database_url("sqlite+aiosqlite:///./5star.db") == "sqlite+aiosqlite:///./5star.db"
