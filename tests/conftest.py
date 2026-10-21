import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base


@pytest.fixture
def run():
    """Run a coroutine to completion (keeps tests free of pytest plugins)."""
    return asyncio.run


@pytest.fixture
def session_factory(tmp_path):
    """Async session factory bound to a fresh SQLite file with all tables."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

    async def create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(create())
    yield async_sessionmaker(engine, expire_on_commit=False)
    asyncio.run(engine.dispose())
