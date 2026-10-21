"""
app/db/session.py
-----------------
Async SQLAlchemy engine, session factory, and FastAPI dependency for the
5Star Predictions project.

Supports both:
  • SQLite   — sqlite+aiosqlite:///./5star.db
  • PostgreSQL — postgresql+asyncpg://user:pass@host/dbname

Usage (FastAPI dependency):
    @router.get("/example")
    async def example(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(MyModel))
        ...

Programmatic initialisation (call once at startup):
    from app.db.session import init_db
    await init_db()
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def _build_engine() -> AsyncEngine:
    """
    Build the async engine with sensible defaults for both SQLite and
    PostgreSQL.  The `connect_args` tweak is SQLite-specific and is skipped
    for other drivers.
    """
    url: str = settings.DATABASE_URL

    # SQLite requires check_same_thread=False when used with an async driver
    # because aiosqlite runs queries on a background thread.
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_async_engine(
        url,
        # Echo SQL statements only in development so production logs stay clean.
        echo=(settings.APP_ENV == "development"),
        # Test pooled connections before use so a dropped PostgreSQL
        # connection doesn't fail the next request.
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    return engine


# Module-level engine singleton — created once when the module is first imported.
engine: AsyncEngine = _build_engine()

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    # Do NOT automatically commit or expire objects after each statement so
    # we can access loaded attributes outside the ORM flush cycle.
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator that yields a database session and guarantees cleanup.

    Inject this with ``Depends(get_db)`` in any FastAPI path operation:

        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            ...

    The session is committed automatically when the request succeeds.
    On any exception, the transaction is rolled back before the session
    is closed, preventing dirty state from leaking between requests.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

# Revision matching the schema the pre-Alembic create_all() produced.
_LEGACY_REVISION = "0001"


def _alembic_config():
    from alembic.config import Config

    config = Config(str(_ALEMBIC_INI))
    # Keep the app's logging setup instead of alembic.ini's.
    config.attributes["configure_logging"] = False
    return config


async def init_db() -> None:
    """
    Bring the database schema up to date by running Alembic migrations.

    Call this once at application startup.  It is a no-op when the schema
    is already current.  A database created by the old ``create_all()``
    startup code (tables present, no ``alembic_version``) is stamped at the
    initial revision first so its tables aren't re-created.
    """
    from alembic import command

    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))

    config = _alembic_config()
    # env.py calls asyncio.run(), so run Alembic in a worker thread.
    if "games" in tables and "alembic_version" not in tables:
        await asyncio.to_thread(command.stamp, config, _LEGACY_REVISION)
    await asyncio.to_thread(command.upgrade, config, "head")
