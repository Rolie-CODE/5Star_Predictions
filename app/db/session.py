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

from collections.abc import AsyncGenerator

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
        # For PostgreSQL: keep a pool of up to 10 connections, 20 overflow.
        # SQLite ignores pool settings (uses NullPool internally via aiosqlite).
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
# Table initialisation
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """
    Create all ORM-mapped tables that do not yet exist in the database.

    Call this once at application startup (e.g. from a FastAPI lifespan
    context manager).  It is a no-op if the tables already exist, so it
    is safe to call on every startup.

    Note: imports are done *inside* the function to avoid circular-import
    issues between models and session at module load time.
    """
    # Import the models module so SQLAlchemy's metadata is populated with
    # the table definitions before we call create_all.
    from app.db import models  # noqa: F401 — side-effect import

    async with engine.begin() as conn:
        # checkfirst=True is implied by create_all (it uses IF NOT EXISTS).
        await conn.run_sync(models.Base.metadata.create_all)
