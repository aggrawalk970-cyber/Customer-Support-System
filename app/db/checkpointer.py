"""
app/db/checkpointer.py
========================
LangGraph PostgreSQL Checkpointer setup.

langgraph-checkpoint-postgres >= 2.0 uses psycopg3 (psycopg) with a
ConnectionPool for persistent, multi-worker-safe checkpointing.

The correct pattern (NOT from_conn_string which is a context manager):
    pool = ConnectionPool(conn_string)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

Falls back to SQLite if psycopg3 or PostgreSQL is unavailable (local dev).
"""
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_checkpointer():
    """
    Return a LangGraph checkpointer backed by PostgreSQL (psycopg3 + connection pool).

    Uses psycopg_pool.ConnectionPool for thread-safe, persistent connections —
    correct for a long-running FastAPI/Celery process.

    Falls back to SQLite automatically if psycopg3 is not available.
    """
    try:
        import psycopg  # noqa: F401 — verify psycopg3 is importable
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        # psycopg3 uses standard postgresql:// DSN (no +psycopg suffix needed)
        conn_string = (
            settings.DATABASE_URL
            .replace("postgresql+psycopg://", "postgresql://")
            .replace("postgres://", "postgresql://")
        )

        # Open the pool immediately (open=True) so connections are ready on startup
        # autocommit=True is REQUIRED for setup() — PostgreSQL does not allow
        # CREATE INDEX CONCURRENTLY inside a transaction block
        pool = ConnectionPool(
            conn_string,
            min_size=1,
            max_size=10,
            open=True,
            kwargs={"autocommit": True},
        )

        checkpointer = PostgresSaver(pool)
        # Creates langgraph checkpoint tables if they don't exist yet
        checkpointer.setup()

        logger.info(
            "LangGraph checkpointer: PostgreSQL (psycopg3 + ConnectionPool)",
            extra={"host": conn_string.split("@")[-1] if "@" in conn_string else "local"},
        )
        return checkpointer

    except Exception as e:
        logger.warning(
            "PostgreSQL checkpointer unavailable — falling back to SQLite. "
            "Set DATABASE_URL in .env to a reachable PostgreSQL instance.",
            extra={"error": str(e)[:400]},
        )
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver

        _conn = sqlite3.connect("./langgraph_checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(_conn)
        logger.warning(
            "Using SQLite checkpointer — NOT suitable for production or multi-worker use."
        )
        return checkpointer
