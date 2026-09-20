"""
app/db/checkpointer.py
========================
LangGraph PostgreSQL Checkpointer setup.

Why PostgreSQL checkpointer (not SQLite)?
  - Survives app restarts and multi-replica deployments
  - Thread-safe: multiple Celery workers can read/write checkpoints safely
  - Same DB as our application data — no extra infrastructure

Package: langgraph-checkpoint-postgres
Driver:  Tries psycopg3 first (psycopg), falls back to psycopg2.
         On Windows, psycopg[binary] requires libpq to be installed.
         psycopg2-binary is a self-contained binary wheel and works out-of-box.

Usage:
    from app.db.checkpointer import get_checkpointer
    checkpointer = get_checkpointer()
    graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["escalation"])

Local dev without Postgres:
    If DATABASE_URL is not configured or Postgres is not reachable,
    the checkpointer falls back to SQLite automatically.
    Set your DATABASE_URL in .env to use PostgreSQL.
"""
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_postgres_checkpointer():
    """
    Try to connect to PostgreSQL using langgraph-checkpoint-postgres.

    Strategy:
      1. Try psycopg3 (psycopg) with postgresql+psycopg:// URI
      2. Fall back to psycopg2 with standard postgresql:// URI
      Both are supported by langgraph-checkpoint-postgres >= 2.0
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    # Strategy 1: psycopg3 (psycopg[binary] or psycopg)
    try:
        import psycopg  # noqa: F401 — just checking availability
        conn_string = (
            settings.DATABASE_URL
            .replace("postgresql://", "postgresql+psycopg://")
            .replace("postgres://", "postgresql+psycopg://")
        )
        checkpointer = PostgresSaver.from_conn_string(conn_string)
        checkpointer.setup()
        logger.info("LangGraph checkpointer: PostgreSQL (psycopg3)")
        return checkpointer
    except Exception as e1:
        logger.warning("psycopg3 unavailable, trying psycopg2...", extra={"reason": str(e1)[:200]})

    # Strategy 2: psycopg2 — uses standard postgresql:// URI
    try:
        import psycopg2  # noqa: F401
        conn_string = settings.DATABASE_URL  # psycopg2 accepts standard postgresql://
        checkpointer = PostgresSaver.from_conn_string(conn_string)
        checkpointer.setup()
        logger.info("LangGraph checkpointer: PostgreSQL (psycopg2)")
        return checkpointer
    except Exception as e2:
        raise RuntimeError(
            f"Both psycopg3 and psycopg2 failed to connect. "
            f"psycopg3 error: {e1}; psycopg2 error: {e2}"
        )


def get_checkpointer():
    """
    Return a LangGraph checkpointer.

    Tries PostgreSQL first (production). Falls back to SQLite (local dev).
    SQLite fallback is automatic — no config change needed.
    Just set DATABASE_URL in .env to switch to PostgreSQL.
    """
    try:
        return _get_postgres_checkpointer()
    except Exception as e:
        logger.warning(
            "PostgreSQL checkpointer unavailable — falling back to SQLite. "
            "Set DATABASE_URL in .env to use PostgreSQL.",
            extra={"error": str(e)[:300]},
        )
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        _conn = sqlite3.connect("./langgraph_checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(_conn)
        logger.warning(
            "Using SQLite checkpointer. This is NOT suitable for production "
            "or multi-worker deployments."
        )
        return checkpointer
