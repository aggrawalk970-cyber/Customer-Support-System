"""
app/db/session.py
==================
SQLAlchemy engine + session factory for PostgreSQL.

This module is the single source of truth for database connectivity.
All other modules that need a DB session should import from here.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# PostgreSQL engine — no connect_args needed for psycopg2 with postgres
# pool_pre_ping=True ensures stale connections are recycled automatically
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
