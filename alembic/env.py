"""
alembic/env.py
===============
Alembic migration environment configuration.

Configured to:
  1. Pull DATABASE_URL from our app settings (not hardcoded in alembic.ini)
  2. Use our SQLAlchemy models' metadata for autogenerate support
  3. Run in online mode against PostgreSQL

Usage:
  alembic revision --autogenerate -m "description"
  alembic upgrade head
  alembic downgrade -1
"""
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Add project root to Python path ──────────────────────────────────────────
# So alembic can import app.* modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import our models and config ──────────────────────────────────────────────
from app.core.config import settings
from app.db.session import Base

# Import all models so Alembic's autogenerate can detect them
import app.db.models  # noqa: F401 — registers Ticket, Order, Approval with Base

# ── Alembic Config ────────────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url with our dynamic DATABASE_URL from .env
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Our model metadata — enables --autogenerate to detect schema changes
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Offline mode: generate SQL scripts without connecting to DB.
    Useful for reviewing migrations before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,      # Detect column type changes
        compare_server_default=True,  # Detect default changes
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Online mode: connect to DB and apply migrations directly.
    Default mode for alembic upgrade head.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
