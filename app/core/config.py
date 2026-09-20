"""
app/core/config.py
==================
Centralised settings for the entire application.

Uses pydantic-settings so every variable is type-checked and can be
overridden by the environment (or a .env file).
"""
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # ── LLM Keys ──────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    # Format: postgresql://user:password@host:5432/dbname
    # Used for: SQLAlchemy ORM (tickets, orders, approvals)
    #           AND the LangGraph PostgreSQL checkpointer
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/customer_support"

    # ── Redis (Upstash) ───────────────────────────────────────────────────────
    # Format: rediss://default:<token>@<host>.upstash.io:6379
    # Used for: Celery result backend + LLM response cache
    REDIS_URL: str = "redis://localhost:6379"

    # ── RabbitMQ / CloudAMQP ──────────────────────────────────────────────────
    # Format: amqps://user:pass@host/vhost  (CloudAMQP free tier)
    # Used for: Celery task broker
    # Falls back to Redis if not set (simpler local dev setup)
    RABBITMQ_URL: str = ""

    # ── Celery ────────────────────────────────────────────────────────────────
    # Derived at runtime: broker = RABBITMQ_URL if set else REDIS_URL
    # result_backend = REDIS_URL always

    # ── LangSmith Observability ───────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "customer-support-agents"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── RAG / Embeddings (kept for future enhancement) ────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── App ───────────────────────────────────────────────────────────────────
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    @property
    def celery_broker_url(self) -> str:
        """Use CloudAMQP/RabbitMQ if configured, else fall back to Redis."""
        return self.RABBITMQ_URL if self.RABBITMQ_URL else self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        """Celery result backend is always Redis (Upstash)."""
        return self.REDIS_URL

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra fields so old .env files with unknown keys don't crash
        extra = "ignore"


settings = Settings()

# Propagate LangSmith env vars for LangChain auto-tracing
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
