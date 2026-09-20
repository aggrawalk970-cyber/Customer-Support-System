"""
app/api/v1/health.py
=====================
Health check endpoint — used by Nginx, load balancers, and Docker health checks.

Returns: {"status": "ok", "version": "2.0.0", "db": "ok", "redis": "ok"}

Why check DB and Redis?
  A simple {"status": "ok"} only tells you the process is alive.
  A proper deep health check tells your orchestrator whether the app
  can actually serve traffic — if Postgres is down, we're not healthy.
"""
from fastapi import APIRouter, Request
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine
from app.services.cache import get_redis
from app.core.rate_limit import limiter

router = APIRouter()


@router.get("/health", summary="Deep health check")
@limiter.limit("120/minute")
def health_check(request: Request):
    """
    Returns service health including DB and Redis connectivity.
    Used by Nginx upstream health checks and Docker HEALTHCHECK directive.
    """
    status = {"status": "ok", "version": settings.APP_VERSION}

    # ── PostgreSQL check ──────────────────────────────────────────────────────
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["db"] = "ok"
    except Exception as e:
        status["db"] = f"error: {str(e)[:100]}"
        status["status"] = "degraded"

    # ── Redis check ───────────────────────────────────────────────────────────
    try:
        redis_client = get_redis()
        if redis_client:
            redis_client.ping()
            status["redis"] = "ok"
        else:
            status["redis"] = "unavailable"
    except Exception as e:
        status["redis"] = f"error: {str(e)[:100]}"

    return status
