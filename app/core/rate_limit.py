"""
app/core/rate_limit.py
=======================
IP-based rate limiting using slowapi (built on `limits` library).

Strategy:
  - Rate limits are enforced per client IP address
  - Uses Redis as the storage backend when available (shared across workers)
  - Falls back to in-memory storage if Redis is not configured (single-process only)

Limits (configurable — adjust based on your SLA):
  - POST /chat:                 20 requests / minute per IP
  - POST /tasks:                20 requests / minute per IP
  - POST /approvals/{id}/resolve: 30 requests / minute per IP
  - GET  /health, /history:    60 requests / minute per IP (less sensitive)

Why IP-based?
  - Prevents DDoS / brute-force on the LLM endpoint (each call costs tokens)
  - Standard first line of defence before auth is added
  - Zero config needed client-side

Why Redis backend for limits?
  - In-memory limits reset if a worker restarts or you run multiple workers
  - Redis makes limits shared across all Celery workers + API replicas
  - Upstash Redis works perfectly here (low-latency counters with TTL)

Interview talking point:
  Rate limiting is one of the first things asked about in backend system design.
  "How do you prevent abuse of your LLM API?" — this is the answer.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_storage_uri() -> str:
    """
    Return the storage URI for the rate limiter.

    Uses Redis (Upstash) when REDIS_URL is configured — limits are then
    shared across all API workers and survive restarts.

    Falls back to in-memory storage (memory://) for local dev without Redis.
    Note: in-memory limits are per-process and reset on restart.
    """
    if settings.REDIS_URL and settings.REDIS_URL not in ("redis://localhost:6379", ""):
        logger.info("Rate limiter: using Redis storage (shared across workers).")
        return settings.REDIS_URL
    logger.warning(
        "Rate limiter: using in-memory storage. "
        "Set REDIS_URL for production (limits shared across workers)."
    )
    return "memory://"


# ── Global Limiter instance ───────────────────────────────────────────────────
# key_func=get_remote_address → rate limit key is the client's IP
# storage_uri → Redis (production) or memory (dev)
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_build_storage_uri(),
    default_limits=["200/minute"],  # Global fallback (per IP)
)
