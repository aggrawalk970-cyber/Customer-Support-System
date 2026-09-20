"""
app/services/cache.py
======================
Redis cache helpers using Upstash Redis.

Used for:
  1. LLM response caching — identical queries return cached answers instantly
     (saves tokens + latency for common questions like "what's your refund policy?")
  2. Rate-limit counters (future)
  3. Any short-lived ephemeral state

Upstash note:
  Upstash Redis uses TLS — the REDIS_URL should start with rediss:// (double-s).
  The redis-py client handles TLS automatically when the URL starts with rediss://.
"""
import json
from typing import Any, Optional

import redis
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level Redis client — created once, reused across requests
# decode_responses=True so we work with strings, not bytes
try:
    _redis_client = redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    _redis_client.ping()
    logger.info("Redis connection established.", extra={"url": settings.REDIS_URL[:30] + "..."})
except Exception as e:
    logger.warning("Redis unavailable — caching disabled.", extra={"error": str(e)})
    _redis_client = None


def get_redis() -> Optional[redis.Redis]:
    """Return the Redis client, or None if unavailable."""
    return _redis_client


def get_cached_response(key: str) -> Optional[str]:
    """
    Retrieve a cached value by key.
    Returns None on cache miss or if Redis is unavailable.
    """
    if not _redis_client:
        return None
    try:
        value = _redis_client.get(key)
        if value:
            logger.info("Cache HIT", extra={"key": key})
        return value
    except Exception as e:
        logger.warning("Cache GET failed", extra={"key": key, "error": str(e)})
        return None


def set_cached_response(key: str, value: Any, ttl: int = 300) -> bool:
    """
    Store a value in Redis with a TTL (seconds).
    Returns True on success, False on failure or if Redis is unavailable.

    Default TTL = 5 minutes. LLM responses are relatively static for common
    support queries — short TTL prevents stale data issues.
    """
    if not _redis_client:
        return False
    try:
        serialised = json.dumps(value) if not isinstance(value, str) else value
        _redis_client.setex(key, ttl, serialised)
        logger.info("Cache SET", extra={"key": key, "ttl": ttl})
        return True
    except Exception as e:
        logger.warning("Cache SET failed", extra={"key": key, "error": str(e)})
        return False


def delete_cached_response(key: str) -> bool:
    """Invalidate a specific cache key."""
    if not _redis_client:
        return False
    try:
        _redis_client.delete(key)
        return True
    except Exception:
        return False


def make_cache_key(prefix: str, *parts: str) -> str:
    """
    Build a consistent, namespaced cache key.
    Example: make_cache_key("chat", thread_id, message_hash) → "chat:abc123:f4d9..."
    """
    return ":".join([prefix] + [str(p) for p in parts])
