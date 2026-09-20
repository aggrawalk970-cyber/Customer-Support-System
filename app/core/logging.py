"""
app/core/logging.py
====================
Structured JSON logging setup for production observability.

Usage:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Agent invoked", extra={"thread_id": "abc", "intent": "billing"})

Why JSON logging?
    Plain text logs are hard to query in aggregators (Datadog, Loki, CloudWatch).
    JSON-structured logs let you filter by field (e.g. intent="billing") directly
    in your log dashboard — no regex needed.
"""
import logging
import sys

from pythonjsonlogger import jsonlogger


def get_logger(name: str = "customer_support") -> logging.Logger:
    """Return a JSON-formatted logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — avoid adding duplicate handlers
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


# Module-level default logger
logger = get_logger("customer_support")
