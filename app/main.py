"""
app/main.py
============
FastAPI Application Entry Point — Multi-Agent Customer Support System

Architecture:
  - Routers registered from app/api/v1/ (chat, tasks, approvals, health)
  - PostgreSQL: application DB (tickets, orders, approvals) + LangGraph checkpoints
  - Redis (Upstash): Celery result backend + response cache + rate limit counters
  - RabbitMQ (CloudAMQP): Celery task broker (async agent execution)
  - IP-based rate limiting via slowapi

Versioned API prefix: /api/v1/
Legacy /api/* routes kept for backward compatibility during transition.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logging import get_logger
from app.core.rate_limit import limiter
from app.api.v1 import chat, tasks, approvals, health, stats

logger = get_logger(__name__)

app = FastAPI(
    title="Multi-Agent Customer Support System",
    description=(
        "Production-grade multi-agent backend using LangGraph. "
        "Features: Triage → Specialist (ReAct) → Escalation/HITL, "
        "PostgreSQL checkpointing, async Celery tasks (RabbitMQ + Redis), "
        "IP-based rate limiting, structured JSON logging, and full debug observability."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate Limiter ───────────────────────────────────────────────────────────────
# Attach the limiter to the app state so slowapi can find it
app.state.limiter = limiter

# Register the 429 Too Many Requests handler
# Returns: {"error": "Rate limit exceeded: X per Y second"} with Retry-After header
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────────────────────
# TODO: restrict allow_origins to your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Depends
from app.core.auth import verify_api_key

# ── Register API Routers ───────────────────────────────────────────────────────
# Versioned prefix: /api/v1/
app.include_router(chat.router,      prefix="/api/v1", tags=["Chat"], dependencies=[Depends(verify_api_key)])
app.include_router(tasks.router,     prefix="/api/v1", tags=["Async Tasks"], dependencies=[Depends(verify_api_key)])
app.include_router(approvals.router, prefix="/api/v1", tags=["Approvals"], dependencies=[Depends(verify_api_key)])
app.include_router(stats.router,     prefix="/api/v1", tags=["Stats"], dependencies=[Depends(verify_api_key)])
app.include_router(health.router,    prefix="/api/v1", tags=["Health"])

# Legacy routes (backward compat) — old /api/chat, /api/health still work
app.include_router(chat.router,      prefix="/api",    tags=["Chat (Legacy)"], dependencies=[Depends(verify_api_key)])
app.include_router(health.router,    prefix="/api",    tags=["Health (Legacy)"])


# ── Startup: Ensure DB Tables Exist ───────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    """
    Create database tables on first run.
    No seed data — this is production-grade. Use Alembic migrations for schema.
    """
    from app.db.session import Base, engine

    logger.info("Starting up — ensuring database tables exist...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified.")
    except Exception as e:
        logger.error("Failed to verify database tables", extra={"error": str(e)})


# ── Legacy Approval Endpoint ───────────────────────────────────────────────────

@app.post("/api/approve", include_in_schema=False)
async def legacy_approve(request: Request):
    """
    Deprecated. Use POST /api/v1/approvals/{thread_id}/resolve instead.
    """
    body = await request.json()
    thread_id = body.get("thread_id", "")
    return JSONResponse(
        status_code=301,
        content={
            "detail": (
                f"This endpoint is deprecated. "
                f"Use POST /api/v1/approvals/{thread_id}/resolve instead."
            )
        },
    )
