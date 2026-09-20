"""
app/api/v1/tasks.py
====================
Async task endpoints — for long-running agent invocations via Celery.

POST /tasks    → Enqueues an agent task, returns task_id immediately (< 50ms)
GET  /tasks/{task_id} → Polls Celery result backend for task status

Rate limits (per IP):
  POST /tasks            → 20 requests / minute
  GET  /tasks/{task_id}  → 120 requests / minute (polling — higher limit)

When to use /tasks vs /chat?
  - /chat:  Quick queries (< 10s), synchronous, simple clients (e.g. curl, Postman)
  - /tasks: Long agent chains (multi-step ReAct), mobile apps, production frontends

Interview talking point:
  This is the standard async job pattern. Client gets a task_id,
  polls until status=SUCCESS, then reads the result. Zero timeout issues.
"""
import uuid
from typing import Optional

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.workers.celery_app import celery_app
from app.workers.tasks import run_agent_task
from app.core.logging import get_logger
from app.core.rate_limit import limiter

router = APIRouter()
logger = get_logger(__name__)


# ── Request / Response Schemas ────────────────────────────────────────────────

class TaskRequest(BaseModel):
    message: str
    user_email: Optional[str] = None
    thread_id: Optional[str] = None  # If not provided, a new UUID is generated


class TaskResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str  # PENDING | STARTED | SUCCESS | FAILURE | RETRY


class TaskStatusResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse, status_code=202, summary="Enqueue an async agent task")
@limiter.limit("20/minute")
def create_task(request: Request, body: TaskRequest):
    """
    Submit a message to be processed asynchronously by the agent graph.

    Returns a task_id immediately. Poll GET /tasks/{task_id} for the result.
    HTTP 202 Accepted = request received, processing in progress.
    """
    thread_id = body.thread_id or str(uuid.uuid4())

    logger.info(
        "Queuing agent task",
        extra={"thread_id": thread_id, "user_email": body.user_email},
    )

    task = run_agent_task.delay(
        thread_id=thread_id,
        message=body.message,
        user_email=body.user_email,
    )

    return TaskResponse(
        task_id=task.id,
        thread_id=thread_id,
        status="PENDING",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse, summary="Poll async task status")
@limiter.limit("120/minute")
def get_task_status(request: Request, task_id: str):
    """
    Poll the status of an async agent task.

    States:
      PENDING  → Task queued, not yet started
      STARTED  → Worker picked it up and is running
      SUCCESS  → Completed — result contains the agent response
      FAILURE  → Task failed — error contains the exception message
      RETRY    → Worker is retrying after a transient failure
    """
    result: AsyncResult = celery_app.AsyncResult(task_id)

    if result.state == "PENDING":
        return TaskStatusResponse(
            task_id=task_id,
            thread_id="",
            status="PENDING",
        )

    if result.state == "FAILURE":
        return TaskStatusResponse(
            task_id=task_id,
            thread_id="",
            status="FAILURE",
            error=str(result.result),
        )

    if result.state == "SUCCESS":
        task_result = result.result or {}
        return TaskStatusResponse(
            task_id=task_id,
            thread_id=task_result.get("thread_id", ""),
            status="SUCCESS",
            result=task_result,
        )

    # STARTED / RETRY / other states
    return TaskStatusResponse(
        task_id=task_id,
        thread_id="",
        status=result.state,
    )
