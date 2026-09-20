"""
app/workers/celery_app.py
==========================
Celery application instance.

Broker:         CloudAMQP (RabbitMQ) via RABBITMQ_URL
                Falls back to Redis if RABBITMQ_URL is not set (local dev).

Result Backend: Upstash Redis via REDIS_URL
                Task results stored here so GET /tasks/{id} can poll them.

Why RabbitMQ as broker?
  RabbitMQ is purpose-built for message queuing: guaranteed delivery,
  message acknowledgment, dead-letter queues, fanout exchanges.
  Redis as a broker works, but it lacks the durability guarantees of AMQP.
  For production customer support (where missing a task = missed customer),
  a proper message broker is the right choice.

Why Redis as result backend?
  Task results are ephemeral (TTL 1 hour by default) and need fast reads.
  Redis is perfect for this — O(1) key lookup, TTL support built-in.
"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "customer_support",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    # ── Serialisation ─────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Result TTL ────────────────────────────────────────────────────────────
    # Keep task results for 1 hour — enough for clients to poll
    result_expires=3600,

    # ── Reliability ───────────────────────────────────────────────────────────
    # Acknowledge the task AFTER it completes (not on receipt)
    # Prevents lost tasks on worker crash
    task_acks_late=True,

    # Reject and re-queue tasks if the worker dies mid-execution
    task_reject_on_worker_lost=True,

    # ── Concurrency ───────────────────────────────────────────────────────────
    # 1 worker process per task (agent graphs are CPU/IO bound, not CPU-parallel)
    worker_concurrency=4,

    # ── Timezone ──────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,

    # ── Redis backend SSL (for Upstash rediss://) ─────────────────────────────
    redis_backend_use_ssl={
        "ssl_cert_reqs": "none"
    } if settings.REDIS_URL.startswith("rediss://") else {},
)
