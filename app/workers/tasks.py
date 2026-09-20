"""
app/workers/tasks.py
=====================
Celery tasks — long-running agent invocations run asynchronously.

Why async agent execution?
  Agent graphs can take 10–30 seconds (multi-step ReAct reasoning + tool calls).
  Blocking the HTTP request for that long:
    - Hits browser/load-balancer timeouts
    - Starves the FastAPI thread pool
  Solution: POST /tasks returns task_id immediately (< 50ms).
  Client polls GET /tasks/{task_id} until status = SUCCESS.

Interview talking point:
  This is the standard pattern for any long-running AI operation in production.
  The API is always fast; the work happens in the background.
"""
import datetime
from typing import Optional

from langchain_core.messages import HumanMessage

from app.workers.celery_app import celery_app
from app.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(
    name="run_agent_task",
    bind=True,
    max_retries=3,
    default_retry_delay=5,  # seconds between retries
    soft_time_limit=120,    # 2-minute soft limit → raises SoftTimeLimitExceeded
    time_limit=150,         # 2.5-minute hard kill
)
def run_agent_task(
    self,
    thread_id: str,
    message: str,
    user_email: Optional[str] = None,
) -> dict:
    """
    Invoke the LangGraph agent graph asynchronously.

    Args:
        thread_id:  LangGraph checkpoint key (identifies the conversation)
        message:    The user's message to process
        user_email: Optional email for personalised responses

    Returns:
        A dict with the final agent response and metadata.
        This is stored in Redis result backend and retrievable via GET /tasks/{id}.
    """
    from app.agents.graph import compiled_graph

    logger.info(
        "Agent task started",
        extra={"task_id": self.request.id, "thread_id": thread_id},
    )

    try:
        config = {"configurable": {"thread_id": thread_id}}
        current_state = compiled_graph.get_state(config)

        if not current_state.values:
            # New conversation thread
            inputs = {
                "messages": [HumanMessage(content=message)],
                "user_email": user_email or "",
                "intent": "general",
                "next_action": "",
                "active_agent": "",
                "escalation_summary": "",
                "approved_by_human": False,
                "logs": [],
                "tool_calls_log": [],
                "retrieval_sources": [],
            }
            compiled_graph.invoke(inputs, config)
        else:
            # Continue existing thread
            updates = {"messages": [HumanMessage(content=message)]}
            if user_email:
                updates["user_email"] = user_email
            compiled_graph.update_state(config, updates)
            compiled_graph.invoke(None, config)

        final_state = compiled_graph.get_state(config)
        values = final_state.values
        needs_approval = len(final_state.next) > 0

        # Extract the last assistant message as the response
        last_ai_message = ""
        for msg in reversed(values.get("messages", [])):
            if msg.__class__.__name__ == "AIMessage":
                content = msg.content
                if isinstance(content, list):
                    last_ai_message = " ".join(
                        str(c.get("text", "")) if isinstance(c, dict) else str(c)
                        for c in content
                    )
                else:
                    last_ai_message = str(content)
                break

        result = {
            "thread_id": thread_id,
            "response": last_ai_message,
            "intent": values.get("intent", "general"),
            "active_agent": values.get("active_agent", ""),
            "needs_approval": needs_approval,
            "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        logger.info(
            "Agent task completed",
            extra={
                "task_id": self.request.id,
                "thread_id": thread_id,
                "intent": result["intent"],
                "needs_approval": needs_approval,
            },
        )
        return result

    except Exception as exc:
        logger.error(
            "Agent task failed",
            extra={"task_id": self.request.id, "error": str(exc)},
        )
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
