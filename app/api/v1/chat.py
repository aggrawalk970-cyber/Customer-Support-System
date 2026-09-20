"""
app/api/v1/chat.py
===================
POST /chat     — Synchronous agent invocation (quick queries, < 10s)
GET  /history  — Fetch a user's ticket + order history

For long-running multi-step agent tasks, use POST /tasks instead.

Rate limits (per IP, enforced via slowapi + Redis):
  POST /chat      → 20 requests / minute
  GET  /history   → 60 requests / minute
"""
from typing import List

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agents.graph import compiled_graph
from app.schemas import (
    ChatRequest, ChatResponse, MessageSchema,
    DebugInfo, ToolCallRecordSchema, RetrievalSourceSchema,
)
from app.core.logging import get_logger
from app.core.rate_limit import limiter

router = APIRouter()
logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_messages(messages) -> List[MessageSchema]:
    """Convert LangChain message objects to serialisable MessageSchema list."""
    formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = "system"

        content = m.content
        if role == "tool":
            content = f"[Tool: {getattr(m, 'name', 'Tool')}] {m.content}"
        elif role == "assistant" and getattr(m, "tool_calls", None):
            calls = [f"{tc['name']}({tc['args']})" for tc in m.tool_calls]
            content = f"[ReAct → {', '.join(calls)}] {m.content}"

        formatted.append(MessageSchema(role=role, content=str(content)))
    return formatted


def build_debug_info(values: dict) -> DebugInfo:
    """Extract observability data from the LangGraph state dict."""
    trajectory = [
        log for log in (values.get("logs") or [])
        if log.startswith("[Triage]") or "specialist" in log.lower() or "[Escalation" in log
    ]
    raw_tool_calls = values.get("tool_calls_log") or []
    tool_calls = [
        ToolCallRecordSchema(
            tool_name=tc.get("tool_name", "unknown"),
            inputs=tc.get("inputs", {}),
            output=tc.get("output", ""),
            timestamp=tc.get("timestamp", ""),
        )
        for tc in raw_tool_calls
    ]
    raw_retrievals = values.get("retrieval_sources") or []
    retrieval_sources = [
        RetrievalSourceSchema(
            query=r.get("query", ""),
            top_chunks=r.get("top_chunks", []),
            scores=r.get("scores", []),
            timestamp=r.get("timestamp", ""),
        )
        for r in raw_retrievals
    ]
    return DebugInfo(
        agent_trajectory=trajectory,
        tool_calls=tool_calls,
        retrieval_sources=retrieval_sources,
        active_agent=values.get("active_agent", ""),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Send a message to the support system")
@limiter.limit("20/minute")
def chat_endpoint(request: Request, body: ChatRequest):
    """
    Main chat endpoint — drives the multi-agent conversation synchronously.

    Flow:
      1. New thread → initialise state and invoke graph from START.
      2. Existing thread → append user message and continue from checkpoint.
      3. If needs_approval=True → conversation paused at HITL interrupt.
         Call POST /approvals/{thread_id}/resolve to continue.

    Thread ID: Generate a UUID client-side per session. This is the LangGraph
    checkpoint key — losing it means losing conversation state.

    Rate limited: 20 requests/minute per IP.
    """
    config = {"configurable": {"thread_id": body.thread_id}}

    logger.info(
        "Chat request",
        extra={"thread_id": body.thread_id, "user_email": body.user_email},
    )

    current_state = compiled_graph.get_state(config)

    if not current_state.values:
        # New conversation — initialise full state
        inputs = {
            "messages": [HumanMessage(content=body.message)],
            "user_email": body.user_email or "",
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
        # Block new messages while waiting for approval
        if current_state.next:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This conversation is paused waiting for human supervisor approval. "
                    "Call POST /approvals/{thread_id}/resolve first."
                ),
            )
        updates = {"messages": [HumanMessage(content=body.message)]}
        if body.user_email:
            updates["user_email"] = body.user_email
        compiled_graph.update_state(config, updates)
        compiled_graph.invoke(None, config)

    final_state = compiled_graph.get_state(config)
    values = final_state.values
    needs_approval = len(final_state.next) > 0

    return ChatResponse(
        messages=format_messages(values.get("messages", [])),
        intent=values.get("intent", "general"),
        next_action=values.get("next_action", "end"),
        active_agent=values.get("active_agent", ""),
        needs_approval=needs_approval,
        logs=values.get("logs", []),
        debug_info=build_debug_info(values),
    )


@router.get("/history/{email}", summary="Get ticket and order history for a user")
@limiter.limit("60/minute")
def get_history_endpoint(request: Request, email: str):
    """
    Fetch all past support tickets and orders for a given email address.
    Used to demonstrate long-term memory across sessions.

    Rate limited: 60 requests/minute per IP.
    """
    from sqlalchemy.orm import Session
    from app.db.session import SessionLocal
    from app.db.models import Ticket, Order

    db: Session = SessionLocal()
    try:
        tickets = (
            db.query(Ticket)
            .filter(Ticket.user_email == email)
            .order_by(Ticket.created_at.desc())
            .all()
        )
        orders = db.query(Order).filter(Order.user_email == email).all()
    finally:
        db.close()

    return {
        "email": email,
        "past_tickets": [
            {
                "id": t.id,
                "issue": t.issue,
                "status": t.status,
                "summary": t.conversation_summary,
                "created_at": t.created_at,
            }
            for t in tickets
        ],
        "orders": [
            {
                "order_id": o.order_id,
                "item_name": o.item_name,
                "price": o.price,
                "status": o.status,
                "delivery_date": o.delivery_date,
            }
            for o in orders
        ],
    }
