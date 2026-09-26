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

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agents.graph import compiled_graph
from app.schemas import (
    ChatRequest, ChatResponse, MessageSchema,
    DebugInfo, ToolCallRecordSchema, RetrievalSourceSchema,
)
from app.core.logging import get_logger
from app.core.rate_limit import limiter
from app.core.guardrails import verify_chat_input

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


def extract_and_log_token_usage(messages):
    prompt_tokens = 0
    completion_tokens = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "usage_metadata") and msg.usage_metadata:
            prompt_tokens += msg.usage_metadata.get("input_tokens", 0)
            completion_tokens += msg.usage_metadata.get("output_tokens", 0)
            
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0:
        return
        
    cost = (prompt_tokens / 1000.0 * 0.001) + (completion_tokens / 1000.0 * 0.002)
    
    logger.info("Token usage", extra={"tokens": total_tokens, "cost_usd": cost})
    
    from app.db.session import SessionLocal
    from app.db.models import ChatStats, Ticket
    db = SessionLocal()
    try:
        stat = ChatStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost
        )
        db.add(stat)
        db.commit()
    except Exception as e:
        logger.error(f"Error saving token stats: {e}")
        db.rollback()
    finally:
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Send a message to the support system")
@limiter.limit("20/minute")
def chat_endpoint(request: Request, body: ChatRequest = Depends(verify_chat_input)):
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
        start_msg_count = 0
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
        start_msg_count = len(current_state.values.get("messages", []))
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
    
    # Cost tracking for new messages
    all_msgs = values.get("messages", [])
    new_msgs = all_msgs[start_msg_count:]
    extract_and_log_token_usage(new_msgs)

    return ChatResponse(
        messages=format_messages(values.get("messages", [])),
        intent=values.get("intent", "general"),
        next_action=values.get("next_action", "end"),
        active_agent=values.get("active_agent", ""),
        needs_approval=needs_approval,
        logs=values.get("logs", []),
        debug_info=build_debug_info(values),
    )


@router.post("/chat/stream", summary="Stream response using Server-Sent Events (SSE)")
@limiter.limit("20/minute")
def chat_stream_endpoint(request: Request, body: ChatRequest = Depends(verify_chat_input)):
    """
    Stream chat response using Server-Sent Events (SSE).
    Frontend receives tokens as they generate.
    """
    config = {"configurable": {"thread_id": body.thread_id}}
    
    current_state = compiled_graph.get_state(config)
    
    if not current_state.values:
        start_msg_count = 0
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
    else:
        start_msg_count = len(current_state.values.get("messages", []))
        if current_state.next:
            raise HTTPException(
                status_code=400,
                detail="This conversation is paused waiting for human supervisor approval. Call POST /approvals/{thread_id}/resolve first."
            )
        updates = {"messages": [HumanMessage(content=body.message)]}
        if body.user_email:
            updates["user_email"] = body.user_email
        compiled_graph.update_state(config, updates)
        inputs = None

    def event_stream():
        try:
            for chunk in compiled_graph.stream(inputs, config, stream_mode="messages"):
                message, metadata = chunk
                if isinstance(message, AIMessage):
                    content = message.content
                    if isinstance(content, list):
                        text = " ".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content])
                    else:
                        text = str(content)
                        
                    if text:
                        yield f"data: {json.dumps({'content': text})}\n\n"
                        
            final_state = compiled_graph.get_state(config)
            
            # Cost tracking for new messages
            all_msgs = final_state.values.get("messages", [])
            new_msgs = all_msgs[start_msg_count:]
            extract_and_log_token_usage(new_msgs)
            
            yield f"data: {json.dumps({'status': 'completed', 'needs_approval': len(final_state.next) > 0})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
