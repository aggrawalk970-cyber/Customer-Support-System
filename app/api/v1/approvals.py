"""
app/api/v1/approvals.py
========================
Human-in-the-Loop approval endpoints.

POST /approvals/{thread_id}/resolve  → Supervisor approves or rejects an escalation
GET  /approvals                       → List pending approvals (for supervisor dashboard)

When does an approval get created?
  When the LangGraph graph hits the escalation node and `interrupt_before=["escalation"]`
  is set, the graph pauses. The chat endpoint returns needs_approval=True.
  The supervisor is notified (e.g. via email/Slack in production) and calls this endpoint
  to resume or reject the escalation.

Interview talking point:
  This is the core of "autonomous but safe" AI design.
  The agent can reason and act, but irreversible or high-impact actions
  (escalating a customer, issuing a refund) require human sign-off.
"""
import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.graph import compiled_graph
from app.api.v1.chat import format_messages, build_debug_info
from app.db.session import SessionLocal
from app.db.models import Approval
from app.schemas import ChatResponse
from app.core.logging import get_logger
from app.core.rate_limit import limiter

router = APIRouter()
logger = get_logger(__name__)


# ── Request Schemas ───────────────────────────────────────────────────────────

class ResolveApprovalRequest(BaseModel):
    approve: bool
    feedback: Optional[str] = None      # Required if approve=False
    approved_by: Optional[str] = None   # Supervisor identifier (future auth)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/approvals/{thread_id}/resolve",
    response_model=ChatResponse,
    summary="Approve or reject a pending escalation",
)
@limiter.limit("30/minute")
def resolve_approval(request: Request, thread_id: str, body: ResolveApprovalRequest):
    """
    Resume a paused LangGraph conversation after human review.

    - approve=True  → Escalation proceeds (ticket created, supervisor notified)
    - approve=False → Rejection message sent to customer, conversation continues

    The thread_id is the LangGraph checkpoint key returned by POST /chat
    when needs_approval=True.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = compiled_graph.get_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    if not state.next:
        raise HTTPException(
            status_code=400,
            detail=f"Thread '{thread_id}' is not waiting for approval.",
        )

    logger.info(
        "Processing approval",
        extra={
            "thread_id": thread_id,
            "approve": body.approve,
            "approved_by": body.approved_by,
        },
    )

    # ── Update LangGraph state ────────────────────────────────────────────────
    if body.approve:
        compiled_graph.update_state(
            config,
            {"approved_by_human": True},
            as_node="triage",
        )
    else:
        feedback = body.feedback or "Escalation rejected by supervisor."
        denial_msg = AIMessage(
            content=(
                f"Your escalation request was reviewed and declined by a human supervisor. "
                f"Reason: {feedback}. We will continue assisting you here."
            )
        )
        compiled_graph.update_state(
            config,
            {"approved_by_human": False, "messages": [denial_msg]},
            as_node="triage",
        )

    # ── Write approval record to DB ───────────────────────────────────────────
    db: Session = SessionLocal()
    try:
        approval = Approval(
            thread_id=thread_id,
            user_email=state.values.get("user_email", ""),
            status="approved" if body.approve else "rejected",
            feedback=body.feedback,
            approved_by=body.approved_by,
            resolved_at=datetime.datetime.utcnow(),
        )
        db.add(approval)
        db.commit()
    except Exception as e:
        logger.warning("Failed to write approval record", extra={"error": str(e)})
    finally:
        db.close()

    # ── Resume graph execution ────────────────────────────────────────────────
    compiled_graph.invoke(None, config)

    new_state = compiled_graph.get_state(config)
    values = new_state.values
    needs_approval = len(new_state.next) > 0

    return ChatResponse(
        messages=format_messages(values.get("messages", [])),
        intent=values.get("intent", "general"),
        next_action=values.get("next_action", "end"),
        active_agent=values.get("active_agent", ""),
        needs_approval=needs_approval,
        logs=values.get("logs", []),
        debug_info=build_debug_info(values),
    )


@router.get("/approvals", summary="List all approval records")
@limiter.limit("60/minute")
def list_approvals(request: Request, status: Optional[str] = None):
    """
    List approval records — useful for a supervisor dashboard.
    Filter by status: pending | approved | rejected
    """
    db: Session = SessionLocal()
    try:
        query = db.query(Approval)
        if status:
            query = query.filter(Approval.status == status)
        approvals = query.order_by(Approval.created_at.desc()).limit(100).all()
        return {
            "approvals": [
                {
                    "id": a.id,
                    "thread_id": a.thread_id,
                    "user_email": a.user_email,
                    "status": a.status,
                    "feedback": a.feedback,
                    "approved_by": a.approved_by,
                    "created_at": a.created_at,
                    "resolved_at": a.resolved_at,
                }
                for a in approvals
            ]
        }
    finally:
        db.close()
