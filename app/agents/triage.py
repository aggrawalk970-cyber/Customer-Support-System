from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from app.agents.state import AgentState
from app.agents.llm import get_llm

# ── Structured Output Schema ──────────────────────────────────────────────────

class TriageDecision(BaseModel):
    """Structured classification output produced by the Triage LLM call.

    Using .with_structured_output() eliminates all fragile regex/JSON string
    parsing. LangChain enforces this schema at the model call boundary via
    function-calling / tool-use under the hood.
    """
    intent: Literal["billing", "technical", "general", "escalation"] = Field(
        description=(
            "Customer's classified intent: "
            "'billing' for payment/refund/subscription issues, "
            "'technical' for network/software/device issues, "
            "'general' for FAQs/shipping/contact info, "
            "'escalation' if the customer demands a human or is highly frustrated."
        )
    )
    email: str = Field(
        default="",
        description="Customer's email address if explicitly mentioned in the conversation, otherwise empty string."
    )
    reason: str = Field(
        description="Brief one-sentence explanation for the classification decision."
    )


# ── System Prompt ─────────────────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = """You are the Triage Agent for a Customer Support System.

Analyze the user's latest query along with the conversation history.
Classify their intent into exactly one of these categories:
- 'billing': pricing, refund requests, payment issues, subscription disputes.
- 'technical': network issues, router reset, software bugs, app update glitches.
- 'general': shipping times, contact information, hours of operation, basic FAQs.
- 'escalation': explicitly asking for a human representative, expressing extreme frustration, requesting a supervisor.

Also extract the customer's email address if they have provided it anywhere in the conversation."""


# ── Triage Node ───────────────────────────────────────────────────────────────

def triage_node(state: AgentState) -> dict:
    """Classify the user's intent using structured output (no JSON parsing needed)."""
    messages = state["messages"]
    logs = state.get("logs", []) or []

    # Fetch long-term memory (Episodic + Semantic)
    history_context = ""
    email_for_lookup = state.get("user_email", "")
    if email_for_lookup:
        from app.db.session import SessionLocal
        from app.db.models import Ticket, UserProfile
        db = SessionLocal()
        try:
            profile = db.query(UserProfile).filter(UserProfile.user_email == email_for_lookup).first()
            tickets = db.query(Ticket).filter(Ticket.user_email == email_for_lookup).order_by(Ticket.created_at.desc()).limit(3).all()
            
            context_parts = []
            if profile and profile.context_summary:
                context_parts.append(f"User Profile/Preferences: {profile.context_summary}")
                
            if tickets:
                ticket_details = "\n".join([f"- Ticket #{t.id} ({t.status}): {t.issue}" for t in tickets])
                context_parts.append(f"Recent Tickets:\n{ticket_details}")
                
            if context_parts:
                history_context = "\n\n--- LONG-TERM MEMORY CONTEXT ---\n" + "\n\n".join(context_parts) + "\n--------------------------------\nUse this context to inform your classification and responses if the user's issue relates to past tickets."
        except Exception as db_e:
            logs.append(f"[Triage] Error fetching history: {db_e}")
        finally:
            db.close()

    # .with_structured_output() uses function-calling to guarantee schema compliance
    llm = get_llm()
    structured_llm = llm.with_structured_output(TriageDecision)

    full_system_prompt = TRIAGE_SYSTEM_PROMPT + history_context
    llm_messages = [SystemMessage(content=full_system_prompt)] + list(messages)

    # Run structured classification — returns a TriageDecision Pydantic object
    try:
        decision: TriageDecision = structured_llm.invoke(llm_messages)
        intent = decision.intent
        reason = decision.reason
        email = decision.email or state.get("user_email", "")
    except Exception as e:
        # Graceful fallback if the LLM doesn't support structured output (e.g. FakeLLM)
        intent = "general"
        reason = f"Structured output failed ({e}), defaulting to general."
        email = state.get("user_email", "")
        last_msg = messages[-1].content.lower() if messages else ""
        if "billing" in last_msg or "refund" in last_msg or "charge" in last_msg or "payment" in last_msg:
            intent = "billing"
        elif "tech" in last_msg or "router" in last_msg or "bug" in last_msg or "reset" in last_msg or "network" in last_msg:
            intent = "technical"
        elif "escalate" in last_msg or "human" in last_msg or "manager" in last_msg or "supervisor" in last_msg:
            intent = "escalation"

    # Determine routing: escalation goes directly, all else route to relevant specialist
    next_action = "escalation" if intent == "escalation" else f"{intent}_specialist"

    log_msg = f"[Triage Node]: Classified intent='{intent}' → next='{next_action}' | Reason: {reason} | Email: '{email}'"
    print(log_msg)

    return {
        "intent": intent,
        "user_email": email,
        "next_action": next_action,
        "active_agent": "triage",
        "logs": logs + [log_msg],
    }
