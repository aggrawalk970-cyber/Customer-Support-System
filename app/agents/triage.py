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

    # .with_structured_output() uses function-calling to guarantee schema compliance
    llm = get_llm()
    structured_llm = llm.with_structured_output(TriageDecision)

    llm_messages = [SystemMessage(content=TRIAGE_SYSTEM_PROMPT)] + list(messages)

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
