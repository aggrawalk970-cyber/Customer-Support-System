"""
app/agents/graph.py
====================
LangGraph StateGraph — wires all agent nodes and compiles the graph.

Checkpointer: PostgreSQL (via langgraph-checkpoint-postgres)
  - Agent conversation state persists across app restarts
  - Multiple workers can safely share the same checkpoint store
  - Falls back to SQLite if Postgres is not reachable (local dev)

interrupt_before=["escalation"]:
  The graph pauses BEFORE running the escalation node.
  The conversation thread stays in the checkpointer.
  A human supervisor calls POST /approvals/{thread_id}/resolve to resume.
  This is the core Human-in-the-Loop (HITL) pattern.
"""
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.triage import triage_node
from app.agents.specialist import billing_node, technical_node, general_node
from app.agents.escalation import escalation_node
from app.db.checkpointer import get_checkpointer


from app.agents.memory import trim_memory_node

from app.agents.summarize import summarize_conversation_node

# ── Router Functions ──────────────────────────────────────────────────────────

def route_after_triage(state: AgentState) -> str:
    """Route to the correct specialist based on classified intent."""
    intent = state.get("intent", "general")
    if intent == "escalation":
        return "escalation"
    elif intent == "billing":
        return "billing_specialist"
    elif intent == "technical":
        return "technical_specialist"
    else:
        return "general_specialist"


def route_after_specialist(state: AgentState) -> str:
    """After a specialist node, either escalate or end the conversation."""
    if state.get("next_action") == "escalation":
        return "escalation"
    return "summarize_conversation"


# ── Graph Construction ────────────────────────────────────────────────────────

# 1. Instantiate the State Graph
workflow = StateGraph(AgentState)

# 2. Register all agent nodes
workflow.add_node("trim_memory", trim_memory_node)
workflow.add_node("triage", triage_node)
workflow.add_node("billing_specialist", billing_node)
workflow.add_node("technical_specialist", technical_node)
workflow.add_node("general_specialist", general_node)
workflow.add_node("escalation", escalation_node)
workflow.add_node("summarize_conversation", summarize_conversation_node)

# 3. Configure flow: entry → trim_memory → triage → conditional specialist routing
workflow.set_entry_point("trim_memory")
workflow.add_edge("trim_memory", "triage")

workflow.add_conditional_edges(
    "triage",
    route_after_triage,
    {
        "billing_specialist": "billing_specialist",
        "technical_specialist": "technical_specialist",
        "general_specialist": "general_specialist",
        "escalation": "escalation",
    }
)

# Each specialist routes to escalation or summarize_conversation
for specialist in ["billing_specialist", "technical_specialist", "general_specialist"]:
    workflow.add_conditional_edges(
        specialist,
        route_after_specialist,
        {
            "escalation": "escalation",
            "summarize_conversation": "summarize_conversation",
        }
    )

workflow.add_edge("escalation", END)
workflow.add_edge("summarize_conversation", END)

# ── Checkpointer: PostgreSQL (persists across app restarts + multi-worker safe) ──
_checkpointer = get_checkpointer()

compiled_graph = workflow.compile(
    checkpointer=_checkpointer,
    interrupt_before=["escalation"],  # HITL: pause before escalating
)
