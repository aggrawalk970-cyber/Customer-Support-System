"""
Partitioned Specialist Agents
─────────────────────────────
Three distinct ReAct agents are compiled ONCE at module load time (not per-request).
Each agent is scoped to its domain with purpose-built tools and a focused system prompt.

Interview Talking Point:
  Compiling react agents once at startup rather than inside each node call avoids
  repeated graph-compilation overhead per request — important for latency at scale.
"""
import datetime
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from app.agents.state import AgentState, ToolCallRecord
from app.agents.llm import get_llm
from app.agents.tools import (
    get_order_status,
    create_ticket,
    get_user_tickets,
    search_knowledge_base,
    send_email,
    get_last_retrieval,
)

# ── System Prompts ────────────────────────────────────────────────────────────

BILLING_PROMPT = """You are the Billing Specialist Agent for a Customer Support System.
You exclusively handle payment, refund, subscription, and billing-related queries.

Your available tools:
- get_order_status: Retrieve details of an order by order ID.
- get_user_tickets: Fetch a customer's past billing tickets by email (long-term memory).
- send_email: Send confirmation emails to customers.

Instructions:
- ALWAYS check ticket history first (get_user_tickets) if you have the customer's email.
- For refund requests, check order status first then refer to the 14-day refund policy.
- If you cannot resolve the issue, say "I am routing you to the Escalation Agent for further assistance."
"""

TECHNICAL_PROMPT = """You are the Technical Specialist Agent for a Customer Support System.
You exclusively handle network, router, software, and device-related technical issues.

Your available tools:
- search_knowledge_base: Search the company knowledge base for troubleshooting guides.
- get_user_tickets: Fetch a customer's past technical tickets by email (long-term memory).

Instructions:
- ALWAYS search the knowledge base to find troubleshooting steps relevant to the issue.
- Check ticket history (get_user_tickets) if you have the customer's email — they may have reported this before.
- Provide step-by-step solutions. If issues persist beyond what the KB covers, say "I am routing you to the Escalation Agent for further assistance."
"""

GENERAL_PROMPT = """You are the General Support Specialist Agent for a Customer Support System.
You handle FAQs, shipping information, contact details, and general inquiries.

Your available tools:
- search_knowledge_base: Search the company knowledge base for policies and FAQ articles.
- send_email: Send informational emails to customers.

Instructions:
- Search the knowledge base to answer every general query accurately.
- If a customer is asking about shipping or delivery status, guide them to check their tracking ID.
- If you cannot find the answer, say "I am routing you to the Escalation Agent for further assistance."
"""

# ── Pre-compiled ReAct Agents (compiled once at module load for performance) ──
_llm = get_llm()

billing_react_agent = create_react_agent(
    model=_llm,
    tools=[get_order_status, get_user_tickets, send_email],
    prompt=BILLING_PROMPT,
)

technical_react_agent = create_react_agent(
    model=_llm,
    tools=[search_knowledge_base, get_user_tickets],
    prompt=TECHNICAL_PROMPT,
)

general_react_agent = create_react_agent(
    model=_llm,
    tools=[search_knowledge_base, send_email],
    prompt=GENERAL_PROMPT,
)


# ── Shared Node Runner ────────────────────────────────────────────────────────

def _run_specialist(
    state: AgentState,
    agent,
    agent_name: str,
) -> dict:
    """Shared execution logic for all three specialist nodes.

    Runs the pre-compiled ReAct agent, captures tool calls into structured
    ToolCallRecord logs, captures RAG retrieval metadata, and checks if the
    final response triggers escalation.
    """
    messages = state["messages"]
    logs = state.get("logs", []) or []
    tool_calls_log = list(state.get("tool_calls_log", []) or [])
    retrieval_sources = list(state.get("retrieval_sources", []) or [])
    email = state.get("user_email", "")

    # Inject the user's email into the context so the agent can use it for tools
    agent_messages = list(messages)
    if email:
        from langchain_core.messages import SystemMessage
        agent_messages.insert(0, SystemMessage(content=f"The current customer's email is: {email}"))

    # Run the ReAct sub-agent on the conversation history
    response = agent.invoke({"messages": agent_messages})
    new_messages = response["messages"]

    # Only keep messages the agent appended (not the input messages)
    newly_added = new_messages[len(messages):]

    # ── Capture Tool Call Records for observability ───────────────────────────
    for msg in newly_added:
        # AIMessage with tool_calls: record the tool invocation
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append(ToolCallRecord(
                    tool_name=tc["name"],
                    inputs=tc["args"],
                    output="[pending]",
                    timestamp=datetime.datetime.utcnow().isoformat(),
                ))
        # ToolMessage: backfill the output into the last matching pending record
        if hasattr(msg, "name") and hasattr(msg, "content") and msg.__class__.__name__ == "ToolMessage":
            for record in reversed(tool_calls_log):
                if record["tool_name"] == getattr(msg, "name", "") and record["output"] == "[pending]":
                    record["output"] = str(msg.content)[:500]  # Truncate very long tool outputs
                    break

    # ── Capture RAG Retrieval Sources ─────────────────────────────────────────
    # After agent runs, check if search_knowledge_base was called.
    # get_last_retrieval() returns metadata about the last FAISS search:
    # {query, top_chunks, scores} — used to debug retrieval vs generation failures.
    last_retrieval = get_last_retrieval()
    if last_retrieval and last_retrieval.get("query"):
        from app.agents.state import RetrievalSource
        retrieval_sources.append(RetrievalSource(
            query=last_retrieval["query"],
            top_chunks=last_retrieval.get("top_chunks", []),
            scores=last_retrieval.get("scores", []),
            timestamp=last_retrieval.get("timestamp", datetime.datetime.utcnow().isoformat() + "Z"),
        ))

    # Check if the specialist is signalling it cannot resolve → escalate
    final_content_raw = new_messages[-1].content if new_messages else ""
    if isinstance(final_content_raw, list):
        final_content = " ".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in final_content_raw]).lower()
    else:
        final_content = str(final_content_raw).lower()
    next_action = "end"
    if "escalat" in final_content or "human representative" in final_content:
        next_action = "escalation"

    log_msg = (
        f"[{agent_name}]: Completed. next_action='{next_action}'. "
        f"tool_calls={len([m for m in newly_added if hasattr(m, 'tool_calls') and m.tool_calls])} | "
        f"rag_searches={len([r for r in retrieval_sources])}"
    )
    print(log_msg)

    return {
        "messages": newly_added,
        "next_action": next_action,
        "active_agent": agent_name,
        "tool_calls_log": tool_calls_log,
        "retrieval_sources": retrieval_sources,
        "logs": logs + [log_msg],
    }


# ── Node Functions (registered into the LangGraph graph) ─────────────────────

def billing_node(state: AgentState) -> dict:
    """Billing Specialist node — handles payment and refund queries."""
    return _run_specialist(state, billing_react_agent, "billing_specialist")


def technical_node(state: AgentState) -> dict:
    """Technical Specialist node — handles network, router, and software issues."""
    return _run_specialist(state, technical_react_agent, "technical_specialist")


def general_node(state: AgentState) -> dict:
    """General Specialist node — handles FAQs, shipping, and contact queries."""
    return _run_specialist(state, general_react_agent, "general_specialist")
