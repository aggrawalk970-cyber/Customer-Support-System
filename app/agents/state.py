from typing import Annotated, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievalSource(TypedDict):
    """Structured record of a single RAG retrieval event.

    Captured for every search_knowledge_base call so we can post-mortem
    whether a wrong answer was a Retrieval bug or a Generation bug.
    """
    query: str                    # The exact query string the agent sent to the vector store
    top_chunks: list[str]         # Top-K retrieved text chunks
    scores: list[float]           # Cosine similarity scores for each chunk (0.0 – 1.0)
    timestamp: str                # ISO-8601 timestamp of the retrieval


class ToolCallRecord(TypedDict):
    """Structured record of a single tool invocation."""
    tool_name: str                # Name of the tool (e.g. get_order_status)
    inputs: dict                  # Input arguments passed to the tool
    output: str                   # Raw string output returned by the tool
    timestamp: str                # ISO-8601 timestamp


class AgentState(TypedDict):
    # ── Short-term memory ───────────────────────────────────────────────
    # LangGraph's add_messages reducer appends new messages automatically
    messages: Annotated[list[BaseMessage], add_messages]

    # ── User metadata ────────────────────────────────────────────────────
    user_email: str

    # ── Routing metadata ─────────────────────────────────────────────────
    # Classified intent: billing | technical | general | escalation
    intent: str

    # Routing destination: billing_specialist | technical_specialist |
    # general_specialist | escalation | end
    next_action: str

    # Which agent node is currently executing (for UI display)
    active_agent: str

    # ── Escalation ───────────────────────────────────────────────────────
    escalation_summary: str
    approved_by_human: bool

    # ── Production Observability ─────────────────────────────────────────
    # High-level logs (node transitions, decisions)
    logs: list[str]

    # Detailed tool call records — used to diagnose generation failures
    tool_calls_log: list[ToolCallRecord]

    # RAG retrieval records — used to diagnose retrieval failures
    retrieval_sources: list[RetrievalSource]
