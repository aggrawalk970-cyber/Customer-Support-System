"""
API Schemas
===========
Defines the Pydantic request/response models for all FastAPI endpoints.

The DebugInfo schema is the core observability payload.
In production, this object lets you audit every agent turn without
opening LangSmith — you can include it in your internal dashboards or
expose it only to internal users via a feature flag.
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Unique conversation thread ID. Generate a UUID per session.")
    message: str = Field(..., description="The user's latest message.")
    user_email: Optional[str] = Field(None, description="Optional user email for long-term memory lookup.")


class ApproveRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID of the conversation awaiting approval.")
    approve: bool = Field(..., description="True to approve escalation; False to reject.")
    feedback: Optional[str] = Field(None, description="Rejection reason shown to the user if approve=False.")


# ── Response Schemas ──────────────────────────────────────────────────────────

class MessageSchema(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', 'tool', or 'system'.")
    content: str = Field(..., description="Message content.")


class RetrievalSourceSchema(BaseModel):
    """Exposes RAG retrieval metadata for each search call.

    Use this to debug the RAG Debugging Quadrant:
    - Low scores (<0.4) → RETRIEVAL failure (bad query or missing docs)
    - Good scores but wrong answer → GENERATION failure (hallucination)
    """
    query: str
    top_chunks: List[str]
    scores: List[float]
    timestamp: str


class ToolCallRecordSchema(BaseModel):
    """Full audit record of a tool invocation.

    Inspect 'inputs' and 'output' together to verify the tool returned
    what the agent needed — isolates tool bugs from LLM bugs.
    """
    tool_name: str
    inputs: Any
    output: str
    timestamp: str


class DebugInfo(BaseModel):
    """Production observability payload returned on every API response.

    This is the structured audit trail for a single conversation turn:
    - agent_trajectory: which nodes the graph visited
    - tool_calls: every tool invoked with inputs + outputs
    - retrieval_sources: every RAG search with similarity scores
    - active_agent: the last agent that ran

    Interview talking point:
    "We embedded observability into the API response so we could debug
    failures without needing access to production logs. A wrong-answer
    complaint is debugged in <5 minutes by inspecting debug_info."
    """
    agent_trajectory: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCallRecordSchema] = Field(default_factory=list)
    retrieval_sources: List[RetrievalSourceSchema] = Field(default_factory=list)
    active_agent: Optional[str] = None


class ChatResponse(BaseModel):
    messages: List[MessageSchema]
    intent: str = ""
    next_action: str = ""
    active_agent: str = ""
    needs_approval: bool = False
    logs: List[str] = Field(default_factory=list)
    debug_info: DebugInfo = Field(default_factory=DebugInfo)
