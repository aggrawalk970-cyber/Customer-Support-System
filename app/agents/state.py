from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Short-term conversation history (automatically appends new messages)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # User metadata
    user_email: str
    
    # Classified intent: billing, technical, general, escalation
    intent: str
    
    # Routing destination: specialist, escalation, end, or human_approval
    next_action: str
    
    # Ticket escalation details if escalation is triggered
    escalation_summary: str
    
    # Human-in-the-loop approval flag
    approved_by_human: bool
    
    # Custom tracing logs for observability (helps demonstrate execution steps)
    logs: list[str]
