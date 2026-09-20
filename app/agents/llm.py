"""
app/agents/llm.py
==================
Backward-compatibility shim.

LLM client has been moved to app.services.gemini_client.
This re-exports get_llm() so existing agent imports don't break.
"""
from app.services.gemini_client import get_llm  # noqa: F401
