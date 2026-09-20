"""
app/services/gemini_client.py
==============================
LLM client factory — wraps app/agents/llm.py with service-layer logging.

This is the single place where LLM model selection happens.
Import get_llm() from here across the entire codebase.

Model strategy:
  - Triage node: could use a cheaper/faster model (e.g. gemini-flash-lite)
  - Specialist agents: use the full Gemini model for complex reasoning
  Currently both use the same model — add `model_tier` param to differentiate later.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_llm(model_tier: str = "default") -> BaseChatModel:
    """
    Return the configured LLM client.

    Priority order:
      1. Gemini (Google) — primary
      2. Groq (Llama 3) — fallback
      3. FakeListChatModel — dev/test fallback (no API keys needed)

    Args:
        model_tier: Reserved for future cost-aware model routing.
                    e.g. "triage" → cheaper model, "specialist" → full model.
    """
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Using Gemini LLM", extra={"model_tier": model_tier})
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )

    if settings.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        logger.info("Using Groq LLM", extra={"model_tier": model_tier})
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
        )

    # Dev fallback — keeps the project runnable without any API keys
    from langchain_core.language_models.fake import FakeListChatModel
    logger.warning("No LLM API keys found — using Mock LLM. Set GEMINI_API_KEY in .env.")
    return FakeListChatModel(
        responses=[
            "I am a mock assistant. Please configure GEMINI_API_KEY or GROQ_API_KEY in .env to enable AI reasoning."
        ]
    )
