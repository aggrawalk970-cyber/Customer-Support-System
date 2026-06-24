from langchain_core.language_models.chat_models import BaseChatModel
from app.config import settings

def get_llm() -> BaseChatModel:
    """Helper function to load the configured LLM client.
    Prioritizes Gemini, falls back to Groq, and uses a FakeListChatModel if no keys are found.
    """
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0
        )
    elif settings.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        # Using Llama 3 8B on Groq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0
        )
    else:
        # Fallback fake model so the project remains runnable out-of-the-box
        from langchain_core.language_models.fake import FakeListChatModel
        print("WARNING: No LLM API keys detected. Falling back to Mock LLM.")
        return FakeListChatModel(
            responses=[
                "Hello, I am a mock assistant. Please configure your GEMINI_API_KEY or GROQ_API_KEY in the .env file to enable actual AI reasoning!"
            ]
        )
