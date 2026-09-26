from langchain_core.messages import SystemMessage, AIMessage
from app.agents.state import AgentState
from app.agents.llm import get_llm
from app.db.session import SessionLocal
from app.db.models import UserProfile

PROFILE_SUMMARIZER_PROMPT = """You are an AI tasked with maintaining a customer profile.
Review the conversation history and generate a single-sentence summary of the user's preferences, setup, or the issue they just had resolved.
Example: "Customer has a Fiber connection and previously had billing issues."
Return only the 1-line summary."""

def summarize_conversation_node(state: AgentState) -> dict:
    messages = state["messages"]
    email = state.get("user_email")
    logs = state.get("logs", []) or []
    
    if not email:
        return {"next_action": "end"}
        
    llm = get_llm()
    summary_messages = [SystemMessage(content=PROFILE_SUMMARIZER_PROMPT)] + list(messages)
    
    try:
        response = llm.invoke(summary_messages)
        content_raw = response.content
        if isinstance(content_raw, list):
            summary_text = " ".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content_raw]).strip()
        else:
            summary_text = str(content_raw).strip()
            
        db = SessionLocal()
        try:
            profile = db.query(UserProfile).filter(UserProfile.user_email == email).first()
            if not profile:
                profile = UserProfile(user_email=email, context_summary=summary_text)
                db.add(profile)
            else:
                # Append or replace? We can just append to keep it brief but rich, or let LLM rewrite it if we feed the old one.
                # Since prompt doesn't have old one, we just append or replace. Overwriting is simpler for a 1-liner.
                profile.context_summary = summary_text
            db.commit()
            log_msg = f"[Summarize Node] Updated user profile for {email} with summary."
            print(log_msg)
            logs.append(log_msg)
        finally:
            db.close()
            
    except Exception as e:
        log_msg = f"[Summarize Node] Error generating summary: {e}"
        print(log_msg)
        logs.append(log_msg)

    return {"next_action": "end", "logs": logs}
