import json
from langchain_core.messages import SystemMessage
from app.agents.state import AgentState
from app.agents.llm import get_llm

TRIAGE_SYSTEM_PROMPT = """You are the Triage Agent for a Customer Support System.
Your job is to analyze the user's latest query along with the conversation history and classify their intent into one of these categories:
- 'billing': pricing, refund requests, payment issues, subscription disputes.
- 'technical': network issues, router reset, software bugs, app update glitches.
- 'general': shipping times, contact information, hours of operation, basic FAQs.
- 'escalation': explicitly asking for a human representative, expressing extreme frustration, or requesting supervisor contact.

You must also scan the conversation history to extract the customer's email address if they have provided it.

Return your classification strictly in JSON format with the following keys:
{
  "intent": "billing" | "technical" | "general" | "escalation",
  "email": "extracted_email_or_empty_string",
  "reason": "brief reason for classification"
}
Do not write any markdown formatting (like ```json or ```). Return ONLY the raw JSON string."""

def triage_node(state: AgentState) -> dict:
    messages = state["messages"]
    logs = state.get("logs", []) or []
    
    llm = get_llm()
    llm_messages = [SystemMessage(content=TRIAGE_SYSTEM_PROMPT)] + messages
    
    # Run LLM classification
    response = llm.invoke(llm_messages)
    content = response.content.strip()
    
    # Clean possible markdown wrapping if the LLM outputs it
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
        
    intent = "general"
    email = state.get("user_email", "")
    reason = "Fallback parse logic used"
    
    try:
        data = json.loads(content)
        intent = data.get("intent", "general")
        reason = data.get("reason", "Parsed successfully")
        new_email = data.get("email", "")
        if new_email:
            email = new_email
    except Exception:
        # Robust fallback parsing (essential for mock LLMs or parser failures)
        last_msg = messages[-1].content.lower() if messages else ""
        if "billing" in last_msg or "refund" in last_msg or "charge" in last_msg:
            intent = "billing"
        elif "tech" in last_msg or "router" in last_msg or "bug" in last_msg or "reset" in last_msg:
            intent = "technical"
        elif "escalate" in last_msg or "human" in last_msg or "manager" in last_msg or "agent" in last_msg:
            intent = "escalation"
        else:
            intent = "general"
            
    # Routing decision
    if intent == "escalation":
        next_action = "escalation"
    else:
        next_action = "specialist"
        
    log_msg = f"[Triage Node]: Classified intent as '{intent}' (reason: {reason}). Email: '{email}'."
    print(log_msg)
    
    return {
        "intent": intent,
        "user_email": email,
        "next_action": next_action,
        "logs": logs + [log_msg]
    }
