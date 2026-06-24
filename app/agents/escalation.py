from langchain_core.messages import AIMessage, SystemMessage
from app.agents.state import AgentState
from app.agents.llm import get_llm
from app.db.connection import SessionLocal
from app.db.models import Ticket

SUMMARIZER_PROMPT = """You are the Escalation Agent.
Analyze the conversation history between the customer and the support agents.
Create a highly concise summary of:
1. What the customer is experiencing or requesting.
2. What troubleshooting or answers were already provided by the Specialist.
3. The reason why the issue remains unresolved and needs human intervention.

Return only the summary text, starting directly with the summary content."""

def escalation_node(state: AgentState) -> dict:
    messages = state["messages"]
    logs = state.get("logs", []) or []
    email = state.get("user_email", "") or "unknown@customer.com"
    approved = state.get("approved_by_human", False)
    
    # If the escalation was rejected by human-in-the-loop supervisor
    if not approved:
        reject_msg = AIMessage(
            content=(
                "The request to escalate this conversation was reviewed and declined by a human supervisor. "
                "We will continue to assist you here. Please let me know how I can help."
            )
        )
        log_msg = "[Escalation Node]: Escalation request rejected by human supervisor."
        print(log_msg)
        return {
            "messages": [reject_msg],
            "next_action": "end",
            "logs": logs + [log_msg]
        }
        
    # If approved (or bypassed), proceed with ticket creation and handoff
    # 1. Generate summary of the conversation
    llm = get_llm()
    summary_messages = [SystemMessage(content=SUMMARIZER_PROMPT)] + messages
    summary_response = llm.invoke(summary_messages)
    summary_text = summary_response.content.strip()
    
    # 2. Simulate Human handoff and create a ticket with status 'escalated' in database
    db = SessionLocal()
    ticket_id = None
    try:
        new_ticket = Ticket(
            user_email=email,
            issue=f"ESCALATED: {messages[-1].content[:200]}...",
            status="escalated",
            conversation_summary=summary_text
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)
        ticket_id = new_ticket.id
    except Exception as e:
        print(f"Error creating escalation ticket in database: {e}")
    finally:
        db.close()
        
    # 3. Notify Supervisor (mock print alert)
    print(f"\n======== [SUPERVISOR NOTIFICATION] ========")
    print(f"ALERT: Ticket #{ticket_id} escalated to human support tier!")
    print(f"Customer Email: {email}")
    print(f"Summary of Issue: {summary_text}")
    print(f"===========================================\n")
    
    # 4. Return final handoff message to customer
    handoff_message = AIMessage(
        content=(
            f"I apologize that I couldn't resolve this issue. I have escalated this matter to a human representative. "
            f"Your support Ticket ID is #{ticket_id}. A supervisor has been notified, and they will follow up with you at "
            f"{email} as soon as possible."
        )
    )
    
    log_msg = f"[Escalation Node]: Created escalated Ticket #{ticket_id} and notified supervisor."
    print(log_msg)
    
    return {
        "messages": [handoff_message],
        "escalation_summary": summary_text,
        "next_action": "end",
        "logs": logs + [log_msg]
    }
