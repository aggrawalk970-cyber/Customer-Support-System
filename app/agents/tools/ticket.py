from langchain_core.tools import tool
from app.db.connection import SessionLocal
from app.db.models import Ticket

@tool
def create_ticket(user_email: str, issue: str) -> str:
    """Create a support ticket in the database for escalation or manual review.
    
    Args:
        user_email: The customer's email address.
        issue: A description of the user's problem.
    """
    db = SessionLocal()
    try:
        new_ticket = Ticket(
            user_email=user_email,
            issue=issue,
            status="open"
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)
        return f"Success! Created Ticket ID {new_ticket.id} for user {user_email} with status 'open'."
    except Exception as e:
        return f"Error creating ticket: {str(e)}"
    finally:
        db.close()

@tool
def get_user_tickets(user_email: str) -> str:
    """Fetch the long-term ticket history for a user using their email address.
    This lets the system know if they have reported this or other issues in the past.
    
    Args:
        user_email: The customer's email address.
    """
    db = SessionLocal()
    try:
        tickets = db.query(Ticket).filter(Ticket.user_email == user_email).order_by(Ticket.created_at.desc()).all()
        if not tickets:
            return f"No prior support tickets found for user {user_email}. This is a new customer."
        
        history = [f"Found {len(tickets)} past ticket(s) for user {user_email}:"]
        for t in tickets:
            history.append(
                f"- Ticket ID {t.id} (Status: {t.status}): '{t.issue}' "
                f"created on {t.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        return "\n".join(history)
    except Exception as e:
        return f"Error fetching ticket history: {str(e)}"
    finally:
        db.close()
