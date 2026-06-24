from .order import get_order_status
from .ticket import create_ticket, get_user_tickets
from .rag import search_knowledge_base
from .email import send_email

# Bundle all available tools for our specialist agent
all_tools = [
    get_order_status,
    create_ticket,
    get_user_tickets,
    search_knowledge_base,
    send_email
]
