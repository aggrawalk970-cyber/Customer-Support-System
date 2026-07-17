import datetime
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.config import settings
from app.db.connection import Base, engine, get_db, SessionLocal
from app.db.models import Order, Ticket
from app.agents.graph import compiled_graph
from app.schemas import ChatRequest, ChatResponse, MessageSchema, ApproveRequest

app = FastAPI(title="Multi-Agent Customer Support System Agentic Pattern")

# Configure CORS so your future frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup hook to create database tables and seed mock data
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed Orders table if empty
        if db.query(Order).count() == 0:
            print("Seeding database with mock orders...")
            mock_orders = [
                Order(order_id="ORD12345", user_email="alice@example.com", item_name="Fiber Wifi Router", price=99.99, status="shipped", delivery_date="2026-06-28"),
                Order(order_id="ORD67890", user_email="bob@example.com", item_name="Ethernet Switch", price=49.50, status="processing", delivery_date="2026-06-30"),
                Order(order_id="ORD11223", user_email="charlie@example.com", item_name="USB-C Hub", price=25.00, status="delivered", delivery_date="2026-06-20")
            ]
            db.add_all(mock_orders)
        
        # Seed Tickets table if empty (allows demonstrating long-term memory lookup)
        if db.query(Ticket).count() == 0:
            print("Seeding database with past ticket for return user...")
            two_weeks_ago = datetime.datetime.now() - datetime.timedelta(days=14)
            mock_ticket = Ticket(
                user_email="alice@example.com",
                issue="Billing query regarding router extra warranty charge.",
                status="resolved",
                conversation_summary="Customer requested refund of extra warranty charge. Refund was manually approved and processed.",
                created_at=two_weeks_ago
            )
            db.add_all([mock_ticket])
        db.commit()
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        db.close()

def format_messages(messages) -> List[MessageSchema]:
    """Helper to convert LangChain messages to a standardized serialization format for APIs."""
    formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = "system"
        # Extract name if present to distinguish tool actions
        content = m.content
        if role == "tool":
            content = f"[Tool Return: {getattr(m, 'name', 'Tool')}] {m.content}"
        elif role == "assistant" and getattr(m, "tool_calls", None):
            calls = [f"{tc['name']}({tc['args']})" for tc in m.tool_calls]
            content = f"[Reasoning & Tool Call: {', '.join(calls)}] {m.content}"
            
        formatted.append(MessageSchema(role=role, content=str(content)))
    return formatted

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Fetch existing execution checkpoint
    current_state = compiled_graph.get_state(config)
    
    if not current_state.values:
        # Brand new conversation thread
        inputs = {
            "messages": [HumanMessage(content=request.message)],
            "user_email": request.user_email or "",
            "intent": "general",
            "next_action": "",
            "escalation_summary": "",
            "approved_by_human": False,
            "logs": []
        }
        compiled_graph.invoke(inputs, config)
    else:
        # Block user messages if current thread is waiting for supervisor approval
        if current_state.next:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This conversation is currently paused waiting for human supervisor approval. "
                    f"Please approve or deny the action at `/api/approve` first."
                )
            )
        
        # Update thread state with new user message
        updates = {"messages": [HumanMessage(content=request.message)]}
        if request.user_email:
            updates["user_email"] = request.user_email
            
        compiled_graph.update_state(config, updates)
        compiled_graph.invoke(None, config)
        
    # Get state after invoking the graph
    final_state = compiled_graph.get_state(config)
    values = final_state.values
    
    # If final_state.next is not empty, it means we hit an interrupt_before checkpoint
    needs_approval = len(final_state.next) > 0
    
    return ChatResponse(
        messages=format_messages(values.get("messages", [])),
        intent=values.get("intent", "general"),
        next_action=values.get("next_action", "end"),
        logs=values.get("logs", []),
        needs_approval=needs_approval
    )

@app.post("/api/approve", response_model=ChatResponse)
def approve_endpoint(request: ApproveRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state = compiled_graph.get_state(config)
    
    if not state.values:
        raise HTTPException(status_code=404, detail="Conversation thread not found.")
        
    if not state.next:
        raise HTTPException(status_code=400, detail="This thread is not waiting for any supervisor approvals.")
        
    # Inject approval state
    if request.approve:
        compiled_graph.update_state(
            config,
            {"approved_by_human": True},
            as_node="specialist"
        )
    else:
        feedback = request.feedback or "Rejected by supervisor."
        denial_message = AIMessage(
            content=f"Human supervisor has rejected this escalation request. Reason: {feedback}"
        )
        compiled_graph.update_state(
            config,
            {
                "approved_by_human": False,
                "messages": [denial_message]
            },
            as_node="specialist"
        )
        
    # Resume graph execution (invoking with None signals it to continue from checkpoint)
    compiled_graph.invoke(None, config)
    
    # Fetch final state post-execution
    new_state = compiled_graph.get_state(config)
    values = new_state.values
    needs_approval = len(new_state.next) > 0
    
    return ChatResponse(
        messages=format_messages(values.get("messages", [])),
        intent=values.get("intent", "general"),
        next_action=values.get("next_action", "end"),
        logs=values.get("logs", []),
        needs_approval=needs_approval
    )

@app.get("/api/history/{email}")
def get_history_endpoint(email: str, db: Session = Depends(get_db)):
    """Fetch database records directly (allows verifying ticket creation and orders)."""
    tickets = db.query(Ticket).filter(Ticket.user_email == email).order_by(Ticket.created_at.desc()).all()
    orders = db.query(Order).filter(Order.user_email == email).all()
    
    return {
        "email": email,
        "past_tickets": [
            {
                "id": t.id,
                "issue": t.issue,
                "status": t.status,
                "summary": t.conversation_summary,
                "created_at": t.created_at
            } for t in tickets
        ],
        "orders": [
            {
                "order_id": o.order_id,
                "item_name": o.item_name,
                "price": o.price,
                "status": o.status,
                "delivery_date": o.delivery_date
            } for o in orders
        ]
    }
