"""
app/db/models.py
=================
SQLAlchemy ORM models — mapped to PostgreSQL tables.

Tables:
  - tickets    : Support tickets created by the escalation agent
  - orders     : Mock e-commerce orders (seed data for demos)
  - approvals  : Human-in-the-loop approval records (new — Stage 2/3)
"""
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, func
from app.db.session import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_email = Column(String(255), index=True, nullable=False)
    issue = Column(Text, nullable=False)
    status = Column(String(50), default="open")           # open | resolved | escalated
    conversation_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(100), primary_key=True, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    item_name = Column(String(255), nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String(50), nullable=False)           # processing | shipped | delivered | canceled
    delivery_date = Column(String(100), nullable=True)


class Approval(Base):
    """
    Tracks human-in-the-loop approval requests.

    When the graph pauses at the escalation interrupt, a record is written here.
    The supervisor calls POST /approvals/{id}/resolve to approve or reject.
    The thread_id links back to the LangGraph checkpoint.

    Interview talking point:
      This table is the audit log for every HITL decision —
      who approved what, when, with what feedback. Essential for compliance.
    """
    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    thread_id = Column(String(255), index=True, nullable=False)
    user_email = Column(String(255), nullable=True)
    status = Column(String(50), default="pending")        # pending | approved | rejected
    feedback = Column(Text, nullable=True)                # Supervisor's rejection reason
    approved_by = Column(String(255), nullable=True)      # Supervisor identifier (future auth)
    created_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)
