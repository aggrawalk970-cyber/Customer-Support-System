from sqlalchemy import Column, Integer, String, Text, Float, DateTime, func
from app.db.connection import Base

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_email = Column(String(255), index=True, nullable=False)
    issue = Column(Text, nullable=False)
    status = Column(String(50), default="open")  # open, resolved, escalated
    conversation_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class Order(Base):
    __tablename__ = "orders"
    
    order_id = Column(String(100), primary_key=True, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    item_name = Column(String(255), nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String(50), nullable=False)  # processing, shipped, delivered, canceled
    delivery_date = Column(String(100), nullable=True)
