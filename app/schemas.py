from typing import List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_email: Optional[str] = None

class MessageSchema(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    messages: List[MessageSchema]
    intent: str
    next_action: str
    logs: List[str]
    needs_approval: bool

class ApproveRequest(BaseModel):
    thread_id: str
    approve: bool
    feedback: Optional[str] = None
