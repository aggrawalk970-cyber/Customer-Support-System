import re
from fastapi import HTTPException, status
from app.schemas import ChatRequest

def verify_chat_input(body: ChatRequest):
    """
    Input guardrail dependency to protect against prompt injection and abuse.
    """
    msg = body.message.strip()
    
    # 1. Max length check (2000 chars)
    if len(msg) > 2000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message exceeds maximum allowed length of 2000 characters."
        )
        
    # 2. Prompt injection patterns
    lower_msg = msg.lower()
    injection_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "you are a developer",
        "you are a helpful assistant",
        "forget all instructions",
        "bypass",
        "disregard previous",
    ]
    
    for pattern in injection_patterns:
        if pattern in lower_msg:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Input flagged by security guardrails."
            )
            
    # 3. Very basic language check (reject non-ascii/latin if needed)
    # This is a very rudimentary check for English/Latin characters
    # If the message is purely non-latin (e.g. all Cyrillic or Chinese), reject it.
    latin_chars = len(re.findall(r'[a-zA-Z]', msg))
    if len(msg) > 5 and latin_chars == 0:
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported language. Please use English."
        )
         
    return body
